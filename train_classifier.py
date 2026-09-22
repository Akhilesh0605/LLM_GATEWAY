# train_classifier.py
"""
Neural Query Complexity Classifier Training Pipeline.
Frozen all-MiniLM-L6-v2 (384-dim) + 2-layer PyTorch MLP (384 -> 64 -> 1).
Tiers:
  - SIMPLE:  score <= 3.0
  - MEDIUM:  3.0 < score < 7.0
  - COMPLEX: score >= 7.0
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    classification_report,
    confusion_matrix,
)

TIER_SIMPLE_MAX = 3.0
TIER_COMPLEX_MIN = 7.0
TIER_LABELS = ["SIMPLE", "MEDIUM", "COMPLEX"]


def score_to_tier(score: float) -> str:
    if score <= TIER_SIMPLE_MAX:
        return "SIMPLE"
    elif score < TIER_COMPLEX_MIN:
        return "MEDIUM"
    return "COMPLEX"


class ComplexityMLP(nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 64, dropout_rate: float = 0.1):
        super(ComplexityMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, scores: np.ndarray):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.scores = torch.tensor(scores, dtype=torch.float32)

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.scores[idx]


def load_dataset(file_path: str):
    texts, scores = [], []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                text = item.get("text") or item.get("query")
                score = item.get("score")
                if text and score is not None:
                    texts.append(str(text).strip())
                    scores.append(float(score))
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(texts):,} valid samples from {file_path}")
    return texts, np.array(scores, dtype=np.float32)


class WeightedHuberLoss(nn.Module):
    def __init__(self, tier_weights: dict, delta: float = 1.0):
        super(WeightedHuberLoss, self).__init__()
        self.delta = delta
        self.weights = {k: torch.tensor(v) for k, v in tier_weights.items()}

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(preds - targets)
        huber_loss = torch.where(
            diff <= self.delta,
            0.5 * (diff ** 2),
            self.delta * (diff - 0.5 * self.delta),
        )
        sample_weights = torch.ones_like(targets, device=targets.device)
        sample_weights[targets <= TIER_SIMPLE_MAX] = self.weights["SIMPLE"].to(targets.device)
        sample_weights[(targets > TIER_SIMPLE_MAX) & (targets < TIER_COMPLEX_MIN)] = self.weights["MEDIUM"].to(targets.device)
        sample_weights[targets >= TIER_COMPLEX_MIN] = self.weights["COMPLEX"].to(targets.device)

        return (huber_loss * sample_weights).mean()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using Compute Device: {device}\n")

    texts, scores = load_dataset(args.data_path)

    tiers = [score_to_tier(s) for s in scores]
    print("\n═══ CLASS DISTRIBUTION DIAGNOSIS ═══")
    for t in TIER_LABELS:
        c = tiers.count(t)
        print(f"  {t:>8}: {c:>5} samples ({c/len(tiers)*100:5.1f}%)")
    print(f"  Score Range: [{scores.min():.2f}, {scores.max():.2f}]")
    print("═" * 36 + "\n")

    print(f"[1/4] Generating 384-dim embeddings via {args.embedding_model}...")
    embedder = SentenceTransformer(args.embedding_model, device=str(device))
    embeddings = embedder.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print(f"\n[2/4] Executing Stratified Split (80/10/10)...")
    tier_indices = np.array([TIER_LABELS.index(score_to_tier(s)) for s in scores])

    X_train, X_temp, y_train, y_temp = train_test_split(
        embeddings, scores, test_size=0.20, random_state=args.seed, stratify=tier_indices
    )
    temp_tier_indices = np.array([TIER_LABELS.index(score_to_tier(s)) for s in y_temp])
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=args.seed, stratify=temp_tier_indices
    )

    print(f"  - Train Partition : {len(X_train):,} samples")
    print(f"  - Val Partition   : {len(X_val):,} samples")
    print(f"  - Test Partition  : {len(X_test):,} samples")

    train_loader = DataLoader(EmbeddingDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(EmbeddingDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(EmbeddingDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)

    train_tiers = [score_to_tier(s) for s in y_train]
    class_counts = {t: max(train_tiers.count(t), 1) for t in TIER_LABELS}
    total_train = sum(class_counts.values())

    tier_weights = {t: min(total_train / (len(TIER_LABELS) * class_counts[t]), 10.0) for t in TIER_LABELS}
    print(f"\n  Class-Weighted Loss Modifiers:")
    for t, w in tier_weights.items():
        print(f"    - {t:<8}: {w:.2f}x")

    model = ComplexityMLP(input_dim=384, hidden_dim=args.hidden_dim, dropout_rate=args.dropout).to(device)
    criterion = WeightedHuberLoss(tier_weights, delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    print(f"\n[3/4] Launching ComplexityMLP Training Loop (Max {args.epochs} epochs)...")
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(batch_y)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item() * len(batch_y)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_mae = mean_absolute_error(val_targets, val_preds)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:03d}/{args.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f}")

        if patience_counter >= args.patience:
            print(f"  [Early Stopping] Triggered at epoch {epoch}. Restoring best weights.")
            break

    print(f"\n[4/4] Evaluating Best Model on Holdout Test Partition...")
    model.load_state_dict(best_model_state)
    model.eval()

    test_preds_list, test_targets_list = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x)
            test_preds_list.extend(preds.cpu().numpy())
            test_targets_list.extend(batch_y.numpy())

    test_preds = np.array(test_preds_list)
    test_targets = np.array(test_targets_list)

    test_mae = mean_absolute_error(test_targets, test_preds)
    test_rmse = np.sqrt(mean_squared_error(test_targets, test_preds))

    true_tiers = [score_to_tier(s) for s in test_targets]
    pred_tiers = [score_to_tier(s) for s in test_preds]
    tier_acc = np.mean([1 if t == p else 0 for t, p in zip(true_tiers, pred_tiers)]) * 100.0

    print("\n" + "=" * 62)
    print("           NEURAL CLASSIFIER TEST EVALUATION REPORT       ")
    print("=" * 62)
    print(f"  Continuous MAE            : {test_mae:.4f} (1.0 - 10.0 scale)")
    print(f"  Continuous RMSE           : {test_rmse:.4f}")
    print(f"  Tier-to-Tier Accuracy     : {tier_acc:.2f}%")
    print("-" * 62)
    print("Class-Level Precision, Recall, and F1 Metrics:\n")
    print(classification_report(true_tiers, pred_tiers, labels=TIER_LABELS, digits=4, zero_division=0))
    print("-" * 62)
    print("Confusion Matrix (Rows: Actual, Cols: Predicted):")
    cm = confusion_matrix(true_tiers, pred_tiers, labels=TIER_LABELS)
    header = f"{'':12}" + "".join([f"{l:>12}" for l in TIER_LABELS])
    print(header)
    for idx, row in enumerate(cm):
        print(f"{TIER_LABELS[idx]:12}" + "".join([f"{val:>12d}" for val in row]))
    print("=" * 62)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    torch.save(best_model_state, args.output_path)
    file_size_kb = os.path.getsize(args.output_path) / 1024.0
    print(f"\n✓ Saved model binary successfully to: '{args.output_path}' ({file_size_kb:.2f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/merged_corpus.jsonl")
    parser.add_argument("--output_path", type=str, default="app/classifier_mlp.pth")
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(args)