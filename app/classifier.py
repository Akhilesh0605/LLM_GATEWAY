"""
PRODUCTION INFERENCE MODULE

Online (no network/DB/LLM) query complexity classification.
Loads once at process start. Deterministic. <2ms latency on single CPU core.

Return type: tuple[int, ComplexityTier]
- int: score 1-10
- ComplexityTier: enum from models.py (SIMPLE | MEDIUM | COMPLEX)

USAGE in app/main.py:
    from app.classifier import ClassificationService, get_classifier
    
    classifier_service = None
    
    @app.on_event("startup")
    async def startup():
        global classifier_service
        classifier_service = get_classifier()
    
    complexity_score, tier = classifier_service.classify(request.query)
"""

import logging
import os
from pathlib import Path
from typing import Tuple, Union, List
from enum import Enum

import numpy as np
import torch

from app.models import ComplexityTier

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False
    logging.warning("sentence-transformers not available")

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIG & THRESHOLDS (Aligned to Router Specs)
# ============================================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

MLP_WEIGHTS_FILE = Path(__file__).parent / "classifier_mlp.pth"
ONNX_MLP_FILE = Path(__file__).parent / "classifier_mlp.onnx"

# Tier Boundaries: Simple <= 3.0, Medium 3.0-7.0, Complex >= 7.0
TIER_SIMPLE_MAX = 3.0
TIER_COMPLEX_MIN = 7.0


# ============================================================================
# MLP DEFINITION (Robust to both 'fc' and 'network' state dict keys)
# ============================================================================

class ComplexityMLP(torch.nn.Module):
    """Lean 2-layer MLP: 384 -> 64 -> 1"""
    def __init__(self, input_dim=384, hidden_dim=64):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)


# ============================================================================
# CLASSIFICATION SERVICE
# ============================================================================

class ClassificationService:
    """
    Load once at startup, use for entire process lifetime.
    
    GUARANTEES:
    - Deterministic: same query -> same score
    - Fast: <2ms on 1 CPU core
    - Offline: zero network, DB, or LLM calls
    - Memory: <500MB (models) + FastAPI overhead
    """
    
    def __init__(self):
        logger.info("=" * 60)
        logger.info("COMPLEXITY CLASSIFIER - INITIALIZATION")
        logger.info("=" * 60)
        
        self.embedder = None
        self.mlp_model = None
        self.onnx_session = None
        self.use_onnx = False
        
        # Load models
        self._load_embedder()
        self._load_mlp()
        
        logger.info("✓ Classifier ready for inference")
        logger.info("=" * 60 + "\n")
    
    def _load_embedder(self) -> None:
        if not HAS_ST:
            logger.error("sentence-transformers not installed!")
            raise ImportError("sentence-transformers is required for classifier")
        
        logger.info(f"Loading embedder: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        logger.info(f"✓ Embedder loaded ({EMBEDDING_DIM}-dim)")
    
    def _load_mlp(self) -> None:
        # 1. Prefer ONNX if compiled
        if HAS_ONNX and ONNX_MLP_FILE.exists():
            try:
                logger.info(f"Loading ONNX MLP: {ONNX_MLP_FILE}")
                self.onnx_session = ort.InferenceSession(str(ONNX_MLP_FILE))
                self.use_onnx = True
                logger.info("✓ ONNX MLP loaded (fastest inference)")
                return
            except Exception as e:
                logger.warning(f"ONNX load failed: {e}; falling back to PyTorch")
        
        # 2. PyTorch Fallback
        if not MLP_WEIGHTS_FILE.exists():
            logger.error(f"✗ MLP weights not found: {MLP_WEIGHTS_FILE}")
            raise FileNotFoundError(f"{MLP_WEIGHTS_FILE} missing. Run: python train_classifier.py")
        
        logger.info(f"Loading PyTorch MLP: {MLP_WEIGHTS_FILE}")
        self.mlp_model = ComplexityMLP(input_dim=EMBEDDING_DIM, hidden_dim=64)
        
        # Load weights and auto-remap state_dict keys if necessary ('network' vs 'fc')
        state_dict = torch.load(str(MLP_WEIGHTS_FILE), map_location='cpu')
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("network.", "fc.")
            cleaned_state_dict[new_key] = v
            
        self.mlp_model.load_state_dict(cleaned_state_dict)
        self.mlp_model.eval()
        self.use_onnx = False
        logger.info("✓ PyTorch MLP loaded successfully")

    def encode(self, query: str) -> np.ndarray:
        """Helper to generate normalized 384-dim embedding once."""
        return self.embedder.encode(query, normalize_embeddings=True, show_progress_bar=False)

    def classify_embedding(self, embedding: Union[np.ndarray, List[float]]) -> Tuple[int, ComplexityTier]:
        """
        Classifies an ALREADY-COMPUTED embedding (avoids re-encoding in cache stage).
        Inference SLA: <0.5ms.
        """
        if isinstance(embedding, list):
            emb_arr = np.array(embedding, dtype=np.float32)
        else:
            emb_arr = embedding.astype(np.float32)
            
        if emb_arr.ndim == 1:
            emb_arr = emb_arr.reshape(1, -1)

        # Forward Pass
        if self.use_onnx:
            score = float(self.onnx_session.run(None, {"embeddings": emb_arr})[0][0][0])
        else:
            with torch.no_grad():
                tensor_input = torch.from_numpy(emb_arr)
                score = float(self.mlp_model(tensor_input).item())

        # Clamp between 1.0 and 10.0
        score = float(np.clip(score, 1.0, 10.0))
        score_int = int(round(score))

        # Map to ComplexityTier Enum
        if score <= TIER_SIMPLE_MAX:
            tier = ComplexityTier.SIMPLE
        elif score < TIER_COMPLEX_MIN:
            tier = ComplexityTier.MEDIUM
        else:
            tier = ComplexityTier.COMPLEX

        return score_int, tier

    def classify(self, query: str) -> Tuple[int, ComplexityTier]:
        """
        End-to-end classification: encodes text -> forwards MLP -> maps tier.
        Inference SLA: <10ms.
        """
        embedding = self.encode(query)
        return self.classify_embedding(embedding)


# ============================================================================
# SINGLETON INSTANCE ACCESS
# ============================================================================

_classifier_instance = None

def get_classifier() -> ClassificationService:
    """Get or create singleton classifier."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = ClassificationService()
    return _classifier_instance

def classify(query: str) -> Tuple[int, ComplexityTier]:
    """Convenience functional wrapper."""
    return get_classifier().classify(query)