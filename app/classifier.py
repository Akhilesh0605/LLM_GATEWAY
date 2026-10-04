"""
PRODUCTION INFERENCE MODULE (ONNX-first)

Online (no network/DB/LLM) query complexity classification.
Loads once at process start. Deterministic. <2ms latency on a single CPU core.

The production image runs the sentence embedder on ONNX Runtime and the tiny
384 -> 64 -> 1 MLP head in NumPy, so PyTorch / sentence-transformers are NOT
required at runtime. If the ONNX artifacts are missing, the service falls back
to torch + sentence-transformers (development only).

Return type: tuple[int, ComplexityTier]
- int: score 1-10
- ComplexityTier: enum from models.py (SIMPLE | MEDIUM | COMPLEX)
"""

import logging
from pathlib import Path
from typing import Tuple, Union, List

import numpy as np

from app.models import ComplexityTier

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
MAX_SEQ_LENGTH = 256

_HERE = Path(__file__).parent
ONNX_DIR = _HERE / "onnx_embedder"
ONNX_MODEL_FILE = ONNX_DIR / "model.onnx"
ONNX_TOKENIZER_FILE = ONNX_DIR / "tokenizer.json"
MLP_NPZ_FILE = _HERE / "classifier_mlp.npz"
MLP_WEIGHTS_FILE = _HERE / "classifier_mlp.pth"

# Tier boundaries: Simple <= 3.0, Medium 3.0-7.0, Complex >= 7.0
TIER_SIMPLE_MAX = 3.0
TIER_COMPLEX_MIN = 7.0

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    from tokenizers import Tokenizer
    HAS_TOKENIZERS = True
except ImportError:
    HAS_TOKENIZERS = False


class ClassificationService:
    """
    Load once at startup, use for the entire process lifetime.

    GUARANTEES:
    - Deterministic: same query -> same score
    - Fast: single-digit ms on 1 CPU core
    - Offline: zero network, DB, or LLM calls
    """

    def __init__(self):
        self.backend = None
        self.session = None
        self.tokenizer = None
        self.input_names = []
        self._npz = None
        self._torch = None
        self._st_model = None
        self._torch_mlp = None

        self._load_embedder()
        self._load_mlp()
        logger.info(f"Classifier ready for inference (backend={self.backend})")

    # -- embedder ---------------------------------------------------------
    def _load_embedder(self) -> None:
        if HAS_ONNX and HAS_TOKENIZERS and ONNX_MODEL_FILE.exists() and ONNX_TOKENIZER_FILE.exists():
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            self.session = ort.InferenceSession(
                str(ONNX_MODEL_FILE), sess_options=opts, providers=["CPUExecutionProvider"]
            )
            self.input_names = [i.name for i in self.session.get_inputs()]
            self.tokenizer = Tokenizer.from_file(str(ONNX_TOKENIZER_FILE))
            self.tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)
            self.backend = "onnx"
            return

        # Development fallback (not installed in the production image)
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "ONNX artifacts are missing and torch/sentence-transformers are not "
                f"installed. Expected: {ONNX_MODEL_FILE} and {ONNX_TOKENIZER_FILE}."
            ) from e
        self._torch = torch
        self._st_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        self.backend = "torch"

    # -- MLP head ---------------------------------------------------------
    def _load_mlp(self) -> None:
        if MLP_NPZ_FILE.exists():
            z = np.load(str(MLP_NPZ_FILE))
            self._npz = (
                z["fc.0.weight"].astype(np.float32),
                z["fc.0.bias"].astype(np.float32),
                z["fc.3.weight"].astype(np.float32),
                z["fc.3.bias"].astype(np.float32),
            )
            return

        if self._torch is None:
            return

        state_dict = self._torch.load(str(MLP_WEIGHTS_FILE), map_location="cpu")
        clean = {k.replace("network.", "fc."): v for k, v in state_dict.items()}
        model = self._torch.nn.Sequential(
            self._torch.nn.Linear(EMBEDDING_DIM, 64),
            self._torch.nn.ReLU(),
            self._torch.nn.Dropout(0.1),
            self._torch.nn.Linear(64, 1),
        )
        model.load_state_dict(clean)
        model.eval()
        self._torch_mlp = model

    # -- inference --------------------------------------------------------
    def encode(self, query: str) -> np.ndarray:
        """Returns an L2-normalized 384-dim float32 embedding."""
        if self.backend == "onnx":
            enc = self.tokenizer.encode(query)
            ids = np.array([enc.ids], dtype=np.int64)
            mask = np.array([enc.attention_mask], dtype=np.int64)

            feed = {}
            for name in self.input_names:
                if name == "input_ids":
                    feed[name] = ids
                elif name == "attention_mask":
                    feed[name] = mask
                elif name == "token_type_ids":
                    feed[name] = np.array([enc.type_ids], dtype=np.int64)

            last = self.session.run(None, feed)[0].astype(np.float32)

            if last.ndim == 3:
                mask_f = mask.astype(np.float32)[..., None]
                summed = (last * mask_f).sum(axis=1)
                counts = np.clip(mask_f.sum(axis=1), 1e-9, None)
                emb = (summed / counts)[0]
            else:
                emb = last[0]

            norm = float(np.linalg.norm(emb))
            if norm > 0:
                emb = emb / norm
            return emb.astype(np.float32)

        return self._st_model.encode(query, normalize_embeddings=True, show_progress_bar=False)

    def _predict(self, emb_arr: np.ndarray) -> float:
        if self._npz is not None:
            W1, b1, W2, b2 = self._npz
            hidden = np.maximum(emb_arr @ W1.T + b1, 0.0)
            return float((hidden @ W2.T + b2).reshape(-1)[0])

        with self._torch.no_grad():
            return float(self._torch_mlp(self._torch.from_numpy(emb_arr)).item())

    def classify_embedding(self, embedding: Union[np.ndarray, List[float]]) -> Tuple[int, ComplexityTier]:
        """Classifies an ALREADY-COMPUTED embedding (avoids re-encoding in cache stage)."""
        if isinstance(embedding, list):
            emb_arr = np.array(embedding, dtype=np.float32)
        else:
            emb_arr = np.asarray(embedding, dtype=np.float32)
        if emb_arr.ndim == 1:
            emb_arr = emb_arr.reshape(1, -1)

        score = float(np.clip(self._predict(emb_arr), 1.0, 10.0))
        score_int = int(round(score))

        if score <= TIER_SIMPLE_MAX:
            tier = ComplexityTier.SIMPLE
        elif score < TIER_COMPLEX_MIN:
            tier = ComplexityTier.MEDIUM
        else:
            tier = ComplexityTier.COMPLEX

        return score_int, tier

    def classify(self, query: str) -> Tuple[int, ComplexityTier]:
        """End-to-end: encode text -> forward MLP -> map tier."""
        return self.classify_embedding(self.encode(query))


_classifier_instance = None


def get_classifier() -> ClassificationService:
    """Get or create the singleton classifier."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = ClassificationService()
    return _classifier_instance


def classify(query: str) -> Tuple[int, ComplexityTier]:
    """Convenience functional wrapper."""
    return get_classifier().classify(query)
