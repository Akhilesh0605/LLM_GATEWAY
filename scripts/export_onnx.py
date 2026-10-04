"""
Build-time helper (runs ONLY in the onnx-builder Docker stage).

1. Exports the all-MiniLM-L6-v2 sentence embedder to ONNX.
2. Converts the trained complexity MLP weights (.pth) to .npz for NumPy inference.

Requires torch + transformers + onnx (build stage only; never shipped).
"""

import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LENGTH = 16  # dummy export length; dynamic axes make runtime length arbitrary


class Encoder(torch.nn.Module):
    """Wraps the HF model so ONNX exports only last_hidden_state."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "/embedder_onnx")
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval()

    dummy = tokenizer(
        "hello world",
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    with torch.no_grad():
        torch.onnx.export(
            Encoder(model),
            (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]),
            str(out_dir / "model.onnx"),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "token_type_ids": {0: "batch", 1: "sequence"},
                "last_hidden_state": {0: "batch", 1: "sequence"},
            },
            opset_version=14,
            do_constant_folding=True,
            dynamo=False,
        )

    tokenizer.save_pretrained(str(out_dir))

    mlp_pth = Path("/tmp/classifier_mlp.pth")
    if mlp_pth.exists():
        state_dict = torch.load(str(mlp_pth), map_location="cpu")
        clean = {k.replace("network.", "fc."): v.detach().cpu().numpy() for k, v in state_dict.items()}
        npz_path = Path("/mlp/classifier_mlp.npz")
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(npz_path), **clean)

    print("ONNX + NPZ export complete")


if __name__ == "__main__":
    main()
