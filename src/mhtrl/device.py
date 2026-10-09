"""Compute device shared by the encoder, the reader and the policy (override with MHTRL_DEVICE)."""
import os

import torch

DEVICE = torch.device(os.environ.get("MHTRL_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))
ON_GPU = DEVICE.type == "cuda"


def to_device(batch: dict) -> dict:
    return {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
