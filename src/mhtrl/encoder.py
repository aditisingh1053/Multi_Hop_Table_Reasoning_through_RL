"""Frozen sentence encoder (all-MiniLM-L6-v2, mean pooling, L2-normalised)."""
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .device import DEVICE, ON_GPU

ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class FrozenEncoder:
    def __init__(self, name: str = ENCODER_NAME, max_length: int = 128, batch_size: int = None):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).eval().to(DEVICE)
        self.max_length = max_length
        self.batch_size = batch_size or (512 if ON_GPU else 128)
        self.dim = self.model.config.hidden_size

    @torch.inference_mode()
    def encode(self, texts, show_progress: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        # sort by length so that padding inside a batch is minimal
        order = np.argsort([len(t) for t in texts])
        rng = range(0, len(texts), self.batch_size)
        if show_progress:
            from tqdm import tqdm
            rng = tqdm(rng, desc="encode", mininterval=10)
        for i in rng:
            idx = order[i:i + self.batch_size]
            batch = self.tok([texts[j] for j in idx], padding=True, truncation=True,
                             max_length=self.max_length, return_tensors="pt")
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            hidden = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            emb = torch.nn.functional.normalize(emb, dim=-1)
            out[idx] = emb.float().cpu().numpy()
        return out
