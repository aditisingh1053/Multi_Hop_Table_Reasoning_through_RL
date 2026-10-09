"""Frozen extractive reader with a persistent per-(question, entity) cache.

Every selected entity (row or passage) is read independently together with the
question; the answer for an evidence set E is the span of the entity with the highest
null-adjusted span score  s(e) = max_span(start + end logit) - (start + end logit of [CLS]).
This open-domain style aggregation (as in DrQA-like retriever-reader systems) makes the
reader output decomposable over entities, so it can be cached and re-used across all
methods, seeds and training iterations.
"""
import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from .device import DEVICE, ON_GPU

DEFAULT_READER = "deepset/minilm-uncased-squad2"
READER_INT8 = os.environ.get("MHTRL_READER_INT8", "0") == "1"


class ExtractiveReader:
    def __init__(self, name: str = DEFAULT_READER, max_length: int = 384, stride: int = 128,
                 max_answer_tokens: int = 30, batch_size: int = None, n_best: int = 20,
                 int8: bool = False):
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(name).eval()
        if int8 and not ON_GPU:  # dynamic int8 quantisation of linear layers (CPU only)
            self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear},
                                                             dtype=torch.qint8)
        self.model.to(DEVICE)
        batch_size = batch_size or int(os.environ.get("MHTRL_READER_BATCH", 128 if ON_GPU else 32))
        self.max_length, self.stride = max_length, stride
        self.max_answer_tokens, self.batch_size, self.n_best = max_answer_tokens, batch_size, n_best

    @torch.inference_mode()
    def read(self, pairs: Sequence[Tuple[str, str]]) -> List[Tuple[str, float]]:
        """pairs: (question, context). Returns (answer span, null-adjusted score) per pair."""
        if not pairs:
            return []
        enc = self.tok([p[0] for p in pairs], [p[1] for p in pairs], truncation="only_second",
                       max_length=self.max_length, stride=self.stride,
                       return_overflowing_tokens=True, return_offsets_mapping=True, padding=False)
        sample_map = enc["overflow_to_sample_mapping"]
        n_win = len(sample_map)
        seq_ids = [enc.sequence_ids(w) for w in range(n_win)]
        # windows are batched by length and padded per batch, so short rows stay cheap
        order = sorted(range(n_win), key=lambda w: len(enc["input_ids"][w]))
        keys = [k for k in ("input_ids", "attention_mask", "token_type_ids") if k in enc]
        starts, ends = [None] * n_win, [None] * n_win
        for i in range(0, n_win, self.batch_size):
            idx = order[i:i + self.batch_size]
            batch = self.tok.pad({k: [enc[k][w] for w in idx] for k in keys}, return_tensors="pt")
            out = self.model(**{k: v.to(DEVICE) for k, v in batch.items()})
            s_all = out.start_logits.float().cpu().numpy()
            e_all = out.end_logits.float().cpu().numpy()
            for j, w in enumerate(idx):
                n = len(enc["input_ids"][w])
                starts[w] = s_all[j, :n]
                ends[w] = e_all[j, :n]

        best = [("", -1e9) for _ in pairs]
        for w in range(n_win):
            ctx = [j for j, s_ in enumerate(seq_ids[w]) if s_ == 1]
            if not ctx:
                continue
            lo, hi = ctx[0], ctx[-1]
            s_log, e_log = starts[w], ends[w]
            null = s_log[0] + e_log[0]
            s_top = lo + np.argsort(s_log[lo:hi + 1])[::-1][: self.n_best]
            e_top = lo + np.argsort(e_log[lo:hi + 1])[::-1][: self.n_best]
            length = e_top[None, :] - s_top[:, None] + 1
            grid = s_log[s_top][:, None] + e_log[e_top][None, :]
            grid = np.where((length >= 1) & (length <= self.max_answer_tokens), grid, -np.inf)
            flat = int(np.argmax(grid))
            cand_score = grid.flat[flat]
            if not np.isfinite(cand_score):
                continue
            s, e = s_top[flat // grid.shape[1]], e_top[flat % grid.shape[1]]
            k = sample_map[w]
            score = float(cand_score - null)
            if score > best[k][1]:
                offsets = enc["offset_mapping"][w]
                a = pairs[k][1][offsets[s][0]: offsets[e][1]]
                best[k] = (a.strip(), score)
        return best


class ReaderCache:
    """Maps (qid, node index) -> (answer, score) for one reader model; persisted with pickle."""

    def __init__(self, reader_name: str, cache_dir: str = None, int8=None):
        cache_dir = cache_dir or os.environ.get("MHTRL_CACHE", "data/cache")
        if int8 is None:   # quantisation is only ever applied to the training reader
            int8 = READER_INT8 and reader_name == DEFAULT_READER
        self.reader_name, self.int8 = reader_name, int8
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        suffix = "_int8" if int8 else ""
        self.path = Path(cache_dir) / f"reader_{reader_name.replace('/', '__')}{suffix}.pkl"
        self.table: Dict[Tuple[str, int], Tuple[str, float]] = {}
        if self.path.exists():
            with open(self.path, "rb") as f:
                self.table = pickle.load(f)
        self._reader = None
        self.n_calls = 0

    @property
    def reader(self) -> ExtractiveReader:
        if self._reader is None:
            if not ON_GPU:
                torch.set_num_threads(min(8, os.cpu_count() or 8))
            self._reader = ExtractiveReader(self.reader_name, int8=self.int8)
        return self._reader

    def ensure(self, items: Iterable[Tuple[str, int, str, str]], chunk: int = 256):
        """items: (qid, node_idx, question, node_text). Reads every pair that is not cached."""
        todo, seen = [], set()
        for qid, node, question, text in items:
            key = (qid, node)
            if key in self.table or key in seen:
                continue
            seen.add(key)
            todo.append((key, question, text))
        # sort by context length so that each reader batch is padded as little as possible
        todo.sort(key=lambda x: len(x[2]))
        for i in range(0, len(todo), chunk):
            part = todo[i:i + chunk]
            outs = self.reader.read([(q, t) for _, q, t in part])
            for (key, _, _), out in zip(part, outs):
                self.table[key] = out
            self.n_calls += len(part)
        return len(todo)

    def answer(self, qid: str, nodes: Sequence[int]) -> str:
        best, best_s = "", -1e18
        for n in nodes:
            a, s = self.table[(qid, n)]
            if s > best_s:
                best, best_s = a, s
        return best

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self.table, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.path)
