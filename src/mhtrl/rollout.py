"""Batched lock-step rollouts, tensor collation and evaluation of selected evidence sets."""
from typing import List, Optional, Sequence

import numpy as np
import torch

from .env import (N_CAND_FEAT, ROW, Bank, EnvConfig, Episode, answer_in_evidence,
                  gold_covered)
from .device import to_device
from .reader import ReaderCache
from .text import exact_match, f1_score


class StepRec:
    __slots__ = ("item", "cands", "feat", "state", "stop_ok", "action", "logp")

    def __init__(self, item, cands, feat, state, stop_ok, action=-1, logp=0.0):
        self.item, self.cands, self.feat, self.state = item, cands, feat, state
        self.stop_ok, self.action, self.logp = stop_ok, action, logp


def collate(bank: Bank, recs: Sequence[StepRec], k_max: int):
    b = len(recs)
    idx = np.full((b, k_max), bank.pad_index, dtype=np.int64)
    feat = np.zeros((b, k_max, N_CAND_FEAT), dtype=np.float32)
    mask = np.zeros((b, k_max), dtype=bool)
    for i, r in enumerate(recs):
        k = len(r.cands)
        if k:
            idx[i, :k] = bank.items[r.item].emb_offset + np.asarray(r.cands)
            feat[i, :k] = r.feat
            mask[i, :k] = True
    return {
        "state": torch.from_numpy(np.stack([r.state for r in recs])),
        "cand_emb": torch.from_numpy(bank.emb_all[idx]),
        "cand_feat": torch.from_numpy(feat),
        "cand_mask": torch.from_numpy(mask),
        "stop_ok": torch.tensor([r.stop_ok for r in recs], dtype=torch.bool),
    }


@torch.no_grad()
def rollout(bank: Bank, item_ids: Sequence[int], actor, env_cfg: EnvConfig, greedy: bool,
            gen: Optional[torch.Generator] = None, chunk: int = 1024):
    """Runs one episode per entry of item_ids. Returns (episodes, step records, stopped flags)."""
    k_max = env_cfg.max_cands
    eps = [Episode(bank.items[i], env_cfg) for i in item_ids]
    recs: List[List[StepRec]] = [[] for _ in eps]
    stopped = [False] * len(eps)
    active = list(range(len(eps)))
    while active:
        pend = []
        for j in active:
            ep = eps[j]
            cands = ep.candidates()
            ok = ep.stop_allowed()
            if not cands and not ok:          # degenerate graph: nothing can be selected
                ep.done = True
                continue
            s, f = ep.features(cands)
            pend.append((j, StepRec(item_ids[j], cands, f, s, ok)))
        for c0 in range(0, len(pend), chunk):
            part = pend[c0:c0 + chunk]
            logits = actor(**to_device(collate(bank, [r for _, r in part], k_max))).float().cpu()
            logp_all = torch.log_softmax(logits, -1)
            if greedy:
                act = logits.argmax(-1)
            else:
                act = torch.multinomial(logp_all.exp(), 1, generator=gen).squeeze(-1)
            lp = logp_all.gather(-1, act.unsqueeze(-1)).squeeze(-1)
            for (j, rec), a, l in zip(part, act.tolist(), lp.tolist()):
                rec.action, rec.logp = a, l
                recs[j].append(rec)
                if a == k_max:
                    eps[j].step(None)
                    stopped[j] = True
                else:
                    eps[j].step(rec.cands[a])
        active = [j for j in active if not eps[j].done]
    return eps, recs, stopped


def predict(bank: Bank, item_ids: Sequence[int], sets: Sequence[Sequence[int]],
            cache: ReaderCache) -> List[str]:
    items = []
    for i, s in zip(item_ids, sets):
        p = bank.items[i]
        items.extend((p.qid, n, p.question, p.text[n]) for n in s)
    cache.ensure(items)
    return [cache.answer(bank.items[i].qid, s) if s else "" for i, s in zip(item_ids, sets)]


def answer_score(gold: str, pred: str, metric: str) -> float:
    return f1_score(gold, pred) if metric == "f1" else exact_match(gold, pred)


def evaluate_sets(bank: Bank, item_ids: Sequence[int], sets: Sequence[Sequence[int]],
                  cache: ReaderCache):
    """Returns (summary dict, per-question rows)."""
    preds = predict(bank, item_ids, sets, cache)
    rows = []
    for i, s, pred in zip(item_ids, sets, preds):
        p = bank.items[i]
        rows.append({
            "qid": p.qid, "pred": pred, "gold": p.answer, "type": p.answer_type,
            "em": exact_match(p.answer, pred), "f1": f1_score(p.answer, pred),
            "ans_in_ev": float(answer_in_evidence(p, s)),
            "gold_hit": float(gold_covered(p, s)) if p.gold_paths else None,
            "n_items": len(s), "n_rows": int(sum(p.kind[n] == ROW for n in s)),
            "words": int(sum(p.word_len[n] for n in s)), "selected": list(map(int, s)),
        })
    return summarize(rows), rows


def summarize(rows):
    def mean(key, subset=None):
        vals = [r[key] for r in (subset if subset is not None else rows) if r[key] is not None]
        return float(np.mean(vals)) if vals else float("nan")

    out = {
        "n": len(rows), "EM": 100 * mean("em"), "F1": 100 * mean("f1"),
        "AnsRecall": 100 * mean("ans_in_ev"), "GoldHit": 100 * mean("gold_hit"),
        "Items": mean("n_items"), "Rows": mean("n_rows"), "Words": mean("words"),
        # right answer although no traced gold path is covered (spurious success)
        "RightWrongEv": 100 * float(np.mean([r["em"] == 1 and r["gold_hit"] == 0
                                             for r in rows if r["gold_hit"] is not None] or [np.nan])),
    }
    for t in ("table", "passage"):
        sub = [r for r in rows if r["type"] == t]
        if sub:
            out[f"{t}_EM"] = 100 * mean("em", sub)
            out[f"{t}_F1"] = 100 * mean("f1", sub)
            out[f"{t}_n"] = len(sub)
    return out
