"""Batched lock-step rollouts, tensor collation and evaluation of selected evidence sets."""
from typing import List, Optional, Sequence

import numpy as np
import torch

from .env import (ROW, Bank, EnvConfig, Episode, answer_in_evidence, gold_covered,
                  uses_loop)
from .device import to_device
from .reader import ReaderCache
from .text import exact_match, f1_score, normalize_answer

NEG_HALF = -1e8    # masked logits are -1e9 (policy.NEG); anything above this is a valid action
NEG_FILL = -1e9


class StepRec:
    __slots__ = ("item", "cands", "feat", "state", "stop_ok", "action", "logp", "commit", "sel")

    def __init__(self, item, cands, feat, state, stop_ok, action=-1, logp=0.0, commit=False, sel=()):
        self.item, self.cands, self.feat, self.state = item, cands, feat, state
        self.stop_ok, self.action, self.logp = stop_ok, action, logp
        self.commit = commit      # True for a COMMIT-phase decision
        self.sel = sel            # entities selected before this step (tuple)


def collate(bank: Bank, recs: Sequence[StepRec], k_max: int):
    b = len(recs)
    n_feat = np.asarray(recs[0].feat).shape[1]
    idx = np.full((b, k_max), bank.pad_index, dtype=np.int64)
    feat = np.zeros((b, k_max, n_feat), dtype=np.float32)
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


def feed_reader(eps: Sequence[Episode], cache: ReaderCache):
    """Reads every selected-but-unread entity of the given episodes (batched) and stores the
    (span, score) outputs in the episodes. These are exactly the reads the reward needs anyway."""
    todo = []
    for ep in eps:
        for n in ep.selected:
            if n not in ep.reader_out:
                todo.append((ep, n))
    if not todo:
        return
    cache.ensure((ep.p.qid, n, ep.p.question, ep.p.text[n]) for ep, n in todo)
    for ep, n in todo:
        ep.reader_out[n] = cache.table[(ep.p.qid, n)]


@torch.no_grad()
def rollout(bank: Bank, item_ids: Sequence[int], actor, env_cfg: EnvConfig, greedy: bool,
            gen: Optional[torch.Generator] = None, chunk: int = 1024,
            cache: Optional[ReaderCache] = None, action_dropout: float = 0.0,
            temperature: float = 1.0):
    """Runs one episode per entry of item_ids. Returns (episodes, step records, stopped flags).

    With the reader-in-the-loop extension the frozen reader is queried between steps for the
    newly selected entities (cache required).

    Sampling only (greedy=False):
      action_dropout  each valid entity candidate is hidden from the *sampler* with this
                      probability (Lin et al., 2018); STOP and COMMIT decisions are never
                      dropped. The recorded log-probability is that of the full policy, so the
                      PPO ratio stays consistent between rollout and update.
      temperature     sampling temperature (test-time self-consistency); logp is untempered.
    """
    k_max = env_cfg.max_cands
    loop = uses_loop(env_cfg)
    if loop and cache is None:
        raise ValueError("reader_loop / commit need the reader cache during rollouts")
    eps = [Episode(bank.items[i], env_cfg) for i in item_ids]
    recs: List[List[StepRec]] = [[] for _ in eps]
    stopped = [False] * len(eps)
    active = list(range(len(eps)))
    while active:
        if loop:
            feed_reader([eps[j] for j in active], cache)
        pend = []
        for j in active:
            ep = eps[j]
            cands = ep.candidates()
            ok = ep.stop_allowed()
            if not cands and not ok:          # degenerate graph: nothing can be selected
                ep.done = True
                continue
            s, f = ep.features(cands)
            pend.append((j, StepRec(item_ids[j], cands, f, s, ok, commit=ep.in_commit,
                                    sel=tuple(ep.selected))))
        for c0 in range(0, len(pend), chunk):
            part = pend[c0:c0 + chunk]
            logits = actor(**to_device(collate(bank, [r for _, r in part], k_max))).float().cpu()
            logp_all = torch.log_softmax(logits, -1)
            if greedy:
                act = logits.argmax(-1)
            else:
                samp = logits / temperature if temperature != 1.0 else logits
                if action_dropout > 0:
                    valid = logits > NEG_HALF
                    keep = torch.rand(logits.shape, generator=gen) >= action_dropout
                    keep[:, -1] = True                                  # never drop STOP
                    is_commit = torch.tensor([r.commit for _, r in part], dtype=torch.bool)
                    keep[is_commit] = True                              # never drop COMMIT options
                    keep = keep & valid
                    empty = ~keep.any(-1)
                    keep[empty] = valid[empty]
                    samp = samp.masked_fill(~keep, NEG_FILL)
                act = torch.multinomial(torch.softmax(samp, -1), 1, generator=gen).squeeze(-1)
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
            cache: ReaderCache, commits: Optional[Sequence[Optional[int]]] = None) -> List[str]:
    """Final answers. Default: span of the most confident selected entity. If commits[k] is an
    entity index, the answer is that entity's span (learned COMMIT)."""
    items = []
    for i, s in zip(item_ids, sets):
        p = bank.items[i]
        items.extend((p.qid, n, p.question, p.text[n]) for n in s)
    cache.ensure(items)
    commits = commits if commits is not None else [None] * len(sets)
    out = []
    for i, s, c in zip(item_ids, sets, commits):
        qid = bank.items[i].qid
        if not s:
            out.append("")
        elif c is not None:
            out.append(cache.table[(qid, c)][0])
        else:
            out.append(cache.answer(qid, s))
    return out


def answer_score(gold: str, pred: str, metric: str) -> float:
    return f1_score(gold, pred) if metric == "f1" else exact_match(gold, pred)


def evaluate_sets(bank: Bank, item_ids: Sequence[int], sets: Sequence[Sequence[int]],
                  cache: ReaderCache, commits: Optional[Sequence[Optional[int]]] = None):
    """Returns (summary dict, per-question rows).

    Besides the final answer, every row records two arbitration diagnostics on the same set:
    f1_maxconf (the reader's max-confidence rule) and f1_best (best span inside the set, an
    oracle arbiter). They split the error into selection loss and aggregation loss."""
    preds = predict(bank, item_ids, sets, cache, commits)
    maxconf = predict(bank, item_ids, sets, cache)
    rows = []
    for k, (i, s, pred) in enumerate(zip(item_ids, sets, preds)):
        p = bank.items[i]
        best = max((f1_score(p.answer, cache.table[(p.qid, n)][0]) for n in s), default=0.0)
        rows.append({
            "qid": p.qid, "pred": pred, "gold": p.answer, "type": p.answer_type,
            "em": exact_match(p.answer, pred), "f1": f1_score(p.answer, pred),
            "f1_maxconf": f1_score(p.answer, maxconf[k]), "f1_best": best,
            "commit": None if commits is None or commits[k] is None else int(commits[k]),
            "commit_changed": float(pred != maxconf[k]),
            "ans_in_ev": float(answer_in_evidence(p, s)),
            "gold_hit": float(gold_covered(p, s)) if p.gold_paths else None,
            "n_items": len(s), "n_rows": int(sum(p.kind[n] == ROW for n in s)),
            "words": int(sum(p.word_len[n] for n in s)), "selected": list(map(int, s)),
        })
    return summarize(rows), rows


@torch.no_grad()
def self_consistency_eval(bank: Bank, actor, env_cfg: EnvConfig, cache: ReaderCache, k: int = 8,
                          temperature: float = 1.0, seed: int = 0, ids=None):
    """Test-time self-consistency over evidence paths (after Wang et al., ICLR 2023).

    Runs the greedy episode plus k-1 sampled episodes per question; each episode yields an
    answer (its COMMIT choice, or max confidence). The final answer is the most frequent
    normalised answer; ties are broken by the summed reader score of its supporting episodes,
    then by the greedy answer. Reports the number of distinct entities read per question, i.e.
    the reader cost of the extra test-time compute.
    """
    ids = list(range(len(bank))) if ids is None else list(ids)
    was_training = actor.training
    actor.eval()
    gen = torch.Generator().manual_seed(seed)
    episodes = [rollout(bank, ids, actor, env_cfg, greedy=True, cache=cache)[0]]
    for _ in range(k - 1):
        episodes.append(rollout(bank, ids, actor, env_cfg, greedy=False, gen=gen, cache=cache,
                                temperature=temperature)[0])
    if was_training:
        actor.train()
    answers = []
    for eps in episodes:
        sets = [e.selected for e in eps]
        commits = [e.committed for e in eps] if env_cfg.commit else None
        answers.append(predict(bank, ids, sets, cache, commits))
    rows = []
    for q, i in enumerate(ids):
        p = bank.items[i]
        votes = {}
        for r, eps in enumerate(episodes):
            a = answers[r][q]
            na = normalize_answer(a)
            if not na:
                continue
            e = eps[q]
            src = e.committed if (env_cfg.commit and e.committed is not None) else \
                max(e.selected, key=lambda n: cache.table[(p.qid, n)][1])
            v = votes.setdefault(na, [0, 0.0, a, r == 0])
            v[0] += 1
            v[1] += cache.table[(p.qid, src)][1]
            v[3] = v[3] or r == 0
        if votes:
            pred = max(votes.values(), key=lambda v: (v[0], v[1], v[3]))[2]
        else:
            pred = answers[0][q]
        read = set()
        for eps in episodes:
            read.update(eps[q].selected)
        greedy_pred = answers[0][q]
        rows.append({
            "qid": p.qid, "pred": pred, "gold": p.answer, "type": p.answer_type,
            "em": exact_match(p.answer, pred), "f1": f1_score(p.answer, pred),
            "f1_greedy": f1_score(p.answer, greedy_pred), "n_votes": len(votes),
            "reads": len(read), "agree": float(normalize_answer(pred) == normalize_answer(greedy_pred)),
        })
    summ = {"n": len(rows), "k": k, "temperature": temperature,
            "EM": 100 * float(np.mean([r["em"] for r in rows])),
            "F1": 100 * float(np.mean([r["f1"] for r in rows])),
            "F1_greedy": 100 * float(np.mean([r["f1_greedy"] for r in rows])),
            "Reads": float(np.mean([r["reads"] for r in rows])),
            "AgreeGreedy": 100 * float(np.mean([r["agree"] for r in rows]))}
    for t in ("table", "passage"):
        sub = [r for r in rows if r["type"] == t]
        if sub:
            summ[f"{t}_F1"] = 100 * float(np.mean([r["f1"] for r in sub]))
    return summ, rows


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
    if rows and "f1_best" in rows[0]:
        out["F1_maxconf"] = 100 * mean("f1_maxconf")
        out["F1_bestinset"] = 100 * mean("f1_best")
        out["CommitChanged"] = 100 * mean("commit_changed")
    for t in ("table", "passage"):
        sub = [r for r in rows if r["type"] == t]
        if sub:
            out[f"{t}_EM"] = 100 * mean("em", sub)
            out[f"{t}_F1"] = 100 * mean("f1", sub)
            out[f"{t}_n"] = len(sub)
    return out
