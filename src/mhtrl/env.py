"""Entity-selection MDP over the per-question table/passage graph.

State   s_t = (question, evidence set E_t, last selected entity, step t)
Actions A_t = C_t U {STOP}, where the candidate set C_t contains
          * every table row that is not yet selected,
          * the k globally best-matching passages (hybrid lexical+dense score), and
          * every passage hyperlinked from an already selected row (graph expansion).
          STOP is unavailable at t = 0 (the reader needs at least one entity).
Episode ends on STOP or after H selections (forced stop).
"""
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .data import PASSAGE, ROW
from .text import contains_answer, content_words, numbers, words

EMB_DIM = 384
N_CAND_FEAT = 14
N_STATE_SCALAR = 8
STATE_DIM = 3 * EMB_DIM + N_STATE_SCALAR

CAND_FEATURES = [
    "is_row", "is_passage", "cos(q,a)", "bm25_norm", "rank_in_type", "linked_to_evidence",
    "linked_to_last", "row_position", "q_overlap", "q_novel_overlap", "number_match",
    "global_topk_passage", "max_cos_to_evidence", "degree",
]
STATE_SCALARS = ["t/H", "rows_selected/H", "passages_selected/H", "q_coverage",
                 "max_cand_cos", "max_cand_bm25", "last_is_row", "last_is_passage"]


@dataclass
class EnvConfig:
    horizon: int = 3
    k_passages: int = 5
    max_cands: int = 48


class Prepared:
    """Static per-question arrays used by the environment and the baselines."""

    def __init__(self, inst: dict, q_emb: np.ndarray, node_emb: np.ndarray, bm25: np.ndarray,
                 emb_offset: int, k_passages: int):
        self.inst = inst
        self.qid, self.question, self.answer = inst["qid"], inst["question"], inst["answer"]
        self.n = len(inst["node_kind"])
        self.n_rows = inst["n_rows"]
        self.kind = np.asarray(inst["node_kind"], dtype=np.int64)
        self.text = inst["node_text"]
        self.adj = [set(a) for a in inst["adj"]]
        self.gold_paths = [tuple(p) for p in inst["gold_paths"]]
        self.answer_type = inst.get("answer_type")
        self.emb_offset = emb_offset
        self.q = q_emb.astype(np.float32)
        self.emb = node_emb.astype(np.float32)
        self.cos = self.emb @ self.q
        mx = float(bm25.max()) if len(bm25) else 0.0
        self.bm25n = bm25 / mx if mx > 0 else np.zeros_like(bm25)
        self.hybrid = self.cos + self.bm25n

        self.rank_feat = np.zeros(self.n, dtype=np.float32)
        for k in (ROW, PASSAGE):
            idx = np.where(self.kind == k)[0]
            if len(idx):
                order = idx[np.argsort(-self.hybrid[idx], kind="stable")]
                self.rank_feat[order] = 1.0 - np.arange(len(idx)) / max(1, len(idx))
        self.row_pos = np.zeros(self.n, dtype=np.float32)
        if self.n_rows > 1:
            self.row_pos[: self.n_rows] = np.arange(self.n_rows) / (self.n_rows - 1)
        deg = np.array([len(a) for a in self.adj], dtype=np.float32)
        self.degree = np.where(self.kind == ROW, np.minimum(deg / 10.0, 1.0),
                               deg / max(1, self.n_rows))

        q_cw = set(content_words(self.question))
        self.n_qcw = max(1, len(q_cw))
        node_words = [set(words(t)) for t in self.text]
        self.qcov = [frozenset(q_cw & w) for w in node_words]
        self.overlap = np.array([len(c) / self.n_qcw for c in self.qcov], dtype=np.float32)
        q_num = numbers(self.question)
        self.num_match = np.array([float(bool(q_num & numbers(t))) for t in self.text],
                                  dtype=np.float32)
        p_idx = np.where(self.kind == PASSAGE)[0]
        top = p_idx[np.argsort(-self.hybrid[p_idx], kind="stable")][:k_passages]
        self.global_top = set(int(i) for i in top)
        self.global_top_arr = np.zeros(self.n, dtype=np.float32)
        self.global_top_arr[list(self.global_top)] = 1.0
        self.contains = np.array([contains_answer(self.answer, t) for t in self.text], dtype=bool) \
            if self.answer else np.zeros(self.n, dtype=bool)
        self.word_len = np.array([len(t.split()) for t in self.text], dtype=np.int64)


class Bank:
    """All prepared questions of one split plus a global embedding matrix for gathering."""

    def __init__(self, path: str, k_passages: int = 5, limit: Optional[int] = None):
        with open(path, "rb") as f:
            payload = pickle.load(f)
        insts = payload["instances"][:limit] if limit else payload["instances"]
        offsets, blocks, pos = {}, [], 0
        for tid, e in payload["table_emb"].items():
            offsets[tid] = pos
            blocks.append(e.astype(np.float32))
            pos += len(e)
        blocks.append(np.zeros((1, EMB_DIM), dtype=np.float32))  # padding row
        self.emb_all = np.concatenate(blocks, 0)
        self.pad_index = pos
        self.items: List[Prepared] = []
        for i, inst in enumerate(insts):
            tid = inst["table_id"]
            self.items.append(Prepared(inst, payload["q_emb"][i], payload["table_emb"][tid],
                                       payload["bm25"][i], offsets[tid], k_passages))

    def __len__(self):
        return len(self.items)


def gold_covered(p: Prepared, selected: Sequence[int]) -> bool:
    s = set(selected)
    return any(all(n in s for n in path) for path in p.gold_paths)


def answer_in_evidence(p: Prepared, selected: Sequence[int]) -> bool:
    return bool(any(p.contains[n] for n in selected))


def evidence_recall(p: Prepared, selected: Sequence[int]) -> float:
    """Any-of gold-path recall psi(E) = max_P |P & E| / |P| over the traced gold paths."""
    if not p.gold_paths:
        return 0.0
    s = set(selected)
    return max(sum(n in s for n in path) / len(path) for path in p.gold_paths)


class Episode:
    def __init__(self, p: Prepared, cfg: EnvConfig):
        self.p, self.cfg = p, cfg
        self.selected: List[int] = []
        self.done = False
        self.truncated_cands = 0

    @property
    def t(self) -> int:
        return len(self.selected)

    def candidates(self) -> List[int]:
        p, sel = self.p, set(self.selected)
        rows = [i for i in range(p.n_rows) if i not in sel]
        psg = set(p.global_top)
        for s in self.selected:
            psg |= {a for a in p.adj[s] if p.kind[a] == PASSAGE}
        psg = sorted((a for a in psg if a not in sel), key=lambda a: -p.hybrid[a])
        cands = rows + psg
        if len(cands) > self.cfg.max_cands:
            linked = set()
            for s in self.selected:
                linked |= p.adj[s]
            pri = {a: p.hybrid[a] + (10.0 if a in linked else 0.0) for a in cands}
            keep = set(sorted(cands, key=lambda a: -pri[a])[: self.cfg.max_cands])
            self.truncated_cands += len(cands) - len(keep)
            cands = [a for a in cands if a in keep]
        return cands

    def stop_allowed(self) -> bool:
        return self.t >= 1

    def features(self, cands: Sequence[int]):
        """Returns (state vector [STATE_DIM], candidate features [len(cands), N_CAND_FEAT])."""
        p, sel, H = self.p, self.selected, self.cfg.horizon
        c = np.asarray(cands, dtype=np.int64)
        f = np.zeros((len(c), N_CAND_FEAT), dtype=np.float32)
        f[:, 0] = p.kind[c] == ROW
        f[:, 1] = p.kind[c] == PASSAGE
        f[:, 2] = p.cos[c]
        f[:, 3] = p.bm25n[c]
        f[:, 4] = p.rank_feat[c]
        covered = frozenset().union(*[p.qcov[s] for s in sel]) if sel else frozenset()
        if sel:
            linked = set().union(*[p.adj[s] for s in sel])
            last_adj = p.adj[sel[-1]]
            f[:, 5] = [a in linked for a in c]
            f[:, 6] = [a in last_adj for a in c]
            f[:, 12] = (p.emb[c] @ p.emb[sel].T).max(1)
        f[:, 7] = p.row_pos[c]
        f[:, 8] = p.overlap[c]
        f[:, 9] = [len(p.qcov[a] - covered) / p.n_qcw for a in c]
        f[:, 10] = p.num_match[c]
        f[:, 11] = p.global_top_arr[c]
        f[:, 13] = p.degree[c]

        s = np.zeros(STATE_DIM, dtype=np.float32)
        s[:EMB_DIM] = p.q
        if sel:
            s[EMB_DIM:2 * EMB_DIM] = p.emb[sel].mean(0)
            s[2 * EMB_DIM:3 * EMB_DIM] = p.emb[sel[-1]]
        k = 3 * EMB_DIM
        n_row_sel = sum(1 for x in sel if p.kind[x] == ROW)
        s[k + 0] = len(sel) / H
        s[k + 1] = n_row_sel / H
        s[k + 2] = (len(sel) - n_row_sel) / H
        s[k + 3] = len(covered) / p.n_qcw
        s[k + 4] = p.cos[c].max() if len(c) else 0.0
        s[k + 5] = p.bm25n[c].max() if len(c) else 0.0
        if sel:
            s[k + 6] = float(p.kind[sel[-1]] == ROW)
            s[k + 7] = float(p.kind[sel[-1]] == PASSAGE)
        return s, f

    def step(self, node: Optional[int]):
        """node=None means STOP."""
        assert not self.done
        if node is None:
            assert self.stop_allowed()
            self.done = True
        else:
            assert node not in self.selected
            self.selected.append(int(node))
            if self.t >= self.cfg.horizon:
                self.done = True


@dataclass
class RewardConfig:
    answer_metric: str = "f1"     # 'f1' or 'em' of the frozen reader's answer
    step_cost: float = 0.02       # paid for every selected entity
    evidence_bonus: float = 0.0   # lambda: terminal bonus lambda * psi(E_T) (changes the objective)
    potential: float = 0.0        # kappa: potential-based shaping with Phi(E) = kappa * psi(E)
    gamma: float = 1.0


def episode_rewards(p: Prepared, selected: Sequence[int], stopped_explicitly: bool,
                    answer_score: float, rc: RewardConfig) -> np.ndarray:
    """Per-step rewards for one finished episode.

    Steps: one per selection, plus one extra step if the episode ended with an explicit STOP.
    Terminal states have potential 0, so with gamma = 1 the shaping terms telescope to
    -Phi(s_0) = 0 and the optimal policy is unchanged (Ng, Harada and Russell, 1999).
    """
    n_steps = len(selected) + (1 if stopped_explicitly else 0)
    r = np.zeros(n_steps, dtype=np.float32)
    if n_steps == 0:          # degenerate graph without selectable entities
        return r
    phi = [0.0]
    for i in range(1, len(selected) + 1):
        phi.append(rc.potential * evidence_recall(p, selected[:i]))
    for i in range(len(selected)):
        r[i] -= rc.step_cost
        nxt = phi[i + 1] if (i + 1 < n_steps) else 0.0   # entering the terminal state
        r[i] += rc.gamma * nxt - phi[i]
    if stopped_explicitly:
        r[-1] += rc.gamma * 0.0 - phi[len(selected)]
    terminal = answer_score + rc.evidence_bonus * evidence_recall(p, selected)
    r[-1] += terminal
    return r
