"""The evidence-selection MDP over one question's graph of table rows and passages.

At each step the agent picks a row or passage (or STOP after the first pick), up to H picks.
With `reader_loop` the reader's answer for each picked entity is added to the observation, and
with `commit` one extra step chooses which picked entity gives the final answer.
"""
import math
import pickle
import re
from dataclasses import dataclass
from functools import cached_property
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .data import PASSAGE, ROW
from .text import STOPWORDS, contains_answer, content_words, normalize_answer, numbers, words

EMB_DIM = 384
N_CAND_FEAT = 14
N_STATE_SCALAR = 8
STATE_DIM = 3 * EMB_DIM + N_STATE_SCALAR

# reader-in-the-loop features, appended after the base ones
LOOP_CAND_FEATURES = [
    "bridge_in_text", "bridge_is_title", "last_bridge_in_text", "bridge_in_row",   # select phase
    "commit_conf", "commit_score", "commit_is_maxconf", "commit_span_in_q",          # commit phase
    "commit_agreement", "commit_span_numeric", "commit_span_len",
]
LOOP_STATE_SCALARS = ["max_conf", "last_conf", "last_span_informative",
                      "q_who", "q_when", "q_number", "q_where", "q_other", "commit_phase"]
N_LOOP_CAND = len(LOOP_CAND_FEATURES)
N_LOOP_STATE = len(LOOP_STATE_SCALARS)
LOOP_SCORE_IDX = N_CAND_FEAT + LOOP_CAND_FEATURES.index("commit_score")
SCORE_SCALE = 5.0   # commit_score = clip(reader score, -20, 20) / SCORE_SCALE

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
    reader_loop: bool = False   # reader feedback (bridge spans, confidences) in the observation
    commit: bool = False        # learned COMMIT step choosing the answer-bearing entity


def uses_loop(cfg: EnvConfig) -> bool:
    return bool(cfg.reader_loop or cfg.commit)


def dims(cfg: EnvConfig) -> dict:
    """Network input sizes for an environment configuration."""
    extra_s, extra_c = (N_LOOP_STATE, N_LOOP_CAND) if uses_loop(cfg) else (0, 0)
    return {"state_dim": STATE_DIM + extra_s, "cand_dim": EMB_DIM + N_CAND_FEAT + extra_c,
            "n_scalar": N_STATE_SCALAR + extra_s, "commit": bool(cfg.commit)}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def _contains(span: str, norm_text: str) -> bool:
    return bool(span) and (" " + span + " ") in (" " + norm_text + " ")


def _informative(span: str) -> bool:
    """A span usable as a bridge: non-empty, short, and not only stop words."""
    toks = span.split()
    return 0 < len(toks) <= 8 and any(t not in STOPWORDS for t in toks)


def question_type(q: str) -> int:
    """0 who, 1 when/year, 2 number, 3 where, 4 other."""
    ql = q.lower()
    if re.search(r"\bhow (many|much|long|old|tall|far|big|large)\b", ql) or "what number" in ql:
        return 2
    if re.search(r"\b(when|what year|which year|what date|in what year)\b", ql):
        return 1
    if re.search(r"\bwho(m|se)?\b", ql):
        return 0
    if re.search(r"\bwhere\b", ql):
        return 3
    return 4


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

    # lazily computed, used only by the reader-in-the-loop features
    @cached_property
    def norm_text(self) -> List[str]:
        return [normalize_answer(t) for t in self.text]

    @cached_property
    def norm_title(self) -> List[str]:
        return [normalize_answer(t.split(" : ", 1)[0]) if k == PASSAGE else ""
                for t, k in zip(self.text, self.kind)]

    @cached_property
    def q_words(self) -> frozenset:
        return frozenset(normalize_answer(self.question).split())

    @cached_property
    def q_type(self) -> int:
        return question_type(self.question)


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
        self.in_commit = False                 # COMMIT phase (cfg.commit only)
        self.committed: Optional[int] = None   # entity whose span is the final answer
        self.reader_out: Dict[int, Tuple[str, float]] = {}   # filled by the rollout

    @property
    def t(self) -> int:
        return len(self.selected)

    @property
    def n_commit_steps(self) -> int:
        """1 if this episode has (or had) a COMMIT decision step."""
        return int(self.cfg.commit and len(self.selected) > 1)

    def candidates(self) -> List[int]:
        if self.in_commit:
            return list(self.selected)
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
        return self.t >= 1 and not self.in_commit

    def features(self, cands: Sequence[int]):
        """Returns (state vector, candidate features [len(cands), n_feat]).

        Without the loop extension: [STATE_DIM] and [len(cands), N_CAND_FEAT] exactly as before.
        With it, N_LOOP_STATE scalars and N_LOOP_CAND features are appended.
        """
        s, f = self._base_features(cands)
        if not uses_loop(self.cfg):
            return s, f
        ls, lf = self._loop_features(cands)
        return np.concatenate([s, ls]), np.concatenate([f, lf], 1)

    def _loop_features(self, cands: Sequence[int]):
        p, sel, H, ro = self.p, self.selected, self.cfg.horizon, self.reader_out
        fx = np.zeros((len(cands), N_LOOP_CAND), dtype=np.float32)
        sx = np.zeros(N_LOOP_STATE, dtype=np.float32)
        # reader outputs are visible while selecting only with reader_loop, and always when committing
        see_reader = self.cfg.reader_loop or self.in_commit
        if not see_reader:
            return sx, fx
        spans = {e: normalize_answer(ro[e][0]) for e in sel if e in ro}
        scores = {e: ro[e][1] for e in sel if e in ro}
        if scores:
            sx[0] = _sigmoid(max(scores.values()) / 4.0)
            sx[1] = _sigmoid(scores.get(sel[-1], -20.0) / 4.0)
            sx[2] = float(_informative(spans.get(sel[-1], "")))
        sx[3 + p.q_type] = 1.0
        sx[8] = float(self.in_commit)

        if self.in_commit:
            best = max(scores, key=scores.get) if scores else None
            for i, e in enumerate(cands):
                sp, sc = spans.get(e, ""), scores.get(e, -20.0)
                toks = sp.split()
                fx[i, 4] = _sigmoid(sc / 4.0)
                fx[i, 5] = max(-20.0, min(20.0, sc)) / SCORE_SCALE
                fx[i, 6] = float(e == best)
                fx[i, 7] = (sum(t in p.q_words for t in toks) / len(toks)) if toks else 1.0
                fx[i, 8] = sum(_contains(sp, p.norm_text[o]) for o in sel if o != e) / H
                fx[i, 9] = float(any(ch.isdigit() for ch in sp))
                fx[i, 10] = min(len(toks), 10) / 10.0
            return sx, fx

        bridges = [s for s in spans.values() if _informative(s)]
        last = spans.get(sel[-1], "") if sel else ""
        last = last if _informative(last) else ""
        if bridges:
            for i, a in enumerate(cands):
                nt = p.norm_text[a]
                hit = any(_contains(b, nt) for b in bridges)
                fx[i, 0] = float(hit)
                if p.kind[a] == PASSAGE:
                    title = p.norm_title[a]
                    fx[i, 1] = float(bool(title) and any(
                        b == title or _contains(b, title) or _contains(title, b) for b in bridges))
                else:
                    fx[i, 3] = float(hit)
                fx[i, 2] = float(_contains(last, nt))
        return sx, fx

    def _base_features(self, cands: Sequence[int]):
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
        """node=None means STOP. In the COMMIT phase, node is the entity to answer from."""
        assert not self.done
        if self.in_commit:
            assert node is not None and node in self.selected
            self.committed = int(node)
            self.done = True
            return
        if node is None:
            assert self.stop_allowed()
            self._end_selection()
        else:
            assert node not in self.selected
            self.selected.append(int(node))
            if self.t >= self.cfg.horizon:
                self._end_selection()

    def _end_selection(self):
        if self.n_commit_steps:
            self.in_commit = True          # one more decision: which entity to answer from
        else:
            if self.cfg.commit and self.selected:
                self.committed = self.selected[0]   # a single entity: nothing to choose
            self.done = True


@dataclass
class RewardConfig:
    answer_metric: str = "f1"     # 'f1' or 'em' of the frozen reader's answer
    step_cost: float = 0.02       # paid for every selected entity
    evidence_bonus: float = 0.0   # lambda: terminal bonus lambda * psi(E_T) (changes the objective)
    potential: float = 0.0        # kappa: potential-based shaping with Phi(E) = kappa * psi(E)
    gamma: float = 1.0


def episode_rewards(p: Prepared, selected: Sequence[int], stopped_explicitly: bool,
                    answer_score: float, rc: RewardConfig, commit_steps: int = 0) -> np.ndarray:
    """Per-step rewards: step cost, potential-based shaping and the answer score at the end.

    The shaping terms sum to zero over an episode (Phi is 0 at the start and the end), so they do
    not change the optimal policy. The COMMIT step, if any, keeps the potential of the final set.
    """
    n_sel = len(selected)
    n_steps = n_sel + (1 if stopped_explicitly else 0) + commit_steps
    r = np.zeros(n_steps, dtype=np.float32)
    if n_steps == 0:          # degenerate graph without selectable entities
        return r
    phi = [0.0]
    for i in range(1, n_sel + 1):
        phi.append(rc.potential * evidence_recall(p, selected[:i]))

    def pot(k):               # potential of the state before step k
        return 0.0 if k >= n_steps else phi[min(k, n_sel)]

    for k in range(n_steps):
        if k < n_sel:
            r[k] -= rc.step_cost
        r[k] += rc.gamma * pot(k + 1) - pot(k)
    terminal = answer_score + rc.evidence_bonus * evidence_recall(p, selected)
    r[-1] += terminal
    return r
