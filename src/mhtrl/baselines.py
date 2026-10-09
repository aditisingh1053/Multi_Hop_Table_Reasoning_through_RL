"""Non-learned evidence selectors. Each returns a list of node indices for one question.

random_walk   uniform choice from the environment's candidate set, fixed budget
topk(score)   the k highest-scoring entities, ignoring the graph (BM25 / dense / hybrid);
              the dense variant is the row/cell retrieval step of TableRAG-style pipelines
greedy_hop    link-then-hop heuristic (HYBRIDER / hybrid-graph style): best row by hybrid
              score, then the best passage hyperlinked from that row
oracle        best traced gold path, judged by the frozen reader (upper bound)
"""
from typing import List

import numpy as np

from .data import PASSAGE
from .env import EnvConfig, Episode, Prepared
from .reader import ReaderCache
from .text import f1_score


def random_walk(p: Prepared, env_cfg: EnvConfig, k: int, rng: np.random.Generator) -> List[int]:
    ep = Episode(p, env_cfg)
    while not ep.done and ep.t < k:
        c = ep.candidates()
        if not c:
            break
        ep.step(int(rng.choice(c)))
    return ep.selected


def topk(p: Prepared, score: np.ndarray, k: int) -> List[int]:
    return [int(i) for i in np.argsort(-score, kind="stable")[:k]]


def greedy_hop(p: Prepared) -> List[int]:
    if p.n_rows == 0:
        return []
    rows = np.arange(p.n_rows)
    r = int(rows[np.argmax(p.hybrid[rows])])
    linked = [a for a in p.adj[r] if p.kind[a] == PASSAGE]
    if not linked:
        return [r]
    a = max(linked, key=lambda x: p.hybrid[x])
    return [r, int(a)]


def oracle(p: Prepared, cache: ReaderCache) -> List[int]:
    if not p.gold_paths:
        return []
    cache.ensure((p.qid, n, p.question, p.text[n]) for path in p.gold_paths for n in path)
    best, best_f1 = list(p.gold_paths[0]), -1.0
    for path in p.gold_paths:
        f1 = f1_score(p.answer, cache.answer(p.qid, path))
        if f1 > best_f1:
            best, best_f1 = list(path), f1
    return best
