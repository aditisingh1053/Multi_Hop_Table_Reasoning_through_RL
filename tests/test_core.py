"""Unit tests for the parts of the pipeline whose silent failure would invalidate results.

Run: python tests/test_core.py   (or: python -m pytest tests)
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mhtrl.env import (EMB_DIM, EnvConfig, Episode, Prepared, RewardConfig,  # noqa: E402
                       episode_rewards)
from mhtrl.policy import Critic, PointerActor  # noqa: E402
from mhtrl.rl.algos import gae, returns_to_go  # noqa: E402
from mhtrl.rollout import StepRec, collate  # noqa: E402
from mhtrl.text import contains_answer, exact_match, f1_score, normalize_answer  # noqa: E402


def toy_instance():
    inst = {
        "qid": "q1", "question": "What is the middle name of the player ranked 2 ?",
        "answer": "Jerry", "table_id": "t", "title": "T", "n_rows": 2,
        "node_kind": [0, 0, 1, 1],
        "node_text": ["T : Rank is 1 ; Player is Emmitt Smith", "T : Rank is 2 ; Player is Walter Payton",
                      "Emmitt Smith : Emmitt James Smith III is a player", "Walter Payton : Walter Jerry Payton was a player"],
        "node_ref": [0, 1, "/wiki/Emmitt_Smith", "/wiki/Walter_Payton"],
        "adj": [[2], [3], [0], [1]],
        "gold_nodes": [], "gold_paths": [(1, 3)], "answer_type": "passage",
    }
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(4, EMB_DIM)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    q = emb[1] * 0.5 + emb[3] * 0.5
    bm25 = np.array([0.1, 1.0, 0.0, 0.5], dtype=np.float32)
    return Prepared(inst, q, emb, bm25, emb_offset=0, k_passages=0)


def test_normalisation_matches_official():
    assert normalize_answer("The  Chicago-Bears!") == "chicagobears"
    assert exact_match("Walter Payton", "walter payton.") == 1.0
    assert abs(f1_score("Walter Jerry Payton", "Jerry Payton") - 0.8) < 1e-9
    assert contains_answer("Jerry", "Walter Jerry Payton")
    assert not contains_answer("Jerr", "Walter Jerry Payton")


def test_candidates_follow_graph():
    p = toy_instance()
    ep = Episode(p, EnvConfig(horizon=3, k_passages=0))
    assert ep.candidates() == [0, 1]                 # only rows before any hop
    assert not ep.stop_allowed()
    ep.step(1)
    assert ep.candidates() == [0, 3]                 # passage linked from row 1 appears
    assert ep.stop_allowed()
    s, f = ep.features(ep.candidates())
    assert f.shape == (2, 14) and f[1, 5] == 1.0 and f[1, 6] == 1.0
    ep.step(3)
    ep.step(0)
    assert ep.done                                   # horizon reached


def test_potential_shaping_telescopes():
    p = toy_instance()
    for stopped, sel in [(True, [1, 3]), (False, [0, 1, 3]), (True, [3]), (False, [0, 2, 1])]:
        base = episode_rewards(p, sel, stopped, 0.7, RewardConfig(potential=0.0))
        shaped = episode_rewards(p, sel, stopped, 0.7, RewardConfig(potential=0.5))
        assert abs(base.sum() - shaped.sum()) < 1e-6, (sel, base, shaped)
        assert len(base) == len(sel) + int(stopped)
    # shaping pays kappa * (psi(E_t+1) - psi(E_t)) per hop and takes it back on termination
    shaped = episode_rewards(p, [1, 3], True, 1.0, RewardConfig(potential=0.5, step_cost=0.0))
    assert np.allclose(shaped, [0.25, 0.25, 0.5])
    # the terminal bonus is not policy-invariant: it adds lambda * psi(E_T) to the return
    bonus = episode_rewards(p, [1], True, 0.0, RewardConfig(evidence_bonus=0.3, step_cost=0.0))
    assert np.allclose(bonus.sum(), 0.15)


def test_gae_limits():
    r = np.array([0.0, -0.02, 1.0], dtype=np.float32)
    v = np.array([0.3, 0.5, 0.8], dtype=np.float32)
    adv1, ret1 = gae(r, v, 1.0, 1.0)
    assert np.allclose(ret1, returns_to_go(r, 1.0))
    assert np.allclose(adv1, returns_to_go(r, 1.0) - v)
    adv0, _ = gae(r, v, 1.0, 0.0)
    assert np.allclose(adv0, [0.0 + 0.5 - 0.3, -0.02 + 0.8 - 0.5, 1.0 - 0.8])


def test_masking_gives_zero_probability_and_gradient():
    torch.manual_seed(0)
    p = toy_instance()
    ep = Episode(p, EnvConfig(horizon=3, k_passages=0, max_cands=4))
    c = ep.candidates()
    s, f = ep.features(c)

    class B:  # minimal bank stub for collate
        items = [p]
        pad_index = 4
        emb_all = np.concatenate([p.emb, np.zeros((1, EMB_DIM), np.float32)])

    batch = collate(B, [StepRec(0, c, f, s, ep.stop_allowed())], 4)
    actor = PointerActor()
    logits = actor(**batch)
    probs = torch.softmax(logits, -1)[0]
    assert torch.all(probs[2:] == 0)                 # 2 padded slots and STOP (t = 0)
    assert abs(probs[:2].sum().item() - 1) < 1e-6
    critic = Critic()
    assert critic(**batch).shape == (1,)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
