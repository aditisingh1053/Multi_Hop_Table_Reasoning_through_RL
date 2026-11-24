"""Unit tests for the parts of the pipeline whose silent failure would invalidate results.

Run: python tests/test_core.py   (or: python -m pytest tests)
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mhtrl.env import (EMB_DIM, EnvConfig, Episode, Prepared, RewardConfig,
                       episode_rewards)
from mhtrl.device import DEVICE, to_device
from mhtrl.policy import Critic, PointerActor
from mhtrl.rl.algos import gae, returns_to_go
from mhtrl.rollout import StepRec, collate
from mhtrl.text import contains_answer, exact_match, f1_score, normalize_answer


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


# ---------------- reader-in-the-loop extension ----------------

TOY_READS = {0: ("Emmitt Smith", 2.0), 1: ("Walter Payton", 3.0), 2: ("James", 1.0), 3: ("Jerry", 0.5)}


class StubCache:
    """Reader cache with fixed outputs (no model download)."""
    reader_name = "stub"

    def __init__(self, qids):
        self.table = {(q, n): v for q in qids for n, v in TOY_READS.items()}

    def ensure(self, items, chunk=256):
        for qid, n, _, _ in items:
            assert (qid, n) in self.table
        return 0

    def answer(self, qid, nodes):
        return max((self.table[(qid, n)] for n in nodes), key=lambda x: x[1])[0]

    def save(self):
        pass


class ToyBank:
    def __init__(self, n=6):
        self.items = []
        for k in range(n):
            p = toy_instance()
            p.qid = f"q{k}"
            self.items.append(p)
        self.emb_all = np.concatenate([self.items[0].emb, np.zeros((1, EMB_DIM), np.float32)])
        self.pad_index = 4

    def __len__(self):
        return len(self.items)


def test_default_dims_unchanged():
    from mhtrl.env import dims
    from mhtrl.policy import make_actor
    d = dims(EnvConfig())
    assert d["state_dim"] == 3 * EMB_DIM + 8 and d["cand_dim"] == EMB_DIM + 14 and not d["commit"]
    # an old checkpoint (PointerActor()) loads into the default make_actor()
    make_actor(EnvConfig()).load_state_dict(PointerActor().state_dict())


def test_bridge_features_and_commit_phase():
    from mhtrl.env import N_CAND_FEAT, N_LOOP_CAND, N_LOOP_STATE, STATE_DIM
    p = toy_instance()
    cfg = EnvConfig(horizon=3, k_passages=0, reader_loop=True, commit=True)
    ep = Episode(p, cfg)
    s, f = ep.features(ep.candidates())
    assert s.shape == (STATE_DIM + N_LOOP_STATE,) and f.shape == (2, N_CAND_FEAT + N_LOOP_CAND)
    assert f[:, N_CAND_FEAT:N_CAND_FEAT + 4].sum() == 0          # nothing read yet
    ep.step(1)
    ep.reader_out[1] = TOY_READS[1]                                # reader: "Walter Payton"
    c = ep.candidates()
    s, f = ep.features(c)
    i3 = c.index(3)
    assert f[i3, N_CAND_FEAT + 1] == 1.0                           # passage titled by the bridge span
    assert s[-1] == 0.0                                            # not in commit phase
    ep.step(3)
    ep.reader_out[3] = TOY_READS[3]
    ep.step(None)                                                  # STOP -> COMMIT phase
    assert ep.in_commit and not ep.done and not ep.stop_allowed()
    c = ep.candidates()
    assert c == [1, 3]
    s, f = ep.features(c)
    assert s[-1] == 1.0
    assert f[c.index(1), N_CAND_FEAT + 6] == 1.0                   # max-confidence entity flagged
    ep.step(3)
    assert ep.done and ep.committed == 3 and ep.n_commit_steps == 1


def test_commit_prior_reproduces_max_confidence():
    from mhtrl.policy import make_actor
    torch.manual_seed(0)
    p = toy_instance()
    cfg = EnvConfig(horizon=3, k_passages=0, max_cands=4, reader_loop=True, commit=True)
    ep = Episode(p, cfg)
    for n in (3, 1):
        ep.step(n)
        ep.reader_out[n] = TOY_READS[n]
    ep.step(None)
    c = ep.candidates()
    s, f = ep.features(c)
    batch = collate(ToyBank(1), [StepRec(0, c, f, s, ep.stop_allowed())], 4)
    logits = make_actor(cfg)(**batch)[0]
    assert c[int(logits.argmax())] == 1                            # entity 1 has the higher reader score
    assert c == [3, 1]                                             # selection order
    # untrained head: logit difference = reader score difference (residual prior)
    assert abs((logits[1] - logits[0]).item() - (TOY_READS[1][1] - TOY_READS[3][1])) < 0.05


def test_rewards_with_commit_step_telescope():
    p = toy_instance()
    for stopped, sel in [(True, [1, 3]), (False, [0, 1, 3])]:
        base = episode_rewards(p, sel, stopped, 0.7, RewardConfig(potential=0.0), commit_steps=1)
        shaped = episode_rewards(p, sel, stopped, 0.7, RewardConfig(potential=0.5), commit_steps=1)
        assert len(base) == len(sel) + int(stopped) + 1
        assert abs(base.sum() - shaped.sum()) < 1e-6
        assert abs(base[-1] - 0.7) < 1e-6                          # answer reward on the COMMIT step


def test_loop_training_end_to_end():
    import tempfile
    from mhtrl.rl.algos import RLConfig, greedy_eval, train_rl
    bank = ToyBank(6)
    cache = StubCache([p.qid for p in bank.items])
    for reader_loop, commit in [(True, True), (True, False), (False, True), (False, False)]:
        cfg = EnvConfig(horizon=3, k_passages=0, max_cands=4, reader_loop=reader_loop, commit=commit)
        rl = RLConfig(iters=2, episodes_per_iter=8, minibatch=4, eval_every=1, actor_lr=3e-4)
        with tempfile.TemporaryDirectory() as d:
            best = train_rl(bank, bank, cache, cfg, RewardConfig(potential=0.5), rl, d, log=lambda *_: None)
        assert best["val_F1"] >= 0
        from mhtrl.policy import make_actor
        actor = make_actor(cfg).to(DEVICE)
        summ, rows = greedy_eval(bank, actor, cfg, cache)
        assert "F1_bestinset" in summ and summ["F1_bestinset"] >= summ["F1_maxconf"] - 1e-9
        if commit:
            assert all(r["commit"] in r["selected"] for r in rows if r["selected"])


# ---------------- optimisation extensions (RiL-PPO++) ----------------

def test_rloo_and_drgrpo_advantages():
    from mhtrl.rl.algos import group_advantages
    R = np.array([1.0, 0.0, 0.5, 0.5, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)
    a = group_advantages(R, 4, "rloo", True)
    # leave-one-out baseline, computed directly
    loo = np.array([R[i] - np.mean(np.delete(R[(i // 4) * 4:(i // 4) * 4 + 4], i % 4)) for i in range(8)])
    assert np.allclose(a, loo, atol=1e-6)
    assert np.allclose(a[4:], 0)                                   # degenerate group: no signal
    dr = group_advantages(R, 4, "grpo", False)
    assert np.allclose(dr, (R.reshape(2, 4) - R.reshape(2, 4).mean(1, keepdims=True)).reshape(-1))
    std = group_advantages(R, 4, "grpo", True)
    assert np.allclose(std[:4], dr[:4] / (R[:4].std() + 1e-6), atol=1e-5)


def test_action_dropout_and_temperature_rollout():
    from mhtrl.policy import make_actor
    from mhtrl.rollout import rollout
    torch.manual_seed(0)
    bank = ToyBank(6)
    cache = StubCache([p.qid for p in bank.items])
    cfg = EnvConfig(horizon=3, k_passages=0, max_cands=4, reader_loop=True, commit=True)
    actor = make_actor(cfg).to(DEVICE)
    gen = torch.Generator().manual_seed(0)
    ids = list(range(6)) * 20
    eps, recs, stopped = rollout(bank, ids, actor, cfg, greedy=False, gen=gen, cache=cache,
                                 action_dropout=0.9, temperature=0.7)
    for e, rs in zip(eps, recs):
        assert e.done and len(set(e.selected)) == len(e.selected)
        if len(e.selected) > 1:
            assert e.committed in e.selected and rs[-1].commit
        for r in rs:                                               # logp is the full (undropped) policy's
            batch = collate(bank, [r], 4)
            lp = torch.log_softmax(actor(**to_device(batch)), -1)[0, r.action].item()
            assert abs(lp - r.logp) < 1e-4
            assert r.action == 4 or r.action < len(r.cands)        # never a padded slot


def test_privileged_features_and_critic():
    from mhtrl.policy import make_critic
    from mhtrl.rl.algos import N_PRIV, privileged_features
    p = toy_instance()
    cache = StubCache([p.qid])
    f = privileged_features(p, (1, 3), [0, 2], cache, "f1")
    assert f.shape == (N_PRIV,)
    assert f[0] == 1.0 and f[1] == 1.0 and f[2] == 1.0 and f[3] == 0.0   # max-conf picks "Walter Payton"
    assert f[4] == 0.0                                             # neither candidate contains "Jerry"
    assert privileged_features(p, (), [3], cache, "f1")[4] == 1.0
    cfg = EnvConfig(horizon=3, k_passages=0, max_cands=4)
    ep = Episode(p, cfg)
    c = ep.candidates()
    s, fe = ep.features(c)
    batch = collate(ToyBank(1), [StepRec(0, c, fe, s, ep.stop_allowed())], 4)
    critic = make_critic(cfg, N_PRIV)
    assert critic(**batch, priv=torch.zeros(1, N_PRIV)).shape == (1,)
    try:
        critic(**batch)
        raise AssertionError("missing priv must raise")
    except ValueError:
        pass


def test_rao_blackwell_commit_gradient_is_exact():
    """d/dtheta sum_a pi(a) f(a) equals the expectation of the score-function estimator."""
    torch.manual_seed(0)
    logits = torch.randn(3, requires_grad=True)
    f = torch.tensor([0.0, 1.0, 0.3])
    exact = (torch.softmax(logits, -1) * f).sum()
    (g_exact,) = torch.autograd.grad(exact, logits)
    p = torch.softmax(logits, -1).detach()
    g_sf = sum(p[a] * f[a] * torch.autograd.grad(torch.log_softmax(logits, -1)[a], logits)[0] for a in range(3))
    assert torch.allclose(g_exact, g_sf, atol=1e-6)


def test_all_extensions_train_end_to_end():
    import json as _json
    import tempfile
    from mhtrl.rl.algos import RLConfig, train_rl
    bank = ToyBank(8)
    cache = StubCache([p.qid for p in bank.items])
    cfg = EnvConfig(horizon=3, k_passages=0, max_cands=4, reader_loop=True, commit=True)
    variants = [
        dict(algo="ppo", commit_rb=True),
        dict(algo="ppo", privileged_critic=True),
        dict(algo="ppo", sil_coef=1.0, sil_batches=2, sil_minibatch=8),
        dict(algo="ppo", action_dropout=0.3),
        dict(algo="ppo", ema=0.5),
        dict(algo="ppo", commit_rb=True, privileged_critic=True, sil_coef=1.0, sil_batches=2,
             sil_minibatch=8, action_dropout=0.1, ema=0.9),
        dict(algo="reinforce", commit_rb=True, privileged_critic=True, sil_coef=0.5, sil_minibatch=8),
        dict(algo="rloo", commit_rb=True, sil_coef=0.5, sil_minibatch=8),
        dict(algo="grpo", grpo_std_norm=False, commit_rb=True),
    ]
    for v in variants:
        rl = RLConfig(iters=3, episodes_per_iter=16, minibatch=8, eval_every=1, actor_lr=3e-4, **v)
        with tempfile.TemporaryDirectory() as d:
            best = train_rl(bank, bank, cache, cfg, RewardConfig(potential=0.5), rl, d, log=lambda *_: None)
            h = _json.load(open(Path(d) / "history.json"))
            assert (Path(d) / "actor_best.pt").exists()
        assert best["kind"] in ("raw", "ema") and 0 <= best["val_F1"] <= 100, v
        last = h["history"][-1]
        assert np.isfinite(last["return"]), v
        if v.get("sil_coef"):
            assert last["sil_buffer"] is not None
        if v.get("ema"):
            assert "val_ema" in last
        if v.get("commit_rb"):
            assert last["answer_rb"] is not None
    # commit_rb without the COMMIT step is a configuration error
    try:
        with tempfile.TemporaryDirectory() as d:
            train_rl(bank, bank, cache, EnvConfig(k_passages=0, max_cands=4), RewardConfig(),
                     RLConfig(iters=1, commit_rb=True), d, log=lambda *_: None)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_self_consistency_eval():
    from mhtrl.policy import make_actor
    from mhtrl.rollout import self_consistency_eval
    torch.manual_seed(0)
    bank = ToyBank(5)
    cache = StubCache([p.qid for p in bank.items])
    for commit in (True, False):
        cfg = EnvConfig(horizon=3, k_passages=0, max_cands=4, reader_loop=True, commit=commit)
        summ, rows = self_consistency_eval(bank, make_actor(cfg).to(DEVICE), cfg, cache, k=4, temperature=1.0)
        assert summ["k"] == 4 and len(rows) == 5
        assert 1 <= summ["Reads"] <= 4 and 0 <= summ["F1"] <= 100
        assert all(r["pred"] in {"Emmitt Smith", "Walter Payton", "James", "Jerry", ""} for r in rows)


def test_configs_load():
    from mhtrl.config import load_config
    root = Path(__file__).resolve().parents[1] / "configs"
    for f in sorted(root.rglob("*.yaml")):
        cfg = load_config(str(f))
        if cfg["rl"].commit_rb:
            assert cfg["env"].commit, f
        assert cfg["rl"].algo in ("ppo", "grpo", "rloo", "reinforce"), f


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
