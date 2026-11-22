"""PPO (clipped surrogate + GAE, learned critic), GRPO / RLOO (group-relative, critic-free) and
REINFORCE with a learned baseline, sharing one rollout/reward/evaluation pipeline.

PPO     A_t = sum_l (gamma*lambda)^l delta_{t+l},  delta_t = r_t + gamma V(s_{t+1}) - V(s_t)
        L   = E[min(rho_t A_t, clip(rho_t, 1-eps, 1+eps) A_t)] + c_H * entropy
GRPO    G episodes per question; A_i = (R_i - mean_G R) / (std_G R + 1e-6) for every step of
        episode i; same clipped surrogate, no critic. grpo_std_norm=False gives Dr. GRPO.
RLOO    A_i = R_i - mean_{j != i} R_j = G/(G-1) (R_i - mean_G R)   (Ahmadian et al., 2024).
REINF.  A_t = G_t - V(s_t) with Monte-Carlo return-to-go G_t; one pass, no ratio, no clip.
Losses are averaged over all steps of a minibatch (no per-episode length normalisation).

Optional extensions (all off by default, so the algorithms above are unchanged):
  commit_rb          Rao-Blackwellised COMMIT. The reader is decomposable and cached, so the
                     answer score f(a) of *every* COMMIT option is known. The COMMIT step is
                     trained with the exact expected objective sum_a pi(a|s) f(a) (an
                     all-action policy gradient with zero sampling variance), and the selection
                     steps receive the conditional expectation E_{a~pi_old}[f(a)] as terminal
                     reward instead of one sampled f(a): same expected return, lower variance.
  privileged_critic  asymmetric actor-critic (Pinto et al., 2018): the critic also sees
                     training-only features (gold-path coverage, reader F1 of the evidence,
                     whether a candidate contains the answer). The actor never sees them.
  sil_coef > 0       self-imitation learning (Oh et al., 2018): a per-question buffer of the best
                     own episode; extra updates on -log pi(a|s) * max(G_t - V(s_t), 0). Imitates
                     what worked *for the reader*, unlike cloning the annotated gold paths.
  action_dropout     exploration by hiding candidates from the sampler (Lin et al., 2018).
  ema > 0            Polyak / exponential moving average of actor weights (Polyak and Juditsky,
                     1992; SWA, Izmailov et al., 2018), evaluated next to the raw iterate.
"""
import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ..device import DEVICE, to_device
from ..env import (Bank, EnvConfig, Prepared, RewardConfig, answer_in_evidence, episode_rewards,
                   evidence_recall)
from ..policy import make_actor, make_critic
from ..reader import ReaderCache
from ..rollout import answer_score, collate, evaluate_sets, predict, rollout

ALGOS = ("ppo", "grpo", "rloo", "reinforce")
N_PRIV = 6
PRIV_FEATURES = ["evidence_recall", "answer_in_evidence", "best_span_score_in_evidence",
                 "maxconf_answer_score", "any_candidate_contains_answer", "frac_candidates_with_answer"]


@dataclass
class RLConfig:
    algo: str = "ppo"             # ppo | grpo | rloo | reinforce
    iters: int = 40
    episodes_per_iter: int = 512
    group_size: int = 4           # grpo / rloo only
    epochs: int = 4
    minibatch: int = 256
    clip: float = 0.2
    actor_lr: float = 1e-4
    critic_lr: float = 5e-4
    ent_coef: float = 0.01
    gae_lambda: float = 0.95
    target_kl: float = 0.03       # stop the epoch loop when approx. KL exceeds 1.5 * target
    max_grad_norm: float = 0.5
    critic_warmup: int = 3        # iterations that train only the critic (when warm-started)
    grpo_std_norm: bool = True
    adv_norm: bool = True
    anneal_lr: bool = True
    eval_every: int = 5
    seed: int = 0
    init_actor: str = ""          # BC checkpoint; empty = random initialisation
    # --- extensions (see module docstring) ---
    commit_rb: bool = False
    commit_coef: float = 1.0
    privileged_critic: bool = False
    sil_coef: float = 0.0
    sil_batches: int = 4
    sil_minibatch: int = 256
    action_dropout: float = 0.0
    ema: float = 0.0


def gae(rewards: np.ndarray, values: np.ndarray, gamma: float, lam: float):
    """GAE for one finished episode; the state after the last step is terminal (V = 0)."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        nxt = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * nxt - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + values


def returns_to_go(rewards: np.ndarray, gamma: float):
    out = np.zeros_like(rewards)
    run = 0.0
    for t in reversed(range(len(rewards))):
        run = rewards[t] + gamma * run
        out[t] = run
    return out


def group_advantages(total: np.ndarray, group_size: int, algo: str, std_norm: bool) -> np.ndarray:
    """Per-episode advantages for the critic-free algorithms (episodes grouped by question)."""
    grp = total.reshape(-1, group_size)
    mu = grp.mean(1, keepdims=True)
    if algo == "rloo":
        a = (grp - mu) * group_size / (group_size - 1)
    else:
        a = grp - mu
        if std_norm:
            a = a / (grp.std(1, keepdims=True) + 1e-6)
    return a.reshape(-1)


def privileged_features(p: Prepared, sel, cands, cache: ReaderCache, metric: str) -> np.ndarray:
    """Training-only critic inputs. Every entity in `sel` has already been read (the reward
    needs it), so this costs no reader calls."""
    f = np.zeros(N_PRIV, dtype=np.float32)
    sel = list(sel)
    if sel:
        f[0] = evidence_recall(p, sel)
        f[1] = float(answer_in_evidence(p, sel))
        outs = [cache.table[(p.qid, n)] for n in sel]
        f[2] = max(answer_score(p.answer, a, metric) for a, _ in outs)
        f[3] = answer_score(p.answer, max(outs, key=lambda x: x[1])[0], metric)
    if len(cands):
        c = np.asarray(cands, dtype=np.int64)
        f[4] = float(p.contains[c].any())
        f[5] = float(p.contains[c].mean())
    return f


def greedy_eval(bank: Bank, actor, env_cfg: EnvConfig, cache: ReaderCache, ids=None):
    ids = list(range(len(bank))) if ids is None else ids
    actor.eval()
    eps, _, _ = rollout(bank, ids, actor, env_cfg, greedy=True, cache=cache)
    actor.train()
    commits = [e.committed for e in eps] if env_cfg.commit else None
    return evaluate_sets(bank, ids, [e.selected for e in eps], cache, commits)


def train_rl(train_bank: Bank, val_bank: Bank, cache: ReaderCache, env_cfg: EnvConfig,
             rc: RewardConfig, cfg: RLConfig, out_dir: str, log=print):
    if cfg.algo not in ALGOS:
        raise ValueError(f"unknown algo {cfg.algo!r}; expected one of {ALGOS}")
    if cfg.commit_rb and not env_cfg.commit:
        raise ValueError("commit_rb needs env.commit = true")
    if cfg.algo in ("grpo", "rloo") and cfg.group_size < 2:
        raise ValueError("group methods need group_size >= 2")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    k_max = env_cfg.max_cands
    metric = rc.answer_metric

    n_priv = N_PRIV if cfg.privileged_critic else 0
    actor, critic = make_actor(env_cfg).to(DEVICE), make_critic(env_cfg, n_priv).to(DEVICE)
    if cfg.init_actor:
        actor.load_state_dict(torch.load(cfg.init_actor, map_location="cpu"))
    use_critic = cfg.algo in ("ppo", "reinforce")
    group_algo = cfg.algo in ("grpo", "rloo")
    opt_a = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr, eps=1e-5)
    opt_c = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr, eps=1e-5)

    ema_state = ema_actor = None
    if cfg.ema > 0:
        ema_state = {k: v.detach().clone() for k, v in actor.state_dict().items()}
        ema_actor = make_actor(env_cfg).to(DEVICE)
    sil_buf = {}   # item index -> (return of the episode, its step records, returns-to-go)

    order, cursor = rng.permutation(len(train_bank)), 0

    def next_questions(n):
        nonlocal order, cursor
        picked = []
        while len(picked) < n:
            if cursor >= len(order):
                order, cursor = rng.permutation(len(train_bank)), 0
            take = min(n - len(picked), len(order) - cursor)
            picked.extend(order[cursor:cursor + take].tolist())
            cursor += take
        return picked

    def priv_of(recs_):
        return np.stack([privileged_features(train_bank.items[r.item], r.sel, r.cands, cache, metric)
                         for r in recs_]) if n_priv else None

    def critic_values(recs_, priv_=None):
        vals = np.zeros(len(recs_), dtype=np.float32)
        with torch.no_grad():
            for c0 in range(0, len(recs_), 1024):
                b = to_device(collate(train_bank, recs_[c0:c0 + 1024], k_max))
                if n_priv:
                    b["priv"] = torch.from_numpy(priv_[c0:c0 + 1024]).to(DEVICE)
                vals[c0:c0 + 1024] = critic(**b).float().cpu().numpy()
        return vals

    history = []
    val_sum, _ = greedy_eval(val_bank, actor, env_cfg, cache)
    # the selected checkpoint is always an RL iterate (never the initialisation itself)
    best = {"iter": 0, "val_F1": -1.0, "kind": "raw", "state": copy.deepcopy(actor.state_dict())}
    history.append({"iter": 0, "episodes": 0, "val": val_sum})
    log(f"[{cfg.algo} seed {cfg.seed}] iter 0 val F1 {val_sum['F1']:.2f} EM {val_sum['EM']:.2f}")
    episodes_seen = 0

    for it in range(1, cfg.iters + 1):
        t0 = time.time()
        if cfg.anneal_lr:
            frac = 1.0 - (it - 1) / cfg.iters
            for g in opt_a.param_groups:
                g["lr"] = cfg.actor_lr * frac
            for g in opt_c.param_groups:
                g["lr"] = cfg.critic_lr * frac

        if group_algo:
            qs = next_questions(cfg.episodes_per_iter // cfg.group_size)
            item_ids = [q for q in qs for _ in range(cfg.group_size)]
        else:
            item_ids = next_questions(cfg.episodes_per_iter)
        episodes_seen += len(item_ids)

        eps, recs, stopped = rollout(train_bank, item_ids, actor, env_cfg, greedy=False, gen=gen,
                                     cache=cache, action_dropout=cfg.action_dropout)
        sets = [e.selected for e in eps]
        commits = [e.committed for e in eps] if env_cfg.commit else None
        preds = predict(train_bank, item_ids, sets, cache, commits)
        ans = [answer_score(train_bank.items[i].answer, pr, metric) for i, pr in zip(item_ids, preds)]

        keep = [j for j in range(len(eps)) if recs[j]]
        flat = [r for j in keep for r in recs[j]]
        if not flat:
            continue
        ep_len = [len(recs[j]) for j in keep]
        bounds = np.cumsum([0] + ep_len)
        owner = np.zeros(len(flat), dtype=np.int64)
        for e, j in enumerate(keep):
            owner[bounds[e]:bounds[e + 1]] = j
        is_commit = np.array([r.commit for r in flat], dtype=bool)

        # Rao-Blackwellised COMMIT: exact value of every COMMIT option, and the expected
        # terminal answer under the behaviour policy for the selection steps
        ans_used = list(ans)
        commit_vals = None
        if cfg.commit_rb:
            commit_vals = np.zeros((len(flat), k_max + 1), dtype=np.float32)
            cidx = np.where(is_commit)[0]
            for fi in cidx:
                r = flat[fi]
                p = train_bank.items[r.item]
                for ai, n in enumerate(r.cands):
                    commit_vals[fi, ai] = answer_score(p.answer, cache.table[(p.qid, n)][0], metric)
            if len(cidx):
                with torch.no_grad():
                    probs = []
                    for c0 in range(0, len(cidx), 1024):
                        part = [flat[i] for i in cidx[c0:c0 + 1024]]
                        lg = actor(**to_device(collate(train_bank, part, k_max))).float()
                        probs.append(torch.softmax(lg, -1).cpu().numpy())
                    probs = np.concatenate(probs, 0)
                exp_val = (probs * commit_vals[cidx]).sum(-1)
                for fi, v in zip(cidx, exp_val):
                    ans_used[owner[fi]] = float(v)

        def rewards_with(answers):
            return [episode_rewards(train_bank.items[i], s, st, a, rc, commit_steps=e.n_commit_steps)
                    for i, s, st, a, e in zip(item_ids, sets, stopped, answers, eps)]

        rewards = rewards_with(ans_used)
        rewards_sampled = rewards_with(ans) if cfg.commit_rb else rewards
        assert all(len(rw) == len(rc_) for rw, rc_ in zip(rewards, recs)), "reward/step mismatch"

        priv = priv_of(flat)
        values = critic_values(flat, priv) if use_critic else np.zeros(len(flat), dtype=np.float32)

        adv = np.zeros(len(flat), dtype=np.float32)
        ret = np.zeros(len(flat), dtype=np.float32)
        if cfg.algo == "ppo":
            for e, j in enumerate(keep):
                a, b = bounds[e], bounds[e + 1]
                adv[a:b], ret[a:b] = gae(rewards[j], values[a:b], rc.gamma, cfg.gae_lambda)
        elif cfg.algo == "reinforce":
            for e, j in enumerate(keep):
                a, b = bounds[e], bounds[e + 1]
                ret[a:b] = returns_to_go(rewards[j], rc.gamma)
                adv[a:b] = ret[a:b] - values[a:b]
        else:  # grpo / rloo: group-relative advantage of the episode return, shared by all its steps
            total = np.array([rewards[j].sum() for j in range(len(eps))], dtype=np.float32)
            a_ep = group_advantages(total, cfg.group_size, cfg.algo, cfg.grpo_std_norm)
            for e, j in enumerate(keep):
                adv[bounds[e]:bounds[e + 1]] = a_ep[j]
        if cfg.adv_norm and not group_algo:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        old_logp = np.array([r.logp for r in flat], dtype=np.float32)
        actions = np.array([r.action for r in flat], dtype=np.int64)
        train_actor = not (cfg.init_actor and it <= cfg.critic_warmup)
        n_epochs = 1 if cfg.algo == "reinforce" else cfg.epochs
        stats = {"kl": [], "clipfrac": [], "ent": [], "pi_loss": [], "v_loss": [], "sil_loss": []}
        kl_stop = False
        if cfg.algo == "reinforce":
            opt_a.zero_grad()   # one gradient step on the whole batch (vanilla policy gradient)
        for epoch in range(cfg.epochs if use_critic else n_epochs):
            perm = rng.permutation(len(flat))
            for c0 in range(0, len(flat), cfg.minibatch):
                mb = perm[c0:c0 + cfg.minibatch]
                batch = to_device(collate(train_bank, [flat[i] for i in mb], k_max))
                if use_critic:
                    extra = {"priv": torch.from_numpy(priv[mb]).to(DEVICE)} if n_priv else {}
                    v = critic(**batch, **extra)
                    v_loss = 0.5 * ((v - torch.from_numpy(ret[mb]).to(DEVICE)) ** 2).mean()
                    opt_c.zero_grad()
                    v_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
                    opt_c.step()
                    stats["v_loss"].append(v_loss.item())
                if not train_actor or kl_stop or epoch >= n_epochs:
                    continue
                logits = actor(**batch)
                logp_all = torch.log_softmax(logits, -1)
                act = torch.from_numpy(actions[mb]).to(DEVICE)
                logp = logp_all.gather(-1, act.unsqueeze(-1)).squeeze(-1)
                probs = logp_all.exp()
                ent = -(probs * logp_all.clamp(min=-1e4)).sum(-1)
                A = torch.from_numpy(adv[mb]).to(DEVICE)
                log_ratio = logp - torch.from_numpy(old_logp[mb]).to(DEVICE)
                if cfg.algo == "reinforce":
                    row = -(logp * A)
                else:
                    ratio = log_ratio.exp()
                    row = -torch.min(ratio * A, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * A)
                    stats["clipfrac"].append(((ratio - 1).abs() > cfg.clip).float().mean().item())
                if cfg.commit_rb:
                    cm = torch.from_numpy(is_commit[mb]).to(DEVICE)
                    if bool(cm.any()):
                        exact = (probs * torch.from_numpy(commit_vals[mb]).to(DEVICE)).sum(-1)
                        row = torch.where(cm, -cfg.commit_coef * exact, row)
                pi_loss = row.mean()
                if cfg.algo == "reinforce":
                    loss = (pi_loss - cfg.ent_coef * ent.mean()) * (len(mb) / len(flat))
                    loss.backward()          # accumulate; the single step is taken below
                else:
                    loss = pi_loss - cfg.ent_coef * ent.mean()
                    opt_a.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
                    opt_a.step()
                with torch.no_grad():
                    kl = ((log_ratio.exp() - 1) - log_ratio).mean().item()
                stats["kl"].append(kl)
                stats["ent"].append(ent.mean().item())
                stats["pi_loss"].append(pi_loss.item())
            if cfg.algo == "reinforce" and epoch == 0 and train_actor:
                torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
                opt_a.step()
            if stats["kl"] and np.mean(stats["kl"][-max(1, len(flat) // cfg.minibatch):]) > 1.5 * cfg.target_kl \
                    and cfg.algo != "reinforce":
                kl_stop = True

        # self-imitation: keep each question's best successful episode, imitate positive-advantage steps
        if cfg.sil_coef > 0:
            for e, j in enumerate(keep):
                if ans[j] <= 0:
                    continue
                G = returns_to_go(rewards_sampled[j], rc.gamma)
                item = item_ids[j]
                if item not in sil_buf or G[0] > sil_buf[item][0] + 1e-9:
                    sil_buf[item] = (float(G[0]), recs[j], G)
            steps = [(r, g) for _, rs, Gs in sil_buf.values() for r, g in zip(rs, Gs)]
            if steps and train_actor:
                for _ in range(cfg.sil_batches):
                    pick = rng.choice(len(steps), size=min(cfg.sil_minibatch, len(steps)), replace=False)
                    srecs = [steps[i][0] for i in pick]
                    sret = np.array([steps[i][1] for i in pick], dtype=np.float32)
                    base = critic_values(srecs, priv_of(srecs)) if use_critic else np.zeros_like(sret)
                    sadv = torch.from_numpy(np.maximum(sret - base, 0.0)).to(DEVICE)
                    if float(sadv.sum()) <= 0:
                        continue
                    sb = to_device(collate(train_bank, srecs, k_max))
                    lp_all = torch.log_softmax(actor(**sb), -1)
                    sact = torch.tensor([r.action for r in srecs], dtype=torch.int64, device=DEVICE)
                    slp = lp_all.gather(-1, sact.unsqueeze(-1)).squeeze(-1)
                    sil_loss = -cfg.sil_coef * (slp * sadv).mean()
                    opt_a.zero_grad()
                    sil_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
                    opt_a.step()
                    stats["sil_loss"].append(sil_loss.item())

        if ema_state is not None and train_actor:
            with torch.no_grad():
                for k, v in actor.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema_state[k].mul_(cfg.ema).add_(v.detach(), alpha=1.0 - cfg.ema)
                    else:
                        ema_state[k].copy_(v)

        ev = float("nan")
        if use_critic:
            var = ret.var()
            ev = float(1 - (ret - values).var() / var) if var > 1e-8 else float("nan")
        rec = {
            "iter": it, "episodes": episodes_seen,
            "return": float(np.mean([r.sum() for r in rewards])),
            "answer": float(np.mean(ans)),
            "answer_rb": float(np.mean(ans_used)) if cfg.commit_rb else None,
            "items": float(np.mean([len(s) for s in sets])),
            "explicit_stop": float(np.mean(stopped)),
            "commit_steps": float(np.mean(is_commit)) if env_cfg.commit else None,
            "zero_adv_groups": float(np.mean(
                np.isclose(np.array([rw.sum() for rw in rewards]).reshape(-1, cfg.group_size).std(1), 0)
            )) if group_algo else None,
            "sil_buffer": len(sil_buf) if cfg.sil_coef > 0 else None,
            "explained_var": ev, "train_actor": train_actor,
            **{k: (float(np.mean(v)) if v else None) for k, v in stats.items()},
            "reader_cache": len(cache.table), "sec": time.time() - t0,
        }
        if it % cfg.eval_every == 0 or it == cfg.iters:
            val_sum, _ = greedy_eval(val_bank, actor, env_cfg, cache)
            rec["val"] = val_sum
            if val_sum["F1"] > best["val_F1"]:
                best = {"iter": it, "val_F1": val_sum["F1"], "kind": "raw",
                        "state": copy.deepcopy(actor.state_dict())}
            if ema_state is not None:
                ema_actor.load_state_dict(ema_state)
                val_ema, _ = greedy_eval(val_bank, ema_actor, env_cfg, cache)
                rec["val_ema"] = val_ema
                if val_ema["F1"] > best["val_F1"]:
                    best = {"iter": it, "val_F1": val_ema["F1"], "kind": "ema",
                            "state": copy.deepcopy(ema_state)}
            cache.save()
        history.append(rec)
        log(f"[{cfg.algo} seed {cfg.seed}] it {it:3d} ret {rec['return']:.3f} ans {rec['answer']:.3f} "
            f"items {rec['items']:.2f} ent {rec['ent'] or 0:.3f} kl {rec['kl'] or 0:.4f} "
            f"ev {ev:.2f} {('val F1 %.2f' % rec['val']['F1']) if 'val' in rec else ''}"
            f"{(' ema %.2f' % rec['val_ema']['F1']) if 'val_ema' in rec else ''} "
            f"({rec['sec']:.0f}s)")

    torch.save({k: v.cpu() for k, v in best["state"].items()}, out / "actor_best.pt")
    torch.save({k: v.cpu() for k, v in actor.state_dict().items()}, out / "actor_final.pt")
    with open(out / "history.json", "w") as f:
        json.dump({"config": asdict(cfg), "env": asdict(env_cfg), "reward": asdict(rc),
                   "best_iter": best["iter"], "best_val_F1": best["val_F1"], "best_kind": best["kind"],
                   "history": history}, f, indent=1)
    cache.save()
    return best
