"""PPO (clipped surrogate + GAE, learned critic), GRPO (group-relative, critic-free) and
REINFORCE with a learned baseline, sharing one rollout/reward/evaluation pipeline.

PPO     A_t = sum_l (gamma*lambda)^l delta_{t+l},  delta_t = r_t + gamma V(s_{t+1}) - V(s_t)
        L   = E[min(rho_t A_t, clip(rho_t, 1-eps, 1+eps) A_t)] + c_H * entropy
GRPO    G episodes per question; A_i = (R_i - mean_G R) / (std_G R + 1e-6) for every step of
        episode i; same clipped surrogate, no critic.
REINF.  A_t = G_t - V(s_t) with Monte-Carlo return-to-go G_t; one pass, no ratio, no clip.
Losses are averaged over all steps of a minibatch (no per-episode length normalisation).
"""
import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ..device import DEVICE, to_device
from ..env import Bank, EnvConfig, RewardConfig, episode_rewards
from ..policy import Critic, PointerActor
from ..reader import ReaderCache
from ..rollout import answer_score, collate, evaluate_sets, predict, rollout


@dataclass
class RLConfig:
    algo: str = "ppo"             # ppo | grpo | reinforce
    iters: int = 40
    episodes_per_iter: int = 512
    group_size: int = 4           # grpo only
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


def greedy_eval(bank: Bank, actor, env_cfg: EnvConfig, cache: ReaderCache, ids=None):
    ids = list(range(len(bank))) if ids is None else ids
    actor.eval()
    eps, _, _ = rollout(bank, ids, actor, env_cfg, greedy=True)
    actor.train()
    return evaluate_sets(bank, ids, [e.selected for e in eps], cache)


def train_rl(train_bank: Bank, val_bank: Bank, cache: ReaderCache, env_cfg: EnvConfig,
             rc: RewardConfig, cfg: RLConfig, out_dir: str, log=print):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    k_max = env_cfg.max_cands

    actor, critic = PointerActor().to(DEVICE), Critic().to(DEVICE)
    if cfg.init_actor:
        actor.load_state_dict(torch.load(cfg.init_actor, map_location="cpu"))
    use_critic = cfg.algo in ("ppo", "reinforce")
    opt_a = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr, eps=1e-5)
    opt_c = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr, eps=1e-5)

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

    history = []
    val_sum, _ = greedy_eval(val_bank, actor, env_cfg, cache)
    # the selected checkpoint is always an RL iterate (never the initialisation itself)
    best = {"iter": 0, "val_F1": -1.0, "state": copy.deepcopy(actor.state_dict())}
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

        if cfg.algo == "grpo":
            qs = next_questions(cfg.episodes_per_iter // cfg.group_size)
            item_ids = [q for q in qs for _ in range(cfg.group_size)]
        else:
            item_ids = next_questions(cfg.episodes_per_iter)
        episodes_seen += len(item_ids)

        eps, recs, stopped = rollout(train_bank, item_ids, actor, env_cfg, greedy=False, gen=gen)
        sets = [e.selected for e in eps]
        preds = predict(train_bank, item_ids, sets, cache)
        ans = [answer_score(train_bank.items[i].answer, pr, rc.answer_metric)
               for i, pr in zip(item_ids, preds)]
        rewards = [episode_rewards(train_bank.items[i], s, st, a, rc)
                   for i, s, st, a in zip(item_ids, sets, stopped, ans)]

        keep = [j for j in range(len(eps)) if recs[j]]
        flat = [r for j in keep for r in recs[j]]
        if not flat:
            continue
        ep_len = [len(recs[j]) for j in keep]
        bounds = np.cumsum([0] + ep_len)

        values = np.zeros(len(flat), dtype=np.float32)
        if use_critic:
            with torch.no_grad():
                for c0 in range(0, len(flat), 1024):
                    values[c0:c0 + 1024] = critic(**to_device(collate(train_bank, flat[c0:c0 + 1024],
                                                                      k_max))).cpu().numpy()

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
        else:  # grpo: group-relative advantage of the episode return, shared by all its steps
            total = np.array([rewards[j].sum() for j in range(len(eps))], dtype=np.float32)
            g = cfg.group_size
            grp = total.reshape(-1, g)
            mu = grp.mean(1, keepdims=True)
            a_ep = grp - mu
            if cfg.grpo_std_norm:
                a_ep = a_ep / (grp.std(1, keepdims=True) + 1e-6)
            a_ep = a_ep.reshape(-1)
            for e, j in enumerate(keep):
                adv[bounds[e]:bounds[e + 1]] = a_ep[j]
        if cfg.adv_norm and cfg.algo != "grpo":
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        old_logp = np.array([r.logp for r in flat], dtype=np.float32)
        actions = np.array([r.action for r in flat], dtype=np.int64)
        train_actor = not (cfg.init_actor and it <= cfg.critic_warmup)
        n_epochs = 1 if cfg.algo == "reinforce" else cfg.epochs
        stats = {"kl": [], "clipfrac": [], "ent": [], "pi_loss": [], "v_loss": []}
        kl_stop = False
        if cfg.algo == "reinforce":
            opt_a.zero_grad()   # one gradient step on the whole batch (vanilla policy gradient)
        for epoch in range(cfg.epochs if use_critic else n_epochs):
            perm = rng.permutation(len(flat))
            for c0 in range(0, len(flat), cfg.minibatch):
                mb = perm[c0:c0 + cfg.minibatch]
                batch = to_device(collate(train_bank, [flat[i] for i in mb], k_max))
                if use_critic:
                    v = critic(**batch)
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
                    pi_loss = -(logp * A).mean()
                    loss = (pi_loss - cfg.ent_coef * ent.mean()) * (len(mb) / len(flat))
                    loss.backward()          # accumulate; the single step is taken below
                else:
                    ratio = log_ratio.exp()
                    surr = torch.min(ratio * A, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * A)
                    pi_loss = -surr.mean()
                    stats["clipfrac"].append(((ratio - 1).abs() > cfg.clip).float().mean().item())
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

        ev = float("nan")
        if use_critic:
            var = ret.var()
            ev = float(1 - (ret - values).var() / var) if var > 1e-8 else float("nan")
        rec = {
            "iter": it, "episodes": episodes_seen,
            "return": float(np.mean([r.sum() for r in rewards])),
            "answer": float(np.mean(ans)),
            "items": float(np.mean([len(s) for s in sets])),
            "explicit_stop": float(np.mean(stopped)),
            "zero_adv_groups": float(np.mean(
                np.isclose(np.array([rw.sum() for rw in rewards]).reshape(-1, cfg.group_size).std(1), 0)
            )) if cfg.algo == "grpo" else None,
            "explained_var": ev, "train_actor": train_actor,
            **{k: (float(np.mean(v)) if v else None) for k, v in stats.items()},
            "reader_cache": len(cache.table), "sec": time.time() - t0,
        }
        if it % cfg.eval_every == 0 or it == cfg.iters:
            val_sum, _ = greedy_eval(val_bank, actor, env_cfg, cache)
            rec["val"] = val_sum
            if val_sum["F1"] > best["val_F1"]:
                best = {"iter": it, "val_F1": val_sum["F1"], "state": copy.deepcopy(actor.state_dict())}
            cache.save()
        history.append(rec)
        log(f"[{cfg.algo} seed {cfg.seed}] it {it:3d} ret {rec['return']:.3f} ans {rec['answer']:.3f} "
            f"items {rec['items']:.2f} ent {rec['ent'] or 0:.3f} kl {rec['kl'] or 0:.4f} "
            f"ev {ev:.2f} {('val F1 %.2f' % rec['val']['F1']) if 'val' in rec else ''} "
            f"({rec['sec']:.0f}s)")

    torch.save({k: v.cpu() for k, v in best["state"].items()}, out / "actor_best.pt")
    torch.save({k: v.cpu() for k, v in actor.state_dict().items()}, out / "actor_final.pt")
    with open(out / "history.json", "w") as f:
        json.dump({"config": asdict(cfg), "env": asdict(env_cfg), "reward": asdict(rc),
                   "best_iter": best["iter"], "best_val_F1": best["val_F1"],
                   "history": history}, f, indent=1)
    cache.save()
    return best
