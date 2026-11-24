"""Behaviour cloning on the traced gold paths.

Gold evidence is noisy, so each step may have several acceptable actions; we maximise the
probability of choosing any of them.
"""
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..device import DEVICE, to_device
from ..env import Bank, EnvConfig, Episode
from ..policy import PointerActor
from ..reader import ReaderCache
from ..rollout import StepRec, collate
from .algos import greedy_eval


def oracle_steps(bank: Bank, env_cfg: EnvConfig):
    k_stop = env_cfg.max_cands
    recs, targets, skipped = [], [], 0
    for i, p in enumerate(bank.items):
        if not p.gold_paths:
            continue
        ep = Episode(p, env_cfg)
        gold_rows = sorted({path[0] for path in p.gold_paths})
        c = ep.candidates()
        tgt = [c.index(r) for r in gold_rows if r in c]
        if not tgt:
            skipped += 1
            continue
        s, f = ep.features(c)
        recs.append(StepRec(i, c, f, s, ep.stop_allowed()))
        targets.append(tgt)

        r_star = max(gold_rows, key=lambda r: p.hybrid[r])
        ep.step(r_star)
        if ep.done:
            continue
        psgs = sorted({path[1] for path in p.gold_paths if len(path) == 2 and path[0] == r_star})
        c = ep.candidates()
        tgt = [c.index(a) for a in psgs if a in c]
        if (r_star,) in p.gold_paths:
            tgt.append(k_stop)
        if not tgt:                      # gold passage cut by the candidate cap
            skipped += 1
            continue
        s, f = ep.features(c)
        recs.append(StepRec(i, c, f, s, ep.stop_allowed()))
        targets.append(tgt)
        reachable = [a for a in psgs if a in c]
        if not reachable or (r_star,) in p.gold_paths:
            continue
        p_star = max(reachable, key=lambda a: p.hybrid[a])
        ep.step(p_star)
        if ep.done:
            continue
        c = ep.candidates()
        s, f = ep.features(c)
        recs.append(StepRec(i, c, f, s, ep.stop_allowed()))
        targets.append([k_stop])
    return recs, targets, skipped


def bc_loss(actor, batch, targets, k_max):
    logits = actor(**batch)
    logp = torch.log_softmax(logits, -1)
    tm = np.zeros(tuple(logp.shape), dtype=bool)
    for i, t in enumerate(targets):
        tm[i, t] = True
    tmask = torch.from_numpy(tm).to(logp.device)
    marg = torch.logsumexp(logp.masked_fill(~tmask, -1e9), -1)
    acc = tmask.gather(-1, logits.argmax(-1, keepdim=True)).float().mean()
    return -marg.mean(), acc


def train_bc(train_bank: Bank, val_bank: Bank, cache: ReaderCache, env_cfg: EnvConfig,
             out_dir: str, seed: int = 0, epochs: int = 8, lr: float = 1e-3,
             minibatch: int = 256, log=print):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    k_max = env_cfg.max_cands
    recs, targets, skipped = oracle_steps(train_bank, env_cfg)
    v_recs, v_targets, _ = oracle_steps(val_bank, env_cfg)
    log(f"[bc seed {seed}] oracle steps: train {len(recs)} (skipped {skipped} questions), val {len(v_recs)}")

    actor = PointerActor().to(DEVICE)
    opt = torch.optim.Adam(actor.parameters(), lr=lr, eps=1e-5)
    history, best = [], {"epoch": 0, "val_F1": -1.0, "state": None}
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        for g in opt.param_groups:
            g["lr"] = lr * (1.0 - (epoch - 1) / epochs)
        perm = rng.permutation(len(recs))
        losses, accs = [], []
        for c0 in range(0, len(recs), minibatch):
            mb = perm[c0:c0 + minibatch]
            batch = to_device(collate(train_bank, [recs[i] for i in mb], k_max))
            loss, acc = bc_loss(actor, batch, [targets[i] for i in mb], k_max)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            accs.append(acc.item())
        with torch.no_grad():
            vl, va = [], []
            for c0 in range(0, len(v_recs), 1024):
                batch = to_device(collate(val_bank, v_recs[c0:c0 + 1024], k_max))
                l, a = bc_loss(actor, batch, v_targets[c0:c0 + 1024], k_max)
                vl.append(l.item() * len(batch["state"]))
                va.append(a.item() * len(batch["state"]))
        val_sum, _ = greedy_eval(val_bank, actor, env_cfg, cache)
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), "train_acc": float(np.mean(accs)),
               "val_loss": sum(vl) / len(v_recs), "val_step_acc": sum(va) / len(v_recs),
               "val": val_sum, "sec": time.time() - t0}
        history.append(rec)
        if val_sum["F1"] > best["val_F1"]:
            best = {"epoch": epoch, "val_F1": val_sum["F1"], "state": copy.deepcopy(actor.state_dict())}
        log(f"[bc seed {seed}] epoch {epoch} loss {rec['train_loss']:.3f} acc {rec['train_acc']:.3f} "
            f"val loss {rec['val_loss']:.3f} step-acc {rec['val_step_acc']:.3f} "
            f"val F1 {val_sum['F1']:.2f} EM {val_sum['EM']:.2f} ({rec['sec']:.0f}s)")
        cache.save()
    torch.save({k: v.cpu() for k, v in best["state"].items()}, out / "actor_best.pt")
    with open(out / "history.json", "w") as f:
        json.dump({"seed": seed, "best_epoch": best["epoch"], "best_val_F1": best["val_F1"],
                   "history": history}, f, indent=1)
    return best
