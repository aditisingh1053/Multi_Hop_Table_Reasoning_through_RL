"""Run the complete experimental suite in one process; resumable.

Order: baselines (val, test) -> behaviour cloning (all seeds) -> PPO / GRPO / REINFORCE (seeds 0-2)
-> ablations (seeds 0-2) -> PPO / GRPO / REINFORCE (remaining seeds) -> second-reader transfer.
Every trained selector is evaluated on the test split right after training. Any step whose output
file already exists is skipped, so after a disconnect the same command simply continues.

Usage: python scripts/run_suite.py [--seeds-main 5] [--seeds-ablation 3]
"""
import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from common import RESULTS, ROOT, Tee, load_bank, save_json

from mhtrl.baselines import greedy_hop, oracle, random_walk, topk
from mhtrl.config import load_config
from mhtrl.device import DEVICE
from mhtrl.env import EnvConfig
from mhtrl.policy import PointerActor
from mhtrl.reader import DEFAULT_READER, ReaderCache
from mhtrl.rl.algos import greedy_eval, train_rl
from mhtrl.rl.bc import train_bc
from mhtrl.rollout import evaluate_sets

SECOND_READER = "deepset/roberta-base-squad2"
SECOND_TAG = "_roberta-base-squad2"
SECOND_LIMIT = 1000
MAIN = ["ppo_main", "grpo", "reinforce"]
ABLATIONS = ["ppo_scratch", "ppo_noshape", "ppo_evbonus", "ppo_em", "ppo_lam0", "ppo_lam1",
             "ppo_nocost", "ppo_cost05", "ppo_cost10"]


def baseline_methods(env_cfg, cache, names=None):
    m = {}
    for seed in range(3):
        for k in (2, 3):
            rng = np.random.default_rng(seed)
            m[f"random_k{k}_s{seed}"] = lambda p, k=k, rng=rng: random_walk(p, env_cfg, k, rng)
    for k in (1, 2, 3):
        m[f"bm25_k{k}"] = lambda p, k=k: topk(p, p.bm25n, k)
        m[f"dense_k{k}"] = lambda p, k=k: topk(p, p.cos, k)
        m[f"hybrid_k{k}"] = lambda p, k=k: topk(p, p.hybrid, k)
    m["greedy_hop"] = greedy_hop
    m["oracle"] = lambda p: oracle(p, cache)
    return {k: v for k, v in m.items() if names is None or k in names}


def run_baselines(bank, split_dir, cache, env_cfg, log, names=None):
    ids = list(range(len(bank)))
    for name, fn in baseline_methods(env_cfg, cache, names).items():
        out = RESULTS / "baselines" / split_dir / f"{name}.json"
        if out.exists():
            continue
        t0 = time.time()
        summary, rows = evaluate_sets(bank, ids, [fn(bank.items[i]) for i in ids], cache)
        cache.save()
        save_json({"method": name, "split": split_dir, "reader": cache.reader_name, "summary": summary,
                   "rows": rows}, out)
        log(f"[baseline] {split_dir:28s} {name:14s} EM {summary['EM']:.2f} F1 {summary['F1']:.2f} "
            f"({time.time() - t0:.0f}s)")


def evaluate_run(run_dir, bank, cache, env_cfg, tag, log, split="test"):
    out = run_dir / f"eval_{split}{tag}.json"
    if out.exists() or not (run_dir / "actor_best.pt").exists():
        return
    actor = PointerActor()
    actor.load_state_dict(torch.load(run_dir / "actor_best.pt", map_location="cpu"))
    actor.to(DEVICE)
    t0 = time.time()
    summary, rows = greedy_eval(bank, actor, env_cfg, cache)
    cache.save()
    save_json({"run": str(run_dir), "split": split, "reader": cache.reader_name, "ckpt": "actor_best.pt",
               "summary": summary, "rows": rows}, out)
    log(f"[eval] {run_dir.name:18s} {split}{tag} EM {summary['EM']:.2f} F1 {summary['F1']:.2f} "
        f"items {summary['Items']:.2f} ({time.time() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds-main", type=int, default=5)
    ap.add_argument("--seeds-ablation", type=int, default=3)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end check (few questions, 2 iterations); writes to results_smoke/")
    args = ap.parse_args()
    global RESULTS
    if args.smoke:
        RESULTS = ROOT / "results_smoke"
        args.seeds_main, args.seeds_ablation = 1, 1
    log = Tee(RESULTS / "logs" / "suite.log")
    env = {"python": platform.python_version(), "torch": torch.__version__,
           "transformers": transformers.__version__, "device": str(DEVICE),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "cpu_count": os.cpu_count(), "platform": platform.platform()}
    save_json(env, RESULTS / "environment.json")
    log(f"environment: {json.dumps(env)}")
    t_start = time.time()

    env_cfg = EnvConfig()
    lim = {"train": 120, "val": 40, "test": 40} if args.smoke else {"train": None, "val": None, "test": None}
    train, val, test = (load_bank(s, env_cfg.k_passages, lim[s]) for s in ("train", "val", "test"))
    cache = ReaderCache(DEFAULT_READER)
    log(f"banks loaded: train {len(train)} val {len(val)} test {len(test)} ({time.time() - t_start:.0f}s)")

    run_baselines(val, "val", cache, env_cfg, log)
    run_baselines(test, "test", cache, env_cfg, log)

    n_bc = max(args.seeds_main, args.seeds_ablation)
    for seed in range(n_bc):
        run_dir = RESULTS / "runs" / f"bc_s{seed}"
        if not (run_dir / "actor_best.pt").exists():
            best = train_bc(train, val, cache, env_cfg, str(run_dir), seed=seed, log=Tee(run_dir / "train.log"),
                            epochs=1 if args.smoke else 8)
            save_json({"best_epoch": best["epoch"], "best_val_F1": best["val_F1"]}, run_dir / "best.json")
        evaluate_run(run_dir, test, cache, env_cfg, "", log)

    first = min(3, args.seeds_main)
    schedule = [(c, s) for s in range(first) for c in MAIN]
    schedule += [(c, s) for s in range(args.seeds_ablation) for c in ABLATIONS]
    schedule += [(c, s) for s in range(first, args.seeds_main) for c in MAIN]
    for name, seed in schedule:
        cfg = load_config(str(ROOT / "configs" / f"{name}.yaml"))
        env_c, rc, rl = cfg["env"], cfg["reward"], cfg["rl"]
        rl.seed = seed
        if args.smoke:
            rl.iters, rl.episodes_per_iter, rl.eval_every, rl.critic_warmup = 2, 32, 1, 1
        if rl.init_actor:
            rl.init_actor = str(RESULTS / "runs" / f"bc_s{seed}" / "actor_best.pt")
        run_dir = RESULTS / "runs" / f"{name}_s{seed}"
        if not (run_dir / "actor_best.pt").exists():
            t0 = time.time()
            best = train_rl(train, val, cache, env_c, rc, rl, str(run_dir), log=Tee(run_dir / "train.log"))
            save_json({"best_iter": best["iter"], "best_val_F1": best["val_F1"]}, run_dir / "best.json")
            log(f"[train] {run_dir.name:18s} best iter {best['iter']} val F1 {best['val_F1']:.2f} "
                f"({(time.time() - t0) / 60:.1f} min)")
        evaluate_run(run_dir, test, cache, env_c, "", log)

    # transfer to a second frozen reader that is never used in training
    test_sub = load_bank("test", env_cfg.k_passages, limit=20 if args.smoke else SECOND_LIMIT)
    cache2 = ReaderCache(SECOND_READER)
    run_baselines(test_sub, "test" + SECOND_TAG, cache2, env_cfg, log,
                  names={"dense_k2", "hybrid_k2", "greedy_hop", "oracle"})
    for name in ["bc"] + MAIN:
        for seed in range(args.seeds_main):
            evaluate_run(RESULTS / "runs" / f"{name}_s{seed}", test_sub, cache2, env_cfg, SECOND_TAG, log)
    log(f"suite finished in {(time.time() - t_start) / 3600:.2f} h")


if __name__ == "__main__":
    main()
