"""Reader-in-the-loop PPO (RiL-PPO) experiments; resumable, writes to results/loop/.

Runs, on the same data, reader and budget as the main suite, in priority order (so that a
time-limited session still ends with the most important results):
  1 baselines  bm25_k3, hybrid_k3, greedy_hop, bridge_hop (non-learned reader-in-the-loop rule),
               read_all (read every entity, max confidence), oracle
  2 core       ppo_scratch (control), loop_full, loop_feedback, loop_commit         (40 it)
  3 plus       plus_all (RiL-PPO++), plus_rb, plus_asym, plus_sil, plus_drop, plus_ema  (40 it)
  4 optimisers opt_rloo, opt_drgrpo, opt_grpo on the reader-in-the-loop environment  (40 it)
  5 long       ppo_scratch_long, loop_full_long, plus_all_long                      (100 it)
  6 test-time  self-consistency (k = 8 evidence paths) for ppo_scratch, loop_full, plus_all
  7 transfer   second reader (RoBERTa-base), first 1,000 test questions
  report       scripts/loop_report.py -> results/loop/REPORT.md and summary.json

Usage: python scripts/run_loop_suite.py [--seeds 3] [--no-long] [--skip-plus] [--skip-opt]
                                        [--no-sc] [--smoke]
"""
import argparse
import json
import platform
import subprocess
import sys
import time

import torch
from common import RESULTS, ROOT, Tee, load_bank, save_json

from mhtrl.baselines import bridge_hop, greedy_hop, oracle, read_all, topk
from mhtrl.config import load_config
from mhtrl.device import DEVICE
from mhtrl.env import EnvConfig
from mhtrl.policy import make_actor
from mhtrl.reader import DEFAULT_READER, ReaderCache
from mhtrl.rl.algos import greedy_eval, train_rl
from mhtrl.rollout import evaluate_sets, self_consistency_eval

SECOND_READER = "deepset/roberta-base-squad2"
SECOND_TAG = "_roberta-base-squad2"
MAIN = ["ppo_scratch", "loop_full", "loop_feedback", "loop_commit"]
PLUS = ["plus_all", "plus_rb", "plus_asym", "plus_sil", "plus_drop", "plus_ema"]
OPT = ["opt_rloo", "opt_drgrpo", "opt_grpo"]
LONG = ["ppo_scratch_long", "loop_full_long", "plus_all_long"]
SC_RUNS = ["ppo_scratch", "loop_full", "plus_all"]
SC_K = 8
SC_TEMPERATURE = 0.7


def run_baselines(bank, split, cache, env_cfg, out_root, log, names=None):
    methods = {
        "bm25_k3": lambda p: topk(p, p.bm25n, 3),
        "hybrid_k3": lambda p: topk(p, p.hybrid, 3),
        "greedy_hop": greedy_hop,
        "bridge_hop": lambda p: bridge_hop(p, cache),
        "read_all": read_all,
        "oracle": lambda p: oracle(p, cache),
    }
    ids = list(range(len(bank)))
    for name, fn in methods.items():
        if names is not None and name not in names:
            continue
        out = out_root / "baselines" / split / f"{name}.json"
        if out.exists():
            continue
        t0 = time.time()
        summary, rows = evaluate_sets(bank, ids, [fn(bank.items[i]) for i in ids], cache)
        cache.save()
        save_json({"method": name, "split": split, "reader": cache.reader_name, "summary": summary,
                   "rows": rows}, out)
        log(f"[baseline] {split:28s} {name:12s} EM {summary['EM']:.2f} F1 {summary['F1']:.2f} "
            f"best-in-set {summary['F1_bestinset']:.2f} ({time.time() - t0:.0f}s)")


def evaluate_run(run_dir, bank, cache, env_cfg, tag, log):
    out = run_dir / f"eval_test{tag}.json"
    if out.exists() or not (run_dir / "actor_best.pt").exists():
        return
    actor = load_actor(run_dir, env_cfg)
    t0 = time.time()
    summary, rows = greedy_eval(bank, actor, env_cfg, cache)
    cache.save()
    save_json({"run": str(run_dir), "split": "test", "reader": cache.reader_name, "summary": summary,
               "rows": rows}, out)
    log(f"[eval] {run_dir.name:22s} test{tag} EM {summary['EM']:.2f} F1 {summary['F1']:.2f} "
        f"(max-conf on same sets {summary['F1_maxconf']:.2f}, best-in-set {summary['F1_bestinset']:.2f}) "
        f"AnsRec {summary['AnsRecall']:.1f} items {summary['Items']:.2f} ({time.time() - t0:.0f}s)")


def load_actor(run_dir, env_cfg):
    actor = make_actor(env_cfg)
    actor.load_state_dict(torch.load(run_dir / "actor_best.pt", map_location="cpu"))
    return actor.to(DEVICE)


def sc_run(run_dir, bank, cache, env_cfg, log, k=SC_K):
    out = run_dir / f"eval_test_sc{k}.json"
    if out.exists() or not (run_dir / "actor_best.pt").exists():
        return
    t0 = time.time()
    summary, rows = self_consistency_eval(bank, load_actor(run_dir, env_cfg), env_cfg, cache, k=k,
                                          temperature=SC_TEMPERATURE)
    cache.save()
    save_json({"run": str(run_dir), "split": "test", "reader": cache.reader_name, "summary": summary,
               "rows": rows}, out)
    log(f"[sc{k}] {run_dir.name:22s} F1 {summary['F1']:.2f} (greedy {summary['F1_greedy']:.2f}) "
        f"reads/question {summary['Reads']:.2f} ({time.time() - t0:.0f}s)")


def train_and_eval(name, seed, root, train, val, test, cache, log, smoke):
    cfg = load_config(str(ROOT / "configs" / f"{name}.yaml"))
    env_c, rc, rl = cfg["env"], cfg["reward"], cfg["rl"]
    rl.seed = seed
    if smoke:
        rl.iters, rl.episodes_per_iter, rl.eval_every = 2, 32, 1
        rl.sil_minibatch = 16
    run_dir = root / "runs" / f"{name}_s{seed}"
    if not (run_dir / "actor_best.pt").exists():
        t0 = time.time()
        best = train_rl(train, val, cache, env_c, rc, rl, str(run_dir), log=Tee(run_dir / "train.log"))
        save_json({"best_iter": best["iter"], "best_val_F1": best["val_F1"], "best_kind": best["kind"]},
                  run_dir / "best.json")
        log(f"[train] {run_dir.name:22s} best iter {best['iter']} ({best['kind']}) val F1 "
            f"{best['val_F1']:.2f} ({(time.time() - t0) / 60:.1f} min)")
    evaluate_run(run_dir, test, cache, env_c, "", log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--no-long", action="store_true", help="skip the 100-iteration runs")
    ap.add_argument("--skip-plus", action="store_true", help="skip the RiL-PPO++ extensions")
    ap.add_argument("--skip-opt", action="store_true", help="skip the critic-free optimiser comparison")
    ap.add_argument("--no-sc", action="store_true", help="skip test-time self-consistency")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end check -> results_smoke/loop")
    args = ap.parse_args()
    root = (ROOT / "results_smoke" if args.smoke else RESULTS) / "loop"
    if args.smoke:
        args.seeds = 1
    log = Tee(root / "logs" / "loop_suite.log")
    env = {"python": platform.python_version(), "torch": torch.__version__, "device": str(DEVICE),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "platform": platform.platform()}
    save_json(env, root / "environment.json")
    log(f"environment: {json.dumps(env)}")
    t_start = time.time()

    base_env = EnvConfig()
    lim = {"train": 120, "val": 40, "test": 40} if args.smoke else {"train": None, "val": None, "test": None}
    train, val, test = (load_bank(s, base_env.k_passages, lim[s]) for s in ("train", "val", "test"))
    cache = ReaderCache(DEFAULT_READER)
    log(f"banks: train {len(train)} val {len(val)} test {len(test)}; reader cache {len(cache.table)} entries")

    run_baselines(test, "test", cache, base_env, root, log)

    stages = [MAIN]
    if not args.skip_plus:
        stages.append(PLUS)
    if not args.skip_opt:
        stages.append(OPT)
    if not args.no_long:
        stages.append([n for n in LONG if not (args.skip_plus and n.startswith("plus"))])
    for names in stages:              # within a stage: seed 0 of every method first
        for seed in range(args.seeds):
            for name in names:
                train_and_eval(name, seed, root, train, val, test, cache, log, args.smoke)

    if not args.no_sc:
        for name in SC_RUNS:
            if args.skip_plus and name.startswith("plus"):
                continue
            env_c = load_config(str(ROOT / "configs" / f"{name}.yaml"))["env"]
            for seed in range(args.seeds):
                sc_run(root / "runs" / f"{name}_s{seed}", test, cache, env_c, log)

    # transfer: a second frozen reader, never used in training. For RiL-PPO this reader is
    # also the one queried *inside* the loop, so the policy reacts to the new reader's outputs.
    test_sub = load_bank("test", base_env.k_passages, limit=20 if args.smoke else 1000)
    cache2 = ReaderCache(SECOND_READER)
    run_baselines(test_sub, "test" + SECOND_TAG, cache2, base_env, root, log,
                  names={"bm25_k3", "bridge_hop", "oracle"})
    for name in ("ppo_scratch", "loop_full", "plus_all"):
        if args.skip_plus and name.startswith("plus"):
            continue
        cfg = load_config(str(ROOT / "configs" / f"{name}.yaml"))
        for seed in range(args.seeds):
            evaluate_run(root / "runs" / f"{name}_s{seed}", test_sub, cache2, cfg["env"], SECOND_TAG, log)
    cache2.save()
    log(f"loop suite finished in {(time.time() - t_start) / 3600:.2f} h")

    if not args.smoke:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "loop_report.py")], check=False)


if __name__ == "__main__":
    main()
