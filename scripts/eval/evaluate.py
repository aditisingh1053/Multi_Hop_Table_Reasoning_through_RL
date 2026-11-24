"""Greedy evaluation of trained selectors on a split.

Usage: python scripts/eval/evaluate.py --runs results/runs/ppo_main_s0 results/runs/bc_s0 --split test
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import Tee, load_bank, save_json, RESULTS

from mhtrl.device import DEVICE
from mhtrl.env import EnvConfig
from mhtrl.policy import make_actor
from mhtrl.reader import DEFAULT_READER, ReaderCache
from mhtrl.rl.algos import greedy_eval


def env_of(run: Path) -> EnvConfig:
    h = run / "history.json"
    if h.exists():
        with open(h) as f:
            env = json.load(f).get("env")
        if env:
            return EnvConfig(**env)
    return EnvConfig()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--reader", default=DEFAULT_READER)
    ap.add_argument("--ckpt", default="actor_best.pt")
    ap.add_argument("--limit", type=int, default=None, help="first N questions of the split")
    args = ap.parse_args()
    log = Tee(RESULTS / "logs" / f"evaluate_{args.split}.log")
    cache = ReaderCache(args.reader)
    tag = "" if args.reader == DEFAULT_READER else "_" + args.reader.split("/")[-1]
    banks = {}
    for run in map(Path, args.runs):
        env_cfg = env_of(run)
        key = env_cfg.k_passages
        if key not in banks:
            banks[key] = load_bank(args.split, env_cfg.k_passages, args.limit)
        actor = make_actor(env_cfg)
        actor.load_state_dict(torch.load(run / args.ckpt, map_location="cpu"))
        actor.to(DEVICE)
        t0 = time.time()
        summary, rows = greedy_eval(banks[key], actor, env_cfg, cache)
        cache.save()
        save_json({"run": str(run), "split": args.split, "reader": args.reader, "ckpt": args.ckpt,
                   "summary": summary, "rows": rows},
                  run / f"eval_{args.split}{tag}{'' if args.ckpt == 'actor_best.pt' else '_' + args.ckpt[:-3]}.json")
        log(f"{run.name:28s} {args.split}{tag} EM {summary['EM']:.2f} F1 {summary['F1']:.2f} "
            f"AnsRec {summary['AnsRecall']:.1f} GoldHit {summary['GoldHit']:.1f} "
            f"items {summary['Items']:.2f} words {summary['Words']:.0f} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
