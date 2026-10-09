"""Evaluate the non-learned selectors with the frozen reader.

Usage: python scripts/run_baselines.py --split test
"""
import argparse
import time

import numpy as np
from common import RESULTS, Tee, load_bank, save_json

from mhtrl.baselines import greedy_hop, oracle, random_walk, topk
from mhtrl.env import EnvConfig
from mhtrl.reader import DEFAULT_READER, ReaderCache
from mhtrl.rollout import evaluate_sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--reader", default=DEFAULT_READER)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--methods", default="all")
    args = ap.parse_args()
    log = Tee(RESULTS / "logs" / f"baselines_{args.split}.log")
    env_cfg = EnvConfig()
    bank = load_bank(args.split, env_cfg.k_passages, args.limit)
    cache = ReaderCache(args.reader)
    ids = list(range(len(bank)))
    tag = "" if args.reader == DEFAULT_READER else "_" + args.reader.split("/")[-1]

    methods = {}
    for seed in range(3):
        for k in (2, 3):
            rng = np.random.default_rng(seed)
            methods[f"random_k{k}_s{seed}"] = lambda p, k=k, rng=rng: random_walk(p, env_cfg, k, rng)
    for k in (1, 2, 3):
        methods[f"bm25_k{k}"] = lambda p, k=k: topk(p, p.bm25n, k)
        methods[f"dense_k{k}"] = lambda p, k=k: topk(p, p.cos, k)
        methods[f"hybrid_k{k}"] = lambda p, k=k: topk(p, p.hybrid, k)
    methods["greedy_hop"] = greedy_hop
    methods["oracle"] = lambda p: oracle(p, cache)
    if args.methods != "all":
        keep = set(args.methods.split(","))
        methods = {k: v for k, v in methods.items() if k in keep}

    for name, fn in methods.items():
        t0 = time.time()
        sets = [fn(bank.items[i]) for i in ids]
        summary, rows = evaluate_sets(bank, ids, sets, cache)
        cache.save()
        save_json({"method": name, "split": args.split, "reader": args.reader, "summary": summary,
                   "rows": rows}, RESULTS / "baselines" / f"{args.split}{tag}" / f"{name}.json")
        log(f"{args.split} {name:16s} EM {summary['EM']:.2f} F1 {summary['F1']:.2f} "
            f"AnsRec {summary['AnsRecall']:.1f} GoldHit {summary['GoldHit']:.1f} "
            f"items {summary['Items']:.2f} words {summary['Words']:.0f} ({time.time() - t0:.0f}s, "
            f"cache {len(cache.table)})")


if __name__ == "__main__":
    main()
