"""Collect evaluation files into result tables (CSV + LaTeX fragments) and bootstrap tests.

Learned methods: mean +- std over seeds. Paired comparisons: per-question scores are first
averaged over seeds, then 10,000 paired bootstrap resamples of the questions give a 95%
interval for the difference and a one-sided p-value.

Usage: python scripts/analysis/aggregate.py --split test
"""
import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RESULTS, save_json

METRICS = ["EM", "F1", "table_EM", "table_F1", "passage_EM", "passage_F1",
           "AnsRecall", "GoldHit", "Items", "Words"]


def load(path):
    with open(path) as f:
        return json.load(f)


def collect(split, tag=""):
    groups = defaultdict(list)          # method -> list of eval dicts (one per seed)
    for f in glob.glob(str(RESULTS / "baselines" / f"{split}{tag}" / "*.json")):
        d = load(f)
        name = re.sub(r"_s\d+$", "", d["method"])
        groups[name].append(d)
    for f in glob.glob(str(RESULTS / "runs" / "*" / f"eval_{split}{tag}.json")):
        d = load(f)
        name = re.sub(r"_s\d+$", "", Path(d["run"]).name)
        groups[name].append(d)
    return groups


def summarize(groups):
    table = {}
    for name, runs in sorted(groups.items()):
        row = {"seeds": len(runs)}
        for m in METRICS:
            vals = [r["summary"].get(m) for r in runs if r["summary"].get(m) is not None]
            if vals:
                row[m] = float(np.mean(vals))
                row[m + "_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        table[name] = row
    return table


def per_question(runs, key="f1"):
    qids = [r["qid"] for r in runs[0]["rows"]]
    mat = np.array([[row[key] for row in r["rows"]] for r in runs], dtype=np.float64)
    for r in runs:
        assert [row["qid"] for row in r["rows"]] == qids, "question order differs between runs"
    return qids, mat.mean(0)


def paired_bootstrap(a, b, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = a - b
    boots = np.concatenate([d[rng.integers(0, len(d), size=(1000, len(d)))].mean(1)
                            for _ in range(n // 1000)])
    return {"diff": float(100 * d.mean()), "ci_low": float(100 * np.percentile(boots, 2.5)),
            "ci_high": float(100 * np.percentile(boots, 97.5)),
            "p_one_sided": float((boots <= 0).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default="")
    ap.add_argument("--compare", nargs="*", default=[
        "ppo_main:bc", "ppo_main:greedy_hop", "ppo_main:hybrid_k2", "ppo_main:grpo",
        "ppo_main:reinforce", "ppo_main:ppo_scratch", "ppo_main:ppo_noshape"])
    args = ap.parse_args()
    groups = collect(args.split, args.tag)
    table = summarize(groups)
    tests = {}
    for pair in args.compare:
        a, b = pair.split(":")
        if a in groups and b in groups:
            qa, sa = per_question(groups[a], "f1")
            qb, sb = per_question(groups[b], "f1")
            assert qa == qb
            ea = per_question(groups[a], "em")[1]
            eb = per_question(groups[b], "em")[1]
            tests[pair] = {"F1": paired_bootstrap(sa, sb), "EM": paired_bootstrap(ea, eb)}
    out = RESULTS / f"summary_{args.split}{args.tag}.json"
    save_json({"table": table, "tests": tests}, out)
    for name, row in table.items():
        print(f"{name:18s} n={row['seeds']} EM {row.get('EM', float('nan')):6.2f}"
              f"±{row.get('EM_std', 0):.2f} F1 {row.get('F1', float('nan')):6.2f}±{row.get('F1_std', 0):.2f}"
              f" AnsRec {row.get('AnsRecall', float('nan')):5.1f} GoldHit {row.get('GoldHit', float('nan')):5.1f}"
              f" items {row.get('Items', float('nan')):.2f} words {row.get('Words', float('nan')):.0f}")
    for pair, t in tests.items():
        print(f"{pair:28s} dF1 {t['F1']['diff']:+.2f} [{t['F1']['ci_low']:+.2f}, {t['F1']['ci_high']:+.2f}]"
              f" p={t['F1']['p_one_sided']:.4f} | dEM {t['EM']['diff']:+.2f}"
              f" [{t['EM']['ci_low']:+.2f}, {t['EM']['ci_high']:+.2f}] p={t['EM']['p_one_sided']:.4f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
