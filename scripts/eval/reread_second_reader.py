"""Second-reader transfer computed in one environment for every selector.

Each selector's evidence is taken from its stored test evaluation (rows[i]["selected"]), so no
policy is re-run and no embedding is recomputed; only the reader changes. The oracle is
recomputed, because its choice among gold paths depends on the reader. Results are written
with the tag "_roberta-base-squad2-cpu" so that they never mix with the GPU-run files.

Usage: python scripts/eval/reread_second_reader.py
"""
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: common.py
from common import RESULTS, Tee, load_bank, save_json

from mhtrl.baselines import oracle
from mhtrl.reader import ReaderCache
from mhtrl.rollout import evaluate_sets

READER = "deepset/roberta-base-squad2"
TAG = "_roberta-base-squad2-cpu"
LIMIT = 1000
RUNS = [f"{n}_s{s}" for n in ("bc", "ppo_main", "grpo", "reinforce") for s in range(5)] + \
       [f"ppo_scratch_s{s}" for s in range(3)]
BASELINES = ["dense_k2", "hybrid_k2", "hybrid_k3", "bm25_k3", "greedy_hop"]


def stored_sets(path, bank):
    with open(path) as f:
        rows = json.load(f)["rows"][:LIMIT]
    assert [r["qid"] for r in rows] == [p.qid for p in bank.items], "question order differs"
    return [r["selected"] for r in rows]


def main():
    log = Tee(RESULTS / "logs" / "reread_second_reader.log")
    bank = load_bank("test", 5, LIMIT)
    ids = list(range(len(bank)))
    cache = ReaderCache(READER, cache_dir="data/cache_reread")
    for name in BASELINES + ["oracle"]:
        out = RESULTS / "baselines" / f"test{TAG}" / f"{name}.json"
        if out.exists():
            continue
        if name == "oracle":
            sets = [oracle(p, cache) for p in bank.items]
        else:
            sets = stored_sets(RESULTS / "baselines" / "test" / f"{name}.json", bank)
        summary, rows = evaluate_sets(bank, ids, sets, cache)
        cache.save()
        save_json({"method": name, "split": f"test{TAG}", "reader": READER, "summary": summary, "rows": rows}, out)
        log(f"[reread] {name:16s} EM {summary['EM']:.2f} F1 {summary['F1']:.2f}")
    for run in RUNS:
        out = RESULTS / "runs" / run / f"eval_test{TAG}.json"
        if out.exists():
            continue
        summary, rows = evaluate_sets(bank, ids, stored_sets(RESULTS / "runs" / run / "eval_test.json", bank), cache)
        cache.save()
        save_json({"run": str(RESULTS / "runs" / run), "split": "test", "reader": READER, "ckpt": "actor_best.pt",
                   "note": "evidence selected in the GPU evaluation, re-read on the laptop CPU",
                   "summary": summary, "rows": rows}, out)
        log(f"[reread] {run:16s} EM {summary['EM']:.2f} F1 {summary['F1']:.2f} items {summary['Items']:.2f}")
    log("reread finished")


if __name__ == "__main__":
    main()
