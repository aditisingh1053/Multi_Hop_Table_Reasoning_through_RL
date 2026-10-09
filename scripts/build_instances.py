"""Build per-question entity graphs, lexical statistics and frozen embeddings.

Splits
  train : questions from HybridQA train on a random subset of tables
  val   : questions from HybridQA train on a *disjoint* set of tables (model selection)
  test  : the full official HybridQA dev set (test labels are hidden by the authors)
BM25 document frequencies are estimated on the train split only.

Usage: python scripts/build_instances.py --n_train 10000 --n_val 1000
"""
import argparse
import pickle
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mhtrl.data import build_instance, load_dev_answer_types, load_questions, load_tables  # noqa: E402
from mhtrl.encoder import FrozenEncoder  # noqa: E402
from mhtrl.text import BM25, words  # noqa: E402


def table_disjoint_split(questions, n_train, n_val, seed):
    by_table = defaultdict(list)
    for q in questions:
        if q.get("answer-node"):          # keep only questions whose answer could be traced
            by_table[q["table_id"]].append(q)
    tables = sorted(by_table)
    random.Random(seed).shuffle(tables)
    train, val, i = [], [], 0
    while len(train) < n_train:
        train.extend(by_table[tables[i]]); i += 1
    while len(val) < n_val:
        val.extend(by_table[tables[i]]); i += 1
    return train[:n_train], val[:n_val]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hybridqa", default="data/raw/HybridQA")
    ap.add_argument("--tables", default="data/processed/wikitables.jsonl")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--n_train", type=int, default=10000)
    ap.add_argument("--n_val", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--test_limit", type=int, default=None, help="debug only")
    args = ap.parse_args()
    t0 = time.time()

    train_q = load_questions(args.hybridqa, "train")
    dev_q = load_questions(args.hybridqa, "dev")
    tr, va = table_disjoint_split(train_q, args.n_train, args.n_val, args.seed)
    dev_types = load_dev_answer_types(args.hybridqa)
    splits = {"train": tr, "val": va, "test": dev_q[:args.test_limit] if args.test_limit else dev_q}
    overlap = len({q["table_id"] for q in dev_q} & {q["table_id"] for q in train_q})
    print(f"train {len(tr)}  val {len(va)}  test(dev) {len(splits["test"])}  "
          f"dev tables also in HybridQA-train: {overlap}")

    needed = {q["table_id"] for qs in splits.values() for q in qs}
    tables = load_tables(args.tables, needed)
    missing = needed - set(tables)
    assert not missing, f"missing tables: {list(missing)[:5]}"
    print(f"loaded {len(tables)} tables in {time.time() - t0:.0f}s")

    instances = {}
    for name, qs in splits.items():
        insts = [build_instance(q, tables[q["table_id"]],
                                dev_types.get(q["question_id"]) if name == "test" else None)
                 for q in qs]
        if name == "train":
            insts = [x for x in insts if x.gold_paths and x.n_rows > 0]
        instances[name] = [x for x in insts if x.n_rows > 0] if name != "test" else insts
        print(f"{name}: {len(instances[name])} instances, "
              f"{sum(bool(x.gold_paths) for x in instances[name])} with a gold path")

    # BM25 statistics from the unique nodes of the training tables only
    seen, corpus = set(), []
    for x in instances["train"]:
        if x.table_id in seen:
            continue
        seen.add(x.table_id)
        corpus.extend(words(t) for t in x.node_text)
    bm25 = BM25(corpus)

    # embed every unique table once, then every question
    enc = FrozenEncoder()
    table_emb = {}
    uniq = {}
    for insts in instances.values():
        for x in insts:
            uniq.setdefault(x.table_id, x.node_text)
    flat, owner = [], []
    for tid, texts in uniq.items():
        flat.extend(texts)
        owner.extend([tid] * len(texts))
    print(f"encoding {len(flat)} node texts from {len(uniq)} tables")
    emb = enc.encode(flat, show_progress=True)
    pos = 0
    for tid, texts in uniq.items():
        table_emb[tid] = emb[pos:pos + len(texts)].astype(np.float16)
        pos += len(texts)

    for name, insts in instances.items():
        q_emb = enc.encode([x.question for x in insts]).astype(np.float16)
        bm = []
        for x in insts:
            qt = words(x.question)
            bm.append(np.array([bm25.score(qt, words(t)) for t in x.node_text], dtype=np.float32))
        payload = {
            "instances": [asdict(x) for x in insts],
            "q_emb": q_emb,
            "bm25": bm,
            "table_emb": {x.table_id: table_emb[x.table_id] for x in insts},
        }
        with open(Path(args.out) / f"{name}.pkl", "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"saved {name}.pkl")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
