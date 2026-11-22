"""Inference demo: run trained selectors on one HybridQA dev question and show every decision.

  python scripts/demo.py                       # a random dev question, RiL-PPO vs plain PPO
  python scripts/demo.py --idx 42              # a specific dev question (0..3465)
  python scripts/demo.py --find "rushing yards"   # first dev question containing this text
  python scripts/demo.py --fresh               # really run the reader now (ignore cached reads)
  python scripts/demo.py --runs results/loop/runs/loop_full_long_s0   # only one model

For each model it prints: every hop (row or passage chosen, with its probability), what the frozen
reader extracted from it, the COMMIT decision (RiL-PPO), the final answer, the gold answer and F1.
"""
import argparse
import random
import tempfile
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import transformers  # noqa: E402

transformers.logging.set_verbosity_error()

import torch
from common import load_bank

from evaluate import env_of
from mhtrl.device import DEVICE
from mhtrl.env import ROW
from mhtrl.policy import make_actor
from mhtrl.reader import DEFAULT_READER, ReaderCache
from mhtrl.rollout import predict, rollout
from mhtrl.text import exact_match, f1_score

DEFAULT_RUNS = ["results/loop/runs/loop_full_long_s0", "results/loop/runs/ppo_scratch_long_s0"]
NAMES = {"loop_full_long": "RiL-PPO (ours, 100 it)", "ppo_scratch_long": "Plain PPO (100 it)",
         "loop_full": "RiL-PPO (ours, 40 it)", "ppo_scratch": "Plain PPO (40 it)"}


def short(t, n=110):
    t = " ".join(t.split())
    return t if len(t) <= n else t[:n] + " ..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--find", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--fresh", action="store_true", help="empty reader cache: the reader runs live")
    args = ap.parse_args()

    t0 = time.time()
    models = []
    for run in map(Path, args.runs):
        cfg = env_of(run)
        actor = make_actor(cfg)
        actor.load_state_dict(torch.load(run / "actor_best.pt", map_location="cpu"))
        models.append((run, cfg, actor.to(DEVICE).eval()))
    bank = load_bank("test", models[0][1].k_passages)
    cache = ReaderCache(DEFAULT_READER, cache_dir=tempfile.mkdtemp() if args.fresh else None)
    print(f"loaded {len(models)} model(s), {len(bank.items)} dev questions, device {DEVICE} "
          f"({time.time() - t0:.1f}s)")

    if args.find:
        hits = [i for i, p in enumerate(bank.items) if args.find.lower() in p.question.lower()]
        if not hits:
            raise SystemExit(f"no dev question contains {args.find!r}")
        idx = hits[0]
    elif args.idx is not None:
        idx = args.idx
    else:
        idx = random.Random(args.seed).randrange(len(bank.items))
    p = bank.items[idx]
    print("\n" + "=" * 100)
    print(f"QUESTION #{idx}: {p.question}")
    print(f"GOLD ANSWER: {p.answer}      ({p.n} candidate entities: {p.n_rows} rows + "
          f"{p.n - p.n_rows} passages)")
    print("=" * 100)

    for run, cfg, actor in models:
        name = NAMES.get(run.name.rsplit("_s", 1)[0], run.name)
        t1 = time.time()
        eps, recs, _ = rollout(bank, [idx], actor, cfg, greedy=True, cache=cache)
        ep, steps = eps[0], recs[0]
        commit = [ep.committed] if cfg.commit else None
        pred = predict(bank, [idx], [ep.selected], cache, commits=commit)[0]
        dt = time.time() - t1
        print(f"\n--- {name}   [{run.name}]")
        for k, r in enumerate(steps, 1):
            prob = torch.tensor(r.logp).exp().item()
            if r.commit:
                n = r.cands[r.action]
                print(f"  COMMIT  answer from entity {n} (p={prob:.2f})")
            elif r.action == cfg.max_cands:
                print(f"  hop {k}   STOP (p={prob:.2f})")
            else:
                n = r.cands[r.action]
                kind = "ROW    " if p.kind[n] == ROW else "PASSAGE"
                print(f"  hop {k}   {kind} {n:3d} (p={prob:.2f}) of {len(r.cands)} candidates: "
                      f"{short(p.text[n])}")
        for n in ep.selected:
            span, score = cache.table[(p.qid, n)]
            mark = "  <- committed" if cfg.commit and n == ep.committed else ""
            print(f"    reader on entity {n:3d}: '{span}' (score {score:.2f}){mark}")
        maxconf = predict(bank, [idx], [ep.selected], cache)[0]
        print(f"  FINAL ANSWER: '{pred}'   gold '{p.answer}'   "
              f"EM {exact_match(p.answer, pred):.0f}  F1 {f1_score(p.answer, pred):.2f}   ({dt:.2f}s)")
        if cfg.commit and maxconf != pred:
            print(f"  (max-confidence alone would have answered '{maxconf}', "
                  f"F1 {f1_score(p.answer, maxconf):.2f})")


if __name__ == "__main__":
    main()
