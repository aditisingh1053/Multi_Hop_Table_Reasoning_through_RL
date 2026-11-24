"""Inference demo: run the trained RiL-PPO selector on one HybridQA dev question and show
every decision in detail.

  python scripts/eval/demo.py --idx 2 --fresh        # showcase question, reader runs live on the GPU
  python scripts/eval/demo.py --find "nationality"   # first dev question containing this text
  python scripts/eval/demo.py                        # a random dev question
  python scripts/eval/demo.py --top 8                # show more candidates per step

Pipeline per question (nothing is trained here: greedy actions, frozen reader):
  1. load the question's entity graph (table rows + linked passages, pre-embedded by MiniLM)
  2. repeat up to 3 hops: the policy scores every candidate + STOP, takes the best one;
     the frozen reader reads the chosen entity and its answer is fed back into the state
  3. COMMIT: the policy chooses which selected entity's answer is final
  4. compare the final answer with the gold answer (EM / F1)
"""
import sys
import argparse
import random
import tempfile
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import transformers  # noqa: E402

transformers.logging.set_verbosity_error()

import torch  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: common.py
from common import load_bank  # noqa: E402

from evaluate import env_of  # noqa: E402
from mhtrl.device import DEVICE, to_device  # noqa: E402
from mhtrl.env import CAND_FEATURES, LOOP_CAND_FEATURES, ROW  # noqa: E402
from mhtrl.policy import make_actor  # noqa: E402
from mhtrl.reader import DEFAULT_READER, ReaderCache  # noqa: E402
from mhtrl.rollout import collate, predict, rollout  # noqa: E402
from mhtrl.text import exact_match, f1_score  # noqa: E402

DEFAULT_RUN = "results/loop/runs/loop_full_long_s0"
FI = {n: i for i, n in enumerate(CAND_FEATURES + LOOP_CAND_FEATURES)}
LINE = "-" * 100


def short(t, n=90):
    t = " ".join(t.split())
    return t if len(t) <= n else t[:n] + " ..."


def kind(p, n):
    return "ROW    " if p.kind[n] == ROW else "PASSAGE"


def flags(f):
    """The human-readable signals behind a candidate's score."""
    out = [f"bm25 {f[FI['bm25_norm']]:.2f}", f"sim {f[FI['cos(q,a)']]:.2f}"]
    if f[FI["linked_to_last"]] > 0:
        out.append("linked-to-last-hop")
    if f[FI["bridge_is_title"]] > 0:
        out.append("BRIDGE=title")
    elif f[FI["bridge_in_text"]] > 0:
        out.append("bridge-in-text")
    if f[FI["bridge_in_row"]] > 0:
        out.append("bridge-in-row")
    return ", ".join(out)


@torch.no_grad()
def probs_of(bank, rec, actor, k_max):
    logits = actor(**to_device(collate(bank, [rec], k_max))).float().cpu()[0]
    return torch.softmax(logits, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--find", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--top", type=int, default=5, help="candidates shown per step")
    ap.add_argument("--fresh", action="store_true", help="empty reader cache: the reader runs live")
    args = ap.parse_args()

    t0 = time.time()
    run = Path(args.run)
    cfg = env_of(run)
    actor = make_actor(cfg)
    actor.load_state_dict(torch.load(run / "actor_best.pt", map_location="cpu"))
    actor = actor.to(DEVICE).eval()
    n_par = sum(p.numel() for p in actor.parameters())
    bank = load_bank("test", cfg.k_passages)
    cache = ReaderCache(DEFAULT_READER, cache_dir=tempfile.mkdtemp() if args.fresh else None)
    print(f"[setup] policy {run.name}/actor_best.pt ({n_par / 1e6:.2f}M params) on {DEVICE}; "
          f"reader {DEFAULT_READER} ({'live' if args.fresh else 'cached reads'}); "
          f"{len(bank.items)} dev questions ({time.time() - t0:.1f}s)")

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
    print(f"GOLD ANSWER (used only for scoring): {p.answer}")
    print(f"GRAPH: {p.n} entities = {p.n_rows} table rows + {p.n - p.n_rows} linked passages; "
          f"horizon {cfg.horizon} hops")
    print("=" * 100)

    t1 = time.time()
    eps, recs, _ = rollout(bank, [idx], actor, cfg, greedy=True, cache=cache)
    ep, steps = eps[0], recs[0]
    k_max = cfg.max_cands
    hop = 0
    for r in steps:
        pr = probs_of(bank, r, actor, k_max)
        if r.commit:
            continue
        hop += 1
        known = [(n, cache.table[(p.qid, n)][0]) for n in r.sel]
        print(f"\nSTEP {hop}  state: {len(r.sel)} entities read so far"
              + (f"; reader answers so far: {', '.join(repr(a) for _, a in known)}" if known else
                 " (nothing read yet: the choice uses question/entity similarity only)"))
        print(f"  policy scores {len(r.cands)} candidates" + (" + STOP" if r.stop_ok else "")
              + f"; top {args.top}:")
        order = sorted(range(len(r.cands)), key=lambda i: -pr[i].item())[:args.top]
        for i in order:
            n = r.cands[i]
            mark = "->" if i == r.action else "  "
            print(f"   {mark} p={pr[i]:.2f}  {kind(p, n)} {n:3d}  {short(p.text[n], 60):64s} [{flags(r.feat[i])}]")
        if r.stop_ok:
            print(f"   {'->' if r.action == k_max else '  '} p={pr[k_max]:.2f}  STOP")
        if r.action == k_max:
            print("  DECISION: STOP (enough evidence)")
            continue
        n = r.cands[r.action]
        span, score = cache.table[(p.qid, n)]
        print(f"  DECISION: read {kind(p, n).strip().lower()} {n}: {short(p.text[n], 110)}")
        print(f"  READER on entity {n}: answer '{span}'"
              f"  -> fed back into the state as the bridge for the next step")
    if len(ep.selected) >= cfg.horizon:
        print(f"\n  (horizon of {cfg.horizon} hops reached: selection ends)")

    commit_rec = [r for r in steps if r.commit]
    print(f"\n{LINE}\nCOMMIT  which selected entity's answer is final?")
    if commit_rec:
        r = commit_rec[0]
        pr = probs_of(bank, r, actor, k_max)
        for i, n in enumerate(r.cands):
            span, score = cache.table[(p.qid, n)]
            f = r.feat[i]
            notes = []
            if f[FI["commit_span_in_q"]] >= 0.5:
                notes.append("answer repeats question words")
            if f[FI["commit_agreement"]] > 0:
                notes.append("answer also appears in other evidence")
            mark = "->" if i == r.action else "  "
            print(f"   {mark} p={pr[i]:.2f}  entity {n:3d} answer '{span}'"
                  + (f"  [{'; '.join(notes)}]" if notes else ""))
    else:
        print(f"   only one entity selected -> its answer is final")
    pred = predict(bank, [idx], [ep.selected], cache, commits=[ep.committed])[0]
    print(LINE)
    print(f"FINAL ANSWER: '{pred}'     GOLD: '{p.answer}'     "
          f"EM {exact_match(p.answer, pred):.0f}   F1 {f1_score(p.answer, pred):.2f}")
    print(f"evidence read: {len(ep.selected)} of {p.n} entities, "
          f"{sum(len(p.text[n].split()) for n in ep.selected)} words; inference time {time.time() - t1:.2f}s")


if __name__ == "__main__":
    main()
