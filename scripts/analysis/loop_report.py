"""Summarise the reader-in-the-loop experiments: results/loop/REPORT.md + summary.json (+ curves.png).

Statistics follow the main report: mean +- std over seeds; paired bootstrap over questions
(per-question scores averaged over seeds, 10,000 resamples) for differences between methods.

Usage: python scripts/analysis/loop_report.py [--root results/loop]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LABEL = {
    "bm25_k3": "BM25 top-3 (best fixed heuristic)",
    "hybrid_k3": "Hybrid top-3",
    "greedy_hop": "Greedy rank-then-hop (HYBRIDER-style)",
    "bridge_hop": "Bridge hop (hand-written reader-in-the-loop rule)",
    "read_all": "Read every entity (~49 reads, max confidence)",
    "oracle": "Oracle gold path (upper bound)",
    "ppo_scratch": "PPO from scratch (previous best)",
    "loop_feedback": "RiL-PPO: bridge feedback only",
    "loop_commit": "RiL-PPO: learned COMMIT only",
    "loop_full": "**RiL-PPO: feedback + COMMIT (ours)**",
    "ppo_scratch_long": "PPO from scratch, 100 iterations",
    "loop_full_long": "**RiL-PPO, 100 iterations (ours)**",
    "plus_rb": "RiL-PPO + Rao-Blackwellised COMMIT",
    "plus_asym": "RiL-PPO + privileged (asymmetric) critic",
    "plus_sil": "RiL-PPO + self-imitation",
    "plus_drop": "RiL-PPO + action dropout",
    "plus_ema": "RiL-PPO + EMA weights",
    "plus_all": "**RiL-PPO++ (all extensions, ours)**",
    "plus_all_long": "**RiL-PPO++, 100 iterations (ours)**",
    "opt_rloo": "RLOO (critic-free), RiL env",
    "opt_drgrpo": "Dr. GRPO (critic-free), RiL env",
    "opt_grpo": "GRPO (critic-free), RiL env",
}
COLS = ["EM", "F1", "F1_maxconf", "F1_bestinset", "AnsRecall", "table_F1", "passage_F1", "Items", "Words"]


def load(path):
    with open(path) as f:
        return json.load(f)


def method_runs(root, tag=""):
    """name -> list of eval dicts (one per seed for learned methods, one for baselines)."""
    out = {}
    split = "test" + tag
    bdir = root / "baselines" / split
    if bdir.exists():
        for f in sorted(bdir.glob("*.json")):
            out[f.stem] = [load(f)]
    rdir = root / "runs"
    if rdir.exists():
        for d in sorted(rdir.iterdir()):
            f = d / f"eval_test{tag}.json"
            if f.exists():
                name = d.name.rsplit("_s", 1)[0]
                out.setdefault(name, []).append(load(f))
    return out


def per_question(evals, key="f1"):
    """qid -> mean over seeds."""
    acc = {}
    for e in evals:
        for r in e["rows"]:
            acc.setdefault(r["qid"], []).append(r[key])
    return {q: float(np.mean(v)) for q, v in acc.items()}


def bootstrap(a, b, n=10000, seed=0):
    qs = sorted(set(a) & set(b))
    d = np.array([a[q] - b[q] for q in qs])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(1)
    return {"delta": 100 * d.mean(), "lo": 100 * np.percentile(means, 2.5),
            "hi": 100 * np.percentile(means, 97.5), "p": float((means <= 0).mean()), "n": len(d)}


def prob_improvement(ea, eb):
    fa = [e["summary"]["F1"] for e in ea]
    fb = [e["summary"]["F1"] for e in eb]
    return float(np.mean([x > y for x in fa for y in fb]))


def ms(evals, key):
    v = [e["summary"].get(key) for e in evals]
    v = [x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not v:
        return None, None
    return float(np.mean(v)), float(np.std(v))


def fmt(m, s, n):
    if m is None:
        return "–"
    return f"{m:.1f} ± {s:.1f}" if n > 1 else f"{m:.1f}"


def unseen_f1(evals, seen):
    vals = []
    for e in evals:
        f = [r["f1"] for r in e["rows"] if r["qid"] not in seen]
        vals.append(100 * float(np.mean(f)) if f else float("nan"))
    return float(np.mean(vals)), float(np.std(vals)), len([r for r in evals[0]["rows"] if r["qid"] not in seen])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "results" / "loop"))
    args = ap.parse_args()
    root = Path(args.root)
    runs = method_runs(root)
    seen_path = ROOT / "results" / "test_qids_seen_table.json"
    seen = set(load(seen_path)) if seen_path.exists() else set()

    order = ["oracle", "read_all", "bm25_k3", "hybrid_k3", "greedy_hop", "bridge_hop",
             "ppo_scratch", "loop_feedback", "loop_commit", "loop_full",
             "plus_rb", "plus_asym", "plus_sil", "plus_drop", "plus_ema", "plus_all",
             "opt_grpo", "opt_drgrpo", "opt_rloo",
             "ppo_scratch_long", "loop_full_long", "plus_all_long"]
    order = [m for m in order if m in runs] + [m for m in runs if m not in order]
    L = ["# Reader-in-the-loop PPO (RiL-PPO): results", "",
         "HybridQA dev set, same frozen reader (MiniLM-SQuAD2) for every selector. Learned methods: "
         "mean ± std over seeds (n = number of seeds). `F1 (max-conf)` = the reader's max-confidence "
         "rule applied to the same selected set; `F1 (best-in-set)` = an oracle arbiter on that set. "
         "The gap AnsRecall → best-in-set is reader error; best-in-set → F1 is aggregation error.", "",
         "## Main table", "",
         "| Selector | n | EM | F1 | F1 (max-conf) | F1 (best-in-set) | Ans. in evidence | F1 table | F1 passage | Entities | Words |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    summary = {"main": {}}
    for m in order:
        ev = runs[m]
        cells = []
        rec = {}
        for c in COLS:
            mu, sd = ms(ev, c)
            rec[c] = [mu, sd]
            if c in ("Items",):
                cells.append("–" if mu is None else f"{mu:.2f}")
            elif c == "Words":
                cells.append("–" if mu is None else f"{mu:.0f}")
            else:
                cells.append(fmt(mu, sd, len(ev)))
        summary["main"][m] = {"n": len(ev), **rec}
        L.append(f"| {LABEL.get(m, m)} | {len(ev)} | " + " | ".join(cells) + " |")

    # paired bootstrap
    pq = {m: per_question(runs[m]) for m in runs}
    comps = [("loop_full", "ppo_scratch"), ("loop_full", "bm25_k3"), ("loop_full", "bridge_hop"),
             ("loop_full", "greedy_hop"), ("loop_full", "read_all"),
             ("loop_feedback", "ppo_scratch"), ("loop_commit", "ppo_scratch"),
             ("loop_full", "loop_feedback"), ("loop_full", "loop_commit"),
             ("bridge_hop", "greedy_hop"),
             ("plus_all", "loop_full"), ("plus_all", "ppo_scratch"), ("plus_all", "bm25_k3"),
             ("plus_rb", "loop_full"), ("plus_asym", "loop_full"), ("plus_sil", "loop_full"),
             ("plus_drop", "loop_full"), ("plus_ema", "loop_full"),
             ("loop_full", "opt_rloo"), ("opt_rloo", "opt_grpo"), ("opt_drgrpo", "opt_grpo"),
             ("loop_full_long", "ppo_scratch_long"), ("loop_full_long", "bm25_k3"),
             ("plus_all_long", "ppo_scratch_long"), ("plus_all_long", "loop_full_long")]
    L += ["", "## Significance (paired bootstrap over questions, 10,000 resamples)", "",
          "| A − B | ΔF1 | 95% CI | p (one-sided) | P(run A > run B) |", "|---|---|---|---|---|"]
    summary["tests"] = {}
    for a, b in comps:
        if a in pq and b in pq:
            t = bootstrap(pq[a], pq[b])
            pi = prob_improvement(runs[a], runs[b])
            summary["tests"][f"{a}-{b}"] = {**t, "P_improve": pi}
            p = "< 0.001" if t["p"] < 0.001 else f"{t['p']:.3f}"
            L.append(f"| {a} − {b} | {t['delta']:+.2f} | [{t['lo']:.2f}, {t['hi']:.2f}] | {p} | {pi:.2f} |")

    # where the gain comes from
    L += ["", "## Where the gain comes from", "",
          "| Selector | Ans. in evidence | Best-in-set F1 | Max-conf F1 | Final F1 | Arbitration gain (F1 − max-conf) | Answer changed by COMMIT (%) |",
          "|---|---|---|---|---|---|---|"]
    for m in ("ppo_scratch", "loop_feedback", "loop_commit", "loop_full", "plus_rb", "plus_all",
              "ppo_scratch_long", "loop_full_long", "plus_all_long"):
        if m not in runs or ms(runs[m], "F1_maxconf")[0] is None:
            continue
        g = lambda k: ms(runs[m], k)[0]
        L.append(f"| {LABEL.get(m, m)} | {g('AnsRecall'):.1f} | {g('F1_bestinset'):.1f} | {g('F1_maxconf'):.1f} | "
                 f"{g('F1'):.1f} | {g('F1') - g('F1_maxconf'):+.1f} | {g('CommitChanged') or 0:.1f} |")

    # unseen tables
    if seen:
        L += ["", f"## Tables never seen in training ({len(seen)} dev questions excluded)", "",
              "| Selector | F1 (unseen tables) |", "|---|---|"]
        summary["unseen"] = {}
        for m in order:
            mu, sd, n = unseen_f1(runs[m], seen)
            summary["unseen"][m] = [mu, sd, n]
            L.append(f"| {LABEL.get(m, m)} | {fmt(mu, sd, len(runs[m]))} |")

    # test-time self-consistency
    sc = {}
    rdir = root / "runs"
    for d in sorted(rdir.iterdir()) if rdir.exists() else []:
        for f in sorted(d.glob("eval_test_sc*.json")):
            sc.setdefault((d.name.rsplit("_s", 1)[0], f.stem.split("_sc")[-1]), []).append(load(f))
    if sc:
        L += ["", "## Test-time self-consistency over evidence paths", "",
              "Greedy episode + (k−1) sampled episodes; majority vote over normalised answers. "
              "`Reads` = distinct entities read per question (reader cost).", "",
              "| Selector | k | F1 greedy | F1 voted | Δ | Reads / question |", "|---|---|---|---|---|---|"]
        summary["self_consistency"] = {}
        for (m, k), ev in sorted(sc.items()):
            g_ = [e["summary"]["F1_greedy"] for e in ev]
            v_ = [e["summary"]["F1"] for e in ev]
            r_ = [e["summary"]["Reads"] for e in ev]
            summary["self_consistency"][f"{m}_k{k}"] = {"greedy": g_, "voted": v_, "reads": r_}
            L.append(f"| {LABEL.get(m, m)} | {k} | {fmt(np.mean(g_), np.std(g_), len(g_))} | "
                     f"{fmt(np.mean(v_), np.std(v_), len(v_))} | {np.mean(v_) - np.mean(g_):+.1f} | "
                     f"{np.mean(r_):.1f} |")

    # optimisation diagnostics from the training logs
    diag = {}
    for d in sorted(rdir.iterdir()) if rdir.exists() else []:
        h = d / "history.json"
        if not h.exists():
            continue
        H = load(h)
        hist = [r for r in H["history"] if r.get("iter", 0) > 0]
        if not hist:
            continue
        tail = hist[-10:]

        def avg(key, rows=tail):
            v = [r.get(key) for r in rows if r.get(key) is not None and not
                 (isinstance(r.get(key), float) and np.isnan(r.get(key)))]
            return float(np.mean(v)) if v else None

        diag.setdefault(d.name.rsplit("_s", 1)[0], []).append({
            "ev": avg("explained_var"), "kl": avg("kl"), "clipfrac": avg("clipfrac"), "ent": avg("ent"),
            "zero_adv": avg("zero_adv_groups", hist), "items": avg("items"), "answer": avg("answer"),
            "ema_best": float(H.get("best_kind") == "ema"), "best_iter": H.get("best_iter")})
    if diag:
        L += ["", "## Optimisation diagnostics (training logs, mean over the last 10 iterations)", "",
              "| Method | train answer F1 | critic expl. var. | KL / update | clip frac. | entropy | "
              "zero-adv. groups | entities | best iterate | EMA chosen |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        summary["diagnostics"] = diag

        def c(v, f="{:.3f}"):
            return "–" if v is None else f.format(v)
        for m in [x for x in order if x in diag] + [x for x in diag if x not in order]:
            rs = diag[m]
            mean_of = lambda k: (np.mean([r[k] for r in rs if r[k] is not None])
                                 if any(r[k] is not None for r in rs) else None)
            L.append(f"| {LABEL.get(m, m)} | {c(mean_of('answer'))} | {c(mean_of('ev'), '{:.2f}')} | "
                     f"{c(mean_of('kl'), '{:.4f}')} | {c(mean_of('clipfrac'), '{:.3f}')} | "
                     f"{c(mean_of('ent'), '{:.2f}')} | {c(mean_of('zero_adv'), '{:.2f}')} | "
                     f"{c(mean_of('items'), '{:.2f}')} | {c(mean_of('best_iter'), '{:.0f}')} | "
                     f"{sum(r['ema_best'] for r in rs):.0f}/{len(rs)} |")

    # second reader
    runs2 = method_runs(root, "_roberta-base-squad2")
    if runs2:
        L += ["", "## Transfer to an unseen reader (RoBERTa-base-SQuAD2, first 1,000 dev questions)", "",
              "For RiL-PPO the new reader is also the one queried inside the loop.", "",
              "| Selector | n | F1 (MiniLM, same questions) | F1 (RoBERTa) |", "|---|---|---|---|"]
        summary["transfer"] = {}
        for m in ["oracle", "bm25_k3", "bridge_hop", "ppo_scratch", "loop_full", "plus_all"]:
            if m not in runs2:
                continue
            qids = {r["qid"] for r in runs2[m][0]["rows"]}
            base = [100 * np.mean([r["f1"] for r in e["rows"] if r["qid"] in qids]) for e in runs.get(m, [])]
            f2 = [e["summary"]["F1"] for e in runs2[m]]
            summary["transfer"][m] = {"minilm": base, "roberta": f2}
            b = f"{np.mean(base):.1f}" if base else "–"
            L.append(f"| {LABEL.get(m, m)} | {len(f2)} | {b} | {fmt(np.mean(f2), np.std(f2), len(f2))} |")

    # learning curves
    curves = {}
    rdir = root / "runs"
    for d in sorted(rdir.iterdir()) if rdir.exists() else []:
        h = d / "history.json"
        if not h.exists():
            continue
        hist = load(h)["history"]
        pts = [(r["iter"], r["val"]["F1"]) for r in hist if "val" in r]
        curves.setdefault(d.name.rsplit("_s", 1)[0], []).append(pts)
    if curves:
        L += ["", "## Validation F1 during training (mean over seeds)", ""]
        iters = sorted({i for v in curves.values() for run in v for i, _ in run})
        show = [i for i in iters if i in (0, 5, 10, 20, 30, 40, 60, 80, 100)]
        L.append("| Method | " + " | ".join(f"it {i}" for i in show) + " |")
        L.append("|---|" + "---|" * len(show))
        for m, rs in curves.items():
            cells = []
            for i in show:
                v = [f for run in rs for it, f in run if it == i]
                cells.append(f"{np.mean(v):.1f}" if v else "")
            L.append(f"| {LABEL.get(m, m)} | " + " | ".join(cells) + " |")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 3.6))
            for m, rs in curves.items():
                its = sorted({i for run in rs for i, _ in run})
                mu = [np.mean([f for run in rs for it, f in run if it == i]) for i in its]
                ax.plot(its, mu, marker="o", ms=3, label=LABEL.get(m, m).replace("*", ""))
            ax.set_xlabel("PPO iteration (1,024 episodes each)")
            ax.set_ylabel("validation F1")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(root / "curves.png", dpi=160)
            L += ["", "![validation curves](curves.png)"]
        except Exception as e:  # plotting is optional
            print("plot skipped:", e)

    root.mkdir(parents=True, exist_ok=True)
    (root / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    with open(root / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("\n".join(L))


if __name__ == "__main__":
    main()
