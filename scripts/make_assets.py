"""Generate every result table (LaTeX), figure (PDF) and number macro used in the report.

Inputs : results/baselines/<split>/*.json, results/runs/<name>_s<seed>/{history.json,eval_*.json},
         data/processed/test.pkl (gold paths, for the answer-location metric)
Outputs: report/generated/*.tex, report/figures/*.pdf, results/assets_summary.json
Usage  : python scripts/make_assets.py
"""
import glob
import json
import pickle
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from common import DATA, RESULTS, ROOT, save_json  # noqa: E402

GEN = ROOT / "report" / "generated"
FIG = ROOT / "report" / "figures"
GEN.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# validated categorical palette (fixed order); heuristics are drawn in neutral inks
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "ink": "#0b0b0b", "ink2": "#52514e", "ink3": "#8a8984", "grid": "#e4e3df"}
SEQ = ["#86b6ef", "#2a78d6", "#104281"]   # ordinal blue ramp (steps 250 / 450 / 650)
SERIES = {"ppo_scratch": ("PPO from scratch", C["blue"], "-"),
          "ppo_main": ("PPO, warm-started", C["orange"], "-"),
          "reinforce": ("REINFORCE, warm-started", C["aqua"], "-"),
          "grpo": ("GRPO, warm-started", C["yellow"], "--")}
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"], "xtick.color": C["ink2"],
    "ytick.color": C["ink2"], "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})

LABEL = {
    "random_k2": "Random walk ($k{=}2$)", "random_k3": "Random walk ($k{=}3$)",
    "bm25_k1": "BM25 top-1", "bm25_k2": "BM25 top-2", "bm25_k3": "BM25 top-3",
    "dense_k1": "Dense top-1", "dense_k2": "Dense top-2 (TableRAG-style)", "dense_k3": "Dense top-3",
    "hybrid_k1": "Hybrid top-1", "hybrid_k2": "Hybrid top-2", "hybrid_k3": "Hybrid top-3",
    "greedy_hop": "Greedy rank-then-hop (HYBRIDER-style)", "oracle": "Oracle gold path",
    "bc": "Behaviour cloning (BC)", "ppo_main": "PPO + GAE, warm-started from BC",
    "reinforce": "REINFORCE + baseline, warm-started", "grpo": "GRPO, warm-started",
    "ppo_scratch": "PPO + GAE from scratch",
    "ppo_noshape": "PPO, no evidence term",
    "ppo_evbonus": "PPO, terminal bonus $\\lambda_{\\mathrm{ev}}{=}0.3$", "ppo_lam0": "PPO, GAE $\\lambda{=}0$",
    "ppo_lam1": "PPO, GAE $\\lambda{=}1$", "ppo_em": "PPO, EM answer reward",
    "ppo_nocost": "PPO, $c{=}0$", "ppo_cost05": "PPO, $c{=}0.05$", "ppo_cost10": "PPO, $c{=}0.1$",
}
LEARNED = {"bc", "ppo_main", "reinforce", "grpo", "ppo_scratch", "random_k2", "random_k3"}
# shorter row labels for the wide main table
SHORT = {"greedy_hop": "Greedy rank-then-hop", "dense_k2": "Dense top-2 (TableRAG)", "bc": "Behaviour cloning",
         "ppo_main": "PPO + GAE, warm-started", "reinforce": "REINFORCE, warm-started", "grpo": "GRPO, warm-started",
         "random_k2": "Random walk, $k{=}2$", "random_k3": "Random walk, $k{=}3$"}

with open(DATA / "test.pkl", "rb") as f:
    _payload = pickle.load(f)
GOLD = {inst["qid"]: [tuple(p) for p in inst["gold_paths"]] for inst in _payload["instances"]}
del _payload


def _load(path):
    with open(path) as f:
        return json.load(f)


def _augment(d):
    """Adds answer-location metrics to an evaluation and checks the stored gold-path coverage."""
    if d.get("_aug"):
        return d
    loc, right_noloc, has = [], [], []
    for r in d["rows"]:
        paths = GOLD[r["qid"]]
        sel = set(r["selected"])
        if paths:
            covered = any(all(n in sel for n in p) for p in paths)
            assert float(covered) == r["gold_hit"], "gold paths differ between machines"
            hit = any(p[-1] in sel for p in paths)
            loc.append(float(hit))
            right_noloc.append(float(r["em"] == 1 and not hit))
    d["summary"]["AnsLoc"] = 100 * float(np.mean(loc))
    d["summary"]["RightNoLoc"] = 100 * float(np.mean(right_noloc))
    d["_aug"] = True
    return d


def evals(name, split="test", tag="", seeds=None):
    """List of evaluation dicts for a method (one per seed)."""
    out = []
    for f in sorted(glob.glob(str(RESULTS / "baselines" / f"{split}{tag}" / f"{name}*.json"))):
        d = _load(f)
        if re.sub(r"_s\d+$", "", d["method"]) == name:
            out.append(d)
    for f in sorted(glob.glob(str(RESULTS / "runs" / f"{name}_s*" / f"eval_{split}{tag}.json"))):
        seed = int(re.search(r"_s(\d+)", f).group(1))
        if seeds is None or seed in seeds:
            out.append(_load(f))
    if split == "test":
        out = [_augment(d) for d in out]
    return out


def histories(name, seeds=None):
    out = []
    for f in sorted(glob.glob(str(RESULTS / "runs" / f"{name}_s*" / "history.json"))):
        seed = int(re.search(r"_s(\d+)", f).group(1))
        if seeds is None or seed in seeds:
            out.append(_load(f))
    return out


def stat(runs, key):
    v = np.array([r["summary"][key] for r in runs if r["summary"].get(key) is not None], dtype=float)
    if len(v) == 0:
        return np.nan, np.nan
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def per_q(runs, key):
    return np.mean([[row[key] for row in r["rows"]] for r in runs], axis=0)


def bootstrap(a, b, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = a - b
    boots = np.concatenate([d[rng.integers(0, len(d), (1000, len(d)))].mean(1) for _ in range(n // 1000)])
    return 100 * d.mean(), 100 * np.percentile(boots, 2.5), 100 * np.percentile(boots, 97.5), float((boots <= 0).mean())


def prob_improvement(x, y):
    """P(X > Y) over all pairs of runs (ties count one half), as in Agarwal et al. (2021)."""
    x, y = np.asarray(x), np.asarray(y)
    return float((x[:, None] > y[None, :]).mean() + 0.5 * (x[:, None] == y[None, :]).mean())


def num(x, d=1, plus=False):
    """A number for LaTeX text: a true minus sign for negatives, optional explicit plus."""
    s = f"{abs(x):.{d}f}"
    if x < 0 and float(s) != 0:
        return "$-$" + s
    return ("+" + s) if plus else s


def fmt(m, s=None, d=1, show_std=True):
    if m is None or np.isnan(m):
        return "--"
    if s is None or not show_std:
        return f"{m:.{d}f}"
    return f"{m:.{d}f}\\,{{\\scriptsize$\\pm${s:.{d}f}}}"


# ----------------------------------------------------------------------------- tables
def main_table(summary):
    groups = [
        ("No learning", ["random_k2", "random_k3", "bm25_k2", "bm25_k3", "dense_k2", "dense_k3",
                         "hybrid_k2", "hybrid_k3", "greedy_hop"]),
        ("Learned selectors", ["bc", "grpo", "ppo_main", "reinforce", "ppo_scratch"]),
        ("Upper bound", ["oracle"]),
    ]
    cols = [("EM", 1), ("F1", 1), ("table_F1", 1), ("passage_F1", 1), ("AnsRecall", 1), ("AnsLoc", 1),
            ("GoldHit", 1), ("Items", 2), ("Words", 0)]
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
             r"\caption{Results on the HybridQA dev set (\nTest{} questions) with the same frozen reader for every "
             r"selector. Learned selectors: mean $\pm$ standard deviation over seeds (number of seeds in brackets). "
             r"F1$_\mathrm{tab}$, F1$_\mathrm{psg}$: questions whose answer is in a cell (\nTestTypeTable) or a passage "
             r"(\nTestTypePassage). AnsRec: answer string present in the evidence. Loc: the entity holding a traced "
             r"answer location is selected. Path: a traced gold path is fully covered (row and passage). "
             r"Items, Words: evidence passed to the reader.}",
             r"\label{tab:main}", r"\setlength{\tabcolsep}{2.9pt}",
             r"\begin{tabular}{@{}lccccccccc@{}}", r"\toprule",
             r"Selector & EM & F1 & F1$_\mathrm{tab}$ & F1$_\mathrm{psg}$ & AnsRec & Loc & Path & Items & Words \\"]
    for gname, names in groups:
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{10}}{{@{{}}l}}{{\textit{{{gname}}}}} \\")
        for n in names:
            runs = evals(n)
            if not runs:
                continue
            show = n in LEARNED
            cells = [fmt(*stat(runs, k), d, show_std=show and k in ("EM", "F1")) for k, d in cols]
            label = SHORT.get(n, LABEL[n]) + (f" [{len(runs)}]" if show else "")
            if n == "ppo_scratch":
                label = r"\textbf{" + label + "}"
                cells = [r"\textbf{" + c + "}" if i < 2 else c for i, c in enumerate(cells)]
            lines.append(label + " & " + " & ".join(cells) + r" \\")
            summary.setdefault("main", {})[n] = {k: stat(runs, k) for k, _ in cols + [("RightNoLoc", 1)]}
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (GEN / "table_main.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


PAIRS = [("ppo_scratch", "bc"), ("ppo_scratch", "ppo_main"), ("ppo_scratch", "reinforce"),
         ("ppo_scratch", "bm25_k3"), ("ppo_scratch", "hybrid_k3"), ("ppo_scratch", "greedy_hop"),
         ("ppo_main", "bc"), ("reinforce", "bc"), ("grpo", "bc"), ("reinforce", "ppo_main")]


def tests_table(summary):
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
             r"\caption{Paired comparisons on the dev set. $\Delta$ is the difference A$-$B in points with a 95\% "
             r"paired-bootstrap interval over questions (10{,}000 resamples, per-question scores averaged over seeds) "
             r"and the one-sided bootstrap $p$-value for A $\le$ B. $P(\mathrm{A}>\mathrm{B})$: probability that a run "
             r"of A has a higher F1 than a run of B (deterministic baselines count as one run).}",
             r"\label{tab:tests}", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
             r"A & B & $\Delta$EM [95\% CI] & $p$ & $\Delta$F1 [95\% CI] & $p$ & $P(\mathrm{A}>\mathrm{B})$ \\",
             r"\midrule"]
    pv = lambda p: "$<$0.001" if p < 0.001 else f"{p:.3f}"  # noqa: E731
    short = {"ppo_scratch": "PPO scratch", "ppo_main": "PPO warm", "bc": "BC", "reinforce": "REINFORCE",
             "grpo": "GRPO", "bm25_k3": "BM25 top-3", "hybrid_k3": "Hybrid top-3", "greedy_hop": "Greedy hop"}
    for a, b in PAIRS:
        ra, rb = evals(a), evals(b)
        if not ra or not rb:
            continue
        dem, lem, hem, pem = bootstrap(per_q(ra, "em"), per_q(rb, "em"))
        df1, lf1, hf1, pf1 = bootstrap(per_q(ra, "f1"), per_q(rb, "f1"))
        pi = prob_improvement([r["summary"]["F1"] for r in ra], [r["summary"]["F1"] for r in rb])
        summary.setdefault("tests", {})[f"{a}:{b}"] = {"dEM": [dem, lem, hem, pem], "dF1": [df1, lf1, hf1, pf1],
                                                       "P_imp": pi}
        sg = lambda x: num(x, 2, plus=True)  # noqa: E731
        lines.append(f"{short[a]} & {short[b]} & {sg(dem)} [{sg(lem)}, {sg(hem)}] & {pv(pem)} & "
                     f"{sg(df1)} [{sg(lf1)}, {sg(hf1)}] & {pv(pf1)} & {pi:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (GEN / "table_tests.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def optim_table(summary):
    names = ["ppo_scratch", "ppo_main", "reinforce", "grpo"]
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
             r"\caption{Training behaviour of the optimisers (same data, reward, evaluation and per-iteration "
             r"episode budget). Init: initial validation F1 of the starting policy. Best val: validation F1 of the "
             r"selected iterate (iteration in brackets, mean over seeds). EV: explained variance of the critic on the "
             r"last batch. Items: entities per training episode at the last iteration. Time: minutes of training "
             r"iterations per run (periodic validation excluded) on one T4 GPU.}",
             r"\label{tab:optim}", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             r"Method & Init & Best val (iter.) & Dev F1 & EV (end) & Items (end) & Time \\", r"\midrule"]
    for n in names:
        hs, runs = histories(n), evals(n)
        if not hs or not runs:
            continue
        init = [h["history"][0]["val"]["F1"] for h in hs]
        val = [h["best_val_F1"] for h in hs]
        it = [h["best_iter"] for h in hs]
        ev = [h["history"][-1].get("explained_var") for h in hs]
        ev = [e for e in ev if e is not None and not np.isnan(e)]
        items = [h["history"][-1]["items"] for h in hs]
        secs = [sum(x.get("sec", 0) for x in h["history"]) / 60 for h in hs]
        degen = [float(np.mean([x["zero_adv_groups"] for x in h["history"] if x.get("zero_adv_groups") is not None]))
                 for h in hs] if n == "grpo" else None
        f1, f1s = stat(runs, "F1")
        lines.append(f"{LABEL[n]} & {np.mean(init):.1f} & {np.mean(val):.1f} ({np.mean(it):.0f}) & {fmt(f1, f1s)} & "
                     f"{(f'{np.mean(ev):.2f}' if ev else 'n/a')} & {np.mean(items):.2f} & {np.mean(secs):.1f} \\\\")
        summary.setdefault("optim", {})[n] = {"init": init, "val": val, "best_iter": it, "ev": ev, "items": items,
                                              "minutes": secs, "degenerate": degen}
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (GEN / "table_optim.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def ablation_table(summary):
    seeds = {0, 1, 2}
    rows = [("Reward form", [("ppo_noshape", "no evidence term"), ("ppo_main", "shaping, $\\kappa=0.5$ (main)"),
                             ("ppo_evbonus", "terminal bonus, $\\lambda_{\\mathrm{ev}}=0.3$"),
                             ("ppo_em", "shaping, EM instead of F1")]),
            ("GAE $\\lambda$", [("ppo_lam0", "$\\lambda=0$"), ("ppo_main", "$\\lambda=0.95$ (main)"),
                                ("ppo_lam1", "$\\lambda=1$")]),
            ("Entity cost $c$", [("ppo_nocost", "$c=0$"), ("ppo_main", "$c=0.02$ (main)"),
                                 ("ppo_cost05", "$c=0.05$"), ("ppo_cost10", "$c=0.1$")])]
    cols = [("EM", 1), ("F1", 1), ("GoldHit", 1), ("RightNoLoc", 1), ("Items", 2)]
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
             r"\caption{Ablations of the warm-started PPO configuration on the dev set (seeds 0--2 for every row, "
             r"including the main configuration). Best it.: validation-selected iteration per seed. Path: a traced "
             r"gold path is covered. R/L: exact match although no traced answer location is selected.}",
             r"\label{tab:ablation}", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule",
             r"Factor & Setting & EM & F1 & Path & R/L & Items & Best it. \\"]
    for factor, items in rows:
        lines.append(r"\midrule")
        first = True
        for n, desc in items:
            runs = evals(n, seeds=seeds)
            hs = histories(n, seeds=seeds)
            if not runs:
                continue
            cells = [fmt(*stat(runs, k), d, show_std=k in ("EM", "F1")) for k, d in cols]
            cells.append(", ".join(str(h["best_iter"]) for h in hs))
            lines.append(f"{factor if first else ''} & {desc} & " + " & ".join(cells) + r" \\")
            summary.setdefault("ablation", {})[f"{factor}|{n}"] = {k: stat(runs, k) for k, _ in cols}
            first = False
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (GEN / "table_ablation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def reader_table(summary, tag):
    names = ["dense_k2", "hybrid_k2", "hybrid_k3", "bm25_k3", "greedy_hop", "bc", "grpo", "ppo_main", "reinforce",
             "ppo_scratch", "oracle"]
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
             r"\caption{Transfer to a second frozen reader (RoBERTa-base SQuAD~2.0) that was never used during "
             r"training, on the first 1{,}000 dev questions (the dev file is ordered by a hashed question id). Each "
             r"selector's evidence is exactly what it selected in the main evaluation; only the reader changes. The "
             r"oracle is recomputed under the new reader. Left: the training reader's scores on the same questions.}",
             r"\label{tab:reader}", r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{Training reader (MiniLM)} & \multicolumn{2}{c}{Unseen reader (RoBERTa)} \\",
             r"\cmidrule(lr){2-3}\cmidrule(l){4-5}", r"Selector & EM & F1 & EM & F1 \\", r"\midrule"]
    any_row = False
    for n in names:
        new = evals(n, tag=tag)
        if not new:
            continue
        qids = {row["qid"] for row in new[0]["rows"]}
        sub = []
        for r in evals(n):
            rows_ = [row for row in r["rows"] if row["qid"] in qids]
            sub.append({"summary": {"EM": 100 * np.mean([x["em"] for x in rows_]),
                                    "F1": 100 * np.mean([x["f1"] for x in rows_])}})
        show = n in LEARNED
        cells = [fmt(*stat(sub, "EM"), show_std=show), fmt(*stat(sub, "F1"), show_std=show),
                 fmt(*stat(new, "EM"), show_std=show), fmt(*stat(new, "F1"), show_std=show)]
        lines.append(LABEL[n] + " & " + " & ".join(cells) + r" \\")
        summary.setdefault("reader", {})[n] = {"old": [stat(sub, "EM"), stat(sub, "F1")],
                                               "new": [stat(new, "EM"), stat(new, "F1")], "n": len(qids)}
        any_row = True
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    if any_row:
        (GEN / "table_reader.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def unseen_table(summary):
    path = RESULTS / "test_qids_seen_table.json"
    if not path.exists():
        return
    seen = set(_load(path))
    names = ["bm25_k3", "hybrid_k3", "greedy_hop", "bc", "grpo", "ppo_main", "reinforce", "ppo_scratch", "oracle"]
    lines, n_un, n_se = [], None, None
    for n in names:
        runs = evals(n)
        if not runs:
            continue
        cell = {}
        for part, keep in (("unseen", lambda q: q not in seen), ("seen", lambda q: q in seen)):
            sub = []
            for r in runs:
                rows_ = [x for x in r["rows"] if keep(x["qid"])]
                sub.append({"summary": {"EM": 100 * np.mean([x["em"] for x in rows_]),
                                        "F1": 100 * np.mean([x["f1"] for x in rows_])}})
                n_un, n_se = (len(rows_), n_se) if part == "unseen" else (n_un, len(rows_))
            cell[part] = (stat(sub, "EM"), stat(sub, "F1"))
        show = n in LEARNED
        lines.append(LABEL[n] + " & " + " & ".join(fmt(*cell[p][i], show_std=show)
                                                    for p in ("unseen", "seen") for i in (0, 1)) + r" \\")
        summary.setdefault("unseen", {})[n] = cell
    head = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
            r"\caption{Dev results split by whether the question's table occurs in the training subset.}",
            r"\label{tab:unseen}", r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
            rf" & \multicolumn{{2}}{{c}}{{Unseen tables ({n_un:,} questions)}} & "
            rf"\multicolumn{{2}}{{c}}{{Seen tables ({n_se:,} questions)}} \\",
            r"\cmidrule(lr){2-3}\cmidrule(l){4-5}", r"Selector & EM & F1 & EM & F1 \\", r"\midrule"]
    (GEN / "table_unseen.tex").write_text("\n".join(head + lines + [r"\bottomrule", r"\end{tabular}",
                                                                    r"\end{table}"]) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- figures
def fig_curves(summary):
    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    for name, (label, col, ls) in SERIES.items():
        hs = histories(name)
        if not hs:
            continue
        x = np.array([v["episodes"] for v in hs[0]["history"] if "val" in v]) / 1000
        y = np.array([[v["val"]["F1"] for v in h["history"] if "val" in v] for h in hs])
        m, s = y.mean(0), y.std(0)
        ax.plot(x, m, color=col, lw=1.8, ls=ls, marker="o", ms=3.5, label=f"{label} ({len(y)} seeds)")
        ax.fill_between(x, m - s, m + s, color=col, alpha=0.14, lw=0)
    for name, txt, ls in [("bc", "behaviour cloning (best epoch)", (0, (4, 2))), ("greedy_hop", "greedy rank-then-hop", ":")]:
        if name == "bc":
            v = np.mean([h["best_val_F1"] for h in histories("bc")])
        else:
            r = evals(name, split="val")
            v = r[0]["summary"]["F1"] if r else None
        if v is not None:
            ax.axhline(v, color=C["ink2"], ls=ls, lw=1.0)
            ax.text(41.5, v, txt, color=C["ink2"], fontsize=7.5, va="center")
    ax.set_xlabel("training episodes (thousands)")
    ax.set_ylabel("validation F1 (greedy policy)")
    ax.set_xlim(0, 41)
    ax.legend(loc="upper center", fontsize=7.5, bbox_to_anchor=(0.5, -0.17), ncol=2)
    fig.savefig(FIG / "curves.pdf")
    plt.close(fig)


def fig_diagnostics(summary):
    panels = [("answer", "(a) training answer F1 (sampled)"), ("items", "(b) entities per episode"),
              ("explained_var", "(c) critic explained variance")]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))
    for ax, (key, title) in zip(axes, panels):
        for n, (label, col, ls) in SERIES.items():
            hs = histories(n)
            if not hs:
                continue
            its = [x["iter"] for x in hs[0]["history"][1:]]
            vals = np.array([[np.nan if x.get(key) is None else x[key] for x in h["history"][1:]] for h in hs],
                            dtype=float)
            if np.all(np.isnan(vals)):
                continue
            ax.plot(its, np.nanmean(vals, 0), color=col, lw=1.5, ls=ls, label=label)
        ax.set_title(title, color=C["ink"], loc="left")
        ax.set_xlabel("iteration")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIG / "diagnostics.pdf")
    plt.close(fig)


def fig_tradeoff(summary):
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    for scorer, mk, label in [("bm25", "^", "BM25 top-$k$"), ("dense", "o", "dense top-$k$"),
                              ("hybrid", "s", "hybrid top-$k$")]:
        pts = [(evals(f"{scorer}_k{k}")[0]["summary"]["Words"], evals(f"{scorer}_k{k}")[0]["summary"]["F1"], k)
               for k in (1, 2, 3) if evals(f"{scorer}_k{k}")]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=C["ink3"], lw=1.0, marker=mk, ms=4.5,
                mfc="white", label=label)
        if scorer == "bm25":
            for w, f, k in pts:
                ax.annotate(f"$k{{=}}{k}$", (w, f), textcoords="offset points", xytext=(4, 4), fontsize=7,
                            color=C["ink2"])
    gh = evals("greedy_hop")[0]["summary"]
    ax.plot(gh["Words"], gh["F1"], marker="D", ms=5, color=C["ink2"], ls="none", label="greedy rank-then-hop")
    bc = evals("bc")
    ax.plot(np.mean([r["summary"]["Words"] for r in bc]), np.mean([r["summary"]["F1"] for r in bc]), marker="X",
            ms=7, color=C["ink"], ls="none", label="behaviour cloning")
    pts = []
    for n, c_ in [("ppo_nocost", 0.0), ("ppo_main", 0.02), ("ppo_cost05", 0.05), ("ppo_cost10", 0.1)]:
        r = evals(n, seeds={0, 1, 2})
        pts.append((np.mean([x["summary"]["Words"] for x in r]), np.mean([x["summary"]["F1"] for x in r]), c_))
    pts.sort()
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=C["orange"], lw=1.6, marker="o", ms=4.5,
            label="PPO warm-started, $c \\in \\{0, .02, .05, .1\\}$")
    sc = evals("ppo_scratch")
    ax.errorbar(np.mean([r["summary"]["Words"] for r in sc]), np.mean([r["summary"]["F1"] for r in sc]),
                yerr=np.std([r["summary"]["F1"] for r in sc], ddof=1), marker="*", ms=11, color=C["blue"],
                ls="none", capsize=2, label="PPO from scratch ($c=0.02$)")
    ax.set_xlabel("evidence passed to the reader (words per question)")
    ax.set_ylabel("dev F1")
    ax.legend(fontsize=7, loc="upper left")
    fig.savefig(FIG / "tradeoff.pdf")
    plt.close(fig)


def fig_sizes(summary):
    names = [("greedy_hop", "greedy rank-then-hop"), ("bc", "behaviour cloning"), ("grpo", "GRPO"),
             ("ppo_main", "PPO, warm-started"), ("reinforce", "REINFORCE"), ("ppo_scratch", "PPO from scratch")]
    data = []
    for n, label in names:
        runs = evals(n)
        sizes = np.concatenate([[row["n_items"] for row in r["rows"]] for r in runs])
        rows = np.concatenate([[row["n_rows"] for row in r["rows"]] for r in runs])
        data.append((label, [100 * np.mean(sizes == k) for k in (1, 2, 3)], float(np.mean(rows)),
                     float(np.mean(sizes - rows))))
    fig, ax = plt.subplots(figsize=(5.8, 0.42 * len(data) + 0.9))
    y = np.arange(len(data))[::-1]
    left = np.zeros(len(data))
    for k in range(3):
        vals = np.array([d[1][k] for d in data])
        ax.barh(y, vals, left=left, color=SEQ[k], edgecolor="white", linewidth=1.5, height=0.62,
                label=f"{k + 1} entit{'y' if k == 0 else 'ies'}")
        for yi, l, v in zip(y, left, vals):
            if v >= 9:
                ax.text(l + v / 2, yi, f"{v:.0f}%", ha="center", va="center", fontsize=7,
                        color="white" if k > 0 else C["ink"])
        left += vals
    for yi, d in zip(y, data):
        ax.text(101.5, yi, f"{d[2]:.2f} rows / {d[3]:.2f} psg", va="center", fontsize=7, color=C["ink2"])
    ax.set_yticks(y)
    ax.set_yticklabels([d[0] for d in data])
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of dev questions (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=3, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.savefig(FIG / "sizes.pdf")
    plt.close(fig)
    summary["sizes"] = data


# ----------------------------------------------------------------------------- number macros
def numbers(summary):
    m, t = summary["main"], summary["tests"]
    mac = {}

    def put(name, val, d=1):
        mac[name] = num(val, d)

    for key, n in [("Scratch", "ppo_scratch"), ("Ppo", "ppo_main"), ("Bc", "bc"), ("Reinforce", "reinforce"),
                   ("Grpo", "grpo"), ("Greedy", "greedy_hop"), ("BmThree", "bm25_k3"), ("HybThree", "hybrid_k3"),
                   ("DenseTwo", "dense_k2"), ("Oracle", "oracle")]:
        put(f"f{key}", m[n]["F1"][0]); put(f"em{key}", m[n]["EM"][0])
        put(f"f{key}Std", m[n]["F1"][1]); put(f"items{key}", m[n]["Items"][0], 2)
        put(f"words{key}", m[n]["Words"][0], 0); put(f"ansrec{key}", m[n]["AnsRecall"][0])
        put(f"loc{key}", m[n]["AnsLoc"][0]); put(f"path{key}", m[n]["GoldHit"][0])
        put(f"ftab{key}", m[n]["table_F1"][0]); put(f"fpsg{key}", m[n]["passage_F1"][0])
        put(f"rnl{key}", m[n]["RightNoLoc"][0])
    for pair, key in [("ppo_scratch:bc", "ScratchBc"), ("ppo_scratch:bm25_k3", "ScratchBm"),
                      ("ppo_scratch:greedy_hop", "ScratchGreedy"), ("ppo_scratch:ppo_main", "ScratchPpo"),
                      ("ppo_main:bc", "PpoBc"), ("reinforce:bc", "ReinforceBc"), ("grpo:bc", "GrpoBc"),
                      ("ppo_scratch:reinforce", "ScratchReinforce")]:
        d, lo, hi, p = t[pair]["dF1"]
        put(f"d{key}", d, 2); put(f"d{key}Lo", lo, 2); put(f"d{key}Hi", hi, 2)
        mac[f"p{key}"] = "$<$0.001" if p < 0.001 else f"{p:.3f}"
        put(f"pi{key}", t[pair]["P_imp"], 2)
        de, del_, deh, pe = t[pair]["dEM"]
        put(f"dem{key}", de, 2)
    o = summary["optim"]
    for key, n in [("Scratch", "ppo_scratch"), ("Ppo", "ppo_main"), ("Reinforce", "reinforce"), ("Grpo", "grpo")]:
        put(f"val{key}", np.mean(o[n]["val"])); put(f"init{key}", np.mean(o[n]["init"]))
        put(f"min{key}", np.mean(o[n]["minutes"])); put(f"itemsEnd{key}", np.mean(o[n]["items"]), 2)
        if o[n]["ev"]:
            put(f"ev{key}", np.mean(o[n]["ev"]), 2)
    put("degenGrpo", 100 * np.mean(o["grpo"]["degenerate"]), 0)
    if "unseen" in summary:
        u = summary["unseen"]
        put("fScratchUnseen", u["ppo_scratch"]["unseen"][1][0]); put("fBcUnseen", u["bc"]["unseen"][1][0])
        put("fBmThreeUnseen", u["bm25_k3"]["unseen"][1][0])
    if "reader" in summary and "ppo_scratch" in summary["reader"]:
        r = summary["reader"]
        for key, n in [("Scratch", "ppo_scratch"), ("Bc", "bc"), ("HybThree", "hybrid_k3"), ("BmThree", "bm25_k3"),
                       ("Greedy", "greedy_hop"), ("Reinforce", "reinforce"), ("Ppo", "ppo_main"), ("Grpo", "grpo"),
                       ("Oracle", "oracle")]:
            if n in r:
                put(f"rob{key}", r[n]["new"][1][0])
                put(f"mini{key}", r[n]["old"][1][0])
    (GEN / "numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(mac.items())),
                                     encoding="utf-8")


def main():
    summary = {}
    main_table(summary)
    tests_table(summary)
    optim_table(summary)
    ablation_table(summary)
    reader_table(summary, "_roberta-base-squad2-cpu")
    unseen_table(summary)
    fig_curves(summary)
    fig_diagnostics(summary)
    fig_tradeoff(summary)
    fig_sizes(summary)
    numbers(summary)
    save_json(summary, RESULTS / "assets_summary.json")
    for k in ("main", "tests", "optim"):
        print(k, json.dumps(summary[k], default=float)[:1500])


if __name__ == "__main__":
    main()
