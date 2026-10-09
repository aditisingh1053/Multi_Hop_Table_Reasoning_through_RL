"""Generate every result table (LaTeX) and figure (PDF) used in the report.

Inputs : results/baselines/<split>/*.json, results/runs/<name>_s<seed>/{history.json,eval_*.json}
Outputs: report/generated/*.tex, report/figures/*.pdf, results/assets_summary.json
Usage  : python scripts/make_assets.py
"""
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from common import RESULTS, ROOT, save_json  # noqa: E402

GEN = ROOT / "report" / "generated"
FIG = ROOT / "report" / "figures"
GEN.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# validated categorical palette (fixed order) and text / reference inks
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df"}
SEQ = ["#86b6ef", "#2a78d6", "#104281"]   # ordinal blue ramp (steps 250 / 450 / 650)
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
    "bc": "Behaviour cloning", "ppo_scratch": "PPO without warm start", "reinforce": "REINFORCE + baseline",
    "grpo": "GRPO", "ppo_main": "PPO + GAE (ours)", "ppo_noshape": "PPO, no evidence term",
    "ppo_evbonus": "PPO, terminal bonus $\\lambda_{\\mathrm{ev}}{=}0.3$", "ppo_lam0": "PPO, GAE $\\lambda{=}0$",
    "ppo_lam1": "PPO, GAE $\\lambda{=}1$", "ppo_em": "PPO, EM answer reward",
    "ppo_nocost": "PPO, $c{=}0$", "ppo_cost05": "PPO, $c{=}0.05$", "ppo_cost10": "PPO, $c{=}0.1$",
}


def _load(path):
    with open(path) as f:
        return json.load(f)


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
    gt = (x[:, None] > y[None, :]).mean()
    eq = (x[:, None] == y[None, :]).mean()
    return float(gt + 0.5 * eq)


def fmt(m, s=None, d=1, show_std=True):
    if np.isnan(m):
        return "--"
    if s is None or not show_std:
        return f"{m:.{d}f}"
    return f"{m:.{d}f}\\,{{\\scriptsize$\\pm${s:.{d}f}}}"


# ----------------------------------------------------------------------------- tables
def main_table(summary):
    groups = [
        ("No learning", ["random_k2", "random_k3", "bm25_k2", "dense_k1", "dense_k2", "dense_k3",
                         "hybrid_k1", "hybrid_k2", "hybrid_k3", "greedy_hop"]),
        ("Learned selectors", ["bc", "ppo_scratch", "reinforce", "grpo", "ppo_main"]),
        ("Upper bound", ["oracle"]),
    ]
    cols = [("EM", 1), ("F1", 1), ("table_F1", 1), ("passage_F1", 1), ("AnsRecall", 1), ("GoldHit", 1),
            ("Items", 2), ("Words", 0)]
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
             r"\caption{Results on the HybridQA dev set (\nTest{} questions) with the same frozen reader for every "
             r"selector. Learned selectors: mean $\pm$ standard deviation over seeds (number of seeds in brackets). "
             r"F1$_\mathrm{tab}$ and F1$_\mathrm{psg}$: questions whose answer is in a cell (\nTestTypeTable) or in a "
             r"passage (\nTestTypePassage). AnsRec: answer string present in the evidence. Gold: a traced gold path "
             r"fully covered. Items, Words: evidence size passed to the reader.}",
             r"\label{tab:main}", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
             r"Selector & EM & F1 & F1$_\mathrm{tab}$ & F1$_\mathrm{psg}$ & AnsRec & Gold & Items & Words \\"]
    for gname, names in groups:
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{9}}{{@{{}}l}}{{\textit{{{gname}}}}} \\")
        for n in names:
            runs = evals(n)
            if not runs:
                continue
            show = n in ("bc", "ppo_scratch", "reinforce", "grpo", "ppo_main", "random_k2", "random_k3")
            cells = []
            for key, d in cols:
                m, s = stat(runs, key)
                cells.append(fmt(m, s, d, show_std=show and key in ("EM", "F1")))
            label = LABEL[n] + (f" [{len(runs)}]" if show else "")
            if n == "ppo_main":
                label = r"\textbf{" + label + "}"
            lines.append(label + " & " + " & ".join(cells) + r" \\")
            summary.setdefault("main", {})[n] = {k: stat(runs, k) for k, _ in cols}
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (GEN / "table_main.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def tests_table(summary):
    ref = evals("ppo_main")
    if not ref:
        return
    comps = ["bc", "greedy_hop", "dense_k2", "hybrid_k2", "reinforce", "grpo", "ppo_scratch", "ppo_noshape"]
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
             r"\caption{Paired comparisons of PPO (ours) with other selectors on the dev set. $\Delta$ is the "
             r"difference in points, with a 95\% paired-bootstrap interval over questions (10{,}000 resamples, "
             r"per-question scores averaged over seeds) and the one-sided bootstrap $p$-value. $P(\mathrm{imp})$ is "
             r"the probability that a PPO run scores a higher F1 than a run of the other method.}",
             r"\label{tab:tests}", r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
             r"PPO versus & $\Delta$EM [95\% CI] & $p$ & $\Delta$F1 [95\% CI] & $p$ & $P(\mathrm{imp})$ \\",
             r"\midrule"]
    a_f1, a_em = per_q(ref, "f1"), per_q(ref, "em")
    ref_f1 = [r["summary"]["F1"] for r in ref]
    for n in comps:
        runs = evals(n)
        if not runs:
            continue
        dem, lem, hem, pem = bootstrap(a_em, per_q(runs, "em"))
        df1, lf1, hf1, pf1 = bootstrap(a_f1, per_q(runs, "f1"))
        pi = prob_improvement(ref_f1, [r["summary"]["F1"] for r in runs])
        summary.setdefault("tests", {})[n] = {"dEM": [dem, lem, hem, pem], "dF1": [df1, lf1, hf1, pf1], "P_imp": pi}
        pv = lambda p: "$<$0.001" if p < 0.001 else f"{p:.3f}"  # noqa: E731
        lines.append(f"{LABEL[n]} & {dem:+.2f} [{lem:+.2f}, {hem:+.2f}] & {pv(pem)} & "
                     f"{df1:+.2f} [{lf1:+.2f}, {hf1:+.2f}] & {pv(pf1)} & {pi:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (GEN / "table_tests.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def last_val(h):
    vals = [x for x in h["history"] if "val" in x]
    return vals


def optim_table(summary):
    names = ["ppo_main", "ppo_noshape", "reinforce", "grpo", "ppo_scratch"]
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
             r"\caption{Optimisers under identical data, rewards, initialisation (except PPO without warm start) "
             r"and evaluation. Val F1: best validation F1 of the selected iterate. EV: explained variance of the "
             r"critic on the last training batch. Degenerate: share of GRPO groups whose four returns are equal. "
             r"Time: wall-clock minutes per run on the laptop CPU.}",
             r"\label{tab:optim}", r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             r"Method & Val F1 & Dev EM & Dev F1 & Best iter. & EV (end) & Time (min) \\", r"\midrule"]
    for n in names:
        hs, runs = histories(n), evals(n)
        if not hs or not runs:
            continue
        val = [h["best_val_F1"] for h in hs]
        it = [h["best_iter"] for h in hs]
        ev = [h["history"][-1].get("explained_var") for h in hs]
        ev = [e for e in ev if e is not None and not np.isnan(e)]
        secs = [sum(x.get("sec", 0) for x in h["history"]) / 60 for h in hs]
        degen = [np.mean([x["zero_adv_groups"] for x in h["history"] if x.get("zero_adv_groups") is not None])
                 for h in hs] if n == "grpo" else None
        em, ems = stat(runs, "EM")
        f1, f1s = stat(runs, "F1")
        lines.append(f"{LABEL[n]} & {fmt(np.mean(val), np.std(val, ddof=1) if len(val) > 1 else 0)} & "
                     f"{fmt(em, ems)} & {fmt(f1, f1s)} & {np.mean(it):.0f} & "
                     f"{(f'{np.mean(ev):.2f}' if ev else ('--' if degen is None else 'n/a'))} & {np.mean(secs):.0f} \\\\")
        summary.setdefault("optim", {})[n] = {"val": val, "best_iter": it, "ev": ev, "minutes": secs,
                                              "degenerate": None if degen is None else [float(d) for d in degen]}
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
    cols = [("EM", 1), ("F1", 1), ("GoldHit", 1), ("RightWrongEv", 1), ("Items", 2), ("Words", 0)]
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
             r"\caption{Ablations of PPO on the dev set (seeds 0--2 for every row, including the main configuration). "
             r"Gold: a traced gold path is covered. R/W: correct answer (EM) although no traced gold path is covered.}",
             r"\label{tab:ablation}", r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule",
             r"Factor & Setting & EM & F1 & Gold & R/W & Items & Words \\"]
    for factor, items in rows:
        lines.append(r"\midrule")
        first = True
        for n, desc in items:
            runs = evals(n, seeds=seeds)
            if not runs:
                continue
            cells = [fmt(*stat(runs, k), d, show_std=k in ("EM", "F1")) for k, d in cols]
            lines.append(f"{factor if first else ''} & {desc} & " + " & ".join(cells) + r" \\")
            summary.setdefault("ablation", {})[f"{factor}|{n}"] = {k: stat(runs, k) for k, _ in cols}
            first = False
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (GEN / "table_ablation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def reader_table(summary, tag):
    names = ["dense_k2", "hybrid_k2", "greedy_hop", "bc", "grpo", "reinforce", "ppo_main", "oracle"]
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
             r"\caption{Transfer to a second frozen reader (RoBERTa-base SQuAD~2.0) that was never used during "
             r"training, on a fixed random subset of the dev set. The training reader's scores on the same "
             r"subset are given for reference.}",
             r"\label{tab:reader}", r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{Training reader (MiniLM)} & \multicolumn{2}{c}{Unseen reader (RoBERTa)} \\",
             r"\cmidrule(lr){2-3}\cmidrule(l){4-5}", r"Selector & EM & F1 & EM & F1 \\", r"\midrule"]
    any_row = False
    for n in names:
        new = evals(n, tag=tag)
        if not new:
            continue
        qids = {row["qid"] for row in new[0]["rows"]}
        old = evals(n)
        sub = []
        for r in old:   # restrict the training-reader evaluation to the same questions
            rows_ = [row for row in r["rows"] if row["qid"] in qids]
            sub.append({"summary": {"EM": 100 * np.mean([x["em"] for x in rows_]),
                                    "F1": 100 * np.mean([x["f1"] for x in rows_])}})
        show = n in ("bc", "grpo", "reinforce", "ppo_main")
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
    names = ["dense_k2", "hybrid_k2", "greedy_hop", "bc", "reinforce", "grpo", "ppo_main", "oracle"]
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
                if part == "unseen":
                    n_un = len(rows_)
                else:
                    n_se = len(rows_)
            cell[part] = (stat(sub, "EM"), stat(sub, "F1"))
        show = n in ("bc", "grpo", "reinforce", "ppo_main")
        lines.append(LABEL[n] + " & " + " & ".join(fmt(*cell[p][i], show_std=show)
                                                    for p in ("unseen", "seen") for i in (0, 1)) + r" \\")
        summary.setdefault("unseen", {})[n] = cell
    if not lines:
        return
    head = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
            r"\caption{Dev results split by whether the question's table occurs in the training subset.}",
            r"\label{tab:unseen}", r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
            rf" & \multicolumn{{2}}{{c}}{{Unseen tables ({n_un:,} questions)}} & "
            rf"\multicolumn{{2}}{{c}}{{Seen tables ({n_se:,} questions)}} \\",
            r"\cmidrule(lr){2-3}\cmidrule(l){4-5}", r"Selector & EM & F1 & EM & F1 \\", r"\midrule"]
    (GEN / "table_unseen.tex").write_text("\n".join(head + lines + [r"\bottomrule", r"\end{tabular}",
                                                                    r"\end{table}"]) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- figures
def curve(name, key="F1"):
    hs = histories(name)
    if not hs:
        return None
    xs = [x["episodes"] for x in hs[0]["history"] if "val" in x]
    ys = np.array([[x["val"][key] for x in h["history"] if "val" in x] for h in hs])
    return np.array(xs), ys


def fig_curves(summary):
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    series = [("ppo_main", "PPO + GAE (ours)", C["blue"], "-"), ("grpo", "GRPO", C["orange"], "-"),
              ("reinforce", "REINFORCE + baseline", C["aqua"], "-"),
              ("ppo_scratch", "PPO without warm start", C["yellow"], "--")]
    for name, label, col, ls in series:
        c = curve(name)
        if c is None:
            continue
        x, y = c
        m, s = y.mean(0), y.std(0)
        ax.plot(x / 1000, m, color=col, lw=1.6, ls=ls, label=f"{label} ({len(y)} seeds)")
        ax.fill_between(x / 1000, m - s, m + s, color=col, alpha=0.15, lw=0)
    bc = [h["best_val_F1"] for h in histories("bc")]
    if bc:
        ax.axhline(np.mean(bc), color=C["ink2"], ls=(0, (4, 2)), lw=1.0)
        ax.text(0.2, np.mean(bc) + 0.3, "behaviour cloning", color=C["ink2"], fontsize=8, va="bottom")
    gh = evals("greedy_hop", split="val")
    if gh:
        ax.axhline(gh[0]["summary"]["F1"], color=C["ink2"], ls=":", lw=1.0)
        ax.text(0.2, gh[0]["summary"]["F1"] + 0.3, "greedy rank-then-hop", color=C["ink2"], fontsize=8, va="bottom")
    ax.set_xlabel("training episodes (thousands)")
    ax.set_ylabel("validation F1 (greedy policy)")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(FIG / "curves.pdf")
    plt.close(fig)


def fig_diagnostics(summary):
    names = [("ppo_main", "PPO + GAE", C["blue"]), ("grpo", "GRPO", C["orange"]),
             ("reinforce", "REINFORCE + baseline", C["aqua"])]
    panels = [("answer", "training answer F1 (sampled)"), ("explained_var", "critic explained variance"),
              ("kl", "approx. KL per update")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for ax, (key, title) in zip(axes, panels):
        for n, label, col in names:
            hs = histories(n)
            if not hs:
                continue
            its = [x["iter"] for x in hs[0]["history"][1:]]
            vals = np.array([[np.nan if x.get(key) is None else x[key] for x in h["history"][1:]] for h in hs],
                            dtype=float)
            if np.all(np.isnan(vals)):
                continue
            ax.plot(its, np.nanmean(vals, 0), color=col, lw=1.4, label=label)
        ax.set_title(title, color=C["ink"])
        ax.set_xlabel("iteration")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "diagnostics.pdf")
    plt.close(fig)


def fig_tradeoff(summary):
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    for scorer, col, label in [("dense", C["orange"], "dense top-$k$"), ("hybrid", C["aqua"], "hybrid top-$k$")]:
        pts = []
        for k in (1, 2, 3):
            r = evals(f"{scorer}_k{k}")
            if r:
                pts.append((r[0]["summary"]["Words"], r[0]["summary"]["F1"], k))
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=1.2, marker="o", ms=4, label=label)
            for w, f, k in pts:
                ax.annotate(f"$k{{=}}{k}$", (w, f), textcoords="offset points", xytext=(4, -9), fontsize=7,
                            color=C["ink2"])
    gh = evals("greedy_hop")
    if gh:
        ax.plot(gh[0]["summary"]["Words"], gh[0]["summary"]["F1"], marker="s", ms=5, color=C["ink2"], ls="none",
                label="greedy rank-then-hop")
    bc = evals("bc")
    if bc:
        ax.errorbar(np.mean([r["summary"]["Words"] for r in bc]), np.mean([r["summary"]["F1"] for r in bc]),
                    yerr=np.std([r["summary"]["F1"] for r in bc], ddof=1), marker="D", ms=5, color=C["yellow"],
                    ls="none", label="behaviour cloning")
    pts = []
    for n, c_ in [("ppo_nocost", 0.0), ("ppo_main", 0.02), ("ppo_cost05", 0.05), ("ppo_cost10", 0.1)]:
        r = evals(n, seeds={0, 1, 2})
        if r:
            pts.append((np.mean([x["summary"]["Words"] for x in r]), np.mean([x["summary"]["F1"] for x in r]),
                        np.std([x["summary"]["F1"] for x in r], ddof=1), c_))
    if pts:
        pts.sort()
        ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts], color=C["blue"], lw=1.6,
                    marker="o", ms=5, capsize=2, label="PPO, cost $c$ varied")
        for w, f, _, c_ in pts:
            ax.annotate(f"$c{{=}}{c_:g}$", (w, f), textcoords="offset points", xytext=(4, 5), fontsize=7,
                        color=C["ink2"])
    ax.set_xlabel("evidence passed to the reader (words per question)")
    ax.set_ylabel("dev F1")
    ax.legend(fontsize=7, loc="lower right")
    fig.savefig(FIG / "tradeoff.pdf")
    plt.close(fig)


def fig_sizes(summary):
    names = [("greedy_hop", "greedy rank-then-hop"), ("bc", "behaviour cloning"), ("ppo_nocost", "PPO, $c=0$"),
             ("ppo_main", "PPO, $c=0.02$"), ("ppo_cost05", "PPO, $c=0.05$"), ("ppo_cost10", "PPO, $c=0.1$")]
    data = []
    for n, label in names:
        runs = evals(n, seeds={0, 1, 2}) if n.startswith("ppo") else evals(n)
        if not runs:
            continue
        sizes = np.concatenate([[row["n_items"] for row in r["rows"]] for r in runs])
        data.append((label, [100 * np.mean(sizes == k) for k in (1, 2, 3)]))
    if not data:
        return
    fig, ax = plt.subplots(figsize=(5.6, 0.42 * len(data) + 0.9))
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
    ax.set_yticks(y)
    ax.set_yticklabels([d[0] for d in data])
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of dev questions (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=3, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.savefig(FIG / "sizes.pdf")
    plt.close(fig)
    summary["sizes"] = data


def main():
    summary = {}
    main_table(summary)
    tests_table(summary)
    optim_table(summary)
    ablation_table(summary)
    reader_table(summary, "_roberta-base-squad2")
    unseen_table(summary)
    fig_curves(summary)
    fig_diagnostics(summary)
    fig_tradeoff(summary)
    fig_sizes(summary)
    save_json(summary, RESULTS / "assets_summary.json")
    print(json.dumps({k: (list(v) if isinstance(v, dict) else v) for k, v in summary.items()}, indent=0)[:3000])


if __name__ == "__main__":
    main()
