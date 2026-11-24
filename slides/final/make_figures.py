"""Figures for the final presentation (reads results/loop). Run from the repo root:
    python slides/final/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results" / "loop" / "runs"
OUT = Path(__file__).resolve().parent / "fig"
OUT.mkdir(exist_ok=True)

OURS, OURS_L, BASE, GREY, INK = "#8C1D18", "#D9A39F", "#2F4B7C", "#A7ADB5", "#222222"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 15, "axes.edgecolor": "#444444",
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200,
    "axes.titlesize": 15,
})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


# ---- Figure: main results (test F1, mean +- std over 3 seeds) ----
rows = [  # label, mean, std, colour (fixed baselines are deterministic: no error bar)
    ("Greedy rank-then-hop", 17.4, 0, GREY),
    ("Bridge hop (hand rule)", 19.0, 0, GREY),
    ("BM25 top-3", 20.8, 0, GREY),
    ("Read every entity (~49)", 21.5, 0, GREY),
    ("PPO, 40 it", 27.6, 1.0, BASE),
    ("PPO, 100 it", 28.0, 1.2, BASE),
    ("RiL-PPO, 40 it (ours)", 29.9, 1.6, OURS_L),
    ("RiL-PPO, 100 it (ours)", 37.5, 1.8, OURS),
]
fig, ax = plt.subplots(figsize=(9.5, 5.2))
y = np.arange(len(rows))
ax.barh(y, [r[1] for r in rows], color=[r[3] for r in rows], height=0.62)
for i, r in enumerate(rows):
    if r[2] > 0:
        ax.errorbar(r[1], i, xerr=r[2], color="#333333", capsize=4, lw=1.2)
for i, r in enumerate(rows):
    ax.text(r[1] + r[2] + 0.6, i, f"{r[1]:.1f}", va="center", fontsize=14, color=INK)
ax.axvline(55.6, color="#555555", ls="--", lw=1.2)
ax.text(55.0, len(rows) - 0.5, "oracle evidence\n55.6", ha="right", va="top", fontsize=13, color="#555555")
ax.set_yticks(y, [r[0] for r in rows])
ax.set_xlim(0, 58)
ax.set_xlabel("Test F1 (HybridQA dev, 3,466 questions)")
save(fig, "main_results.png")

# ---- Figure: validation curves, mean +- std over seeds ----
def curves(prefix):
    per_seed = []
    for s in range(3):
        h = json.load(open(RUNS / f"{prefix}_s{s}" / "history.json"))["history"]
        pts = {r["iter"]: r["val"]["F1"] for r in h if r.get("val")}
        per_seed.append(pts)
    its = sorted(set.intersection(*[set(p) for p in per_seed]))
    a = np.array([[p[i] for i in its] for p in per_seed])
    return np.array(its), a.mean(0), a.std(0)

fig, ax = plt.subplots(figsize=(9.5, 5.0))
for prefix, lab, col in [("ppo_scratch_long", "PPO", BASE), ("loop_full_long", "RiL-PPO (ours)", OURS)]:
    it, m, s = curves(prefix)
    ax.plot(it, m, color=col, lw=2.4, label=lab, marker="o", ms=4)
    ax.fill_between(it, m - s, m + s, color=col, alpha=0.15, lw=0)
ax.set_xlabel("PPO iteration (1,024 episodes each)")
ax.set_ylabel("Validation F1 (1,000 questions)")
ax.legend(frameon=False, loc="lower right")
ax.grid(axis="y", color="#E5E5E5")
save(fig, "curves.png")

# ---- Figure: where the gain comes from ----
cats = ["Answer in\nselected evidence", "Best answer\nin the set", "Reader's\nmax-confidence", "Final answer\n(F1)"]
ppo = [45.3, 37.9, 28.0, 28.0]
ril = [57.4, 47.4, 29.5, 37.5]
x = np.arange(len(cats))
fig, ax = plt.subplots(figsize=(9.5, 5.0))
b1 = ax.bar(x - 0.19, ppo, 0.38, color=BASE, label="PPO, 100 it")
b2 = ax.bar(x + 0.19, ril, 0.38, color=OURS, label="RiL-PPO, 100 it (ours)")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"{b.get_height():.1f}",
                ha="center", fontsize=13)
ax.set_xticks(x, cats)
ax.set_ylabel("Score on test (%)")
ax.set_ylim(0, 65)
ax.legend(frameon=False, loc="upper right")
save(fig, "decomposition.png")

# ---- Figure: training-signal variants (40 iterations) ----
var = [  # label, mean, std
    ("GRPO", 17.7, 0.5), ("Self-imitation", 26.5, 1.6), ("Privileged critic", 26.6, 1.5),
    ("Dr. GRPO", 28.4, 1.0), ("All extensions", 28.5, 1.9), ("RiL-PPO (PPO+GAE)", 29.9, 1.6),
    ("+ RB COMMIT", 30.1, 1.2), ("RLOO", 30.2, 1.6), ("+ Action dropout", 31.1, 2.6),
]
fig, ax = plt.subplots(figsize=(9.5, 5.2))
y = np.arange(len(var))
cols = [OURS if v[0].startswith("RiL") else (GREY if v[1] < 29.9 else BASE) for v in var]
ax.barh(y, [v[1] for v in var], xerr=[v[2] for v in var], color=cols, height=0.62,
        error_kw=dict(ecolor="#333333", capsize=4, lw=1.2))
ax.axvline(29.9, color=OURS, ls=":", lw=1.4)
for i, v in enumerate(var):
    ax.text(v[1] + v[2] + 0.5, i, f"{v[1]:.1f}", va="center", fontsize=14)
ax.set_yticks(y, [v[0] for v in var])
ax.set_xlim(0, 38)
ax.set_xlabel("Test F1, 40 iterations, mean ± std over 3 seeds")
save(fig, "variants.png")
