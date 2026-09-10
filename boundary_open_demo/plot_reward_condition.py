"""Concept figure for the reward condition (main-text Fig. 4).

What learning does under the corpus's modal reward, on the same network, with only the
boundary changed. Both curves are the ten-seed run of Section V (four_cells_curves.json); nothing
is redrawn by hand.

(a) Episode return, the quantity a training curve normally reports. Both conditions converge,
    and the boundary-open agent converges to the higher return of the two: leaving after a
    step or two accumulates almost no travel-time penalty.
(b) OD trip completion, which almost no report states. The boundary-closed agent rises to 99.9%.
    The boundary-open agent falls to zero and stays there.

The pair is the paper's claim in one figure: a converged return certifies nothing about
arrival, and only a completion measure separates the two.

Drawn at the final placement width (6.9 in) in 8 pt Times, so Word inserts it at 1.0x.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

FS = 8.0
plt.rcParams.update({
    "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS,
    "xtick.labelsize": FS - 0.5, "ytick.labelsize": FS - 0.5, "legend.fontsize": FS - 1.0,
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})

HERE = Path(__file__).parent
# 2026-09-10 (T-1996): the output was Path(__file__).resolve().parents[2]/"manuscript"/"figures",
# which reaches handoff/manuscript/figures only through the handoff/experiments symlink and only
# unresolved; .resolve() follows the link into archive/experiments_dup/, where no builder reads.
# The tree that owns handoff/manuscript/figures is found by walking up instead (T-1984 class).
def _figures():
    d = Path(__file__).resolve().parent
    for _ in range(8):
        if (d / "handoff" / "manuscript" / "figures").is_dir():
            return d / "handoff" / "manuscript" / "figures"
        d = d.parent
    # In the deposited package no manuscript tree exists beside this file; the figure lands here.
    return Path(__file__).resolve().parent


OUT = _figures() / "fig_reward_condition.png"
INK = "#222222"
CLOSED = "#08519c"
OPEN = "#a50f15"

# Drawn from the run that produces Table VIII. results_10seed.npz supplied these panels until
# 2026-08-18 and predates the thread pinning and the four-cell rerun: its completion curve
# ended at 99.5% against the 99.9% of the table, so the body carried two figures for one cell
# four lines apart. The band is the 95% percentile bootstrap over the seeds, the study's single
# dispersion convention, rather than the standard deviation drawn before.
import json

RAW = json.loads((HERE / "four_cells_curves.json").read_text(encoding="utf-8"))
steps = np.asarray(RAW["closed_time_min"]["steps"])


def _boot(a, resamples=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.shape[0], size=(resamples, a.shape[0]))
    draws = a[idx].mean(axis=1)
    return np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)


def band(ax, cell, field, color, label, marker="o"):
    a = np.asarray(RAW[cell][field])
    m = a.mean(0)
    lo, hi = _boot(a)
    ax.plot(steps, m, "-", color=color, lw=1.4, marker=marker, ms=2.6,
            markevery=3, label=label)
    ax.fill_between(steps, lo, hi, color=color, alpha=0.13, lw=0)
    return m


# One column of the two-column page holds both panels, so they stack vertically.
# Side by side across the page. Stacked in one column the pair ran 4.9 in tall and took
# roughly half a page for two line plots that share an x axis.
fig = plt.figure(figsize=(6.9, 2.75))
axA = fig.add_axes([0.075, 0.215, 0.395, 0.735])
axB = fig.add_axes([0.590, 0.215, 0.395, 0.735])

# ---------------- (a) episode return ----------------
mc = band(axA, "closed_time_min", "returns", CLOSED, "boundary-closed")
mo = band(axA, "open_time_min", "returns", OPEN, "boundary-open", marker="s")
axA.set_xlabel("Training episodes", fontsize=FS)
axA.set_ylabel("Episode return", fontsize=FS)
axA.grid(True, alpha=0.3, lw=0.5)
axA.legend(loc="lower right", frameon=False, fontsize=FS - 1.0, handlelength=1.6)
# The note sits in the clear band above both curves rather than across them.
axA.set_ylim(min(mc.min(), mo.min()) * 1.12, 34)
# 2026-09-02, minor m8: the in-panel note repeated the Section V-C sentence stating that the
# boundary-open agent converges to the higher return of the two. Removed from the drawing.

# ---------------- (b) OD trip completion ----------------
cc = band(axB, "closed_time_min", "curves", CLOSED, "boundary-closed")
co = band(axB, "open_time_min", "curves", OPEN, "boundary-open", marker="s")
axB.set_xlabel("Training episodes", fontsize=FS)
axB.set_ylabel("OD trip completion (%)", fontsize=FS)
axB.set_ylim(-6, 108)
axB.grid(True, alpha=0.3, lw=0.5)
# 2026-08-26: panel (b) repeated panel (a)'s legend for the same two conditions, and carried an
# italic "never arrives" pointing at the flat curve the 0.0% label already names. Both are gone;
# the manuscript bars italics outside a coined term, and a duplicated key costs plot area.
axB.text(steps[-1], cc[-1] - 5, f"{cc[-1]:.1f}%", ha="right", va="top",
         fontsize=FS - 0.5, color=CLOSED, fontweight="bold")
axB.text(steps[-1], 3.0, f"{co[-1]:.1f}%", ha="right", va="bottom",
         fontsize=FS - 0.5, color=OPEN, fontweight="bold")

# A subfigure label belongs below the graphic it names, as in Fig. 8 and Fig. 9, and it is
# set clear of the x-axis label above it.
for ax, label, y in ((axA, "(a) Episode return", 0.012),
                     (axB, "(b) OD trip completion", 0.012)):
    x = (ax.get_position().x0 + ax.get_position().x1) / 2.0
    fig.text(x, y, label, ha="center", va="bottom", fontsize=FS)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=600, facecolor="white")
print(f"closed: return {mc[-1]:.1f}, completion {cc[-1]:.1f}%")
print(f"open  : return {mo[-1]:.1f}, completion {co[-1]:.1f}%")
print("wrote", OUT)
