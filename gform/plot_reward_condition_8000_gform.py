# -*- coding: utf-8 -*-
"""Fig. 5 at the 8000-episode budget: the travel-time reward under both boundary conditions.

Two curves, as the present figure draws, over the longer budget. The legend names the boundary condition of each curve, and the end-of-curve value labels of
the present figure are not drawn. The panel arrangement, the axis names, the panel labels, the band, the figure width
and the font size are those of boundary_open_demo/plot_reward_condition.py.

Data: four_cells_curves_8000_gform.json, cells closed_time_min and open_time_min, ten
seeds, a checkpoint every 150 episodes.

    python3 plot_reward_condition_8000_gform.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

# 2026-09-21: one column wide, the two panels stacked. Drawn at 3.42 in and placed at
# the column width, so 7.0 pt prints at 7.2 pt, the size of the text inside Fig. 2.
FS = 7.0
plt.rcParams.update({
    "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS,
    "xtick.labelsize": FS - 0.5, "ytick.labelsize": FS - 0.5, "legend.fontsize": FS - 1.0,
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})

HERE = os.path.dirname(os.path.abspath(__file__))
# 2026-09-19: this generator draws the placed Fig. 5, so it writes into the manuscript's
# figures directory under the name the builder places. check_figure_generators reads the same path.
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "manuscript", "figures",
                                   "fig_reward_condition.png"))
RAW = json.load(open(os.path.join(HERE, "four_cells_curves_8000_gform.json"), encoding="utf-8"))
STEPS = np.asarray(RAW["closed_time_min"]["steps"])
CLOSED, OPEN = "#08519c", "#a50f15"


def _boot(a, resamples=10000, seed=0):
    """The band of plot_reward_condition.py, unchanged."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.shape[0], size=(resamples, a.shape[0]))
    draws = a[idx].mean(axis=1)
    return np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)


def band(ax, cell, field, color, label, marker="o"):
    a = np.asarray(RAW[cell][field])
    m = a.mean(0)
    lo, hi = _boot(a)
    ax.plot(STEPS, m, "-", color=color, lw=1.4, marker=marker, ms=2.6, markevery=6, label=label)
    ax.fill_between(STEPS, lo, hi, color=color, alpha=0.13, lw=0)
    return m


fig = plt.figure(figsize=(3.42, 4.25))
axA = fig.add_axes([0.175, 0.630, 0.795, 0.345])
axB = fig.add_axes([0.175, 0.135, 0.795, 0.345])

mc = band(axA, "closed_time_min", "returns", CLOSED, "boundary-closed network")
mo = band(axA, "open_time_min", "returns", OPEN, "boundary-open network", marker="s")
axA.set_xlabel("Training episodes", fontsize=FS)
axA.set_ylabel("Episode return", fontsize=FS)
axA.grid(True, alpha=0.3, lw=0.5)
axA.legend(loc="lower right", bbox_to_anchor=(1.0, 0.09), frameon=False,
           fontsize=FS - 1.0, handlelength=1.6)
# 2026-09-19: the rule of plot_reward_condition.py, the minimum times 1.12 against a
# fixed top of 34, left the top quarter of the panel empty at this budget, since no return of the
# travel-time reward is positive. Panel (b) sets its range five units outside the 0 to 100 its
# data covers, and panel (a) follows it: five units outside 0 to -100, with the ticks stopping at
# the round values the data reaches.
axA.set_ylim(-105, 5)
axA.set_yticks(list(range(-100, 1, 20)))

cc = band(axB, "closed_time_min", "curves", CLOSED, "boundary-closed network")
co = band(axB, "open_time_min", "curves", OPEN, "boundary-open network", marker="s")
axB.set_xlabel("Training episodes", fontsize=FS)
axB.set_ylabel("OD trip completion (%)", fontsize=FS)
axB.set_ylim(-6, 108)
axB.grid(True, alpha=0.3, lw=0.5)
# 2026-09-19: panel (b) carried no key, since the end-of-curve values it used to print
# named its two curves. Those labels are gone, so the key is drawn here as well. Both panels put
# it in the same corner at the same height, lifted clear of the boundary-open curve lying on zero.
axB.legend(loc="lower right", bbox_to_anchor=(1.0, 0.09), frameon=False,
           fontsize=FS - 1.0, handlelength=1.6)

for ax, label in ((axA, "(a) Episode return"), (axB, "(b) OD trip completion")):
    p = ax.get_position()
    fig.text((p.x0 + p.x1) / 2.0, p.y0 - 0.105, label, ha="center", va="bottom", fontsize=FS)

fig.savefig(OUT, dpi=600, facecolor="white")
print("return  closed %.2f to %.2f | open %.2f to %.2f" % (mc.min(), mc.max(), mo.min(), mo.max()))
_lo_c, _hi_c = _boot(np.asarray(RAW["closed_time_min"]["returns"]))
_lo_o, _hi_o = _boot(np.asarray(RAW["open_time_min"]["returns"]))
print("band closed %.2f to %.2f | band open %.2f to %.2f | panel (a) range -105 to 5, ticks -100 to 0"
      % (_lo_c.min(), _hi_c.max(), _lo_o.min(), _hi_o.max()))
print("gap between the two return curves: first %.2f, min %.2f at %d, last %.2f"
      % (abs(mc[0] - mo[0]), np.abs(mc - mo).min(), STEPS[int(np.abs(mc - mo).argmin())],
         abs(mc[-1] - mo[-1])))
print("completion closed first %.1f last %.1f | open first %.1f last %.1f"
      % (cc[0], cc[-1], co[0], co[-1]))
print("wrote", os.path.basename(OUT))
