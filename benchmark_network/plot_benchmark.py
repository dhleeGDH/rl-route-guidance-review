"""Completion-collapse figure across a bespoke grid and two standard benchmarks.

Three panels (bespoke 5x5 grid, Sioux Falls, Nguyen-Dupuis). In each panel the OD trip
completion rate is plotted as the boundary goes from closed to open, once for a
travel-time-minimizing reward and once for a destination-aligned reward. The travel-time
line collapses toward zero when the boundary opens; the aligned line holds high. Values
are annotated on every point. The identical collapse across three unrelated topologies
shows the Section V collapse is a property of the reward-and-boundary configuration, not
of the bespoke grid.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
plt.rcParams.update({"font.size": 8.0, "axes.titlesize": 8.0, "axes.labelsize": 8.0,
                     "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5})

TT = "#c0392b"    # travel-time reward (collapses)
AL = "#1f6fb4"    # destination-aligned reward (holds)


def _boot_err(x, b=10000, seed=20260719):
    """Half-widths of the 95% percentile bootstrap interval, as (lower, upper) for errorbar."""
    x = np.asarray(x, float)
    rng = np.random.RandomState(seed)
    means = np.mean(rng.choice(x, size=(b, len(x)), replace=True), axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (max(0.0, x.mean() - lo), max(0.0, hi - x.mean()))


def load():
    """Return dict net -> {reward: (closed_mean, closed_err, open_mean, open_err)}.

    The dispersion follows Table IV cell for cell: a 95% percentile bootstrap interval on the
    10-seed lattice and one standard deviation on the 5-seed benchmark networks. Plotting a
    standard deviation for the lattice put "31 +/- 18%" beside Table IV's "30.8 [19.9-42.6]"
    for the same cell, which reads as two different results.
    """
    d = {}
    # bespoke grid: read the 10 seed-level evaluation completions, the quantity Table IV
    # reports and the quantity panels (b) and (c) below plot. Reading the last point of the
    # training curve instead drew panel (a) at 99.4 and 17.9 against the 98.8 and 15.1 of
    # Table IV, which put two measures of the same four cells a page apart.
    _g = json.loads((HERE / ".." / "boundary_open_demo"
                     / "four_cells_boundary_dest.json").read_text(encoding="utf-8"))
    d["(a) Bespoke"] = {}
    for rw in ("time_min", "aligned"):
        c = np.asarray(_g["closed_%s" % rw]["per_seed"])
        o = np.asarray(_g["open_%s" % rw]["per_seed"])
        d["(a) Bespoke"][rw] = (c.mean(), _boot_err(c), o.mean(), _boot_err(o))
    try:
        z = np.load(str(HERE / "benchmark_results.npz"))
        for key, label in [("sioux_falls", "(b) Sioux Falls"),
                           ("nguyen_dupuis", "(c) Nguyen-Dupuis")]:
            d[label] = {}
            for rw in ("time_min", "aligned"):
                c = 100 * z[f"{key}_closed_{rw}"]; o = 100 * z[f"{key}_open_{rw}"]
                d[label][rw] = (c.mean(), c.std(), o.mean(), o.std())
    except FileNotFoundError:
        pass
    return d


def annotate(ax, x, y, sd, color, dx, dy, ha):
    txt = f"{y:.1f}%"
    ax.annotate(txt, (x, y), textcoords="offset points", xytext=(dx, dy),
                ha=ha, va="center", fontsize=7.5, fontweight="bold", color=color)


def annotate_column(ax, x, points):
    """Label both series at one x, set outside the pair of lines rather than across them.

    points: [(value, sd, colour), ...]. The labels go left of the closed column and right of
    the open one, so no label crosses a line. Where the two values are close, the higher one
    is raised and the lower dropped so the pair does not print on top of itself.
    """
    dx, ha = (-7, "right") if x == 0 else (7, "left")
    hi, lo = sorted(points, key=lambda p: -p[0])
    gap = hi[0] - lo[0]
    offs = (0.0, 0.0) if gap > 14 else (6.0, -6.0)
    for (v, sd, color), dy in zip((hi, lo), offs):
        annotate(ax, x, v, sd, color, dx, dy, ha)


def main():
    data = load()
    nets = list(data.keys())
    fig, axes = plt.subplots(1, len(nets), figsize=(6.9, 2.3), sharey=True)
    if len(nets) == 1:
        axes = [axes]
    x = [0, 1]
    for ax, net in zip(axes, nets):
        closed, opened = [], []
        for rw, color, lbl in [("time_min", TT, "Travel-time reward"),
                               ("aligned", AL, "Destination-aligned reward")]:
            cm, cs, om, os = data[net][rw]
            # cs/os are a standard deviation on the benchmark panels and a (lower, upper)
            # bootstrap half-width on the lattice panel, so the bars are built per shape.
            if isinstance(cs, tuple):
                yerr = np.array([[cs[0], os[0]], [cs[1], os[1]]])
            else:
                yerr = np.array([cs, os], float)
            ax.errorbar(x, [cm, om], yerr=yerr, color=color, marker="o", ms=4,
                        lw=1.3, capsize=2.5, label=lbl, zorder=3)
            closed.append((cm, cs, color)); opened.append((om, os, color))
        annotate_column(ax, 0, closed)
        annotate_column(ax, 1, opened)
        ax.set_xticks(x); ax.set_xticklabels(["Boundary\nclosed", "Boundary\nopen"])
        ax.set_xlim(-0.62, 1.62); ax.set_ylim(-6, 112)
        ax.grid(axis="y", ls=":", alpha=0.5)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("OD trip completion (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.19, 1, 1))
    # the panel identifier belongs under its panel, not over it, and the legend sits below the
    # identifiers: sharing one line puts the centre legend straight through panel (b)'s label
    for ax, net in zip(axes, nets):
        cx = (ax.get_position().x0 + ax.get_position().x1) / 2.0
        fig.text(cx, 0.115, net, ha="center", va="bottom", fontsize=8.0)
    out = Path(__file__).resolve().parent / "figures"
    fig.savefig(str(out), dpi=600)
    print("wrote", out.name)


if __name__ == "__main__":
    main()
