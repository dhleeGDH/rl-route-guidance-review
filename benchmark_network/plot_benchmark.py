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
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 8.0, "axes.titlesize": 8.0, "axes.labelsize": 8.0,
                     # Liberation Serif is metric-compatible with Times New Roman and matches its upright percent
                     # sign and digits; Nimbus Roman and FreeSerif draw an oldstyle slanted percent
                     # that does not match the other eight figures, which were made on Windows.
                     "font.family": "serif",
                     "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5})

TT = "#c0392b"    # travel-time reward (collapses)
AL = "#1f6fb4"    # destination-aligned reward (holds)


sys.path.insert(0, str(HERE.parent))
from table6_rows import ci, cell_key            # noqa: E402


def _boot_err(x, key):
    """Half-widths of the 95% percentile bootstrap interval, as (lower, upper) for errorbar.

    The interval comes from the same function Table VI is derived through, keyed on the same cell
    identity, so a bar in this figure and a bracket in that table can never disagree for one cell.
    A local copy of the bootstrap drew "31 +/- 18%" beside the table's own "30.8 [19.9-42.8]".
    """
    m, lo, hi = ci(x, key)
    return (max(0.0, m - lo), max(0.0, hi - m))


def load():
    """Return dict net -> {reward: (closed_mean, closed_err, open_mean, open_err)}.

    Every panel rests on ten seeds and one dispersion convention, the 95% percentile bootstrap
    interval Table VI reports. A standard deviation on the benchmark panels against an interval on
    the lattice panel put two conventions inside one figure.
    """
    d = {}
    # bespoke grid: read the 10 seed-level evaluation completions, the quantity Table IV
    # reports and the quantity panels (b) and (c) below plot. Reading the last point of the
    # training curve instead drew panel (a) at 99.4 and 17.9 against the 98.8 and 15.1 of
    # Table IV, which put two measures of the same four cells a page apart.
    _g = json.loads((HERE / ".." / "boundary_open_demo"
                     / "four_cells_boundary_dest.json").read_text(encoding="utf-8"))
    _m8k = {}
    for _c in ("closed_time_min", "open_time_min", "closed_aligned", "open_aligned"):
        _f = HERE / ".." / "boundary_open_demo" / ("matched_%s.json" % _c)
        if not _f.exists():
            _f = HERE / ".." / "boundary_open_demo" / "matched_budget_grid.json"
        _m8k.update(json.loads(_f.read_text(encoding="utf-8"))["cells"])
    # All three panels run at 8000 episodes. Three panels at three budgets invited exactly the
    # cross-panel comparison the text disclaims, so the budget is matched and the shorter-budget
    # cells are reported in the supplement instead.
    # The episode budget is stated once in the caption. Repeating it in all three identifiers
    # ran the third one off the figure and printed it over the second.
    lab_a = "(a) Bespoke grid, 150 OD pairs"
    d[lab_a] = {}
    for rw in ("time_min", "aligned"):
        c = np.asarray(_m8k["closed_%s" % rw]["per_seed"])
        o = np.asarray(_m8k["open_%s" % rw]["per_seed"])
        d[lab_a][rw] = (c.mean(), _boot_err(c, cell_key("grid", rw, 8000, "closed")),
                        o.mean(), _boot_err(o, cell_key("grid", rw, 8000, "open")))
    # Read as one comparison, the three panels invite the grid's lower open aligned value
    # to be taken for a topology effect. The panels run at the budget each network was
    # trained on, so each identifier now carries it. Each benchmark reads its own ten-seed
    # file: one shared file was overwritten key for key by whichever network wrote last.
    # Nguyen-Dupuis carried a third panel of equal prominence on 4 distinct OD pairs against 150
    # and 114. Narrowing the axis and shading the field stated that difference weakly, and the
    # panel read as a third replication regardless. Its cells are reported in the supplement.
    for key, fname, eps, label in [
            ("sioux_falls", "benchmark_results_10seed.npz", 8000,
             "(b) Sioux Falls, 114 OD pairs")]:
        f = HERE / fname
        if not f.exists():
            continue
        z = np.load(str(f))
        d[label] = {}
        for rw in ("time_min", "aligned"):
            c = 100 * z[f"{key}_closed_{rw}"]; o = 100 * z[f"{key}_open_{rw}"]
            d[label][rw] = (c.mean(), _boot_err(c, cell_key(key, rw, eps, "closed")),
                            o.mean(), _boot_err(o, cell_key(key, rw, eps, "open")))
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
    # 6 pt each way left the pair 12 pt apart, which is barely more than the 7.5 pt type and
    # read as one block wherever the two series meet. Panel (c) is the worst case: both
    # networks complete 100.0% on the closed boundary, so the gap is exactly zero.
    offs = (0.0, 0.0) if gap > 14 else (9.5, -9.5)
    for (v, sd, color), dy in zip((hi, lo), offs):
        annotate(ax, x, v, sd, color, dx, dy, ha)


def main():
    data = load()
    nets = list(data.keys())
    fig, axes = plt.subplots(1, len(nets), figsize=(4.8, 2.3), sharey=True,
                             gridspec_kw={"width_ratios": [1.0, 1.0][:len(nets)]})
    if len(nets) == 1:
        axes = [axes]
    x = [0, 1]
    BW = 0.34          # grouped-bar width; the pair spans 0.68 of the unit spacing
    for ax, net in zip(axes, nets):
        tops = {}
        # A line drawn between two categorical levels reads as an interpolation, and no state
        # lies between a closed and an open boundary. Grouped bars carry the same four values
        # without implying a path from one condition to the other. Colour alone vanishes in a
        # greyscale print, so a hatch separates the two series as well.
        for i, (rw, color, lbl, hatch) in enumerate(
                [("time_min", TT, "Travel-time reward", ""),
                 ("aligned", AL, "Destination-aligned reward", "///")]):
            cm, cs, om, os = data[net][rw]
            # cs/os are a standard deviation on the benchmark panels and a (lower, upper)
            # bootstrap half-width on the lattice panel, so the bars are built per shape.
            if isinstance(cs, tuple):
                yerr = np.array([[cs[0], os[0]], [cs[1], os[1]]])
                upper = [cs[1], os[1]]
            else:
                yerr = np.array([cs, os], float)
                upper = [cs, os]
            xs = [p + (i - 0.5) * BW for p in x]
            ax.bar(xs, [cm, om], width=BW, color=color, hatch=hatch, edgecolor="white",
                   linewidth=0.6, label=lbl, zorder=3)
            ax.errorbar(xs, [cm, om], yerr=yerr, fmt="none", ecolor="0.25",
                        elinewidth=0.9, capsize=2.5, zorder=4)
            # the label states the value and is POSITIONED above the error bar: passing the
            # sum as the y would print value + error, which is a different number
            for gi, (xp, v, e) in enumerate(zip(xs, [cm, om], upper)):
                # keyed by the GROUP, not by the bar: keying on the bar x put every label in a
                # bucket of its own and the collision rule below could never fire
                tops.setdefault(gi, []).append((xp, v, v + e, color))
        # Two bars of a group whose tops are close would print their labels side by side and
        # collide, since a group is narrower than one label. The lower label is raised over
        # the higher one instead, which keeps each label centred on its own bar.
        for gi in sorted(tops):
            grp = tops[gi]
            for j, (xp, v_, top_, color_) in enumerate(grp):
                # the second label of a group is lifted clear only where the OTHER bar of that
                # same group is close enough to collide with it. Comparing against the group
                # ceiling instead lifted the open-boundary label, whose partner bar is at zero.
                other = [q[2] for k, q in enumerate(grp) if k != j]
                dy = 15 if (j and other and abs(other[0] - top_) < 12) else 6
                ax.annotate("%.1f%%" % v_, (xp, top_), textcoords="offset points",
                            xytext=(0, dy), ha="center", va="center",
                            fontsize=7.0, fontweight="bold", color=color_)
        ax.set_xticks(x); ax.set_xticklabels(["Boundary\nclosed", "Boundary\nopen"])
        ax.set_xlim(-0.62, 1.62); ax.set_ylim(0, 118)
        ax.grid(axis="y", ls=":", alpha=0.5)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        # Panel (c) rests on 4 distinct OD pairs against 150 and 114. A narrower axis stated that
        # difference weakly, and a shaded field states it at a glance.
        if net.startswith("(c)"):
            ax.set_facecolor("#f0f0f0")
    axes[0].set_ylabel("OD trip completion (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.19, 1, 1))
    # the panel identifier belongs under its panel, not over it, and the legend sits below the
    # identifiers: sharing one line puts the centre legend straight through panel (b)'s label
    for ax, net in zip(axes, nets):
        cx = (ax.get_position().x0 + ax.get_position().x1) / 2.0
        fig.text(cx, 0.115, net, ha="center", va="bottom", fontsize=7.5)
    # resolved against this file so the script runs wherever the tree sits
    out = HERE / ".." / ".." / "manuscript" / "figures" / "fig_benchmark_inversion.png"
    fig.savefig(str(out), dpi=600)
    print("wrote", out.name)


if __name__ == "__main__":
    main()
