"""Design-space taxonomy of the 94 reviewed studies (main-text Fig. 2).

Three panels. Panel (a) organizes the corpus along two recorded axes at once: algorithm family
(rows) and reward alignment (stacked segments), with the count of forecast-conditioned studies
in each family annotated at the right. Panel (b) gives the application classification of
Section II-C, an exploratory title-and-abstract field. Panel (c) draws the design commitments
across the three periods of the corpus, the trend the prose of Section II-C describes. Counts
are from the released extraction (experiments/corpus/corpus_v9_coded.csv) joined with the pool
years, N = 94; every number matches its sentence in the manuscript.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent

plt.rcParams.update({
    "font.size": 8.0, "axes.titlesize": 8.0, "axes.labelsize": 8.0,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    # Liberation Serif is metric-compatible with Times New Roman and matches its upright
    # percent sign and digits. Times New Roman is absent on Linux, and the other serif
    # substitutes here draw an oldstyle slanted percent that does not match the figures
    # already made on Windows.
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})

# family: {individual, mixed, system, unclear}, forecast-conditioned count, total
DATA = [
    ("Multi-agent",          [8, 14, 6, 1],  3, 29),
    ("Value-based deep",     [20, 4, 3, 0],  8, 27),
    ("Tabular Q / Sarsa",    [12, 0, 0, 1],  3, 13),
    ("Policy-gradient / AC", [11, 0, 1, 0],  2, 12),
    ("Model-based / hybrid", [3, 3, 0, 0],   0, 6),
    ("Distributional",       [3, 0, 0, 0],   0, 3),
    ("Unspecified",          [3, 0, 0, 1],   1, 4),
]
SEG = ["Individual", "Mixed", "System-level", "Unclear"]
COL = ["#1f6fb4", "#e08a1e", "#c0392b", "#b8b8b8"]

# Section II-C, exploratory classification from titles and abstracts
APPS = [("General road-network routing", 58), ("Connected / automated vehicles", 13),
        ("Eco-routing", 9), ("Emergency routing", 6), ("Signal-adjacent task", 4),
        ("Electric-vehicle routing", 3), ("Ride service", 1)]

# periods through 2018, 2019-2022, 2023 onward; sizes 6 / 26 / 62
PERIODS = ["-2018\n(n=6)", "2019-22\n(n=26)", "2023-\n(n=62)"]
TRENDS = [("Value-based deep", [0, 6, 21], "#1f6fb4"),
          ("Policy-gradient / AC", [0, 2, 10], "#7b52a5"),
          ("Tabular Q / Sarsa", [2, 7, 4], "#5d8a3c"),
          ("Graph encoder", [0, 2, 14], "#e08a1e"),
          ("Candidate-path action", [0, 1, 13], "#c0392b")]


def main():
    fams = [d[0] for d in DATA]
    y = np.arange(len(DATA))[::-1]  # largest at top
    fig = plt.figure(figsize=(6.9, 4.7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.12, 1.0], hspace=0.62, wspace=0.42)
    ax = fig.add_subplot(gs[0, :])
    left = np.zeros(len(DATA))
    vals = np.array([d[1] for d in DATA], float)
    for j in range(4):
        ax.barh(y, vals[:, j], left=left, color=COL[j], label=SEG[j], height=0.62)
        for i in range(len(DATA)):
            if vals[i, j] >= 1:
                ax.text(left[i] + vals[i, j] / 2, y[i], f"{int(vals[i, j])}",
                        ha="center", va="center",
                        fontsize=10 if vals[i, j] >= 2 else 7.5,
                        color="white" if j != 3 else "#333")
        left += vals[:, j]
    # the annotations sat at the end of each bar, so their distance from the axis varied with
    # bar length and a reader could not tell which row a label belonged to
    for i, (_, _, fc, tot) in enumerate(DATA):
        ax.text(31.0, y[i], f"n={tot}", va="center", ha="left", fontsize=9.5, color="#333")
        ax.text(35.0, y[i], f"forecast-conditioned: {fc}", va="center", ha="left",
                fontsize=9.5, color="#333")
    ax.set_yticks(y); ax.set_yticklabels(fams)
    ax.set_xlabel("Number of studies (reward alignment)")
    ax.set_xlim(0, 52)
    # no in-figure title: the manuscript caption (Fig. 2) carries it, avoiding duplication
    ax.legend(loc="lower right", ncol=4, frameon=False, bbox_to_anchor=(1.0, -0.34))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_title("(a) Algorithm family and reward alignment", fontsize=8.5, pad=4)

    # ---- panel (b): applications ----
    axb = fig.add_subplot(gs[1, 0])
    yb = np.arange(len(APPS))[::-1]
    axb.barh(yb, [n for _, n in APPS], color="#1f6fb4", height=0.6)
    for i, (_, n) in enumerate(APPS):
        axb.text(n + 0.8, yb[i], str(n), va="center", ha="left", fontsize=8, color="#333")
    axb.set_yticks(yb); axb.set_yticklabels([a for a, _ in APPS], fontsize=7.5)
    axb.set_xlim(0, 66)
    axb.set_xlabel("Number of studies")
    axb.spines["top"].set_visible(False); axb.spines["right"].set_visible(False)
    axb.set_title("(b) Application classification", fontsize=8.5, pad=4)

    # ---- panel (c): design commitments over the three periods ----
    axc = fig.add_subplot(gs[1, 1])
    x = np.arange(3)
    for label, vals, col in TRENDS:
        axc.plot(x, vals, marker="o", ms=3.5, lw=1.2, color=col, label=label)
    axc.set_xticks(x); axc.set_xticklabels([p.replace("\\n", "\n") for p in PERIODS], fontsize=7.5)
    axc.set_ylabel("Number of studies")
    axc.set_ylim(0, 24)
    axc.legend(frameon=False, fontsize=6.8, loc="upper left")
    axc.spines["top"].set_visible(False); axc.spines["right"].set_visible(False)
    axc.set_title("(c) Design commitments by period", fontsize=8.5, pad=4)

    # resolved against this file so the script runs wherever the tree sits; the original
    # wrote to a Windows path that does not exist on this machine
    out = HERE / ".." / ".." / "manuscript" / "figures" / "fig_taxonomy.png"
    fig.savefig(str(out), dpi=600, bbox_inches="tight")
    print("wrote", out.name)


if __name__ == "__main__":
    main()
