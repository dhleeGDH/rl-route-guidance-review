"""Design-space taxonomy of the 94 charted systems (main-text Fig. 2).

Organizes the corpus along two charted axes at once: algorithm family (rows) and reward
alignment (stacked segments), with the count of forecast-conditioned systems in each family
annotated at the right. The result maps the design space of the field before the two
conditions are read against it. Counts are from the released charting
(data/screened/corpus_v9_coded.csv), N = 94.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.size": 8.0, "axes.titlesize": 8.0, "axes.labelsize": 8.0,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
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


def main():
    fams = [d[0] for d in DATA]
    y = np.arange(len(DATA))[::-1]  # largest at top
    fig, ax = plt.subplots(figsize=(6.9, 2.9))
    left = np.zeros(len(DATA))
    vals = np.array([d[1] for d in DATA], float)
    for j in range(4):
        ax.barh(y, vals[:, j], left=left, color=COL[j], label=SEG[j], height=0.62)
        for i in range(len(DATA)):
            if vals[i, j] >= 2:
                ax.text(left[i] + vals[i, j] / 2, y[i], f"{int(vals[i, j])}",
                        ha="center", va="center", fontsize=10,
                        color="white" if j != 3 else "#333")
        left += vals[:, j]
    for i, (_, _, fc, tot) in enumerate(DATA):
        ax.text(tot + 0.4, y[i], f"n={tot}   forecast-conditioned: {fc}",
                va="center", fontsize=10.5, color="#333")
    ax.set_yticks(y); ax.set_yticklabels(fams)
    ax.set_xlabel("Number of systems (reward alignment)")
    ax.set_xlim(0, 44)
    # no in-figure title: the manuscript caption (Fig. 2) carries it, avoiding duplication
    ax.legend(loc="lower right", ncol=4, frameon=False, bbox_to_anchor=(1.0, -0.28))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "figures"
    fig.savefig(str(out), dpi=600, bbox_inches="tight")
    print("wrote", out.name)


if __name__ == "__main__":
    main()
