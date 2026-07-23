"""What a boundary-closed network teaches (main-text Fig. 7).

Reads singleton_env_results.npz, written by singleton_env.py. Both conditions are the
Section V setup, boundary-closed with the travel-time reward; the only difference is the set
of destinations seen in training. Evaluation is the same two held-out OD sets for both.

The point of the figure is the pair of bars on the left: a boundary-closed evaluation run on
the training destination scores both conditions at 100%, and separates nothing. Only the
unseen destinations tell them apart.

Drawn at the final placement width (3.3 in, one column) in 8 pt Times, so Word inserts it at
1.0x.
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
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})

HERE = Path(__file__).parent
OUT = Path(__file__).resolve().parents[2] / "05_writing" / "figures" / "fig_singleton.png"
INK = "#222222"
SING = "#c0392b"     # one destination in training
MULTI = "#1f6fb4"    # a destination drawn per episode

d = np.load(HERE / "singleton_env_results.npz")
groups = ["trained\ndestination", "unseen\ndestinations"]
sing = [d["singleton__seen"] * 100, d["singleton__unseen"] * 100]
mult = [d["multi__seen"] * 100, d["multi__unseen"] * 100]

fig, ax = plt.subplots(figsize=(3.3, 2.6))
x = np.arange(2)
w = 0.34

for off, vals, col, lab, hatch in (
        (-w / 2, sing, SING, "trained on one fixed destination", "xx"),
        (+w / 2, mult, MULTI, "trained on a new destination each episode", "//")):
    m = [v.mean() for v in vals]
    s = [v.std() for v in vals]
    ax.bar(x + off, m, width=w, yerr=s, color=col, alpha=0.85, hatch=hatch,
           edgecolor="black", linewidth=0.6, capsize=2.5,
           error_kw={"elinewidth": 0.6}, label=lab)
    for xi, mi, si in zip(x + off, m, s):
        ax.text(xi, mi + si + 2.5, f"{mi:.0f}", ha="center", va="bottom",
                fontsize=FS - 0.5, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=FS - 0.5)
ax.set_ylabel("OD trip completion (%)", fontsize=FS)
ax.set_ylim(0, 112)
ax.set_yticks([0, 25, 50, 75, 100])
ax.grid(True, axis="y", alpha=0.3, lw=0.5)
# The legend sits above the axes. Anchoring its upper edge inside the axes once dropped it
# onto the "100" printed over the second bar. Each entry names its own training condition, so
# the legend carries no title of its own.
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False,
          fontsize=FS - 1.0, handlelength=1.5, ncol=1, borderaxespad=0.0)

fig.tight_layout(pad=0.4)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=600, facecolor="white")
print(f"singleton: seen {sing[0].mean():.1f}  unseen {sing[1].mean():.1f}")
print(f"multi    : seen {mult[0].mean():.1f}  unseen {mult[1].mean():.1f}")
print("wrote", OUT)
