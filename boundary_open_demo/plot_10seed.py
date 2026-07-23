"""Regenerate the Section V bar figure (main text Fig. 2) and the training-curve figure
(supplement Fig. S-1) from the ten-seed run results_10seed.npz, with the 15pt styling the
manuscript figures use.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot import STYLE, ORDER, load

plt.rcParams.update({"font.size": 15, "axes.labelsize": 15, "xtick.labelsize": 13,
                     "ytick.labelsize": 13, "legend.fontsize": 12.5})

FIGDIR = r"D:\review_paper\drl-rgs-review\05_writing\figures"
data = load("results_10seed.npz")

# --- Fig. 2 (main): final completion bars ---
fig, ax = plt.subplots(figsize=(7.2, 4.4))
finals = [data[k]["comp"][:, -1] * 100.0 for k in ORDER]
means = [f.mean() for f in finals]; stds = [f.std() for f in finals]
x = np.arange(len(ORDER))
for i, key in enumerate(ORDER):
    _, color, _, _, hatch = STYLE[key]
    ax.bar(x[i], means[i], yerr=stds[i], color=color, alpha=0.85, hatch=hatch,
           edgecolor="black", linewidth=0.9, capsize=5)
    ax.text(x[i], means[i] + stds[i] + 2, f"{means[i]:.0f}%", ha="center", va="bottom", fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(["closed\ntime-min\n(typical)", "open\ntime-min\n(typical)",
                    "open\naligned\n(control)", "closed\naligned"], fontsize=12)
ax.set_ylabel("Final OD trip completion rate (%)")
ax.set_ylim(0, 112); ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(FIGDIR + r"\fig_v_final_completion.png", dpi=300)
print("wrote fig_v_final_completion.png")

# --- Fig. S-1 (supplement): training curves ---
fig, ax = plt.subplots(figsize=(7.2, 4.4))
for key in ORDER:
    label, color, ls, mk, _ = STYLE[key]
    steps = data[key]["steps"]; comp = data[key]["comp"] * 100.0
    mean, std = comp.mean(0), comp.std(0)
    ax.plot(steps, mean, ls, color=color, marker=mk, markersize=6,
            markevery=max(1, len(steps) // 12), label=label, linewidth=2.0)
    ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.12)
ax.set_xlabel("Training episodes"); ax.set_ylabel("OD trip completion rate (%)")
ax.set_ylim(-3, 108); ax.grid(True, alpha=0.3)
ax.legend(loc="center right", fontsize=11, framealpha=0.9)
fig.tight_layout(); fig.savefig(FIGDIR + r"\fig_v_completion_curves.png", dpi=300)
print("wrote fig_v_completion_curves.png")
