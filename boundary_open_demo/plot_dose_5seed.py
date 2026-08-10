"""Regenerate Fig. 4 (reward alignment dose-response) from the five-seed sweep
(dose_response_5seed.npy), replacing the earlier three-seed figure. Styling matches the
other Section V figures (global 15pt) used in the manuscript.
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 15, "axes.titlesize": 15, "axes.labelsize": 15,
                     "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12.5})

# Paths resolve against this file. The originals carried the Windows figure directory and
# loaded their inputs from the working directory, so the script ran only from inside its
# own folder on the machine it was written on.
HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parents[1] / "manuscript" / "figures"
OUT = str(FIGDIR / "fig_v_dose_response.png")
data = np.load(HERE / "dose_response_5seed.npy")   # rows: [lambda, mean, std]
lam, mean, std = data[:, 0], data[:, 1] * 100.0, data[:, 2] * 100.0

fig, ax = plt.subplots(figsize=(7.0, 4.3))
ax.errorbar(lam, mean, yerr=std, marker="o", markersize=7, linewidth=2.0,
            capsize=5, color="#1f4e79", ecolor="#1f4e79")
ax.set_xlabel("Reward alignment strength")
ax.set_ylabel("OD trip completion rate (%)")
ax.set_ylim(-4, 108)
ax.set_xlim(-0.05, 1.05)
ax.grid(True, alpha=0.3)
for x, y in zip(lam, mean):
    ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(6, 8),
                fontsize=12)
fig.tight_layout()
fig.savefig(OUT, dpi=300)
print("wrote", OUT)
