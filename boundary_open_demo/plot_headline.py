"""Completion on the boundary-open network against two things the failure could depend on
(main-text Fig. 6).

Panel (a) sweeps the reward-alignment strength. Panel (b) varies the exit geometry instead,
thinning the share of peripheral nodes that offer an absorbing exit. The second panel answers
the objection that giving every peripheral node an exit is the geometry most favourable to the
failure: the collapse survives cutting the exits to a quarter.

The four-cell endpoints panel (a) runs between are reported in Table IV, so only the sweep
itself is drawn here.

Drawn at the final placement width (3.3 in, one column) in 8 pt Times, so Word inserts it at
1.0x and the in-figure type matches the 10 pt body.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W_COL = 3.3           # placement width in the manuscript, inches
FS = 8.0              # in-figure type size at 1.0x

plt.rcParams.update({
    "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS,
    "xtick.labelsize": FS - 0.5, "ytick.labelsize": FS - 0.5, "legend.fontsize": FS - 0.5,
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})

FIGDIR = r"D:\review_paper\drl-rgs-review\05_writing\figures"

import json

dose = np.load("dose_response_5seed.npy")   # rows: [lambda, mean, std]
lam, dm, ds = dose[:, 0], dose[:, 1] * 100.0, dose[:, 2] * 100.0

ed = json.load(open("exit_density_results.json", encoding="utf-8"))
DENS = [("25%", 25), ("50%", 50), ("100%", 100)]
x_d = [v for _, v in DENS]
tt_m = [ed["%s_time_min" % k]["completion_mean"] for k, _ in DENS]
tt_s = [ed["%s_time_min" % k]["completion_std"] for k, _ in DENS]
al_m = [ed["%s_aligned" % k]["completion_mean"] for k, _ in DENS]
al_s = [ed["%s_aligned" % k]["completion_std"] for k, _ in DENS]

fig, (axA, axB) = plt.subplots(2, 1, figsize=(W_COL, 3.85))

axA.errorbar(lam, dm, yerr=ds, marker="o", markersize=3, linewidth=1.2,
             color="#1f4e79", ecolor="#1f4e79", capsize=2.5, elinewidth=0.6)
axA.set_xlabel("reward-alignment strength", fontsize=FS)
axA.set_ylabel("OD trip completion (%)", fontsize=FS)
axA.set_ylim(-4, 112)
axA.set_xlim(-0.05, 1.05)
axA.grid(True, alpha=0.3, linewidth=0.5)

axB.errorbar(x_d, al_m, yerr=al_s, marker="s", markersize=3, linewidth=1.2,
             color="#1f4e79", ecolor="#1f4e79", capsize=2.5, elinewidth=0.6,
             label="destination-aligned")
axB.errorbar(x_d, tt_m, yerr=tt_s, marker="o", markersize=3, linewidth=1.2,
             color="#c0392b", ecolor="#c0392b", capsize=2.5, elinewidth=0.6,
             label="travel-time")
axB.set_xlabel("peripheral nodes with an exit (%)", fontsize=FS)
axB.set_ylabel("OD trip completion (%)", fontsize=FS)
axB.set_ylim(-4, 112)
axB.set_xticks(x_d)
axB.set_xlim(15, 110)
axB.grid(True, alpha=0.3, linewidth=0.5)
axB.legend(loc="center right", frameon=False, fontsize=FS - 1.5, handlelength=1.6)

fig.tight_layout(pad=0.4)
# subcaptions sit under the panel they name
fig.subplots_adjust(hspace=0.51, bottom=0.165, top=0.985)
# The label is set clear of the x-axis label above it, and the panels are drawn closer
# together than the label needs, so the pair reads as one figure.
for ax, label in ((axA, "(a) Varying the reward"),
                  (axB, "(b) Varying the exit geometry")):
    bb = ax.get_position()
    fig.text((bb.x0 + bb.x1) / 2.0, bb.y0 - 0.115, label,
             ha="center", va="top", fontsize=FS)
out = FIGDIR + r"\fig_headline.png"
fig.savefig(out, dpi=600, facecolor="white")
print(f"(a) endpoints: strength 0 -> {dm[0]:.1f}%   strength 1 -> {dm[-1]:.1f}%")
print("(b) travel-time  25/50/100%%: %s" % [round(v, 1) for v in tt_m])
print("(b) aligned      25/50/100%%: %s" % [round(v, 1) for v in al_m])
print("wrote", out)
