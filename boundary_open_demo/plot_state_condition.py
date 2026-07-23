"""Concept figure for the state condition (main-text Fig. 3).

Everything shown is a real instance drawn from the same time-varying 5x5 grid and the same two
planners as the Section V demonstration: the cost model, the snapshot-extrapolated planner and
the time-consistent planner are imported from state_condition_demo, and one OD pair is
selected for illustration. No quantity is invented for the figure.

(a) The two committed routes. Both are fixed before departure; they differ only because the
    states that produced them differ.
(b) Cost accumulated along each commitment. The snapshot planner's own prediction assumes the
    departure-time costs persist, so its curve departs from what the vehicle actually pays,
    and the gap widens link by link until it surfaces as the ETA error at D.

Drawn at the final placement width (6.9 in) in 8 pt Times, so Word inserts it at 1.0x.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from state_condition_demo import (N, cost, snapshot_route, forecast_route, realized,
                                  planned_snapshot)

FS = 8.0
plt.rcParams.update({
    "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS,
    "xtick.labelsize": FS - 0.5, "ytick.labelsize": FS - 0.5, "legend.fontsize": FS - 1.0,
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})

OUT = Path(__file__).resolve().parents[2] / "05_writing" / "figures" / "fig_state_condition.png"
INK = "#222222"
# The cells Fig. 3 uses, in (row, col) form: origin an interior cell, destination the node
# the one outgoing boundary link leaves from. Keeping them equal makes the two figures read
# as the same network.
O_CELL = (3, 1)
D_CELL = (2, N - 1)
STUB = 0.62           # length of the destination link protruding from the boundary node
SNAP = "#c0392b"      # the snapshot-extrapolated commitment
FCST = "#1f6fb4"      # the forecast-conditioned commitment
PLAN = "#7f7f7f"      # what the snapshot planner predicted for its own route


def pick_instance():
    """Search the same generator for a clear instance: the two commitments differ, the
    snapshot route is clearly slower, and its ETA error is visible.

    The OD pair is held at the cells Fig. 3 draws, so both figures show one network: the
    origin an interior cell, the destination the outgoing link at a boundary node. Only the
    cost field varies over the search, and every quantity plotted is that instance's own."""
    best = None
    for seed in range(6000):
        rng = np.random.RandomState(seed)
        o, d = O_CELL, D_CELL
        field = rng.rand(N, N)
        phase = rng.uniform(0, 2 * np.pi, size=(N, N))
        rs = snapshot_route(o, d, field, phase)
        rf = forecast_route(o, d, field, phase)
        if rs == rf:
            continue
        sr, fr = realized(rs, field, phase), realized(rf, field, phase)
        sp = planned_snapshot(rs, field, phase)
        if sr - fr < 1.0 or sr - sp < 1.5:
            continue
        best = (seed, o, d, field, phase, rs, rf, sr, fr, sp)
        break
    if best is None:
        raise SystemExit("no illustrative instance found")
    return best


seed, o, d, field, phase, rs, rf, sr, fr, sp = pick_instance()
print(f"instance seed={seed} O={o} D={d}  snapshot realized {sr:.2f} (planned {sp:.2f}) "
      f"| forecast realized {fr:.2f}")

# The two panels sit side by side across the page. Stacked in one column they ran 5.15 in tall
# for two small plots, most of it the empty upper rows of the lattice, and cost roughly half a
# page. Each panel is still raised to clear the subcaption line beneath it: a subfigure label
# belongs below the graphic it names, not as a title above it.
fig = plt.figure(figsize=(6.9, 2.75))
axA = fig.add_axes([0.045, 0.175, 0.40, 0.78])
axB = fig.add_axes([0.585, 0.225, 0.395, 0.73])


def xy(cell):
    return cell[1], N - 1 - cell[0]


# ---------------- (a) the two committed routes ----------------
for i in range(N):
    for j in range(N):
        if j < N - 1:
            axA.plot([j, j + 1], [N - 1 - i, N - 1 - i], "-", color="0.8", lw=0.8, zorder=1)
        if i < N - 1:
            axA.plot([j, j], [N - 1 - i, N - 2 - i], "-", color="0.8", lw=0.8, zorder=1)
for i in range(N):
    for j in range(N):
        axA.plot(j, N - 1 - i, "o", ms=2.6, mfc="white", mec=INK, mew=0.7, zorder=3)

for route, col, off in ((rs, SNAP, -0.045), (rf, FCST, 0.045)):
    pts = np.array([xy(c) for c in route], dtype=float) + off
    axA.plot(pts[:, 0], pts[:, 1], "-", color=col, lw=2.5, zorder=4,
             solid_capstyle="round", alpha=0.95)

axA.plot(*xy(o), "s", ms=5.2, mfc=INK, mec=INK, zorder=6)
axA.text(xy(o)[0], xy(o)[1] - 0.22, "O", ha="center", va="top", fontsize=FS,
         fontweight="bold")
# The destination is the outgoing link at the boundary node the routes reach, drawn as in
# Fig. 3: a bold stub leaving the lattice with a star at its far end.
dx, dy = xy(d)
axA.plot([dx, dx + STUB], [dy, dy], "-", color=INK, lw=1.8, zorder=5,
         solid_capstyle="butt")
axA.plot(dx + STUB, dy, "*", ms=9, mfc=INK, mec=INK, zorder=6)
axA.text(dx + STUB + 0.18, dy, "D", ha="left", va="center", fontsize=FS,
         fontweight="bold")
axA.set_xlim(-0.5, N + 0.35); axA.set_ylim(-0.3, N - 0.75)
axA.set_aspect("equal"); axA.axis("off")
axA.plot([], [], "-", color=SNAP, lw=2.2, label="from a snapshot-extrapolated state")
axA.plot([], [], "-", color=FCST, lw=2.2, label="from a forecast-conditioned state")
axA.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False,
           fontsize=FS - 1.5, handlelength=1.4, borderaxespad=0.0)

# ---------------- (b) cost accumulated along each commitment ----------------


def cum_realized(route):
    return np.concatenate([[0.0], np.cumsum(
        [cost(route[k + 1], k, field, phase) for k in range(len(route) - 1)])])


def cum_planned(route):
    return np.concatenate([[0.0], np.cumsum(
        [cost(route[k + 1], 0, field, phase) for k in range(len(route) - 1)])])


srp, srr = cum_planned(rs), cum_realized(rs)
frr = cum_realized(rf)
ks, kf = np.arange(len(srr)), np.arange(len(frr))

axB.plot(ks, srp, "--", color=PLAN, lw=1.2, marker="s", ms=2.6,
         label="snapshot route, as its own state predicts")
axB.plot(ks, srr, "-", color=SNAP, lw=1.4, marker="o", ms=3.0,
         label="snapshot route, as actually paid")
axB.plot(kf, frr, "-", color=FCST, lw=1.4, marker="^", ms=3.0,
         label="forecast route, as actually paid")

axB.fill_between(ks, srp, srr, color=SNAP, alpha=0.10)
# The gap is measured just clear of the last markers, and its label sits above the curves
# rather than across them.
xgap = ks[-1] + 0.30
axB.annotate("", xy=(xgap, srr[-1]), xytext=(xgap, srp[-1]),
             arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8))
axB.text(xgap + 0.14, srr[-1] + 0.35, "ETA error at D", ha="right", va="bottom",
         fontsize=FS - 1.0, color=INK, fontweight="bold")

axB.set_xlabel("links traversed", fontsize=FS)
axB.set_ylabel("cumulative travel time", fontsize=FS)
axB.set_xticks(np.arange(0, max(len(srr), len(frr))))
axB.set_xlim(-0.25, ks[-1] + 0.55)
axB.set_ylim(0, max(srr[-1], frr[-1], srp[-1]) * 1.45)
axB.grid(True, alpha=0.3, lw=0.5)
axB.legend(loc="upper left", frameon=False, fontsize=FS - 1.5, handlelength=1.6,
           borderaxespad=0.2)

# Side by side, both subcaptions sit on one line at the foot of the figure. The vertical
# offsets below belonged to the stacked layout and put (a) inside the lattice.
for ax, label, y in ((axA, "(a) Routes committed at departure", 0.012),
                     (axB, "(b) Cost accumulated along the committed route", 0.012)):
    x = (ax.get_position().x0 + ax.get_position().x1) / 2.0
    fig.text(x, y, label, ha="center", va="bottom", fontsize=FS)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=600, facecolor="white")
print("wrote", OUT)
