"""The two networks the controlled experiment compares (main-text Fig. 3).

(a) Boundary-closed: only the origin and destination stubs leave the lattice, so a vehicle
    has nowhere else to go and a route ends on arrival or at the step budget.
(b) Boundary-open: every peripheral node carries a stub, each an absorbing exit, so a route
    can end without arrival.

The panels are stacked vertically and drawn at their final placement width (2.35 in, set
narrower than the 3.3 in column so the figure and the prose that introduces it share one
column) with 8 pt type, so Word inserts the file at 1.0x.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FS = 8.0
plt.rcParams.update({
    "font.size": FS, "axes.titlesize": FS,
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
})

N = 5
STUB = 0.62          # length of a stub link protruding from a boundary node
OUT = Path(__file__).resolve().parents[2] / "05_writing" / "figures" / "fig_experiment_design.png"


def boundary_stubs():
    """(node, outward direction) for every peripheral node of the lattice."""
    s = []
    for k in range(N):
        s.append(((k, N - 1), (0, 1)))      # top
        s.append(((k, 0), (0, -1)))         # bottom
        s.append(((0, k), (-1, 0)))         # left
        s.append(((N - 1, k), (1, 0)))      # right
    return s


def draw_stub(ax, node, d, color="black", lw=1.4, tip=True):
    x, y = node
    dx, dy = d
    ax.plot([x, x + dx * STUB], [y, y + dy * STUB], "-", color=color, lw=lw, zorder=4)
    if tip:
        ax.plot([x + dx * STUB], [y + dy * STUB], "o", ms=3.0, mfc="white",
                mec=color, mew=1.0, zorder=5)


def draw_panel(ax, open_boundary, title):
    # lattice links
    for i in range(N):
        for j in range(N):
            if i < N - 1:
                ax.plot([i, i + 1], [j, j], "-", color="0.6", lw=0.9, zorder=1)
            if j < N - 1:
                ax.plot([i, i], [j, j + 1], "-", color="0.6", lw=0.9, zorder=1)
    # nodes
    for i in range(N):
        for j in range(N):
            ax.plot(i, j, "o", ms=3.4, mfc="white", mec="black", mew=0.8, zorder=3)

    # the destination is an outgoing boundary link; the origin is an interior cell
    O_CELL = (1, 1)
    D_NODE, D_DIR = (N - 1, 2), (1, 0)      # right-leaving stub at a boundary cell

    if open_boundary:
        # other perimeter links also leave the network, each an absorbing wrong exit
        for node, d in boundary_stubs():
            if node == D_NODE and d == D_DIR:
                continue
            draw_stub(ax, node, d, color="0.45", lw=1.0)
        ax.text(N / 2 - 0.5, -1.00, "other perimeter links also leave the network",
                ha="center", va="bottom", fontsize=7, style="italic", color="0.25")
    else:
        ax.text(N / 2 - 0.5, -1.00, "the destination link is the only way out",
                ha="center", va="bottom", fontsize=7, style="italic", color="0.25")

    # A subfigure label belongs below the graphic it names, as in Fig. 8 and Fig. 9. It is
    # drawn in the panel's own coordinates so that it tracks the drawing rather than the slot.
    ax.text(N / 2 - 0.5, -1.68, title, ha="center", va="bottom", fontsize=FS)

    # origin: an interior cell
    ax.plot(O_CELL[0], O_CELL[1], "s", ms=5.0, mfc="black", mec="black", zorder=6)
    ax.text(O_CELL[0], O_CELL[1] - 0.34, "O", ha="center", va="top",
            fontsize=8, fontweight="bold")

    # destination: the one outgoing boundary link, drawn as a bold stub with a star tip
    draw_stub(ax, D_NODE, D_DIR, color="black", lw=1.6, tip=False)
    ax.plot(D_NODE[0] + STUB, D_NODE[1], "*", ms=8.0, mfc="black", mec="black", zorder=6)
    ax.text(D_NODE[0] + STUB + 0.22, D_NODE[1], "D", ha="left", va="center",
            fontsize=8, fontweight="bold")

    ax.set_xlim(-0.95, N + 0.45)
    ax.set_ylim(-2.05, N - 0.05)
    ax.set_aspect("equal")
    ax.axis("off")


# One column of the two-column page holds the pair, so the panels stack vertically.
fig, (a, b) = plt.subplots(2, 1, figsize=(2.35, 4.75))
draw_panel(a, False, "(a) Boundary-closed network")
draw_panel(b, True, "(b) Boundary-open network")
fig.tight_layout(pad=0.15, h_pad=0.8)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=600, facecolor="white")
print("wrote", OUT)
