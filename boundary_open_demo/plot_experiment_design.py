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
    # Liberation Serif is metric-compatible with Times New Roman and matches its upright
    # percent sign and digits. Times New Roman is absent on Linux, and the other serif
    # substitutes here draw an oldstyle slanted percent that does not match the figures
    # already made on Windows.
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
})

N = 5
STUB = 0.62          # length of a stub link protruding from a boundary node
# The Windows tree held the figures under 05_writing; this package holds them under
# manuscript. The old path did not exist here and mkdir below created it silently, so the
# script reported success while writing where nothing reads.
# 2026-09-10 (T-1996): the output was Path(__file__).resolve().parents[2]/"manuscript"/"figures",
# which reaches handoff/manuscript/figures only through the handoff/experiments symlink and only
# unresolved; .resolve() follows the link into archive/experiments_dup/, where no builder reads.
# The tree that owns handoff/manuscript/figures is found by walking up instead (T-1984 class).
def _figures():
    d = Path(__file__).resolve().parent
    for _ in range(8):
        if (d / "handoff" / "manuscript" / "figures").is_dir():
            return d / "handoff" / "manuscript" / "figures"
        d = d.parent
    # In the deposited package no manuscript tree exists beside this file; the figure lands here.
    return Path(__file__).resolve().parent


OUT = (_figures()
       / "fig_experiment_design.png")


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
    # 2026-09-02, minor m8: the two italic notes under the panels read as speech and repeated
    # Section III-A, which already states that the destination link is the one exit on the
    # closed variant and that the gray stubs remove a vehicle on the open one. A figure carries
    # the drawing; the prose carries the sentence.

    # A subfigure label belongs below the graphic it names, as in Fig. 8 and Fig. 9. It is
    # drawn in the panel's own coordinates so that it tracks the drawing rather than the slot.
    ax.text(N / 2 - 0.5, -1.68, title, ha="center", va="bottom", fontsize=FS)

    # origin: an interior cell
    ax.plot(O_CELL[0], O_CELL[1], "s", ms=5.0, mfc="black", mec="black", zorder=6)
    ax.text(O_CELL[0], O_CELL[1] - 0.34, "O", ha="center", va="top",
            fontsize=8, fontweight="bold")

    # Destination. A label D at the STUB TIP once read as a destination node, so it was replaced
    # by the words "destination link". 2026-09-01: D returns, at the boundary NODE rather than at
    # the tip, on the author's instruction that the panel carry O and D. Section III-A now states
    # that the destination link is the outgoing link of the destination node, so the node bears the
    # label and the bold stub is the link, which is what Section II-A's destination g in V means.
    ax.plot(D_NODE[0], D_NODE[1], "s", ms=5.0, mfc="white", mec="black", mew=1.3, zorder=6)
    draw_stub(ax, D_NODE, D_DIR, color="black", lw=2.0, tip=True)
    ax.text(D_NODE[0], D_NODE[1] - 0.34, "D", ha="center", va="top",
            fontsize=8, fontweight="bold")

    ax.set_xlim(-0.95, N + 0.45)
    ax.set_ylim(-2.05, N - 0.05)
    ax.set_aspect("equal")
    ax.axis("off")


# One column of the two-column page holds the pair, so the panels stack vertically.
# The two panels differ only in their peripheral links, so they read as a pair. Stacked, the
# figure ran to 76% of a column at \columnwidth and drove the surrounding text into stretched
# vertical glue. Side by side the same panels occupy a fifth of that height.
fig, (a, b) = plt.subplots(1, 2, figsize=(3.45, 1.95))
draw_panel(a, False, "(a) Boundary-closed network")
draw_panel(b, True, "(b) Boundary-open network")
fig.tight_layout(pad=0.15, w_pad=1.0)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=600, facecolor="white")
print("wrote", OUT)
