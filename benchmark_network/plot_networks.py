# -*- coding: utf-8 -*-
"""Fig. 8: the three road networks the controlled experiment runs on.

Panel (a) is the bespoke 5x5 lattice, panel (b) Sioux Falls, panel (c) Nguyen-Dupuis.

Both benchmark panels use the schematic layout each network is published in, which is how
readers of the transportation literature recognize them. Plotting Sioux Falls at its raw
geographic node coordinates instead collapses nodes 9/10, 16/17 and 21/22 onto each other and
draws a shape no reader identifies. The layout is a drawing choice; the arc set is not, and
every arc drawn here comes from the link table the experiment runs on.

Drawn at the final placement width (6.9 in, full width) in 8 pt Times, so Word inserts it at
1.0x.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_demo import SF_LINKS, SF_COORD, ND_LINKS, NETWORKS  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "05_writing" / "figures" / "fig_networks.png"
FS = 8.0
INK = "#222222"
EDGE = "#8a8a8a"
NODE = "#1f4e79"
BND = "#c0392b"

plt.rcParams.update({
    "font.size": FS, "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"], "text.color": INK,
})

N_SIDE = 5
BESPOKE_BOUNDARY = {(r, c) for r in range(N_SIDE) for c in range(N_SIDE)
                    if r in (0, N_SIDE - 1) or c in (0, N_SIDE - 1)}
# Both panels read their boundary set from the experiment rather than restating it, so the
# colours cannot drift from the networks the runs use.
SF_BOUNDARY = NETWORKS["sioux_falls"]["boundary"]
ND_BOUNDARY = NETWORKS["nguyen_dupuis"]["boundary"]

# Schematic node placement for Sioux Falls, the layout the network is published and reproduced
# in. It rectifies the geography onto a lattice: node 1 stays north-west, node 13 south-west,
# nodes 7 and 18 east, exactly as the raw coordinates place them, at spacings a reader can
# follow. Every arc becomes a lattice line or a short diagonal.
SF_LAYOUT = {
    1: (1, 8), 2: (5, 8),
    3: (1, 7), 4: (3, 7), 5: (4, 7), 6: (5, 7),
    7: (7, 6), 8: (5, 6), 9: (4, 6),
    12: (1, 5), 11: (3, 5), 10: (4, 5), 16: (5, 5), 18: (7, 5),
    17: (5, 4),
    14: (3, 3), 15: (4, 3), 19: (5, 3),
    22: (4, 2), 23: (3, 2),
    # Node 20 shares the rank of node 21, which is where the published layout places it.
    13: (1, 1), 24: (3, 1), 21: (4, 1), 20: (5, 1),
}


def bespoke():
    pos = {(r, c): (c, -r) for r in range(N_SIDE) for c in range(N_SIDE)}
    links = []
    for r in range(N_SIDE):
        for c in range(N_SIDE):
            for dr, dc in ((0, 1), (1, 0)):
                nb = (r + dr, c + dc)
                if nb in pos:
                    links.append(((r, c), nb))
    return pos, links, BESPOKE_BOUNDARY


def nd_layout():
    """Node placement of the published Nguyen-Dupuis figure (Nguyen and Dupuis, 1984).

    Four ranks. Nodes 1 and 12 sit on top; 4, 5, 6, 7 and 8 form the second; 9, 10, 11 and
    destination 2 the third; and 13 with destination 3 the bottom. Arc 12-8 runs diagonally
    over node 7 and arcs 4-9 and 9-13 step down to the left of the grid, as published.
    """
    return {1: (2.0, 3.0), 12: (3.0, 3.0),
            4: (1.0, 2.0), 5: (2.0, 2.0), 6: (3.0, 2.0), 7: (4.0, 2.0), 8: (5.0, 2.0),
            9: (2.0, 1.0), 10: (3.0, 1.0), 11: (4.0, 1.0), 2: (5.0, 1.0),
            13: (3.0, 0.0), 3: (4.0, 0.0)}


def draw(ax, pos, links, boundary, labels=False):
    for a, b in links:
        if a in pos and b in pos:
            (x1, y1), (x2, y2) = pos[a], pos[b]
            ax.plot([x1, x2], [y1, y2], "-", color=EDGE, lw=0.6, zorder=1)
    for n, (x, y) in pos.items():
        on = n in boundary
        ax.plot(x, y, "o", ms=3.4 if on else 2.8, color=BND if on else NODE, zorder=2)
    # The prose names individual nodes of each benchmark network, so the panels number them.
    if labels:
        for n, (x, y) in pos.items():
            ax.annotate(str(n), (x, y), textcoords="offset points", xytext=(3.2, 2.6),
                        fontsize=FS - 2.0, color=INK, zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.margins(0.10)


assert set(SF_LAYOUT) == set(SF_COORD), "Sioux Falls layout must place every node"
assert set(nd_layout()) == {n for e in ND_LINKS for n in e}, "layout must place every ND node"

fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.0))
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.20, wspace=0.06)

p, l, b = bespoke()
draw(axes[0], p, l, b)
draw(axes[1], SF_LAYOUT, SF_LINKS, SF_BOUNDARY, labels=True)
draw(axes[2], nd_layout(), ND_LINKS, ND_BOUNDARY, labels=True)

# Equal aspect leaves each panel a different height, so a label anchored to its own axes sits
# at its own baseline. Figure coordinates put all three subcaptions on one line.
for ax, label in zip(axes, ("(a) Bespoke 5x5 lattice", "(b) Sioux Falls", "(c) Nguyen-Dupuis")):
    x = (ax.get_position().x0 + ax.get_position().x1) / 2.0
    fig.text(x, 0.115, label, ha="center", va="bottom", fontsize=FS)

# The two node colours carry the distinction the experiment turns on, so the figure states
# them itself rather than leaving them to the caption.
handles = [Line2D([], [], marker="o", ms=3.0, color=BND, ls="none", label="boundary node"),
           Line2D([], [], marker="o", ms=3.0, color=NODE, ls="none", label="interior node")]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2,
           frameon=False, fontsize=FS, handlelength=1.0, handletextpad=0.4,
           columnspacing=2.2, borderaxespad=0.0)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=600, facecolor="white")
print("nodes: bespoke %d, sioux falls %d, nguyen-dupuis %d"
      % (len(p), len(SF_COORD), len(nd_layout())))
print("wrote", OUT)
