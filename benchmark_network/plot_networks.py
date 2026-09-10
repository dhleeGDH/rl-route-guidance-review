# -*- coding: utf-8 -*-
"""Fig. 4: three of the six road networks the controlled experiment runs on.

Panel (a) is the bespoke 5x5 grid, panel (b) Sioux Falls, panel (c) Anaheim.

2026-08-28. Two defects the author raised on the printed figure. The boundary and interior
nodes differed by colour and size alone, which a grayscale print collapses; they now differ by
MARKER, a filled square against a small circle, so the distinction survives without colour.
The panel labels carried no size, so a reader could not see that panel (c) holds sixteen times
the nodes of panel (a) while occupying a panel of similar height; each label now states its
node count.

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

# The original wrote to 05_writing, a directory of the previous tree that does not exist here,
# so the generator produced nothing and the shipped figure came from somewhere else.
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


OUT = _figures() / "fig_networks.png"
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


def draw(ax, pos, links, boundary, labels=False, inner=None):
    """inner is a subset of boundary drawn as a filled marker against an open one.

    Cold review of v809 asked for the two Anaheim borders to be separable on the page: the paper
    compares a 13-node convex hull with a 95-node band of a tenth and the figure drew their union
    as one set, so a reader could not see which nodes belong to which. The two are distinguished
    by MARKER FILL rather than by colour, since the author's 2026-08-28 correction on this figure
    was that a grayscale print collapses a colour distinction.
    """
    _dense = len(pos) > 100
    for a, b in links:
        if a in pos and b in pos:
            (x1, y1), (x2, y2) = pos[a], pos[b]
            ax.plot([x1, x2], [y1, y2], "-", color=EDGE,
                    lw=0.25 if _dense else 0.6, alpha=0.45 if _dense else 1.0, zorder=1)
    for n, (x, y) in pos.items():
        on = n in boundary
        # 2026-09-02, minor m8: the hull and the band were drawn at one marker size, so on the
        # 416-node panel a filled square and an open one of equal size read as one class. The
        # hull is now the larger of the two.
        band_only = on and inner is not None and n not in inner
        ms = ((2.4 if band_only else 3.8) if on else 1.3) if _dense else (3.2 if on else 2.6)
        if band_only:
            ax.plot(x, y, "s", ms=ms, mfc="white", mec=BND, mew=0.7, ls="none", zorder=3)
        else:
            ax.plot(x, y, "s" if on else "o", ms=ms,
                    color=BND if on else NODE,
                    alpha=1.0 if on else (0.55 if _dense else 1.0),
                    zorder=3 if on else 2)
    # Round 266: only the Sioux Falls panel carried node numbers, which read as a difference
    # between the three networks rather than as a property of one drawing. Anaheim has 416 nodes
    # and cannot carry them, so none of the three does. The `labels` argument is kept for a
    # single-panel use and defaults off.
    if labels:
        for n, (x, y) in pos.items():
            ax.annotate(str(n), (x, y), textcoords="offset points", xytext=(3.2, 2.6),
                        fontsize=FS - 2.0, color=INK, zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.margins(0.10)


assert set(SF_LAYOUT) == set(SF_COORD), "Sioux Falls layout must place every node"
assert set(nd_layout()) == {n for e in ND_LINKS for n in e}, "layout must place every ND node"

def anaheim():
    """The fourth network, at its published node coordinates.

    Anaheim carries the value iteration of Section V-C and the retraining of a released
    implementation, so it belongs beside the three the learner runs on. Its border is the set
    Section IV-A defines, which is the one every Anaheim figure in this study uses.
    """
    import json as _json
    exp = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(exp))
    from exposure_ratio_real_networks import _hull, load_coords, read_tntp
    coords = load_coords(exp / "networks" / "anaheim_nodes.geojson")
    _, _, links = read_tntp(exp / "networks" / "Anaheim_net.tntp")
    xs = [c[0] for c in coords.values()]
    ys = [c[1] for c in coords.values()]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    dx, dy = (x1 - x0) * 0.10, (y1 - y0) * 0.10
    hull = _hull([(k, v[0], v[1]) for k, v in coords.items()])
    band = hull | {k for k, (x, y) in coords.items()
                   if x <= x0 + dx or x >= x1 - dx or y <= y0 + dy or y >= y1 - dy}
    assert len(hull) == 13 and len(band) == 95, (len(hull), len(band))
    return coords, links, band, hull


# The three drawings have unlike aspect ratios, so equal-width panels left the grid small against
# a wide Anaheim and the gaps between the three unequal. The widths below are the drawn extents,
# which makes the three read at one scale of node spacing and evens the space between them.
fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.35),
                         gridspec_kw={"width_ratios": [1.0, 1.0, 2.0]})
fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.24, wspace=0.14)

p, l, b = bespoke()
draw(axes[0], p, l, b)
draw(axes[1], SF_LAYOUT, SF_LINKS, SF_BOUNDARY)
_ap, _al, _ab, _ah = anaheim()
draw(axes[2], _ap, _al, _ab, inner=_ah)

# Equal aspect leaves each panel a different height, so a label anchored to its own axes sits
# at its own baseline. Figure coordinates put all three subcaptions on one line.
for ax, label in zip(axes, ("(a) Bespoke 5\u00d75 grid, 25 nodes",
                            "(b) Sioux Falls, 24 nodes",
                            "(c) Anaheim, 416 nodes, 13-node hull in a 95-node band")):
    x = (ax.get_position().x0 + ax.get_position().x1) / 2.0
    fig.text(x, 0.125, label, ha="center", va="bottom", fontsize=FS)

# The two node colours carry the distinction the experiment turns on, so the figure states
# them itself rather than leaving them to the caption.
handles = [Line2D([], [], marker="s", ms=3.8, color=BND, ls="none",
                  label="boundary node"),
           Line2D([], [], marker="s", ms=2.4, mfc="white", mec=BND, mew=0.7, ls="none",
                  label="Anaheim band of a tenth, beyond the hull"),
           Line2D([], [], marker="o", ms=2.6, color=NODE, ls="none", label="interior node")]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3,
           frameon=False, fontsize=FS, handlelength=1.0, handletextpad=0.4,
           columnspacing=2.2, borderaxespad=0.0)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=600, facecolor="white")
print("nodes: bespoke %d, sioux falls %d" % (len(p), len(SF_COORD)))
print("wrote", OUT)
