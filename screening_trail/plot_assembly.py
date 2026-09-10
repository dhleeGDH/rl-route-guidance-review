# -*- coding: utf-8 -*-
"""Fig. 1, the assembly of the reviewed corpus, drawn from assembly_stages.json.

Every count is read from the stage record; none is written here.

WHY THIS SHAPE. The figure this replaces was PRISMA 2020's flow diagram in everything but its
three stage bands: one vertical column of five boxes, exclusions in boxes down the right margin,
and box wording taken from the standard ("Records identified from", "Reports sought for
retrieval", "Reports assessed for eligibility", "Studies included in review"). T-1980 checked it
box by box and found the correspondence one to one. This review is neither registered nor
PRISMA-compliant, so the resemblance claimed a protocol the review does not run, and three cold
rounds read the labels and asked for the checklist.

What replaces it is a two-band drawing, four levels deep and two columns wide:

  1. the two routes side by side, unequal on purpose. The April-June collection and citation
     searching passed their own scope screening and joined as 86; the 5 July export entered
     unscreened at 1,084. The screening state is a line inside each box, so the asymmetry is
     structural rather than a remark in the caption.
  2. the pool, decomposed as the record decomposes it: 71 from the collection, 1,069 from the
     export, 15 in both. The old top box printed 1,170, which sums a screened remnant against an
     unscreened export and states a scale the review never had.
  3. one reading band, split left and right rather than stacked, so the two readings stay
     distinct without a level of their own. Records leaving are stated inside the band as a
     subtraction, never in a box in the right margin.
  4. the corpus, split by how each study was read.

Studies recorded from an abstract alone are drawn dashed from the point they part from the
full-text path through to the terminal box, which is the distinction the body draws and the old
figure left to its caption.

    python3 plot_assembly.py
"""
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
# 2026-09-10 (T-1984): the output path was HERE.parents[1]/"manuscript"/"figures", which holds
# only when this file is reached through the handoff/experiments symlink AND the path is not
# resolved. Path.resolve() follows the link, so the old form wrote into archive/experiments_dup/
# and the figure the builder places was never the one this script produced. The tree that owns
# handoff/manuscript/figures is found by walking up instead.
def _figures():
    for d in [HERE] + list(HERE.parents):
        cand = d / "handoff" / "manuscript" / "figures"
        if cand.is_dir():
            return cand
    # 2026-09-10 (T-1986): in the deposited package there is no manuscript tree beside this
    # file, and a reader running it wants the figure, not an exit. It lands here instead.
    return HERE


OUT = _figures() / "fig_flow.png"
S = json.load(io.open(HERE / "assembly_stages.json", encoding="utf-8"))

# Liberation Serif matches the other figures, which were made on Windows against Times New Roman.
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Liberation Serif", "DejaVu Serif"]
matplotlib.rcParams["font.size"] = 7.0

INK, EDGE, GREY = "#1b2430", "#33415c", "#5b6a7d"
SOLID = dict(boxstyle="round,pad=0.020", linewidth=0.8, edgecolor=EDGE, facecolor="#f2f5fa")
OPEN_ = dict(boxstyle="round,pad=0.020", linewidth=0.8, edgecolor=GREY, facecolor="#ffffff")
DASH = dict(boxstyle="round,pad=0.020", linewidth=0.8, edgecolor=GREY, facecolor="#ffffff",
            linestyle=(0, (2.6, 1.8)))

# The export and the collection are 1069 and 71 records of the pool, with 15 in both. Printing the
# pair beside the pooled total is what shows a reader that one route is two orders larger than the
# other and that only the smaller one was screened before pooling.
ONLY_2 = S["route_2"] - S["both"]
ONLY_13 = S["route_13"] - S["both"]

fig, ax = plt.subplots(figsize=(7.16, 2.07))
ax.set_xlim(0, 100); ax.set_ylim(0, 29); ax.axis("off")


def box(cx, hw, ytop, h, lines, kw, fs=7.0, color=INK, weight="normal"):
    ax.add_patch(FancyBboxPatch((cx - hw, ytop - h), 2 * hw, h, zorder=2, **kw))
    ax.text(cx, ytop - h / 2.0, "\n".join(lines), ha="center", va="center", zorder=3,
            linespacing=1.34, fontsize=fs, color=color, fontweight=weight)


def arrow(x0, y0, x1, y1, lw=0.85, color=EDGE, ls="solid"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                 linewidth=lw, color=color, zorder=1, linestyle=ls))


# ---- level 1: the two routes, unequal ------------------------------------------------------
# The vertical budget is 28.5 units over 2.07 in, at 14 units to the inch, which is one
# 7 pt line at 1.34 leading every 1.83 units. Every level states its own top and height so a
# later edit cannot push the terminal box below the axis, which the first draft did.
L_CX, L_HW = 25.0, 23.5
R_CX, R_HW = 74.0, 22.0
TOP, H1 = 28.5, 7.0
box(L_CX, L_HW, TOP, H1,
    ["April to June 2026 collection and citation searching",
     "514 queries over three interfaces, five citation rounds",
     "screened on scope by the routes themselves: %d records" % S["route_13"]], SOLID)
box(R_CX, R_HW, TOP, H1,
    ["IEEE Xplore export, 5 July 2026",
     "%s unique records" % format(S["route_2"], ","),
     "not screened before pooling"], OPEN_)

# ---- level 2: the pool ---------------------------------------------------------------------
P_CX, P_HW, P_TOP, H2 = 50.0, 33.0, 19.3, 4.8
arrow(L_CX, TOP - H1, P_CX - 11, P_TOP)
arrow(R_CX, TOP - H1, P_CX + 11, P_TOP)
box(P_CX, P_HW, P_TOP, H2,
    ["Pooled: %s records" % format(S["pooled"], ","),
     "%d from the collection, %s from the export, %d in both"
     % (ONLY_13, format(ONLY_2, ","), S["both"])], SOLID)

# ---- level 3: one reading band, split left and right ---------------------------------------
B_L, B_R, B_TOP, H3 = 4.0, 96.0, 13.2, 6.2
arrow(P_CX, P_TOP - H2, P_CX, B_TOP)
ax.add_patch(FancyBboxPatch((B_L, B_TOP - H3), B_R - B_L, H3, boxstyle="round,pad=0.020",
                            linewidth=0.8, edgecolor=EDGE, facecolor="#ffffff", zorder=2))
ax.plot([50.0, 50.0], [B_TOP - H3 + 0.6, B_TOP - 0.6], lw=0.6, color=GREY, zorder=3)
ax.text(27.0, B_TOP - H3 / 2.0, "read at title and abstract\n%s records, less %s"
        % (format(S["pooled"], ","), format(S["excluded_title_abstract"], ",")),
        ha="center", va="center", fontsize=7.0, color=INK, linespacing=1.34, zorder=3)
ax.text(73.0, B_TOP - H3 / 2.0,
        "read at full text\n%d records, less %d on scope and %d merged versions\n"
        "%d not obtained and read at an abstract"
        % (S["screened_in"], S["excluded_full_text"], S["versions_merged"], S["not_obtained"]),
        ha="center", va="center", fontsize=7.0, color=INK, linespacing=1.34, zorder=3)

# ---- level 4: the corpus --------------------------------------------------------------------
C_CX, C_HW, C_TOP, H4 = 50.0, 31.0, 5.6, 4.7
arrow(C_CX, B_TOP - H3, C_CX, C_TOP)
box(C_CX, C_HW, C_TOP, H4,
    ["Reviewed corpus: %d studies" % S["corpus"],
     "%d read at full text, %d read at an abstract"
     % (S["full_text_recorded"], S["abstract_only"])], SOLID, weight="bold")

# the abstract-only path parts from the full-text reading and rejoins at the corpus, drawn dashed
# so a reader sees which of the two terminal counts was never verified at full text
CH_X, CY = 92.0, C_TOP - H4 / 2.0
ax.plot([CH_X, CH_X], [B_TOP - H3, CY], lw=0.75, color=GREY, linestyle=(0, (2.6, 1.8)), zorder=1)
ax.plot([CH_X, C_CX + C_HW], [CY, CY], lw=0.75, color=GREY, linestyle=(0, (2.6, 1.8)), zorder=1)

fig.tight_layout(pad=0.10)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=600, facecolor="white")
print("wrote %s" % OUT)
