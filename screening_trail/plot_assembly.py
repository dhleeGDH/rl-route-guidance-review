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
  4. the corpus, one terminal count.

2026-09-10 (T-1995): the abstract-only path is gone with the studies it drew. T-1991 read two of
the three at full text and removed the third, so every reviewed study is a full text and the
terminal box states one number. Every count a reader would otherwise have to subtract is printed:
the pool states 1,155, the screening band states the 1,052 leaving it, the full-text band states
the 103 reaching it and the 10 leaving it, and the terminal box states the 93.

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

INK, EDGE = "#1b2430", "#33415c"
# 2026-09-10 (T-1988): every box is white. A tint on some boxes and not others read as a grouping
# the flow does not have.
BOX = dict(boxstyle="square,pad=0.0", linewidth=0.8, edgecolor=EDGE, facecolor="#ffffff")

ONLY_2 = S["route_2"] - S["both"]
ONLY_13 = S["route_13"] - S["both"]

# The layout is stated in inches and converted at UPI units to the inch, so a change to the type
# size or the leading moves the boxes rather than overflowing them. LINE is one line of 7 pt at
# 1.20 leading; PAD is what a box adds above and below its text; GAP is the run each connector
# needs to draw a shaft and a head rather than a head alone.
UPI = 40.0
LINE, PAD, GAP, MARGIN = 0.1167 * UPI, 0.040 * UPI, 0.130 * UPI, 0.030 * UPI
H_IN = 2.30
fig, ax = plt.subplots(figsize=(3.42, H_IN))
ax.set_xlim(0, 100); ax.set_ylim(0, H_IN * UPI); ax.axis("off")


def box(cx, hw, ytop, lines):
    h = PAD + LINE * len(lines)
    ax.add_patch(FancyBboxPatch((cx - hw, ytop - h), 2 * hw, h, zorder=2, **BOX))
    ax.text(cx, ytop - h / 2.0, "\n".join(lines), ha="center", va="center", zorder=3,
            linespacing=1.20, fontsize=7.0, color=INK)
    return ytop - h


def down(x, y0, y1, ls="solid", color=EDGE, lw=0.85):
    """A connector with a shaft. FancyArrowPatch over a 3-unit gap drew a head and nothing else."""
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=6.0,
                                 linewidth=lw, color=color, zorder=1, linestyle=ls,
                                 shrinkA=0, shrinkB=0))


TOP = H_IN * UPI - MARGIN
LX, RX, HW = 27.0, 73.0, 21.0
PX, PHW = 50.0, 44.0

# ---- the two routes, side by side and unequal ----------------------------------------------
# 2026-09-10 (T-1995): the two lines read "Screened on scope: 86 records" against "Not screened:
# 1,084 records", which put the verb first and left a reader to work out what each number counts.
# Both count the records the route sends to the pool. The count leads and the clause after it says
# what was done to those records, so the asymmetry between the routes is legible from the boxes.
lb = box(LX, HW, TOP, ["Collection and citation search",
                       "514 queries, five rounds",
                       "%d records, screened on scope" % S["route_13"]])
# T-1996: "none screened" said the records were never screened at all; they were, downstream, at
# title and abstract like every pooled record. What the route did not do is screen before pooling.
rb = box(RX, HW, TOP, ["IEEE Xplore export, 5 July 2026",
                       "%s records" % format(S["route_2"], ","),
                       "not screened before pooling"])

# ---- the pool ------------------------------------------------------------------------------
# Each connector drops straight from the bottom centre of its own box onto the top edge of the
# pool, which the pool spans. A slanted pair met the pool off centre and read as leaving the far
# end of each box.
p_top = min(lb, rb) - GAP
down(LX, lb, p_top)
down(RX, rb, p_top)
pb = box(PX, PHW, p_top, ["Pooled: %s records" % format(S["pooled"], ","),
                          "%d from the collection, %s from the export, %d in both"
                          % (ONLY_13, format(ONLY_2, ","), S["both"])])

# ---- the two readings, in sequence ---------------------------------------------------------
# Section II-B screens at title and abstract and assesses at full text. "Read at title and
# abstract" inverted the object and the place: what is read is the record, at those two fields.
a_top = pb - GAP
down(PX, pb, a_top)
ab = box(PX, PHW, a_top, ["Screened at title and abstract",
                          "Excluded: %s records" % format(S["excluded_title_abstract"], ",")])
b_top = ab - GAP
down(PX, ab, b_top)
# The two records leaving at this band for want of a full text and on scope are printed as one
# subtraction. Supplementary Table S-6 keeps them apart and the paragraph beside it gives the
# reason for the one whose full text was never obtained. The sum is taken from the record so the
# box cannot drift from it.
bb = box(PX, PHW, b_top,
         ["Assessed at full text: %d records" % S["screened_in"],
          "Excluded on scope: %d records" % (S["excluded_full_text"] + S["not_obtained"]),
          "Versions merged: %d records" % S["versions_merged"]])

# ---- the corpus ----------------------------------------------------------------------------
c_top = bb - GAP
down(PX, bb, c_top)
cb = box(PX, PHW, c_top, ["Reviewed corpus: %d studies" % S["corpus"]])

ax.set_ylim(cb - MARGIN, H_IN * UPI)
# 2026-09-10 (T-1995): H_IN sizes the canvas before the drawing is laid out, so every line removed
# left the boxes stretched over the same inches rather than the figure shorter. The height is cut
# to what the drawing actually occupies, which holds the vertical scale at UPI units to the inch.
fig.set_size_inches(3.42, (H_IN * UPI - cb + MARGIN) / UPI)
fig.tight_layout(pad=0.06)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=600, facecolor="white")
print("wrote %s" % OUT)
