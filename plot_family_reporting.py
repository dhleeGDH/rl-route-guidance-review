# -*- coding: utf-8 -*-
"""Fig. 4: the six reporting fields by algorithm family, replacing Table IV.

WHY THIS EXISTS. Table IV printed a 7x7 grid of counts and Section IV-D argued one claim from it,
that no family label predicts any of the six fields. A grid of counts states that claim only to a
reader who computes seven ratios. The same counts are drawn here against the family total, so the
shortfall is read directly, and every count is printed in its cell, so no value is lost.

2026-09-01. Redrawn as a matrix on the third cold review, which found 42 bars in one panel
unreadable in print. Three defects the bars carried: six shades of one hue, which a grayscale print
collapses exactly as the author said of Fig. 5; value labels printed on top of each other wherever
two fields of a family were equal; and an absolute axis, on which a family of 3 and a family of 29
cannot be compared, which is the claim the panel exists to make.

Each cell prints its COUNT and is shaded by that count against the family total in the row label.
The shading is a reading aid for the row, not a share of an assessable denominator: generalization,
completion and code are assessable on 89, 91 and 91 studies corpus-wide, and the three abstract-only
studies lower three family totals by at most one.

    python3 plot_family_reporting.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
                     "font.size": 11, "axes.linewidth": 0.8})

# 2026-09-10 (T-1996): the path was HERE/../../manuscript/figures, which resolves outside the
# repository from scripts/ and into archive/experiments_dup/ from the archive copy, so no run of
# either could refresh the figure the builder reads. T-1993 moved ROWS and the PNG stayed at the
# corpus of 94 for that reason. The tree owning handoff/manuscript/figures is found by walking up.
def _figures():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        cand = os.path.join(d, "handoff", "manuscript", "figures")
        if os.path.isdir(cand):
            return cand
        d = os.path.dirname(d)
    # In the deposited package no manuscript tree exists beside this file; the figure lands here.
    return os.path.dirname(os.path.abspath(__file__))


OUT = os.path.join(_figures(), "fig_family_reporting.png")

FIELDS = ["Graph\nencoder", "Candidate\npath", "Destination\nin state",
          "Generalization", "Completion", "Code"]
# Family, total, then the six counts in the order of FIELDS. Table IV of the manuscript.
ROWS = [("Multi-agent", 29, [3, 3, 17, 2, 6, 4]),
        ("Value-based deep", 28, [9, 5, 15, 6, 6, 0]),
        ("Tabular", 12, [0, 2, 4, 0, 1, 0]),
        ("Policy-gradient", 12, [2, 2, 5, 4, 2, 3]),
        ("Model-based hybrid", 6, [0, 1, 2, 0, 1, 0]),
        ("Unspecified", 3, [1, 2, 1, 1, 0, 0]),
        ("Distributional", 3, [1, 0, 1, 1, 0, 3])]

fig, ax = plt.subplots(figsize=(7.1, 3.5))
M = np.array([[c / float(tot) for c in counts] for _, tot, counts in ROWS])
ax.imshow(M, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")

for i, (name, tot, counts) in enumerate(ROWS):
    for j, c in enumerate(counts):
        share = c / float(tot)
        ax.text(j, i, "%d" % c, ha="center", va="center", fontsize=10.5,
                color="white" if share > 0.55 else "#1a1a1a")

ax.set_xticks(np.arange(len(FIELDS)))
ax.set_xticklabels(FIELDS, fontsize=9.5)
ax.set_yticks(np.arange(len(ROWS)))
ax.set_yticklabels(["%s (%d)" % (r[0], r[1]) for r in ROWS], fontsize=10)
ax.set_xticks(np.arange(-0.5, len(FIELDS), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(ROWS), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.4)
ax.tick_params(which="minor", length=0)
ax.tick_params(which="major", length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xlabel("Studies of the family recording the field, shaded against the family total "
              "in the row label", fontsize=9.5)
fig.tight_layout()
fig.savefig(OUT, dpi=400, bbox_inches="tight")
print("wrote %s" % os.path.normpath(OUT))
