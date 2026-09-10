# -*- coding: utf-8 -*-
"""Fig. 3: reward alignment and the six reporting fields by algorithm family, in one grid.

T-00 queue, T-12. Fig. 3 drew the reward alignment of each family as stacked bars and Fig. 4 drew
the six reporting fields of the same seven families as a shaded grid. Two figures over one row
index cost a float each and asked a reader to carry the family order between them. The two are
merged here, with reward alignment as the first column group, so the row of a family reads across
from what the family rewards to what the family discloses.

Adapted from plot_family_reporting.py of the archived experiment tree, which supplied the six
field counts. The reward alignment and the forecast-conditioned counts come from plot_taxonomy.py
of repo_v11, which drew the figure this one absorbs. No count is changed; both column groups sum
to the totals scripts/countcheck.py checks against Table A-1.

    python3 scripts/plot_family_merged.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
                     "font.size": 11, "axes.linewidth": 0.8})

HERE = os.path.dirname(os.path.abspath(__file__))
# The figures directory of the working tree when this file sits in scripts/, and the directory
# of the script itself in the deposited package, where no manuscript tree exists beside it.
_WT = os.path.join(HERE, "..", "handoff", "manuscript", "figures")
OUT = os.path.join(_WT if os.path.isdir(_WT) else HERE, "fig_family_grid.png")

# The first four columns are the reward alignment of Fig. 3 and sum to the family total. The fifth
# is the forecast-conditioned count that figure carried at its right edge. The last six are the
# fields of Fig. 4.
# Set on one line and rotated. Two-line labels collided from "Destination in state" rightward at
# eleven columns, which no reduction in type size separated.
REWARD = ["Individual", "Mixed", "System-level", "Unclear"]
FIELDS = ["Forecast-conditioned", "Graph encoder", "Candidate path",
          "Destination in state", "Generalization", "Completion", "Code"]

# Family, total, reward alignment, then forecast-conditioned and the six fields.
ROWS = [("Multi-agent", 29, [8, 14, 6, 1], [3, 3, 3, 17, 2, 6, 4]),
        ("Value-based deep", 28, [21, 4, 3, 0], [8, 9, 5, 15, 6, 6, 0]),
        ("Tabular", 12, [12, 0, 0, 0], [3, 0, 2, 4, 0, 1, 0]),
        ("Policy-gradient", 12, [11, 0, 1, 0], [2, 2, 2, 5, 4, 2, 3]),
        ("Model-based hybrid", 6, [3, 3, 0, 0], [0, 0, 1, 2, 0, 1, 0]),
        ("Unspecified", 3, [2, 0, 0, 1], [1, 1, 2, 1, 1, 0, 0]),
        ("Distributional", 3, [3, 0, 0, 0], [0, 1, 0, 1, 1, 0, 3])]

# A miscount here is invisible on the page: every cell still prints and every row still shades.
# The three totals the manuscript states are asserted instead.
assert sum(t for _, t, _, _ in ROWS) == 93
assert [sum(r[2][j] for r in ROWS) for j in range(4)] == [60, 21, 10, 2]
assert sum(r[3][0] for r in ROWS) == 17
assert [sum(r[3][j] for r in ROWS) for j in range(1, 7)] == [16, 15, 45, 14, 16, 10]

LABELS = REWARD + FIELDS
counts = [r[2] + r[3] for r in ROWS]

fig, ax = plt.subplots(figsize=(7.1, 3.6))
M = np.array([[c / float(tot) for c in row]
              for row, (_, tot, _, _) in zip(counts, ROWS)])
ax.imshow(M, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")

for i, ((name, tot, _, _), row) in enumerate(zip(ROWS, counts)):
    for j, c in enumerate(row):
        ax.text(j, i, "%d" % c, ha="center", va="center", fontsize=10.5,
                color="white" if c / float(tot) > 0.55 else "#1a1a1a")

ax.set_xticks(np.arange(len(LABELS)))
ax.set_xticklabels(LABELS, fontsize=9.5, rotation=35, ha="right",
                   rotation_mode="anchor")
ax.set_yticks(np.arange(len(ROWS)))
ax.set_yticklabels(["%s (%d)" % (r[0], r[1]) for r in ROWS], fontsize=10)
ax.set_xticks(np.arange(-0.5, len(LABELS), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(ROWS), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.4)
ax.tick_params(which="minor", length=0)
ax.tick_params(which="major", length=0)
for s in ax.spines.values():
    s.set_visible(False)

# The reward alignment columns partition the family; the rest do not. The rule says so without a
# legend, which a two-group grid otherwise needs.
ax.axvline(len(REWARD) - 0.5, color="#1a1a1a", linewidth=1.1)
ax.text((len(REWARD) - 1) / 2.0, -0.85, "Reward alignment", ha="center", va="center", fontsize=9.5)
ax.text(len(REWARD) + (len(FIELDS) - 1) / 2.0, -0.85, "Design choice and disclosure",
        ha="center", va="center", fontsize=9.5)
ax.set_ylim(len(ROWS) - 0.5, -1.2)

ax.set_xlabel("Studies of the family at each value, shaded against the family total "
              "in the row label", fontsize=9.5)
fig.tight_layout()
fig.savefig(OUT, dpi=400, bbox_inches="tight")
print("wrote %s" % os.path.normpath(OUT))
