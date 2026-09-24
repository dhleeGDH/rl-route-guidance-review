# -*- coding: utf-8 -*-
"""Fig. 3: the studies at each recorded value, by algorithm family.

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
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
                     "font.size": 11, "axes.linewidth": 0.8})

HERE = os.path.dirname(os.path.abspath(__file__))
# The figures directory of the working tree when this file sits beside the manuscript, found by
# walking up, and the directory of the script itself in the deposited package, where no manuscript
# tree exists beside it.
def _figures():
    d = HERE
    for _ in range(8):
        cand = os.path.join(d, "manuscript", "figures")
        if os.path.isdir(cand):
            return cand
        d = os.path.dirname(d)
    return HERE


OUT = os.path.join(_figures(), "fig_family_grid.png")

# 2026-09-13: the columns follow the field order of Table I in five groups, the state,
# the action, the reward, the evaluation environment and the reporting practice, which is the
# order of the second paragraph of Section II-B. The boundary-condition column is new and sits
# between the reward and the reporting groups. Every count is computed from the study record
# below rather than written in by hand.
# 2026-09-13: the groups are separated by a gap between their columns rather than by a
# rule, which crossed the headings and the cells; each heading sits over the middle of its own
# group and is wrapped where the group is narrower than the name.
GROUPS = [("State", ["Forecast-conditioned", "Graph encoder", "Destination in the state"]),
          ("Action", ["Candidate-path action"]),
          ("Reward alignment", ["Individual", "Mixed", "System-level", "Unclear"]),
          ("Evaluation environment", ["Boundary condition"]),
          ("Reporting practice", ["Generalization", "Completion", "Code"])]

FAMILY = {"MARL": "Multi-agent", "DQN-family": "Value-based deep RL", "Tabular-Q-Sarsa": "Tabular",
          "Policy-gradient-AC": "Policy-gradient", "Model-based-Hybrid": "Model-based hybrid",
          "Unspecified": "Unspecified", "Distributional-RL": "Distributional"}
ORDER = ["Multi-agent", "Value-based deep RL", "Tabular", "Policy-gradient", "Model-based hybrid",
         "Unspecified", "Distributional"]
# The record sits beside this file in the package.
RECORD = os.path.join(HERE, "study_record_reviewed_studies.csv")
VALUE = [
    ("Forecast-conditioned", lambda r: r["predictive_representation"] in ("local-prediction", "rollout-capable")),
    ("Graph encoder", lambda r: r["state_representation"] == "graph-encoder"),
    ("Destination in the state", lambda r: r["OD_conditioned"] == "yes"),
    ("Candidate-path action", lambda r: r["action_granularity"] == "candidate-path"),
    ("Individual", lambda r: r["reward_alignment"] == "individual"),
    ("Mixed", lambda r: r["reward_alignment"] == "mixed"),
    ("System-level", lambda r: r["reward_alignment"] == "system"),
    ("Unclear", lambda r: r["reward_alignment"] == "unclear"),
    ("Boundary condition", lambda r: r["boundary_condition"] != "not-addressed"),
    ("Generalization", lambda r: r["generalization_mechanism"] not in ("none", "unclear")),
    ("Completion", lambda r: r["trip_completion_reported"] == "reported"),
    ("Code", lambda r: r["code_release"] == "released"),
]


def rows_from_record():
    import csv
    with io.open(RECORD, encoding="utf-8-sig") as f:
        rec = [r for r in csv.DictReader(f) if r["in_reviewed_studies"] == "yes"]
    out = []
    for fam in ORDER:
        sub = [r for r in rec if FAMILY[r["algorithm_family_coarse"]] == fam]
        out.append((fam, len(sub), [sum(1 for r in sub if pred(r)) for _, pred in VALUE]))
    return out


ROWS = rows_from_record()

# A miscount here is invisible on the page: every cell still prints and every row still shades.
# The totals the manuscript states are asserted instead.
assert sum(t for _, t, _ in ROWS) == 93
_col = lambda j: sum(r[2][j] for r in ROWS)
assert [_col(j) for j in range(4)] == [17, 16, 45, 15], [_col(j) for j in range(4)]
assert [_col(j) for j in range(4, 8)] == [60, 21, 10, 2], [_col(j) for j in range(4, 8)]
assert _col(8) == 10, _col(8)
assert [_col(j) for j in range(9, 12)] == [14, 16, 10], [_col(j) for j in range(9, 12)]

def _wrap(name, ncol):
    """A heading folded so that it stays inside the columns of its own group.

    One column carries about eleven characters at this type size, and a heading wider than its
    group ran over the rule beside it. The name is folded at spaces to that width.
    """
    width = max(11 * ncol, len(max(name.split(), key=len)))
    out, line = [], ""
    for word in name.split():
        trial = (line + " " + word).strip()
        if len(trial) > width and line:
            out.append(line); line = word
        else:
            line = trial
    out.append(line)
    return "\n".join(out)


LABELS = [lab for _, labs in GROUPS for lab in labs]
counts = [r[2] for r in ROWS]

# One empty slot between two groups, so the eye separates them without a rule over the cells.
GAP = 0.6
XS, _x = [], 0.0
for _gi, (_g, _labs) in enumerate(GROUPS):
    if _gi:
        _x += GAP
    for _ in _labs:
        XS.append(_x); _x += 1.0
SPAN = [(XS[sum(len(l) for _, l in GROUPS[:i])],
         XS[sum(len(l) for _, l in GROUPS[:i + 1]) - 1]) for i in range(len(GROUPS))]

# 2026-09-21: the type is reduced by a fifth. The figure is placed at 0.85 of the text
# width, and at that reduction the cells and the axis names print at 7.4 and 7.0 pt,
# which is the size of the text inside Fig. 2. At the earlier size they printed at 10.3 pt, larger
# than every other figure of the manuscript.
fig, ax = plt.subplots(figsize=(7.1, 3.6))
for i, ((name, tot, _), row) in enumerate(zip(ROWS, counts)):
    for j, c in enumerate(row):
        share = c / float(tot)
        ax.add_patch(plt.Rectangle((XS[j] - 0.5, i - 0.5), 1.0, 1.0,
                                   facecolor=plt.get_cmap("Blues")(share), edgecolor="none"))
        ax.text(XS[j], i, "%d" % c, ha="center", va="center", fontsize=8.5,
                color="white" if share > 0.55 else "#1a1a1a")

ax.set_xticks(XS)
ax.set_xticklabels(LABELS, fontsize=8.0, rotation=35, ha="right",
                   rotation_mode="anchor")
ax.set_yticks(np.arange(len(ROWS)))
ax.set_yticklabels(["%s (%d)" % (r[0], r[1]) for r in ROWS], fontsize=8.5)
ax.set_yticks(np.arange(-0.5, len(ROWS), 1), minor=True)
ax.grid(which="minor", axis="y", color="white", linewidth=1.4)
ax.tick_params(which="minor", length=0)
ax.tick_params(which="major", length=0)
ax.set_xlim(XS[0] - 0.5, XS[-1] + 0.5)
for s in ax.spines.values():
    s.set_visible(False)

# A rule between the groups and the group name above each, so a reader reads the row across the
# five parts of the evaluation design in the order Section II-B states them.
for (gname, labs), (x0, x1) in zip(GROUPS, SPAN):
    ax.text((x0 + x1) / 2.0, -1.05, _wrap(gname, len(labs)), ha="center", va="center",
            fontsize=7.5, linespacing=1.15)
ax.set_ylim(len(ROWS) - 0.5, -1.75)

# 2026-09-13: the footer is dropped; the body states what the cells are.
fig.tight_layout()
fig.savefig(OUT, dpi=400, bbox_inches="tight")
print("wrote %s" % os.path.normpath(OUT))
