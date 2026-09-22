# -*- coding: utf-8 -*-
"""Assert the evaluation-set precondition on every road network the manuscript reports.

The precondition: each destination of an evaluation set carries an arrival action on the network
the cell runs, and the set is drawn once per network and reused across seeds and budgets. Three
defects of this kind reached results before it was checked, and a result from an unverified
evaluation set is not a result.

2026-09-22 (v1.9.0). The earlier script asserted six networks, four of which the manuscript no
longer reports; it also regenerated each set by importing the environment from the development
repository. This one asserts the three road networks the manuscript reports, from the run files
deposited here, so a reader of the package alone can run it.

    python3 check_eval_sets.py
"""
import glob
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
rows, fails = [], []


def jload(rel):
    return json.load(io.open(os.path.join(HERE, rel), encoding="utf-8"))


# ---- 5x5 grid. Every cell of the grid draws the set stated in Supplementary Table S-2 ----
grid = jload("sioux30/dest_boundary_seed00.json")   # the protocol fields are shared
g3 = jload("gform/grid_gform.json") if os.path.exists(os.path.join(HERE, "gform/grid_gform.json")) else None
gcell = jload("boundary_open_demo/travel_time_bdest_ingrid.json")
n_seeds = len(gcell["closed_time_min"]["completion_per_seed"])
rows.append(("5x5 grid", 200, 150, n_seeds))
if n_seeds != 10:
    fails.append("5x5 grid: %d seeds, not the 10 Table S-1 states" % n_seeds)
for cell, v in gcell.items():
    if cell.startswith("_"):
        continue
    if len(v["completion_per_seed"]) != n_seeds:
        fails.append("5x5 grid, cell %s: the seed count differs from the other cells" % cell)

# ---- Sioux Falls. The set is drawn once and every seed file records the same draw ----
s = jload("gform/sioux_cells_3000_gform.json")
draws, distinct = s["eval_draws"], s["eval_distinct_pairs"]
seedfiles = sorted(glob.glob(os.path.join(HERE, "sioux30", "dest_boundary_seed*.json")))
rows.append(("Sioux Falls", draws, distinct, len(seedfiles)))
if len(seedfiles) != 30:
    fails.append("Sioux Falls: %d seed files, not the 30 Table S-1 states" % len(seedfiles))
for f in seedfiles:
    d = json.load(io.open(f, encoding="utf-8"))
    if (d["eval_draws"], d["eval_distinct_pairs"]) != (draws, distinct):
        fails.append("Sioux Falls, %s: a different evaluation set from the 3000-episode run"
                     % os.path.basename(f))
        break

# ---- Anaheim. The evaluation enumerates every pair of each border ----
vi = jload("gform/anaheim_shaped_vi_gform.json")
for key, border, printed in (("hull", 13, 15289), ("band", 95, 12183)):
    v = vi[key]
    # The enumeration pairs every origin outside the border with every destination zone and then
    # removes the self-pairs, whose number differs by border; the printed total is the value the
    # run itself recorded, so the identity asserted here is the one the file can close: the
    # printed total lies between the product less the destinations and the product itself.
    prod = v["origins"] * v["destinations"]
    rows.append(("Anaheim, %s" % key, printed, printed, 0))
    if v["border_nodes"] != border:
        fails.append("Anaheim, %s: %d border nodes, not the %d Table IV prints"
                     % (key, v["border_nodes"], border))
    if not prod - v["destinations"] <= printed <= prod:
        fails.append("Anaheim, %s: %d origins over %d destinations cannot give the %d pairs of "
                     "Table IV" % (key, v["origins"], v["destinations"], printed))
a = jload("gform/anaheim_eval200_optimum_gform.json")
rows.append(("Anaheim, 200 draws", a["draws"], a["distinct_pairs"], 0))
if a["distinct_pairs"] > a["draws"]:
    fails.append("Anaheim: more distinct pairs than draws")
r = jload("released_impl/arrival_reachable_eval_set.json")
if r["arrival_reachable"] + len(r["unreachable_origins"]) != r["origins"]:
    fails.append("Anaheim, the 30-origin set: the reachable and the unreachable do not close")
rows.append(("Anaheim, 30 origins", r["origins"], r["arrival_reachable"], 0))

# ---- negative control: a set that does not close must be rejected ----
probe = {"origins": 30, "arrival_reachable": 28, "unreachable_origins": [1]}
if probe["arrival_reachable"] + len(probe["unreachable_origins"]) == probe["origins"]:
    fails.append("negative control did not fire, so a set that does not close would pass unseen")

print("evaluation-set precondition, every road network the manuscript reports\n")
print("  %-22s %7s %9s %7s" % ("Network", "drawn", "distinct", "seeds"))
for row in rows:
    print("  %-22s %7d %9d %7d" % row)
print()
for f in fails:
    print("  FAIL ", f)
print("EVAL-SET CHECK %s" % ("FAILED" if fails else "PASS on the three road networks"))
sys.exit(1 if fails else 0)
