# -*- coding: utf-8 -*-
"""Every learned cell of Table VI at one budget and one seed count.

WHY THIS EXISTS. Table VI mixed three protocols in one comparison: the two headline rows at 3000
episodes over ten seeds with a min-max range, a recovery row at 8000 episodes over five seeds with a
standard deviation, and the SUMO rows at five seeds. A reviewer objected that a table whose rows
differ in budget, in seed count and in dispersion convention cannot carry a within-table comparison,
since any difference between two rows is confounded with the protocol that produced them. The
objection is correct and no wording repairs it.

This runs all four learned grid cells, boundary closed and open crossed with the travel-time and
destination-aligned rewards, at a single budget of 8000 episodes over ten seeds, and reports the
standard deviation for each. The exact-optimum rows carry no dispersion, since value iteration is
deterministic.

    OMP_NUM_THREADS=1 python3 matched_budget_grid.py
"""
import argparse
import json

import numpy as np
import torch

from sweep_extra import train_eval, make_eval_od

torch.set_num_threads(1)

EVAL = make_eval_od(5)
CELLS = [
    ("closed_time_min", dict(boundary="closed", reward="time_min")),
    ("open_time_min",   dict(boundary="open",   reward="time_min")),
    ("closed_aligned",  dict(boundary="closed", reward="aligned")),
    ("open_aligned",    dict(boundary="open",   reward="aligned")),
]

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--episodes", type=int, default=8000)
ap.add_argument("--out", default="matched_budget_grid.json")
ap.add_argument("--only", default=None, help="one cell name, so the four run as four processes")
a = ap.parse_args()

print("5x5 grid, every learned cell at %d episodes over %d seeds"
      % (a.episodes, a.seeds), flush=True)
res = {}
CELLS = [c for c in CELLS if a.only is None or c[0] == a.only]
assert CELLS, "no cell named %s" % a.only
for name, kw in CELLS:
    comps = [train_eval(kw, s, a.episodes, EVAL) for s in range(a.seeds)]
    v = 100 * np.asarray(comps, dtype=float)
    res[name] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                 "min": float(v.min()), "max": float(v.max()),
                 "per_seed": [round(float(x), 1) for x in v],
                 "seeds": a.seeds, "episodes": a.episodes}
    print("  %-16s %5.1f%% (sd %4.1f)  range %.1f-%.1f"
          % (name, v.mean(), v.std(ddof=1), v.min(), v.max()), flush=True)

json.dump({"network": "bespoke 5x5 grid", "episodes": a.episodes, "seeds": a.seeds,
           "eval_pairs": len(EVAL), "cells": res}, open(a.out, "w"), indent=1)
print("wrote %s" % a.out)
