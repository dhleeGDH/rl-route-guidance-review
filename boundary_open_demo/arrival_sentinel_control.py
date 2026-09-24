# -*- coding: utf-8 -*-
"""Is "the arrival term alone reaches nothing" a property of the objective or of the encoding?

Table IX reports an optimum at arrival on every pair under an arrival term alone, against a
learner completing 0.0% at 3000 episodes and again at 8000. The state presents an off-grid move
with a cost slot of 9.0, above every real cost, so the arriving action is always observed as the
most expensive move available. On that reading the 0.0% measures the encoding rather than the
objective, and the two are separated below.

The same arm runs here under both encodings, nothing else differing:

  high      the published behavior, an off-grid slot of 9.0
  matched   an off-grid slot of 1.0, which is what the reward pays for that move

Both arms go through four_cells_boundary_dest.run_cell, the function behind the published cells.
The negative control is the high arm, which must return the 0.0 on every seed at both budgets
that arrival_term_budget.json records; a departure there means this script is not running the
published configuration and the comparison is void.

    OMP_NUM_THREADS=1 python3 arrival_sentinel_control.py --seeds 10
"""
import argparse
import io
import json
import os

import numpy as np
import torch

from four_cells_boundary_dest import run_cell
from env_boundary_dest import make_eval_od

torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
PUBLISHED = os.path.join(HERE, "arrival_term_budget.json")
# the arrival term alone, as CONDS names it in ablation_bdest.py
TERMS = dict(beta=0.0, r_goal=10.0, r_exit=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--budgets", type=int, nargs="+", default=[3000, 8000])
    ap.add_argument("--out", default=os.path.join(HERE, "arrival_sentinel_control.json"))
    a = ap.parse_args()

    eval_od = make_eval_od(n=200, seed=12345)
    pub = json.load(io.open(PUBLISHED, encoding="utf-8"))
    res = {}
    print("arrival term alone, boundary-open grid, %d seeds" % a.seeds)
    for sentinel in ("high", "matched"):
        for episodes in a.budgets:
            per_seed = [run_cell("open", "aligned", s, episodes, eval_od,
                                 state_sentinel=sentinel, **TERMS) for s in range(a.seeds)]
            key = "%s_%d" % (sentinel, episodes)
            res[key] = {"per_seed": per_seed, "mean": float(np.mean(per_seed)),
                        "sd": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0}
            line = "  %-8s %5d ep  mean %6.1f  sd %5.1f" % (
                sentinel, episodes, res[key]["mean"], res[key]["sd"])
            if sentinel == "high" and str(episodes) in pub and a.seeds == 10:
                same = np.allclose(pub[str(episodes)]["per_seed"], per_seed, atol=1e-6)
                res[key]["reproduces_published"] = bool(same)
                line += "   negative control: %s" % ("MATCH" if same else "MISMATCH")
            print(line, flush=True)
    json.dump({"seeds": a.seeds, "eval_pairs": len(eval_od), "terms": TERMS, "cells": res},
              io.open(a.out, "w", encoding="utf-8"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
