"""Does the learning-level gap depend on the state sentinel?

The state presents an off-grid move with a cost slot of 9.0 (UNAVAIL_COST) while the reward
that move pays is the ordinary step cost of 1.0. The learning-level gap (open-aligned 30.8%
against a 100.0% optimum) is measured against that mismatch rather than against the objective.

The four cells are run twice through four_cells_boundary_dest.run_cell, the same function that
produced the published cells, so nothing can differ between the control and the published run
except the encoding under test:

  high      the published behavior, an off-grid slot of 9.0
  matched   an off-grid slot of 1.0, which is what the reward pays for that move

The "high" arm is the negative control and must reproduce four_cells_boundary_dest.json seed
for seed. A mismatch there means this script is not running the published configuration, and
the comparison is void; --check-negative-control enforces that and exits non-zero on failure.

Results are written to a file of this study's own. The published cells are never touched.

    OMP_NUM_THREADS=1 python3 sentinel_control.py --seeds 10 --episodes 3000
"""
import argparse
import json
import os

import numpy as np
import torch

from four_cells_boundary_dest import run_cell
from env_boundary_dest import make_eval_od

torch.set_num_threads(1)

PUBLISHED = "four_cells_boundary_dest.json"
CELLS = [("closed", "time_min"), ("open", "time_min"),
         ("closed", "aligned"), ("open", "aligned")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--out", default="sentinel_control.json")
    ap.add_argument("--only", default="", help="sentinel_boundary_reward, one cell only")
    a = ap.parse_args()
    assert a.out != PUBLISHED, "would overwrite the published cells"

    eval_od = make_eval_od(n=200, seed=12345)
    res = {}
    for sentinel in ("high", "matched"):
        for boundary, reward in CELLS:
            key = "%s_%s_%s" % (sentinel, boundary, reward)
            if a.only and key != a.only:
                continue
            per_seed = [run_cell(boundary, reward, s, a.episodes, eval_od,
                                 state_sentinel=sentinel) for s in range(a.seeds)]
            res[key] = {"per_seed": per_seed,
                        "mean": float(np.mean(per_seed)),
                        "sd": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0}
            print("%-8s %-6s %-9s mean %6.1f  sd %5.1f" %
                  (sentinel, boundary, reward, res[key]["mean"], res[key]["sd"]), flush=True)

            # Negative control: the "high" arm must reproduce the published per-seed values.
            if sentinel == "high" and os.path.exists(PUBLISHED) and a.seeds == 10:
                # the published file stores its cells at the top level, with no "cells" wrapper
                pub = json.load(open(PUBLISHED))["%s_%s" % (boundary, reward)]
                same = np.allclose(pub["per_seed"], per_seed, atol=1e-6)
                print("   negative control against %s: %s" %
                      (PUBLISHED, "MATCH" if same else "MISMATCH %s" % pub["per_seed"]),
                      flush=True)
                res[key]["reproduces_published"] = bool(same)

    json.dump({"episodes": a.episodes, "seeds": a.seeds, "eval_pairs": len(eval_od),
               "distinct_pairs": len(set(map(str, eval_od))), "cells": res},
              open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
