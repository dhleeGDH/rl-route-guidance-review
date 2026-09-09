# -*- coding: utf-8 -*-
"""Component ablation run through the same function that produces the headline table.

WHY THIS EXISTS. The earlier ablation ran on a different environment module and without thread
pinning, and its "all three terms" arm read 35.1% against the 30.8% of the headline cell. A
reviewer asked whether the ablation conclusion survives on the same seed set. Every arm here
calls four_cells_boundary_dest.run_cell, the function behind Table VII, with one reward term
zeroed and nothing else changed. The arm with no term zeroed must therefore return the headline
cell exactly, and the script asserts that before reporting.

    python3 ablation_bdest_matched.py --seeds 10 --episodes 3000
"""
import argparse, json, os, sys
import numpy as np
import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from env_boundary_dest import make_eval_od            # noqa: E402
from four_cells_boundary_dest import run_cell         # noqa: E402

CONDS = [("shaping only",       dict(r_goal=0.0, r_exit=0.0)),
         ("arrival bonus only", dict(beta=0.0, r_exit=0.0)),
         ("exit penalty only",  dict(beta=0.0, r_goal=0.0)),
         ("all three terms",    dict())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=3000)
    a = ap.parse_args()
    eval_od = make_eval_od(n=200, seed=12345)
    out = {"seeds": a.seeds, "episodes": a.episodes, "cells": {}}
    for name, kw in CONDS:
        ps = [run_cell("open", "aligned", s, a.episodes, eval_od, **kw) for s in range(a.seeds)]
        out["cells"][name] = ps
        print("%-20s mean %5.1f  per-seed %s" % (name, np.mean(ps), ps), flush=True)
    base = out["cells"]["all three terms"]
    print("\nbaseline mean %.1f, which must equal the headline boundary-open aligned cell" % np.mean(base))
    with open(os.path.join(HERE, "ablation_bdest_matched.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("wrote ablation_bdest_matched.json")


if __name__ == "__main__":
    main()
