# -*- coding: utf-8 -*-
"""Training curves for the four cells, recorded by the run that produces Table VIII.

Fig. 5 of the manuscript and Fig. S-1 of the supplement were drawn from results_10seed.npz,
a run predating both the thread pinning and the four-cell rerun. Its final points read 30.8%
on the boundary-open aligned cell and 99.5% on the boundary-closed travel-time cell, against
the 35.1% and 99.9% Table VIII reports from four_cells_boundary_dest.json. One document then
carried two measurements of one cell. This runner records the curve inside the run that
produces the table, through the same run_cell function, so the last checkpoint of a curve and
the cell of the table cannot disagree.

Cells run in parallel across processes at one linear-algebra thread each, which is the
arrangement the supplement records.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from env_boundary_dest import make_eval_od          # noqa: E402
from four_cells_boundary_dest import run_cell       # noqa: E402

CELLS = [("closed", "time_min"), ("open", "time_min"),
         ("closed", "aligned"), ("open", "aligned")]


def one_seed(args):
    boundary, reward, seed, episodes, every = args
    eval_od = make_eval_od(n=200, seed=12345)
    curve = []
    final = run_cell(boundary, reward, seed, episodes, eval_od,
                     curve_every=every, curve=curve)
    return boundary, reward, seed, final, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--every", type=int, default=150)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="four_cells_curves.json")
    a = ap.parse_args()

    jobs = [(b, r, s, a.episodes, a.every)
            for (b, r) in CELLS for s in range(a.seeds)]
    out = {"%s_%s" % (b, r): {"steps": None, "curves": [None] * a.seeds,
                              "returns": [None] * a.seeds,
                              "finals": [None] * a.seeds} for (b, r) in CELLS}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for boundary, reward, seed, final, curve in ex.map(one_seed, jobs):
            key = "%s_%s" % (boundary, reward)
            out[key]["steps"] = [int(e) for e, _, _ in curve]
            out[key]["curves"][seed] = [float(v) for _, v, _ in curve]
            out[key]["returns"][seed] = [float(g) for _, _, g in curve]
            out[key]["finals"][seed] = float(final)
            print("== %-6s %-9s seed %d  final %5.1f%%"
                  % (boundary, reward, seed, final), flush=True)

    for key, rec in out.items():
        rec["mean_final"] = float(np.mean(rec["finals"]))
        print("%-18s mean final %5.1f%%" % (key, rec["mean_final"]))
    with open(os.path.join(HERE, a.out), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
