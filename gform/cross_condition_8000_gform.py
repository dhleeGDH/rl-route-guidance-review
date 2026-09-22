# -*- coding: utf-8 -*-
"""The cross-condition evaluation at the 8000-episode budget, published form.

WHY THIS EXISTS. exit_sentinel_control.json and zero_shot_transfer.json carry the cross-condition
cell at 3000 episodes, and the four cells of the rewritten Table V and the curves of Fig. 5 are at
8000. A paragraph quoting both would mix two budgets. This repeats the same computation at 8000.

WHAT IS UNCHANGED. The job is the one exit_sentinel_control.py runs, copied line for line:

    run_cell("closed", reward, seed, episodes, od, state_sentinel=sent)
    run_cell("closed", reward, seed, episodes, od, state_sentinel=sent, eval_boundary="open")

Two calls of the same function at the same seed train the same policy, since eval_boundary enters
only the final evaluation, so the pair scores one policy on both boundaries. The evaluation set is
make_eval_od(n=200, seed=12345), the sentinel is the published "high", the learner, the step budget
and every other setting are run_cell's defaults. The mixin of the M2-C variant is not imported, so
this is the published form of the shaping term.

    python3 cross_condition_8000_gform.py --episodes 8000 --seeds 10 --workers 16
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = "/home/dhlee/review_paper/handoff/experiments/boundary_open_demo"
sys.path.insert(0, DEMO)

from env_boundary_dest import make_eval_od          # noqa: E402
from four_cells_boundary_dest import run_cell       # noqa: E402  (pins torch to one thread)

B, SEED_A = 10000, 20260719                          # convention A of table6_rows.ci()


def ci(x, label):
    x = np.asarray(x, float)
    rng = np.random.RandomState((SEED_A + zlib.crc32(label.encode("utf-8"))) % (2 ** 32))
    means = np.mean(rng.choice(x, size=(B, len(x)), replace=True), axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _one(job):
    reward, seed, episodes = job
    od = make_eval_od(n=200, seed=12345)
    t0 = time.time()
    closed = run_cell("closed", reward, seed, episodes, od, state_sentinel="high")
    opened = run_cell("closed", reward, seed, episodes, od, state_sentinel="high",
                      eval_boundary="open")
    return reward, seed, closed, opened, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(HERE, "cross_condition_8000_gform.json"))
    a = ap.parse_args()
    jobs = [(rw, s, a.episodes) for rw in ("time_min", "aligned") for s in range(a.seeds)]
    cells = {rw: {"scored_closed": [None] * a.seeds, "scored_open": [None] * a.seeds,
                  "seconds": [None] * a.seeds} for rw in ("time_min", "aligned")}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for reward, seed, closed, opened, secs in ex.map(_one, jobs):
            cells[reward]["scored_closed"][seed] = closed
            cells[reward]["scored_open"][seed] = opened
            cells[reward]["seconds"][seed] = secs
            print("== %-9s seed %d: closed %5.1f%%  open %5.1f%%  [%.0fs]"
                  % (reward, seed, closed, opened, secs), flush=True)
    out = {"trained_on": "closed", "episodes": a.episodes, "seeds": a.seeds,
           "variant": "published", "state_sentinel": "high",
           "eval_set": "make_eval_od(n=200, seed=12345)", "cells": cells,
           "convention": "A, table6_rows.ci(), 10000 resamples, seed 20260719 + crc32(key)"}
    for rw in ("time_min", "aligned"):
        for field in ("scored_closed", "scored_open"):
            m, lo, hi = ci(cells[rw][field], "grid|%s|%d|%s" % (rw, a.episodes, field))
            cells[rw][field + "_mean"] = m
            cells[rw][field + "_ci_A"] = [lo, hi]
            print("%-9s %-14s %.1f [%.1f-%.1f]" % (rw, field, m, lo, hi))
    paired = [g - t for g, t in zip(cells["aligned"]["scored_open"],
                                    cells["time_min"]["scored_open"])]
    m, lo, hi = ci(paired, "grid|cross|%d|paired" % a.episodes)
    out["paired_open"] = {"per_seed": paired, "mean": m, "ci_A": [lo, hi]}
    paired_c = [g - t for g, t in zip(cells["aligned"]["scored_closed"],
                                      cells["time_min"]["scored_closed"])]
    m2, lo2, hi2 = ci(paired_c, "grid|cross|%d|paired_closed" % a.episodes)
    out["paired_closed"] = {"per_seed": paired_c, "mean": m2, "ci_A": [lo2, hi2]}
    print("paired, scored open   %.1f [%.1f-%.1f]" % (m, lo, hi))
    print("paired, scored closed %.1f [%.1f-%.1f]" % (m2, lo2, hi2))
    out["wall_seconds"] = time.time() - t0
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("wrote %s in %.0f s" % (os.path.basename(a.out), out["wall_seconds"]))


if __name__ == "__main__":
    main()
