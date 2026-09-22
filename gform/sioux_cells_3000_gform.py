# -*- coding: utf-8 -*-
"""The four Sioux Falls cells at the 3000-episode budget, published form.

Table V carries the bespoke grid at 3000 and at 8000 and Sioux Falls at 8000
alone, so the two networks do not hold the same pair of budgets. This runs the Sioux Falls cells
at 3000 through the path sioux30/run_cell.py uses.

WHAT IS UNCHANGED. benchmark_demo is imported and not edited; the cells go through
benchmark_demo.train_eval, which carries the learner of the deposited run: QNet 64-64, Adam at
1e-3, gamma 0.99, buffer 20000, batch 64, target network every 200 gradient steps, epsilon from
1.0 to 0.05 over the first 60% of the episodes. The network entry is a copy of
NETWORKS["sioux_falls"], so the border (the 12-node outer face), the destination rule
(dest_boundary), the step budget (60) and the congestion field are the deposited ones. The
evaluation set is make_eval_od(net), the same 200 draws at seed 999. Thirty seeds, as deposited.
Travel time comes back from train_eval, which excludes the arriving move: its accumulator adds a
cost only where the action is not the exit slot, and arrival is that slot at the destination.

The one value that differs from the deposited run is the budget, which is what this run moves:
net["episodes"] is set to 3000 on the copy, never on benchmark_demo's own dictionary.

    python3 sioux_cells_3000_gform.py --episodes 3000 --seeds 30 --workers 16
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
import torch

torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(os.environ.get("REPO_DIR", "."), "benchmark_network")
sys.path.insert(0, BENCH)

import benchmark_demo as B          # noqa: E402  (imported, never written)

CELLS = (("closed", "time_min"), ("closed", "aligned"),
         ("open", "time_min"), ("open", "aligned"))
B_RES, SEED_A = 10000, 20260719      # convention A of table6_rows.ci()


def ci(x, label):
    x = np.asarray([v for v in x if v is not None and not np.isnan(v)], float)
    if x.size == 0:
        return None, None, None
    rng = np.random.RandomState((SEED_A + zlib.crc32(label.encode("utf-8"))) % (2 ** 32))
    means = np.mean(rng.choice(x, size=(B_RES, len(x)), replace=True), axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def one_seed(args):
    seed, episodes = args
    torch.set_num_threads(1)
    net = dict(B.NETWORKS["sioux_falls"])      # a copy; benchmark_demo's own entry is untouched
    net["od_mode"] = "dest_boundary"
    net["episodes"] = episodes                 # the one value this run moves
    od = B.make_eval_od(net)
    t0 = time.time()
    out = {}
    for boundary, reward in CELLS:
        r, tt, n = B.train_eval(net, boundary, reward, seed, episodes, od,
                                return_travel_time=True)
        out["%s_%s" % (boundary, reward)] = {
            "completion_pct": 100.0 * r,
            "travel_time": None if not np.isfinite(tt) else float(tt),
            "completing_trips": int(n)}
    return seed, out, len(od), len(set(od)), net["max_steps"], time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(HERE, "sioux_cells_3000_gform.json"))
    a = ap.parse_args()
    acc = {"%s_%s" % (b, r): {"completion_per_seed": [None] * a.seeds,
                              "travel_time_per_seed": [None] * a.seeds,
                              "completing_trips": [None] * a.seeds}
           for (b, r) in CELLS}
    seconds = [None] * a.seeds
    draws = distinct = max_steps = None
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for seed, res, draws, distinct, max_steps, secs in ex.map(
                one_seed, [(s, a.episodes) for s in range(a.seeds)]):
            seconds[seed] = secs
            for k, v in res.items():
                acc[k]["completion_per_seed"][seed] = v["completion_pct"]
                acc[k]["travel_time_per_seed"][seed] = v["travel_time"]
                acc[k]["completing_trips"][seed] = v["completing_trips"]
            print("== seed %2d  %s  [%.0fs]"
                  % (seed, "  ".join("%s %5.1f%%" % (k, res[k]["completion_pct"]) for k in acc),
                     secs), flush=True)
    for k in acc:
        b, r = k.split("_", 1)
        m, lo, hi = ci(acc[k]["completion_per_seed"], "sioux|%s|%d|%s" % (r, a.episodes, b))
        acc[k]["completion_mean"], acc[k]["completion_ci_A"] = m, [lo, hi]
        m2, lo2, hi2 = ci(acc[k]["travel_time_per_seed"],
                          "sioux|%s|%d|%s|tt" % (r, a.episodes, b))
        acc[k]["travel_time_mean"] = m2
        acc[k]["travel_time_ci_A"] = None if m2 is None else [lo2, hi2]
        print("%-16s completion %5.1f [%.1f-%.1f]   travel time %s"
              % (k, m, lo, hi, "n/a" if m2 is None else "%.2f [%.2f-%.2f]" % (m2, lo2, hi2)))
    out = {"network": "Sioux Falls, 24 nodes, 76 links",
           "border": "the 12-node outer face of the published drawing",
           "od_mode": "dest_boundary", "episodes": a.episodes, "seeds": a.seeds,
           "eval_draws": draws, "eval_distinct_pairs": distinct, "max_steps": max_steps,
           "variant": "published",
           "travel_time": "excludes the arriving move, as train_eval accumulates it",
           "convention": "A, table6_rows.ci(), 10000 resamples, seed 20260719 + crc32(key)",
           "cells": acc, "seconds_per_seed": seconds, "wall_seconds": time.time() - t0}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("wrote %s in %.0f s" % (os.path.basename(a.out), out["wall_seconds"]))


if __name__ == "__main__":
    main()
