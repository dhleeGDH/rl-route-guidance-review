# Runs the published Sioux Falls cells at one seed and writes a JSON shard.
# The benchmark package is imported, never written. Training is benchmark_demo.train_eval unchanged.
# Set BENCHMARK_DIR to the benchmark_network directory if it is not at the default relative path.
import argparse, json, os, sys
REPO = os.environ.get("BENCHMARK_DIR",
       os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, REPO)
import torch
torch.set_num_threads(1)
import numpy as np
import benchmark_demo as B

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--od-mode", default="dest_boundary")
ap.add_argument("--outdir", required=True)
a = ap.parse_args()

net = dict(B.NETWORKS["sioux_falls"])
net["od_mode"] = a.od_mode
od = B.make_eval_od(net)
eps = net["episodes"]
res = {"seed": a.seed, "od_mode": a.od_mode, "episodes": eps,
       "max_steps": net["max_steps"], "eval_draws": len(od),
       "eval_distinct_pairs": len(set(od)), "cells": {}}
for boundary in ("closed", "open"):
    for reward in ("time_min", "aligned"):
        r, tt, n = B.train_eval(net, boundary, reward, a.seed, eps, od,
                                return_travel_time=True)
        res["cells"]["%s_%s" % (boundary, reward)] = {
            "completion_pct": 100.0 * r,
            "travel_time": None if not np.isfinite(tt) else float(tt),
            "completing_trips": int(n)}
os.makedirs(a.outdir, exist_ok=True)
with open(os.path.join(a.outdir, "%s_seed%02d.json" % (a.od_mode, a.seed)), "w") as f:
    json.dump(res, f, indent=1)
print("done", a.od_mode, a.seed, flush=True)
