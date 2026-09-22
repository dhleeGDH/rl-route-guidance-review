# -*- coding: utf-8 -*-
"""The Anaheim optimum of each reward on the 200 draws the trained cell is scored on.

Table IV states the optimum over the 15,289 enumerated pairs of the hull. The trained cell of
anaheim_cell_gform.py is scored on 200 draws, the protocol the grid and Sioux Falls cells use.
The two are different pair sets, so this scores the same exact policies on the 200 draws, which
is what a trained rate is read against. The attainable maximum of the draw is computed here as
well, by the reachability of anaheim_arrival_term_gform.py restricted to the same pairs.

    python3 anaheim_eval200_optimum_gform.py
"""
import json
import os
import sys
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import anaheim_shaped_vi_gform as A     # noqa: E402
import anaheim_cell_gform as C          # noqa: E402


def reach_avoiding(dest, radj, blocked):
    seen, q = {dest}, deque([dest])
    while q:
        u = q.popleft()
        for v in radj.get(u, ()):
            if v in seen or (v in blocked and v != dest):
                continue
            seen.add(v)
            q.append(v)
    return seen


def main():
    ns = A.load(0.0)
    solve = A.build_solver(ns)
    ods = C.make_eval_od(ns)
    dests = sorted({d for _, d in ods})
    out = {"draws": len(ods), "distinct_pairs": len(set(ods)), "destinations": len(dests),
           "cells": {}}
    arms = [("closed", "time_min", 0.0, "none"), ("open", "time_min", 0.0, "none"),
            ("closed", "aligned", 1.0, "hops"), ("open", "aligned", 1.0, "hops")]
    for boundary, reward, beta, kind in arms:
        oc = {d: solve(d, boundary, reward, beta, kind) for d in dests}
        ok = sum(1 for o, d in ods if oc[d](o) == "arrive")
        key = "%s_%s" % (boundary, reward)
        out["cells"][key] = {"arrive": ok, "draws": len(ods),
                             "arrive_pct": round(100.0 * ok / len(ods), 1)}
        print("%-18s optimum on the 200 draws: %5.1f%% (%d/%d)"
              % (key, 100.0 * ok / len(ods), ok, len(ods)))
    border, radj = set(ns["BORDER"]), ns["RADJ"]
    rb = {d: reach_avoiding(d, radj, border) for d in dests}
    free = sum(1 for o, d in ods if o in rb[d])
    out["attainable_maximum_pct"] = round(100.0 * free / len(ods), 1)
    out["attainable_maximum_pairs"] = free
    print("attainable maximum on the same draws: %.1f%% (%d/%d)"
          % (100.0 * free / len(ods), free, len(ods)))
    json.dump(out, open(os.path.join(HERE, "anaheim_eval200_optimum_gform.json"), "w"), indent=1)
    print("wrote anaheim_eval200_optimum_gform.json")


if __name__ == "__main__":
    main()
