# -*- coding: utf-8 -*-
"""Exact optimum of each reward on Sioux Falls, under the convention of the trained cells.

WHY THIS EXISTS. Table IV of the rewrite states the optimum of each reward on every network of
the experiment. The grid optima come from boundary_open_demo/optimal_vi_boundary_dest.py and the
Anaheim optima from anaheim_vi_perimeter.py, and no exact computation was deposited for Sioux
Falls: the network carries trained cells alone. This computes it under the convention the Sioux
Falls cells are trained and evaluated with, which is benchmark_demo.GraphRouteEnv:

  border      the 12-node outer face of the published drawing, NETWORKS["sioux_falls"]["boundary"]
  exit        an action offered at a border node on the open network, taken at one flat unit;
              a route may pass a border node without ending there
  arrival     the exit action taken at the destination node, which is drawn on the border
  cost        the congestion field of Eq. (9), replaced here by its expected value 1.15, which is
              the deterministic surrogate optimal_vi_boundary_dest.py solves on the grid
  evaluation  benchmark_demo.make_eval_od(net): 200 draws, seed 999, destinations on the border,
              minimum separation of three links, 127 of them distinct

Value iteration at gamma = 1 over the 24 nodes, then a greedy rollout from each origin. No
learner, no seed and no simulation enter the computation.

    python3 sioux_vi_gform.py
"""
import json
import os
import sys

BENCH = "/home/dhlee/review_paper/handoff/experiments/benchmark_network"
sys.path.insert(0, BENCH)

import benchmark_demo as B   # noqa: E402

E_COST = 1.0 + 0.6 * 0.5 * 0.5          # 1.15, as optimal_vi_boundary_dest.py
R_GOAL, R_EXIT, BETA = 10.0, 5.0, 1.0
NEG = -1e12

NET = dict(B.NETWORKS["sioux_falls"])
ADJ, RADJ = B.build(NET)
N = NET["n_nodes"]
BORDER = set(NET["boundary"])


def phi_to(dst):
    d = B.hop_dist_to(dst, RADJ, N)
    return {u: min(d[u], N) for u in range(1, N + 1)}


def solve(dst, boundary, reward, iters=4000, tol=1e-12):
    """V[u] is the optimal return-to-go from node u. Terminal actions carry no successor."""
    phi = phi_to(dst)
    exits = (BORDER - {dst}) if boundary == "open" else set()
    V = {u: 0.0 for u in range(1, N + 1)}
    for _ in range(iters):
        nv = {}
        for u in range(1, N + 1):
            best = NEG
            for v in ADJ[u]:
                rw = -E_COST + (BETA * (phi[u] - phi[v]) if reward == "aligned" else 0.0)
                best = max(best, rw + V[v])
            if u == dst:
                rw = -1.0 + (R_GOAL + BETA * phi[u] if reward == "aligned" else 0.0)
                best = max(best, rw)
            elif u in exits:
                rw = -1.0 - (R_EXIT if reward == "aligned" else 0.0)
                best = max(best, rw)
            nv[u] = best
        delta = max(abs(nv[u] - V[u]) for u in V)
        V = nv
        if delta < tol:
            break
    return V, phi, exits


def rollout(o, dst, V, phi, exits, reward, cap=None):
    """The greedy policy of V. Returns (arrived, steps taken inside the network)."""
    cap = cap or (4 * N)
    u = o
    for t in range(cap):
        best, arg = NEG, None
        for v in ADJ[u]:
            rw = -E_COST + (BETA * (phi[u] - phi[v]) if reward == "aligned" else 0.0)
            q, tgt = rw + V[v], v
            if q > best:
                best, arg = q, tgt
        if u == dst:
            rw = -1.0 + (R_GOAL + BETA * phi[u] if reward == "aligned" else 0.0)
            if rw > best:
                best, arg = rw, "ARR"
        elif u in exits:
            rw = -1.0 - (R_EXIT if reward == "aligned" else 0.0)
            if rw > best:
                best, arg = rw, "EXIT"
        if arg == "ARR":
            return True, t
        if arg in ("EXIT", None):
            return False, t
        u = arg
    return False, cap


def reachable_share(ods):
    """The attainable maximum under this convention: a border node does not end a trip, so a
    pair keeps every route it has on the closed graph."""
    ok = 0
    for o, d in ods:
        dd = B.hop_dist_to(d, RADJ, N)
        ok += o != d and dd[o] < 1e8
    return 100.0 * ok / len(ods)


def main():
    ods = B.make_eval_od(NET)
    distinct = sorted(set(ods))
    out = {"network": "sioux_falls", "nodes": N, "links": len(NET["links"]),
           "border_nodes": sorted(BORDER), "border_size": len(BORDER),
           "exit_convention": "a trip ends by taking the exit action at a border node; "
                              "a route may pass a border node without ending there",
           "surrogate_link_cost": E_COST, "gamma": 1.0,
           "r_goal": R_GOAL, "r_exit": R_EXIT, "beta": BETA,
           "eval_draws": len(ods), "eval_distinct_pairs": len(distinct),
           "max_steps_of_the_trained_cell": NET["max_steps"],
           "attainable_maximum_draws": reachable_share(ods),
           "attainable_maximum_distinct": reachable_share(distinct),
           "cells": {}}
    print("Sioux Falls, exact optimum by boundary condition and reward")
    print("  border %s" % sorted(BORDER))
    print("  %d evaluation draws, %d distinct pairs" % (len(ods), len(distinct)))
    for boundary in ("closed", "open"):
        for reward in ("time_min", "aligned"):
            cache = {}
            ok = okd = 0
            longest = 0
            for o, d in ods:
                key = (d, boundary, reward)
                if key not in cache:
                    cache[key] = solve(d, boundary, reward)
                V, phi, ex = cache[key]
                a, steps = rollout(o, d, V, phi, ex, reward)
                ok += a
                longest = max(longest, steps)
            for o, d in distinct:
                V, phi, ex = cache[(d, boundary, reward)]
                okd += rollout(o, d, V, phi, ex, reward)[0]
            lab = "%s %s" % (boundary, reward)
            out["cells"][lab] = {"completion_draws_pct": round(100.0 * ok / len(ods), 1),
                                 "arrived_draws": ok, "draws": len(ods),
                                 "completion_distinct_pct": round(100.0 * okd / len(distinct), 1),
                                 "arrived_distinct": okd, "distinct": len(distinct),
                                 "longest_optimal_route_steps": longest}
            print("  %-18s draws %5.1f%% (%d/%d)   distinct %5.1f%% (%d/%d)   longest route %d steps"
                  % (lab, 100.0 * ok / len(ods), ok, len(ods),
                     100.0 * okd / len(distinct), okd, len(distinct), longest))
    print("  attainable maximum, this convention: %.1f%% of the draws, %.1f%% of the distinct pairs"
          % (out["attainable_maximum_draws"], out["attainable_maximum_distinct"]))
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sioux_vi_gform.json")
    json.dump(out, open(p, "w"), indent=1)
    print("wrote %s" % os.path.basename(p))


if __name__ == "__main__":
    main()
