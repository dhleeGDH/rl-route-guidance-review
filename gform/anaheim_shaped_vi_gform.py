# -*- coding: utf-8 -*-
"""Anaheim optima of Eq. (3) and Eq. (4) with the potential term of Eq. (4) included.

anaheim_vi_perimeter.py solves the destination-aligned reward with the arrival
bonus and the exit penalty alone: its `solve()` adds `R_GOAL * SCALE` at the destination and
subtracts `R_EXIT * SCALE` at an exit, and carries no shaping term. Eq. (4) of the manuscript
carries three terms, the third being the potential on the links. The grid solver
(boundary_open_demo/optimal_vi_boundary_dest.py) and the Sioux Falls solver (sioux_vi_gform.py)
both include it at beta = 1. This computes the Anaheim rows under the same three-term reward, so
that one reward is solved on all three networks.

The shaping follows the manuscript's own reading of Eq. (4):

  ordinary move u -> v      -c(u,v) + beta * (Phi(u) - Phi(v))
  move onto the destination -c(u,d) + beta *  Phi(u)                + R_d * SCALE
  move onto another border  -c(u,v)                                 - R_x * SCALE
                            ("On an exit the potential retains the value at the link of
                             departure", so the shaping increment of an exit transition is zero)

with gamma = 1, at which the two shaping forms of the record (published and M2-C / gform) agree.
Phi is reported under two readings, since the manuscript defines it as "the distance to the
destination" and Anaheim is the one network whose link costs are not one unit per link:

  hops   the number of links to the destination, scaled by SCALE, which is the direct transfer
         of the grid convention, where one link costs about one unit and R_d, R_x are already
         scaled by SCALE in the deposited Anaheim solver
  cost   the least travel time to the destination in the units of the link file itself

beta = 0 reproduces the deposited run and is asserted against it rather than assumed.

Everything else is the deposited construction: this file execs anaheim_vi_perimeter.py up to its
first statement for the border rule, cost table, origin set and destination set. No deposited
output is read as a value and none is written.

    python3 anaheim_shaped_vi_gform.py
"""
import heapq
import json
import os
import sys
from collections import deque

SRC = os.path.join(os.environ.get("DEPOSIT_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "anaheim_vi_perimeter.py")
HERE = os.path.dirname(os.path.abspath(__file__))
NEG = -1e18

# the deposited cells, as published, for the beta = 0 regression
PUBLISHED = {
    "hull": {"closed time_min": 100.0, "closed aligned": 100.0,
             "open time_min": 21.1, "open aligned": 94.1},
    "band": {"closed time_min": 100.0, "closed aligned": 100.0,
             "open time_min": 5.2, "open aligned": 45.6},
}


def load(band):
    sys.argv = ["anaheim_vi_perimeter.py", "--band", str(band)]
    sys.path.insert(0, os.path.dirname(SRC))
    src = open(SRC, encoding="utf-8").read()
    ns = {"__file__": SRC, "__name__": "anaheim_vi_perimeter_gform"}
    exec(compile(src[:src.index('print("Anaheim: %d nodes')], SRC, "exec"), ns)
    return ns


def hop_phi(dest, RADJ, NODES):
    d = {dest: 0}
    q = deque([dest])
    while q:
        u = q.popleft()
        for v in RADJ.get(u, ()):
            if v not in d:
                d[v] = d[u] + 1
                q.append(v)
    return {n: d.get(n, NODES) for n in range(1, NODES + 1)}


def cost_phi(dest, RADJ, COST, NODES):
    dist = {dest: 0.0}
    pq = [(0.0, dest)]
    while pq:
        dv, v = heapq.heappop(pq)
        if dv > dist.get(v, 1e18) + 1e-15:
            continue
        for u in RADJ.get(v, ()):
            nd = dv + COST[(u, v)]
            if nd < dist.get(u, 1e18) - 1e-15:
                dist[u] = nd
                heapq.heappush(pq, (nd, u))
    big = max(dist.values()) if dist else 0.0
    return {n: dist.get(n, big) for n in range(1, NODES + 1)}


def build_solver(ns):
    BORDER, ADJ, COST, SCALE = ns["BORDER"], ns["ADJ"], ns["COST"], ns["SCALE"]
    NODES, RADJ = ns["NODES"], ns["RADJ"]
    R_GOAL, R_EXIT = ns["R_GOAL"], ns["R_EXIT"]

    def phi_of(dest, kind):
        if kind == "none":
            return {n: 0.0 for n in range(1, NODES + 1)}
        if kind == "hops":
            h = hop_phi(dest, RADJ, NODES)
            return {n: SCALE * h[n] for n in range(1, NODES + 1)}
        return cost_phi(dest, RADJ, COST, NODES)

    def step_value(u, v, dest, exits, reward, phi, beta):
        """One transition of Eq. (3) or Eq. (4). Returns (immediate reward, successor or None)."""
        step = -COST[(u, v)]
        if reward != "aligned" or beta == 0.0:
            shape_move = 0.0
            shape_arr = 0.0
        else:
            shape_move = beta * (phi[u] - phi[v])
            shape_arr = beta * phi[u]                      # Phi is zero at the destination link
        if v == dest:
            return step + shape_arr + (R_GOAL * SCALE if reward == "aligned" else 0.0), None
        if v in exits:
            return step - (R_EXIT * SCALE if reward == "aligned" else 0.0), None
        return step + shape_move, v

    def solve(dest, boundary, reward, beta=0.0, kind="none"):
        exits = (BORDER - {dest}) if boundary == "open" else set()
        phi = phi_of(dest, kind if reward == "aligned" else "none")
        V = {n: NEG for n in range(1, NODES + 1)}
        V[dest] = 0.0
        converged = False
        for _ in range(NODES + 2):
            stable = True
            for u in range(1, NODES + 1):
                if u == dest or u in exits:
                    continue
                best = NEG
                for v in ADJ.get(u, ()):
                    rw, nxt = step_value(u, v, dest, exits, reward, phi, beta)
                    q = rw if nxt is None else rw + V[nxt]
                    if q > best:
                        best = q
                if best > V[u] + 1e-12:
                    V[u] = best
                    stable = False
            if stable:
                converged = True
                break
        if not converged:
            raise SystemExit("value iteration did not converge: dest %s %s" % (dest, boundary))

        def outcome(o):
            seen, cur = set(), o
            for _ in range(NODES + 2):
                if cur == dest:
                    return "arrive"
                if cur in exits:
                    return "exit"
                if cur in seen or cur not in ADJ:
                    return "stuck"
                seen.add(cur)
                best, arg = NEG, None
                for v in ADJ[cur]:
                    rw, nxt = step_value(cur, v, dest, exits, reward, phi, beta)
                    q = rw if nxt is None else rw + V[nxt]
                    if q > best:
                        best, arg = q, v
                cur = arg
            return "stuck"
        return outcome
    return solve


def score(ns, solve, boundary, reward, beta, kind):
    ORIGINS, DESTS, reachable = ns["ORIGINS"], ns["DESTS"], ns["reachable"]
    tot = {"arrive": 0, "exit": 0, "stuck": 0}
    for dest in DESTS:
        oc, reach = solve(dest, boundary, reward, beta, kind), reachable(dest)
        for o in ORIGINS:
            if o != dest and o in reach:
                tot[oc(o)] += 1
    n = sum(tot.values())
    return {"pairs": n, "arrive": tot["arrive"], "exit": tot["exit"], "stuck": tot["stuck"],
            "arrive_pct": round(100.0 * tot["arrive"] / n, 1)}


def run(band, label):
    ns = load(band)
    solve = build_solver(ns)
    out = {"band": band, "border_nodes": len(ns["BORDER"]), "origins": len(ns["ORIGINS"]),
           "destinations": len(ns["DESTS"]), "mean_link_cost": ns["SCALE"],
           "r_goal": ns["R_GOAL"], "r_exit": ns["R_EXIT"], "gamma": 1.0, "cells": {}}
    print("== %s : border %d, origins %d, destinations %d"
          % (label, len(ns["BORDER"]), len(ns["ORIGINS"]), len(ns["DESTS"])))
    arms = [("beta 0 (deposited reward)", "time_min", 0.0, "none"),
            ("beta 0 (deposited reward)", "aligned", 0.0, "none"),
            ("beta 1, hop potential", "aligned", 1.0, "hops"),
            ("beta 1, cost potential", "aligned", 1.0, "cost")]
    for boundary in ("closed", "open"):
        for arm, reward, beta, kind in arms:
            r = score(ns, solve, boundary, reward, beta, kind)
            key = "%s %s | %s" % (boundary, reward, arm)
            out["cells"][key] = r
            print("   %-58s arrive %6.1f%%  exit %6d  pairs %6d"
                  % (key, r["arrive_pct"], r["exit"], r["pairs"]))
    # regression against the deposited run at beta = 0
    for boundary in ("closed", "open"):
        for reward in ("time_min", "aligned"):
            got = out["cells"]["%s %s | beta 0 (deposited reward)" % (boundary, reward)]["arrive_pct"]
            want = PUBLISHED[label]["%s %s" % (boundary, reward)]
            mark = "OK  " if abs(got - want) < 0.05 else "FAIL"
            print("   regression %s %-9s %-7s deposited %5.1f  recomputed %5.1f"
                  % (mark, boundary, reward, want, got))
            out.setdefault("regression", {})["%s %s" % (boundary, reward)] = {
                "deposited": want, "recomputed": got, "pass": abs(got - want) < 0.05}
    return out


def main():
    res = {"hull": run(0.0, "hull"), "band": run(0.10, "band")}
    json.dump(res, open(os.path.join(HERE, "anaheim_shaped_vi_gform.json"), "w"), indent=1)
    print("wrote anaheim_shaped_vi_gform.json")


if __name__ == "__main__":
    main()
