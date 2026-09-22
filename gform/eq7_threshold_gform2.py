# -*- coding: utf-8 -*-
# 2026-09-16. The right-hand side of Eq. (7) as the manuscript writes it.
#
# The body prints  gamma > 1 - Delta / [(K - k_min) R_d].  Two things separate this from the
# deposited eq7_route_sets.py: that file carries the Phi_max denominator of Supplementary S-I.A,
# (K - k_min) R_g + (K - 1) beta Phi_max, and it measures Delta on the UNDISCOUNTED problem by a
# deliberate choice recorded in its own comment. The author's definition for this run is the one
# the body's where-clause states: Delta is the smallest positive gap in the travel-time return AT
# gamma between an optimal path and the best path outside the optimal set, and K and k_min are the
# transition counts of those two paths. Delta therefore moves with gamma and so does the bound.
#
# Writes eq7_threshold_gform.json next to this file. Reads the deposited networks; writes nothing
# else.
import io
import json
import os
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
NETS = [("Sioux Falls", os.path.join(BASE, "networks", "SiouxFalls_net.tntp")),
        ("Anaheim", os.path.join(BASE, "networks", "Anaheim_net.tntp"))]
GAMMAS = [0.90, 0.95, 0.99, 0.999, 0.9999]
# Delta is measured in the raw cost units of the network, so R_d must be in the same
# units. The deposited comparison (eq7_pairs_c.py, and eq7_route_sets.py before it) sets the
# arrival bonus at R_GOAL = 10 IN UNITS OF THE MEAN TRAVERSAL COST and multiplies by that mean, so
# the arrival bonus in raw units is 10 * mean(cost). An earlier form put the literal 10 against a raw-unit
# Delta and mixed the two scales; the threshold below is in one scale throughout.
R_D_UNITS = 10.0         # Section IV-B: an arrival bonus of 10 in units of one step's cost
TOL = 1e-9


def read(path):
    zones = nodes = None
    links = []
    for ln in io.open(path, encoding="utf-8", errors="replace"):
        s = ln.strip()
        if s.startswith("<NUMBER OF ZONES>"):
            zones = int(s.split(">")[1])
        elif s.startswith("<NUMBER OF NODES>"):
            nodes = int(s.split(">")[1])
        elif s and not s.startswith(("~", "<")):
            f = s.rstrip(";").split()
            if len(f) >= 6:
                try:
                    links.append((int(f[0]), int(f[1]), float(f[4])))
                except ValueError:
                    pass
    return zones, nodes, links


def hop_potential(dest, radj, nodes):
    d = {dest: 0}
    q = deque([dest])
    while q:
        u = q.popleft()
        for p in radj.get(u, ()):
            if p not in d:
                d[p] = d[u] + 1
                q.append(p)
    far = max(d.values()) + 1 if d else 1
    return {n: d.get(n, far) for n in range(1, nodes + 1)}, far


def vi_time(dest, adj, cost, nodes, gamma):
    V = {n: -1e18 for n in range(1, nodes + 1)}
    V[dest] = 0.0
    for _ in range(nodes + 2):
        stable = True
        for u in range(1, nodes + 1):
            if u == dest:
                continue
            best = -1e18
            for v in adj.get(u, ()):
                nxt = -cost[(u, v)] + (0.0 if v == dest else gamma * V[v])
                if nxt > best:
                    best = nxt
            if best > V[u] + 1e-12:
                V[u] = best
                stable = False
        if stable:
            break
    return V


def hops_of_greedy(u, dest, V, adj, cost, gamma, nodes):
    """Transitions along the greedy optimal route, or None where it does not complete."""
    cur, k, seen = u, 0, set()
    while k <= nodes + 1:
        if cur == dest:
            return k
        if cur in seen or cur not in adj:
            return None
        seen.add(cur)
        best, arg = -1e18, None
        for v in sorted(adj[cur]):
            q = -cost[(cur, v)] + (0.0 if v == dest else gamma * V[v])
            if q > best + 1e-12:
                best, arg = q, v
        cur = arg
        k += 1
    return None


def greedy_hops_all(dest, adj, cost, V, gamma, nodes):
    """Transitions of the greedy optimal route from every node, memoized along the way."""
    k = {dest: 0}

    def walk(u):
        chain, cur = [], u
        seen = set()
        while cur not in k:
            if cur in seen or cur not in adj:
                for c in chain:
                    k[c] = None
                k[cur] = None
                return
            seen.add(cur)
            chain.append(cur)
            best, arg = -1e18, None
            for v in sorted(adj[cur]):
                q = -cost[(cur, v)] + (0.0 if v == dest else gamma * V[v])
                if q > best + 1e-12:
                    best, arg = q, v
            if arg is None:
                for c in chain:
                    k[c] = None
                return
            cur = arg
        base = k[cur]
        for c in reversed(chain):
            base = None if base is None else base + 1
            k[c] = base

    for u in range(1, nodes + 1):
        if u not in k:
            walk(u)
    return k


def analyse(name, path, gamma):
    """Delta at gamma, with K and k_min the extreme transition counts of the two paths across
    every OD pair, which is what the where-clause of Eq. (7) states."""
    zones, nodes, links = read(path)
    adj, radj, cost = {}, {}, {}
    for u, v, c in links:
        c = c if c > 0 else 1e-3
        adj.setdefault(u, []).append(v)
        radj.setdefault(v, []).append(u)
        cost[(u, v)] = c
    scale = sum(cost.values()) / len(cost)
    r_d = R_D_UNITS * scale
    delta, delta_at = None, None
    K, k_min = 0, 10 ** 9
    for dest in range(1, zones + 1):
        phi, far = hop_potential(dest, radj, nodes)
        reach = [n for n in range(1, nodes + 1) if phi[n] < far and n != dest]
        V = vi_time(dest, adj, cost, nodes, gamma)
        kopt = greedy_hops_all(dest, adj, cost, V, gamma, nodes)
        for u in reach:
            qs = [(v, -cost[(u, v)] + (0.0 if v == dest else gamma * V[v]))
                  for v in adj.get(u, ())]
            if len(qs) < 2 or kopt.get(u) is None:
                continue
            top = max(q for _, q in qs)
            for v, q in qs:
                gap = top - q
                if gap <= TOL:
                    continue
                kd = 1 if v == dest else (None if kopt.get(v) is None else kopt[v] + 1)
                if kd is None:
                    continue
                K = max(K, kopt[u], kd)
                k_min = min(k_min, kopt[u], kd)
                if delta is None or gap < delta:
                    delta, delta_at = gap, (dest, u, v, kopt[u], kd)
    dest, u, v, k_opt, k_dev = delta_at
    span = K - k_min
    rhs = 1.0 - delta / (span * r_d) if span > 0 else float("-inf")
    return {"network": name, "gamma": gamma, "delta": delta,
            "delta_at": {"destination": dest, "node": u, "deviating_to": v,
                         "k_optimal_path": k_opt, "k_best_non_optimal_path": k_dev},
            "K": K, "k_min": k_min, "mean_traversal_cost": scale,
            "R_d_raw": r_d, "R_d_units": R_D_UNITS,
            "eq7_rhs": rhs, "condition_gamma_gt_rhs": bool(gamma > rhs)}


def main():
    out = []
    print("%-12s %-8s %10s %4s %6s %6s %12s  %s"
          % ("network", "gamma", "Delta", "K", "k_min", "R_d raw", "Eq.(7) RHS", "gamma > RHS"))
    for name, path in NETS:
        for g in GAMMAS:
            r = analyse(name, path, g)
            out.append(r)
            print("%-12s %-8s %10.6f %4d %6d %6.1f %12.6f  %s"
                  % (name, g, r["delta"], r["K"], r["k_min"], r["R_d_raw"], r["eq7_rhs"],
                     r["condition_gamma_gt_rhs"]), flush=True)
    json.dump(out, io.open(os.path.join(HERE, "eq7_threshold_gform2.json"), "w",
                           encoding="utf-8"), indent=1)
    print("wrote eq7_threshold_gform2.json")


if __name__ == "__main__":
    main()
