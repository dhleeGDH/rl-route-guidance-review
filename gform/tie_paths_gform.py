# -*- coding: utf-8 -*-
# T-2069a supplement [H]. Two censuses behind the Table S-2 divergences, under the Eq. (4) form.
#
#  (i) states where the Eq. (4) optimal action set is a PROPER SUBSET of the Eq. (3) one and every
#      action of the Eq. (3) set carries the same travel-time return: the aligned reward has only
#      broken a tie, and no reader-visible travel time can separate the two rewards there;
# (ii) states where the Eq. (3) optimal set itself contains actions whose greedy routes differ in
#      transition count: a tie in return across routes of different length.
#
# Writes tie_paths_gform.json. Reads the deposited networks and writes nothing else.
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from eq7_threshold_gform import read, hop_potential, vi_time, greedy_hops_all   # noqa: E402

NETS = [("Sioux Falls", os.path.join(BASE, "networks", "SiouxFalls_net.tntp")),
        ("Anaheim", os.path.join(BASE, "networks", "Anaheim_net.tntp"))]
GAMMAS = [0.95, 0.99, 0.999, 0.9999, 1.0]
R_GOAL_UNITS, BETA_UNITS, TOL = 10.0, 1.0, 1e-7


def vi_shaped(dest, adj, cost, nodes, gamma, phi, beta, r_goal):
    V = {n: -1e18 for n in range(1, nodes + 1)}
    V[dest] = 0.0
    for _ in range(nodes + 2):
        stable = True
        for u in range(1, nodes + 1):
            if u == dest:
                continue
            best = -1e18
            for v in adj.get(u, ()):
                r = -cost[(u, v)] + beta * (phi[u] - gamma * phi[v])
                if v == dest:
                    r += r_goal
                q = r + (0.0 if v == dest else gamma * V[v])
                if q > best:
                    best = q
            if best > V[u] + 1e-12:
                V[u] = best
                stable = False
        if stable:
            break
    return V


def sets_at(u, dest, adj, cost, gamma, V, phi, beta, r_goal):
    vals = []
    for v in adj.get(u, ()):
        r = -cost[(u, v)]
        if phi is not None:
            r += beta * (phi[u] - gamma * phi[v])
            if v == dest:
                r += r_goal
        vals.append((v, r + (0.0 if v == dest else gamma * V[v])))
    top = max(q for _, q in vals)
    return frozenset(v for v, q in vals if q > top - TOL), dict(vals)


def analyse(name, path, gamma):
    zones, nodes, links = read(path)
    adj, radj, cost = {}, {}, {}
    for u, v, c in links:
        c = c if c > 0 else 1e-3
        adj.setdefault(u, []).append(v)
        radj.setdefault(v, []).append(u)
        cost[(u, v)] = c
    scale = sum(cost.values()) / len(cost)
    beta, r_goal = BETA_UNITS * scale, R_GOAL_UNITS * scale
    proper_subset_tied = same_return_diff_hops = states = differing = 0
    for dest in range(1, zones + 1):
        phi, far = hop_potential(dest, radj, nodes)
        reach = [n for n in range(1, nodes + 1) if phi[n] < far and n != dest]
        Vt = vi_time(dest, adj, cost, nodes, gamma)
        Va = vi_shaped(dest, adj, cost, nodes, gamma, phi, beta, r_goal)
        kopt = greedy_hops_all(dest, adj, cost, Vt, gamma, nodes)
        for u in reach:
            a, qa = sets_at(u, dest, adj, cost, gamma, Vt, None, beta, r_goal)
            b, _ = sets_at(u, dest, adj, cost, gamma, Va, phi, beta, r_goal)
            states += 1
            if a != b:
                differing += 1
            if b < a:
                vals = [qa[v] for v in a]
                if max(vals) - min(vals) <= TOL:
                    proper_subset_tied += 1
            hops = {(1 if v == dest else (None if kopt.get(v) is None else kopt[v] + 1))
                    for v in a}
            hops.discard(None)
            if len(hops) > 1:
                same_return_diff_hops += 1
    return {"network": name, "gamma": gamma, "states": states, "states_differing": differing,
            "aligned_proper_subset_with_tied_travel_time": proper_subset_tied,
            "travel_time_optimal_set_spans_different_hop_counts": same_return_diff_hops}


def grid_case(gamma):
    """The same two censuses on the 5x5 boundary-closed grid of Fig. 1, node-based, uniform cost."""
    N, E_COST, R_D, BETA = 5, 1.0 + 0.6 * 0.5 * 0.5, 10.0, 1.0
    nodes = [(r, c) for r in range(N) for c in range(N)]

    def nb(u):
        r, c = u
        return [(r + a, c + b) for a, b in ((-1, 0), (0, 1), (1, 0), (0, -1))
                if 0 <= r + a < N and 0 <= c + b < N]

    def man(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def vi(dest, shaped):
        V = {u: -1e18 for u in nodes}
        V[dest] = 0.0
        for _ in range(4 * len(nodes)):
            stable = True
            for u in nodes:
                if u == dest:
                    continue
                best = -1e18
                for v in nb(u):
                    r = -E_COST
                    if shaped:
                        r += BETA * (man(u, dest) - gamma * man(v, dest))
                        if v == dest:
                            r += R_D
                    q = r + (0.0 if v == dest else gamma * V[v])
                    best = max(best, q)
                if best > V[u] + 1e-12:
                    V[u] = best
                    stable = False
            if stable:
                break
        return V

    subset_tied = tie_hops = states = differing = 0
    for dest in nodes:
        Vt, Va = vi(dest, False), vi(dest, True)
        kopt = {}
        for u in nodes:
            cur, k, seen = u, 0, set()
            while cur != dest and k <= 4 * len(nodes) and cur not in seen:
                seen.add(cur)
                best, arg = -1e18, None
                for v in sorted(nb(cur)):
                    q = -E_COST + (0.0 if v == dest else gamma * Vt[v])
                    if q > best + 1e-12:
                        best, arg = q, v
                cur, k = arg, k + 1
            kopt[u] = k if cur == dest else None
        for u in nodes:
            if u == dest:
                continue
            qa = {v: -E_COST + (0.0 if v == dest else gamma * Vt[v]) for v in nb(u)}
            ta = max(qa.values())
            a = frozenset(v for v, q in qa.items() if q > ta - TOL)
            qb = {}
            for v in nb(u):
                r = -E_COST + BETA * (man(u, dest) - gamma * man(v, dest))
                if v == dest:
                    r += R_D
                qb[v] = r + (0.0 if v == dest else gamma * Va[v])
            tb = max(qb.values())
            b = frozenset(v for v, q in qb.items() if q > tb - TOL)
            states += 1
            if a != b:
                differing += 1
            if b < a and max(qa[v] for v in a) - min(qa[v] for v in a) <= TOL:
                subset_tied += 1
            hops = {(1 if v == dest else (None if kopt.get(v) is None else kopt[v] + 1))
                    for v in a}
            hops.discard(None)
            if len(hops) > 1:
                tie_hops += 1
    return {"network": "bespoke grid", "gamma": gamma, "states": states,
            "states_differing": differing,
            "aligned_proper_subset_with_tied_travel_time": subset_tied,
            "travel_time_optimal_set_spans_different_hop_counts": tie_hops}


def main():
    out = []
    print("%-12s %-8s %8s %10s %14s %16s"
          % ("network", "gamma", "states", "differing", "subset+tied", "tie across hops"))
    for name, path in NETS:
        for g in GAMMAS:
            r = analyse(name, path, g)
            out.append(r)
            print("%-12s %-8s %8d %10d %14d %16d"
                  % (name, g, r["states"], r["states_differing"],
                     r["aligned_proper_subset_with_tied_travel_time"],
                     r["travel_time_optimal_set_spans_different_hop_counts"]), flush=True)
    for g in GAMMAS:
        r = grid_case(g)
        out.append(r)
        print("%-12s %-8s %8d %10d %14d %16d"
              % (r["network"], g, r["states"], r["states_differing"],
                 r["aligned_proper_subset_with_tied_travel_time"],
                 r["travel_time_optimal_set_spans_different_hop_counts"]), flush=True)
    json.dump(out, io.open(os.path.join(HERE, "tie_paths_gform.json"), "w", encoding="utf-8"),
              indent=1)
    print("wrote tie_paths_gform.json")


if __name__ == "__main__":
    main()
