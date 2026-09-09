# -*- coding: utf-8 -*-
"""T-1903 step 7: the pair-level statistic the manuscript prints, under the three variants.

Section V-B prints "22.6% of the completing pairs of Anaheim differ in travel time by 14.8% at
most" at gamma = 0.99. That is a share of OD pairs, not the node-level share of identical action
sets. Both are computed here, from the same solver as eq7_c.py.
"""
import json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARC = ROOT / "archive/experiments_dup/handoff_experiments"
sys.path.insert(0, str(ARC))
from eq7_route_sets import read, hop_potential, BETA, R_GOAL     # noqa: E402
import eq7_c                                                      # noqa: E402


def travel_time_of_greedy(o, dest, V, adj, cost, gamma, phi, beta, r_goal, shaping, nodes):
    """Undiscounted travel time of the route the greedy policy of V takes from o."""
    g = gamma if shaping == "c" else 1.0
    cur, tt, seen = o, 0.0, set()
    for _ in range(nodes + 2):
        if cur == dest:
            return tt
        if cur in seen:
            return None
        seen.add(cur)
        best, arg = -1e18, None
        for v in adj.get(cur, ()):
            r = -cost[(cur, v)]
            if phi is not None:
                r += beta * (phi[cur] - g * phi[v])
                if v == dest:
                    r += r_goal
            q = r + (0.0 if v == dest else gamma * V[v])
            if q > best:
                best, arg = q, v
        if arg is None:
            return None
        tt += cost[(cur, arg)]
        cur = arg
    return None


def analyse_pairs(path, gamma, shaping):
    zones, nodes, links = read(path)
    adj, radj, cost = {}, {}, {}
    for u, v, c in links:
        c = c if c > 0 else 1e-3
        adj.setdefault(u, []).append(v)
        radj.setdefault(v, []).append(u)
        cost[(u, v)] = c
    scale = sum(cost.values()) / len(cost)
    beta, r_goal = BETA * scale, R_GOAL * scale
    both, differ, rels = 0, 0, []
    for dest in range(1, zones + 1):
        phi, far = hop_potential(dest, radj, nodes)
        reach = [n for n in range(1, nodes + 1) if phi[n] < far and n != dest]
        Vtt = eq7_c.value_iteration(dest, adj, cost, nodes, gamma, None, beta, r_goal, shaping)
        Vda = eq7_c.value_iteration(dest, adj, cost, nodes, gamma, phi, beta, r_goal, shaping)
        for o in reach:
            a = travel_time_of_greedy(o, dest, Vtt, adj, cost, gamma, None, beta, r_goal, shaping, nodes)
            b = travel_time_of_greedy(o, dest, Vda, adj, cost, gamma, phi, beta, r_goal, shaping, nodes)
            if a is None or b is None:
                continue
            both += 1
            if abs(a - b) > 1e-9:
                differ += 1
                rels.append(abs(a - b) / max(a, 1e-9))
    return dict(pairs=both, differing=differ,
                share=round(100.0 * differ / both, 1) if both else None,
                max_rel=round(100.0 * max(rels), 1) if rels else 0.0)


if __name__ == "__main__":
    out = {}
    for name, path in eq7_c.NETS:
        for shaping in ("published", "c"):
            for gamma in (0.99, 0.95):
                r = analyse_pairs(path, gamma, shaping)
                out["%s|%s|%s" % (name, shaping, gamma)] = r
                print("%-12s %-10s gamma=%-6s pairs %5d  differing %5.1f%%  max relative %5.1f%%"
                      % (name, shaping, gamma, r["pairs"], r["share"], r["max_rel"]), flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "eq7_pairs_c.json"), "w"), indent=1)
