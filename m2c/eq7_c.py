# -*- coding: utf-8 -*-
"""T-1903 step 3: Table S-2 under the discount-consistent shaping.

The solver is the one behind Table S-2, archive/.../eq7_route_sets.py, re-expressed here with a
shaping switch. The comparison is boundary-closed, so no exit terminal enters and variant C and the
T-1901 variant coincide for this table. archive/ is read and never written.
"""
import json, os, sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARC = ROOT / "archive/experiments_dup/handoff_experiments"
sys.path.insert(0, str(ARC))
from eq7_route_sets import read, hop_potential, BETA, R_GOAL   # noqa: E402

NETS = [("Sioux Falls", ARC / "networks" / "SiouxFalls_net.tntp"),
        ("Anaheim", ARC / "networks" / "Anaheim_net.tntp")]


def value_iteration(dest, adj, cost, nodes, gamma, phi, beta, r_goal, shaping):
    g = gamma if shaping == "c" else 1.0
    V = {n: -1e18 for n in range(1, nodes + 1)}
    V[dest] = 0.0
    for _ in range(nodes + 2):
        stable = True
        for u in range(1, nodes + 1):
            if u == dest:
                continue
            best = -1e18
            for v in adj.get(u, ()):
                r = -cost[(u, v)]
                if phi is not None:
                    r += beta * (phi[u] - g * phi[v])
                    if v == dest:
                        r += r_goal
                nxt = r + (0.0 if v == dest else gamma * V[v])
                if nxt > best:
                    best = nxt
            if best > V[u] + 1e-12:
                V[u] = best
                stable = False
        if stable:
            break
    return V


def argmax_set(u, V, adj, cost, dest, gamma, phi, beta, r_goal, shaping):
    g = gamma if shaping == "c" else 1.0
    vals = []
    for v in adj.get(u, ()):
        r = -cost[(u, v)]
        if phi is not None:
            r += beta * (phi[u] - g * phi[v])
            if v == dest:
                r += r_goal
        vals.append((v, r + (0.0 if v == dest else gamma * V[v])))
    if not vals:
        return frozenset()
    top = max(q for _, q in vals)
    return frozenset(v for v, q in vals if q > top - 1e-7)


def analyse(path, gamma, shaping):
    zones, nodes, links = read(path)
    adj, radj, cost = {}, {}, {}
    for u, v, c in links:
        c = c if c > 0 else 1e-3
        adj.setdefault(u, []).append(v)
        radj.setdefault(v, []).append(u)
        cost[(u, v)] = c
    scale = sum(cost.values()) / len(cost)
    beta, r_goal = BETA * scale, R_GOAL * scale
    same = diff = 0
    for dest in range(1, zones + 1):
        phi, far = hop_potential(dest, radj, nodes)
        reach = [n for n in range(1, nodes + 1) if phi[n] < far and n != dest]
        Vtt = value_iteration(dest, adj, cost, nodes, gamma, None, beta, r_goal, shaping)
        Vda = value_iteration(dest, adj, cost, nodes, gamma, phi, beta, r_goal, shaping)
        for u in reach:
            a = argmax_set(u, Vtt, adj, cost, dest, gamma, None, beta, r_goal, shaping)
            b = argmax_set(u, Vda, adj, cost, dest, gamma, phi, beta, r_goal, shaping)
            if a == b:
                same += 1
            else:
                diff += 1
    return same, diff, round(100.0 * same / (same + diff), 2)


if __name__ == "__main__":
    out = {}
    print("%-12s %-10s %-8s %s" % ("network", "variant", "gamma", "identical optimal action sets"))
    for name, path in NETS:
        for shaping in ("published", "c"):
            for gamma in (1.0, 0.9999, 0.999, 0.99, 0.95):
                s, d, pct = analyse(path, gamma, shaping)
                out["%s|%s|%s" % (name, shaping, gamma)] = dict(same=s, diff=d, share=pct)
                print("%-12s %-10s %-8s %6.2f%%  (%d of %d nodes differ)"
                      % (name, shaping, gamma, pct, d, s + d), flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "eq7_c.json"), "w"), indent=1)
