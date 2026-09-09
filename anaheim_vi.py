# -*- coding: utf-8 -*-
"""Exact value iteration on a real road network under both rewards, boundary closed and open.

Section V-B settles the 5x5 grid this way: the analytical bound is a worst-case sufficient
condition and the value iteration sharpens it. The same computation is run here on Anaheim, 416
nodes and 914 links, to answer whether the sharpening still finds leaving optimal at real scale.

Construction mirrors the grid exactly. Zone nodes are the trip ends. The destination is one zone.
Under the boundary-closed condition it is the only terminal. Under the boundary-open condition every
other zone also terminates the trip, which is the counterpart of a peripheral link removing a
vehicle. Traversal cost is the link's free-flow time.

    python3 anaheim_vi.py
"""
import json
from collections import deque
from pathlib import Path

NET = Path(__file__).resolve().parent / "networks" / "Anaheim_net.tntp"
R_GOAL, R_EXIT = 10.0, 5.0          # aligned reward, in units of the mean traversal cost

def read(p):
    z = n = None
    L = []
    for ln in open(p):
        s = ln.strip()
        if s.startswith("<NUMBER OF ZONES>"): z = int(s.split(">")[1])
        elif s.startswith("<NUMBER OF NODES>"): n = int(s.split(">")[1])
        elif s and not s.startswith(("~", "<")):
            f = s.rstrip(";").split()
            if len(f) >= 6:
                try: L.append((int(f[0]), int(f[1]), float(f[4])))
                except ValueError: pass
    return z, n, L

ZONES, NODES, LINKS = read(NET)
cost = {}
adj = {}
for u, v, c in LINKS:
    if c <= 0: c = 1e-3
    adj.setdefault(u, []).append(v)
    cost[(u, v)] = c
scale = sum(cost.values()) / len(cost)          # mean traversal cost, the unit for R_GOAL/R_EXIT

def solve(dest, boundary, reward):
    """Optimal value by backward induction on a deterministic MDP. Returns the optimal action map."""
    exits = set(range(1, ZONES + 1)) - {dest} if boundary == "open" else set()
    V = {n: -1e18 for n in range(1, NODES + 1)}
    V[dest] = 0.0
    for _ in range(NODES + 2):                   # costs are positive, so this converges
        stable = True
        for u in range(1, NODES + 1):
            if u == dest or u in exits:
                continue
            best = -1e18
            for v in adj.get(u, ()):
                step = -cost[(u, v)]
                if v == dest:
                    nxt = step + (R_GOAL * scale if reward == "aligned" else 0.0)
                elif v in exits:
                    nxt = step - (R_EXIT * scale if reward == "aligned" else 0.0)
                else:
                    nxt = step + V[v]
                if nxt > best: best = nxt
            if best > V[u] + 1e-12:
                V[u] = best; stable = False
        if stable: break
    # walk the optimal policy from each origin and record how the trip ends
    def outcome(o):
        seen, cur = set(), o
        for _ in range(NODES + 2):
            if cur == dest: return "arrive"
            if cur in exits: return "exit"
            if cur in seen or cur not in adj: return "stuck"
            seen.add(cur)
            best, arg = -1e18, None
            for v in adj[cur]:
                step = -cost[(cur, v)]
                if v == dest: nxt = step + (R_GOAL * scale if reward == "aligned" else 0.0)
                elif v in exits: nxt = step - (R_EXIT * scale if reward == "aligned" else 0.0)
                else: nxt = step + V[v]
                if nxt > best: best, arg = nxt, v
            cur = arg
        return "stuck"
    return outcome

def hops(dest):
    radj = {}
    for u, v, _ in LINKS: radj.setdefault(v, []).append(u)
    d, q = {dest: 0}, deque([dest])
    while q:
        x = q.popleft()
        for y in radj.get(x, ()):
            if y not in d: d[y] = d[x] + 1; q.append(y)
    return d

print("Anaheim: %d nodes, %d links, %d zones, mean traversal cost %.3f min\n" %
      (NODES, len(LINKS), ZONES, scale))
DESTS = [1, 10, 20, 30, 38]
out = {}
for boundary in ("closed", "open"):
    for reward in ("time_min", "aligned"):
        tot = {"arrive": 0, "exit": 0, "stuck": 0}
        for dest in DESTS:
            oc = solve(dest, boundary, reward)
            hp = hops(dest)
            for o in range(ZONES + 1, NODES + 1):        # interior origins
                if o in hp: tot[oc(o)] += 1
        n = sum(tot.values())
        print("%-7s %-9s arrive %5.1f%%   exit %5.1f%%   stuck %4.1f%%   (%d origin-destination pairs)" %
              (boundary, reward, 100*tot["arrive"]/n, 100*tot["exit"]/n, 100*tot["stuck"]/n, n))
        out["%s/%s" % (boundary, reward)] = {k: round(100*v/n, 1) for k, v in tot.items()}
json.dump(out, open(Path(__file__).resolve().parent / "anaheim_vi.json", "w"), indent=1)
