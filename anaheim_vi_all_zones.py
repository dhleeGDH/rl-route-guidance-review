# -*- coding: utf-8 -*-
"""Exact value iteration on Anaheim over EVERY zone, not a sample of five.

WHY THIS EXISTS. anaheim_vi.py ran five destinations, "1, 10, 20, 30 and 38", chosen by spacing
across the zone numbering. A reviewer objected that the exit rate is a function of where the
destination sits relative to the perimeter, so a numbering-spaced sample of 5 of 38 can move the
headline figure, and that value iteration has no cost barrier to running all of them. Both points
are correct: the full sweep below is 38 destinations against 378 interior origins, 14,364 pairs,
and it completes in under a second.

It also reports what the five-zone run left implicit:
  - the exit rate per destination zone, so the spread across zones is visible rather than averaged;
  - each zone's hop distance to its nearest other zone, which is the quantity the exit-dominance
    bound depends on, against the mean hop distance from the origins that reach it;
  - the traversal-cost spread, since the bound's threshold scales with the largest link cost.

    python3 anaheim_vi_all_zones.py
"""
import json
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
NET = HERE / "networks" / "Anaheim_net.tntp"
OUT = HERE / "anaheim_vi_all_zones.json"
R_GOAL, R_EXIT = 10.0, 5.0


def read(p):
    z = n = None
    L = []
    for ln in open(p):
        s = ln.strip()
        if s.startswith("<NUMBER OF ZONES>"):
            z = int(s.split(">")[1])
        elif s.startswith("<NUMBER OF NODES>"):
            n = int(s.split(">")[1])
        elif s and not s.startswith(("~", "<")):
            f = s.rstrip(";").split()
            if len(f) >= 6:
                try:
                    L.append((int(f[0]), int(f[1]), float(f[4])))
                except ValueError:
                    pass
    return z, n, L


ZONES, NODES, LINKS = read(NET)
cost, adj, radj = {}, {}, {}
for u, v, c in LINKS:
    if c <= 0:
        c = 1e-3
    adj.setdefault(u, []).append(v)
    radj.setdefault(v, []).append(u)
    cost[(u, v)] = c
scale = sum(cost.values()) / len(cost)
CMAX = max(cost.values())


def solve(dest, boundary, reward):
    exits = set(range(1, ZONES + 1)) - {dest} if boundary == "open" else set()
    V = {n: -1e18 for n in range(1, NODES + 1)}
    V[dest] = 0.0
    for _ in range(NODES + 2):
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
                if nxt > best:
                    best = nxt
            if best > V[u] + 1e-12:
                V[u] = best
                stable = False
        if stable:
            break

    def outcome(o):
        seen, cur = set(), o
        for _ in range(NODES + 2):
            if cur == dest:
                return "arrive"
            if cur in exits:
                return "exit"
            if cur in seen or cur not in adj:
                return "stuck"
            seen.add(cur)
            best, arg = -1e18, None
            for v in adj[cur]:
                step = -cost[(cur, v)]
                if v == dest:
                    nxt = step + (R_GOAL * scale if reward == "aligned" else 0.0)
                elif v in exits:
                    nxt = step - (R_EXIT * scale if reward == "aligned" else 0.0)
                else:
                    nxt = step + V[v]
                if nxt > best:
                    best, arg = nxt, v
            cur = arg
        return "stuck"
    return outcome


def hops_to(dest):
    d, q = {dest: 0}, deque([dest])
    while q:
        x = q.popleft()
        for y in radj.get(x, ()):
            if y not in d:
                d[y] = d[x] + 1
                q.append(y)
    return d


ALL = list(range(1, ZONES + 1))
INTERIOR = list(range(ZONES + 1, NODES + 1))
print("Anaheim: %d nodes, %d links, %d zones" % (NODES, len(LINKS), ZONES))
print("traversal cost: mean %.3f, max %.3f, ratio %.1f\n" % (scale, CMAX, CMAX / scale))

# rho per zone: hops from that zone's reachable interior to the NEAREST other zone
per_zone = {}
for dest in ALL:
    hp = hops_to(dest)
    reach = [o for o in INTERIOR if o in hp]
    mean_hops = sum(hp[o] for o in reach) / max(1, len(reach))
    nearest_other = min((min(hops_to(z).get(o, 10 ** 6) for z in ALL if z != dest)
                         for o in reach), default=None)
    per_zone[dest] = {"reachable_origins": len(reach), "mean_hops_to_dest": mean_hops,
                      "min_hops_to_any_other_zone": nearest_other}

res = {}
for boundary in ("closed", "open"):
    for reward in ("time_min", "aligned"):
        tot = {"arrive": 0, "exit": 0, "stuck": 0}
        by_zone = {}
        for dest in ALL:
            oc, hp = solve(dest, boundary, reward), hops_to(dest)
            z = {"arrive": 0, "exit": 0, "stuck": 0}
            for o in INTERIOR:
                if o in hp:
                    z[oc(o)] += 1
                    tot[oc(o)] += 1
            by_zone[dest] = z
        n = sum(tot.values())
        key = "%s_%s" % (boundary, reward)
        res[key] = {"total": tot, "pairs": n,
                    "exit_rate": 100.0 * tot["exit"] / n, "arrive_rate": 100.0 * tot["arrive"] / n,
                    "by_zone": by_zone}
        print("%-7s %-9s arrive %5.1f%%   exit %5.1f%%   stuck %4.1f%%   (%d pairs over %d zones)"
              % (boundary, reward, 100.0 * tot["arrive"] / n, 100.0 * tot["exit"] / n,
                 100.0 * tot["stuck"] / n, n, ZONES))

op = res["open_time_min"]["by_zone"]
rates = sorted(100.0 * op[z]["exit"] / max(1, sum(op[z].values())) for z in ALL)
print("\nopen travel-time exit rate by destination zone: min %.1f%%, median %.1f%%, max %.1f%%"
      % (rates[0], rates[len(rates) // 2], rates[-1]))
print("zones at 100%% exit: %d of %d" % (sum(1 for r in rates if r >= 99.95), ZONES))
nn = [per_zone[z]["min_hops_to_any_other_zone"] for z in ALL
      if per_zone[z]["min_hops_to_any_other_zone"] is not None]
print("hops from an origin to its nearest zone other than the destination: min %d, max %d" % (min(nn), max(nn)))

# the arrival term at two and a half times the strength the grid uses, over the same pairs
R_GOAL, R_EXIT = 25.0, 12.5
tot25 = {"arrive": 0, "exit": 0, "stuck": 0}
for dest in ALL:
    oc, hp = solve(dest, "open", "aligned"), hops_to(dest)
    for o in INTERIOR:
        if o in hp:
            tot25[oc(o)] += 1
n25 = sum(tot25.values())
print("open    aligned   at 2.5x arrival strength: arrive %5.1f%%   (%d pairs)"
      % (100.0 * tot25["arrive"] / n25, n25))
res["open_aligned_2.5x"] = {"total": tot25, "pairs": n25,
                            "arrive_rate": 100.0 * tot25["arrive"] / n25}

json.dump({"zones": ZONES, "nodes": NODES, "links": len(LINKS), "mean_cost": scale,
           "max_cost": CMAX, "per_zone_geometry": per_zone, "results": res},
          open(OUT, "w"), indent=1)
print("\nwrote %s" % OUT.name)
