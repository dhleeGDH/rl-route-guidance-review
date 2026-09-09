# -*- coding: utf-8 -*-
"""Exact value iteration on Anaheim with the boundary Section IV-A defines.

WHY THIS EXISTS. `anaheim_vi_all_zones.py` opened the network by treating every zone other than the
destination as an exit. Section IV-A defines the boundary as the peripheral links at the edge of a
study area, and `exposure_ratio_real_networks.py` withdrew the zone list as a border on exactly that
ground: zones are trip ends distributed through a network rather than its outside. The two
constructions therefore disagreed on one network, and the 97.4% they produced answered a question the
manuscript's own definition does not ask.

This recomputes the exit-dominance share under the definition the manuscript uses. The border is the
convex hull of the published node coordinates together with every node within a tenth of the bounding
box of an edge, which is the set `exposure_ratio_real_networks.py` already applies. Destinations stay
the published zones, since those are the network's real trip ends. A trip ends by reaching its
destination or by leaving at any border node other than the destination.

    python3 anaheim_vi_perimeter.py
"""
import json
from collections import deque
from pathlib import Path

from exposure_ratio_real_networks import _hull, load_coords, read_tntp

HERE = Path(__file__).resolve().parent
NET = HERE / "networks" / "Anaheim_net.tntp"
COORD = HERE / "networks" / "anaheim_nodes.geojson"
import argparse as _ap
_A = _ap.ArgumentParser()
_A.add_argument("--band", type=float, default=0.10,
                help="border band as a fraction of the bounding box; 0 is the convex hull alone")
# Round 64 M5: Proposition 1 assumes an exit the policy may decline, whereas this network was
# opened by absorption, where touching a border node ends the trip whatever the policy prefers.
# Under "optional" a border node offers a leaving action at one ordinary traversal cost and the
# route may pass through instead, which is the semantics the grids use. "absorbing" is the
# published default and is unchanged.
_A.add_argument("--exit-mode", choices=("absorbing", "optional"), default="absorbing")
_ARGS, _ = _A.parse_known_args()
OUT = HERE / ("anaheim_vi_perimeter.json" if _ARGS.band == 0.10
              else "anaheim_vi_perimeter_band%s.json" % ("hull" if _ARGS.band == 0 else str(_ARGS.band)))
if _ARGS.exit_mode == "optional":
    OUT = HERE / ("anaheim_vi_perimeter_optional_band%s.json"
                  % ("hull" if _ARGS.band == 0 else str(_ARGS.band)))
R_GOAL, R_EXIT = 10.0, 5.0


def cost_table(path):
    zones = nodes = None
    cost, adj, radj = {}, {}, {}
    for ln in open(path):
        s = ln.strip()
        if s.startswith("<NUMBER OF ZONES>"):
            zones = int(s.split(">")[1])
        elif s.startswith("<NUMBER OF NODES>"):
            nodes = int(s.split(">")[1])
        elif s and not s.startswith(("~", "<")):
            f = s.rstrip(";").split()
            if len(f) >= 6:
                try:
                    u, v, c = int(f[0]), int(f[1]), float(f[4])
                except ValueError:
                    continue
                c = c if c > 0 else 1e-3
                adj.setdefault(u, []).append(v)
                radj.setdefault(v, []).append(u)
                cost[(u, v)] = c
    return zones, nodes, cost, adj, radj


def perimeter(coords, band=0.10):
    xs = [c[0] for c in coords.values()]
    ys = [c[1] for c in coords.values()]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    dx, dy = (x1 - x0) * band, (y1 - y0) * band
    hull = _hull([(k, v[0], v[1]) for k, v in coords.items()])
    edge = {k for k, (x, y) in coords.items()
            if x <= x0 + dx or x >= x1 - dx or y <= y0 + dy or y >= y1 - dy}
    return hull | edge


ZONES, NODES, COST, ADJ, RADJ = cost_table(NET)
COORDS = load_coords(COORD)
BORDER = perimeter(COORDS, band=_ARGS.band)
SCALE = sum(COST.values()) / len(COST)
DESTS = sorted(set(range(1, ZONES + 1)))
ORIGINS = sorted(n for n in range(1, NODES + 1) if n not in BORDER)


def solve(dest, boundary, reward):
    exits = (BORDER - {dest}) if boundary == "open" else set()
    optional = _ARGS.exit_mode == "optional"

    def leave_value(reward):
        return -SCALE - (R_EXIT * SCALE if reward == "aligned" else 0.0)
    V = {n: -1e18 for n in range(1, NODES + 1)}
    V[dest] = 0.0
    for _ in range(NODES + 2):
        stable = True
        for u in range(1, NODES + 1):
            if u == dest or (u in exits and not optional):
                continue
            best = leave_value(reward) if (optional and u in exits) else -1e18
            for v in ADJ.get(u, ()):
                step = -COST[(u, v)]
                if v == dest:
                    nxt = step + (R_GOAL * SCALE if reward == "aligned" else 0.0)
                elif v in exits and not optional:
                    nxt = step - (R_EXIT * SCALE if reward == "aligned" else 0.0)
                else:
                    nxt = step + V[v]
                best = max(best, nxt)
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
            if cur in exits and not optional:
                return "exit"
            if cur in seen or cur not in ADJ:
                return "stuck"
            seen.add(cur)
            best, arg = (leave_value(reward), "LEAVE") if (optional and cur in exits) \
                else (-1e18, None)
            for v in ADJ[cur]:
                step = -COST[(cur, v)]
                if v == dest:
                    nxt = step + (R_GOAL * SCALE if reward == "aligned" else 0.0)
                elif v in exits and not optional:
                    nxt = step - (R_EXIT * SCALE if reward == "aligned" else 0.0)
                else:
                    nxt = step + V[v]
                if nxt > best:
                    best, arg = nxt, v
            if arg == "LEAVE":
                return "exit"
            cur = arg
        return "stuck"
    return outcome


def reachable(dest):
    d, q = {dest}, deque([dest])
    while q:
        x = q.popleft()
        for y in RADJ.get(x, ()):
            if y not in d:
                d.add(y)
                q.append(y)
    return d


print("Anaheim: %d nodes, %d links, %d zones" % (NODES, len(COST), ZONES))
print("border nodes %d, interior origins %d, destinations %d (the published zones)"
      % (len(BORDER), len(ORIGINS), len(DESTS)))
res = {}
for boundary in ("closed", "open"):
    for reward in ("time_min", "aligned"):
        tot = {"arrive": 0, "exit": 0, "stuck": 0}
        for dest in DESTS:
            oc, reach = solve(dest, boundary, reward), reachable(dest)
            for o in ORIGINS:
                if o != dest and o in reach:
                    tot[oc(o)] += 1
        n = sum(tot.values())
        res["%s_%s" % (boundary, reward)] = {
            "total": tot, "pairs": n,
            "arrive_rate": round(100.0 * tot["arrive"] / n, 1),
            "exit_rate": round(100.0 * tot["exit"] / n, 1)}
        print("  %-7s %-9s arrive %5.1f%%  exit %5.1f%%  stuck %4.1f%%  (%d pairs)"
              % (boundary, reward, 100.0 * tot["arrive"] / n, 100.0 * tot["exit"] / n,
                 100.0 * tot["stuck"] / n, n))

# per-destination spread, and the arrival term at two and a half times the grid's strength
oc_by_zone = {}
for dest in DESTS:
    oc, reach = solve(dest, "open", "time_min"), reachable(dest)
    z = {"arrive": 0, "exit": 0, "stuck": 0}
    for o in ORIGINS:
        if o != dest and o in reach:
            z[oc(o)] += 1
    oc_by_zone[dest] = z
rates = sorted(100.0 * v["exit"] / max(1, sum(v.values())) for v in oc_by_zone.values())
print("open travel-time exit rate by destination: min %.1f%%, median %.1f%%, max %.1f%%"
      % (rates[0], rates[len(rates) // 2], rates[-1]))

R_GOAL, R_EXIT = 25.0, 12.5
tot25 = {"arrive": 0, "exit": 0, "stuck": 0}
for dest in DESTS:
    oc, reach = solve(dest, "open", "aligned"), reachable(dest)
    for o in ORIGINS:
        if o != dest and o in reach:
            tot25[oc(o)] += 1
n25 = sum(tot25.values())
print("open aligned at 2.5x arrival strength: arrive %.1f%% (%d pairs)"
      % (100.0 * tot25["arrive"] / n25, n25))
res["open_aligned_2.5x"] = {"total": tot25, "pairs": n25,
                            "arrive_rate": round(100.0 * tot25["arrive"] / n25, 1)}
res["per_zone_exit_rate"] = {"min": round(rates[0], 1), "median": round(rates[len(rates) // 2], 1),
                             "max": round(rates[-1], 1)}

json.dump({"nodes": NODES, "links": len(COST), "zones": ZONES,
           "border_nodes": len(BORDER), "origins": len(ORIGINS), "destinations": len(DESTS),
           "border_rule": "convex hull of published coordinates plus a %g-of-bounding-box band" % _ARGS.band,
           "band": _ARGS.band, "border_nodes": len(BORDER),
           "results": res}, open(OUT, "w"), indent=1)
print("wrote %s" % OUT.name)
