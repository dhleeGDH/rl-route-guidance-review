# -*- coding: utf-8 -*-
"""Where the 38 Anaheim destination zones sit relative to each border, and what the travel-time
optimum does on the pairs whose destination is an interior zone.

WHY THIS EXISTS. anaheim_vi_perimeter.json and anaheim_vi_perimeter_bandhull.json report the
optimum of each reward over all pairs and the spread of the exit rate across destinations, but
neither records which of the 38 published zones lies on the border it was solved with. A
destination on the border and a destination inside the network are different geometries, and the
rewritten Section IV-B states the two separately.

The construction is the deposited one: this file execs anaheim_vi_perimeter.py up to its first
statement, which gives the published border rule, cost table, origin set, destination set,
solve() and reachable() exactly as deposited, and then accounts the outcomes by destination
zone. Nothing is written by the exec'd prefix and no deposited output is touched.

    python3 anaheim_zone_geometry_gform.py
"""
import json
import os
import sys

SRC = "/home/dhlee/review_paper/handoff/deposit_next/v1.8.0/anaheim_vi_perimeter.py"
HERE = os.path.dirname(os.path.abspath(__file__))


def load(band):
    """The deposited module up to its first executable statement, at the given border band."""
    sys.argv = ["anaheim_vi_perimeter.py", "--band", str(band)]
    sys.path.insert(0, os.path.dirname(SRC))
    src = open(SRC, encoding="utf-8").read()
    cut = src.index('print("Anaheim: %d nodes')
    ns = {"__file__": SRC, "__name__": "anaheim_vi_perimeter_gform"}
    exec(compile(src[:cut], SRC, "exec"), ns)
    return ns


def run(band, label):
    ns = load(band)
    BORDER, ORIGINS, DESTS = ns["BORDER"], ns["ORIGINS"], ns["DESTS"]
    solve, reachable = ns["solve"], ns["reachable"]
    on_border = sorted(z for z in DESTS if z in BORDER)
    interior = sorted(z for z in DESTS if z not in BORDER)
    out = {"band": band, "border_rule": label, "border_nodes": len(BORDER),
           "origins": len(ORIGINS), "zones": len(DESTS),
           "zones_on_border": len(on_border), "zones_interior": len(interior),
           "zones_on_border_list": on_border, "zones_interior_list": interior,
           "r_goal_units": "R_GOAL x mean link cost", "mean_link_cost": ns["SCALE"],
           "by_group": {}}
    print("== %s : %d border nodes, %d zones on the border, %d zones inside"
          % (label, len(BORDER), len(on_border), len(interior)))
    for reward in ("time_min", "aligned"):
        grp = {"border_dest": {"arrive": 0, "exit": 0, "stuck": 0},
               "interior_dest": {"arrive": 0, "exit": 0, "stuck": 0}}
        per_zone = {}
        for dest in DESTS:
            oc, reach = solve(dest, "open", reward), reachable(dest)
            z = {"arrive": 0, "exit": 0, "stuck": 0}
            for o in ORIGINS:
                if o != dest and o in reach:
                    z[oc(o)] += 1
            per_zone[dest] = z
            key = "border_dest" if dest in BORDER else "interior_dest"
            for k in z:
                grp[key][k] += z[k]
        for key in ("border_dest", "interior_dest"):
            n = sum(grp[key].values())
            grp[key]["pairs"] = n
            grp[key]["exit_pct"] = round(100.0 * grp[key]["exit"] / n, 1) if n else None
            grp[key]["arrive_pct"] = round(100.0 * grp[key]["arrive"] / n, 1) if n else None
            print("   open %-9s %-13s pairs %6d  exit %6d (%5.1f%%)  arrive %6d (%5.1f%%)"
                  % (reward, key, n, grp[key]["exit"], grp[key]["exit_pct"],
                     grp[key]["arrive"], grp[key]["arrive_pct"]))
        out["by_group"][reward] = grp
        out.setdefault("per_zone_open_%s" % reward, {str(k): v for k, v in per_zone.items()})
    return out


def main():
    res = {"hull": run(0.0, "convex hull of the published coordinates, 13 nodes"),
           "band": run(0.10, "the hull plus a tenth-of-bounding-box band, 95 nodes")}
    p = os.path.join(HERE, "anaheim_zone_geometry_gform.json")
    json.dump(res, open(p, "w"), indent=1)
    print("wrote %s" % os.path.basename(p))


if __name__ == "__main__":
    main()
