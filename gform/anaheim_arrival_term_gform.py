# -*- coding: utf-8 -*-
"""The Anaheim arrival term at 2.5x, and the attainable maximum of the same border, recomputed.

Section IV-B quotes 97.8% at the hull and 46.7% at the band for an arrival term
of 25 with an exit penalty of 12.5, and quotes the same two numbers as the attainable maximum of
those borders. The first pair is stored in the deposited value-iteration files and the second in
arrival_ceiling_networks.json, written by two different programs. This recomputes both in one
pass, so the coincidence of the two is checked rather than assumed. The deposited outputs are
read, never rewritten: the construction is obtained by exec'ing anaheim_vi_perimeter.py up to its
first statement, as anaheim_zone_geometry_gform.py does.

    python3 anaheim_arrival_term_gform.py
"""
import json
import os
import sys
from collections import deque

SRC = os.path.join(os.environ.get("DEPOSIT_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "anaheim_vi_perimeter.py")
HERE = os.path.dirname(os.path.abspath(__file__))


def load(band):
    sys.argv = ["anaheim_vi_perimeter.py", "--band", str(band)]
    sys.path.insert(0, os.path.dirname(SRC))
    src = open(SRC, encoding="utf-8").read()
    ns = {"__file__": SRC, "__name__": "anaheim_vi_perimeter_gform"}
    exec(compile(src[:src.index('print("Anaheim: %d nodes')], SRC, "exec"), ns)
    return ns


def reach_avoiding(dest, radj, blocked):
    seen, q = {dest}, deque([dest])
    while q:
        u = q.popleft()
        for v in radj.get(u, ()):
            if v in seen or (v in blocked and v != dest):
                continue
            seen.add(v)
            q.append(v)
    return seen


def run(band, label, r_goal, r_exit):
    ns = load(band)
    BORDER, ORIGINS, DESTS = ns["BORDER"], ns["ORIGINS"], ns["DESTS"], 
    RADJ, reachable = ns["RADJ"], ns["reachable"]
    ns["R_GOAL"], ns["R_EXIT"] = r_goal, r_exit      # solve() reads them from the module
    solve = ns["solve"]
    tot = {"arrive": 0, "exit": 0, "stuck": 0}
    pairs = free = 0
    for dest in DESTS:
        oc, reach = solve(dest, "open", "aligned"), reachable(dest)
        rb = reach_avoiding(dest, RADJ, BORDER)
        for o in ORIGINS:
            if o != dest and o in reach:
                tot[oc(o)] += 1
                pairs += 1
                free += o in rb
    out = {"band": band, "border_rule": label, "border_nodes": len(BORDER),
           "r_goal": r_goal, "r_exit": r_exit,
           "reward_units": "multiples of the mean link cost %.6f min" % ns["SCALE"],
           "pairs": pairs, "aligned_arrive": tot["arrive"], "aligned_exit": tot["exit"],
           "aligned_arrive_pct": round(100.0 * tot["arrive"] / pairs, 1),
           "arrival_reachable": free,
           "attainable_maximum_pct": round(100.0 * free / pairs, 1)}
    print("%-9s border %3d  pairs %6d | aligned at %.1f/%.1f arrives %6d (%.1f%%) | "
          "reachable %6d (%.1f%%)"
          % (label, len(BORDER), pairs, r_goal, r_exit, tot["arrive"],
             out["aligned_arrive_pct"], free, out["attainable_maximum_pct"]))
    return out


def main():
    res = {"hull_2.5x": run(0.0, "hull", 25.0, 12.5),
           "band_2.5x": run(0.10, "band 0.1", 25.0, 12.5)}
    json.dump(res, open(os.path.join(HERE, "anaheim_arrival_term_gform.json"), "w"), indent=1)
    print("wrote anaheim_arrival_term_gform.json")


if __name__ == "__main__":
    main()
