# -*- coding: utf-8 -*-
"""The arrival ceiling a cordon imposes on an open-boundary evaluation.

WHY THIS EXISTS. Opening the boundary of a study area is the obvious remedy for an evaluation
that cannot see a policy's preference between arriving and leaving. It is not sufficient. Once
the cordon is open, every border node other than the destination absorbs a trip, so a pair can
arrive only where some route to its destination avoids the whole border set. The share of pairs
that keep such a route is the ceiling any completion rate on that network is measured against,
and it is a property of the network and the cordon rather than of a policy: no reward, no
learner and no simulation enter the computation.

The ceiling is not a constant. It falls as the cordon is drawn wider, and it differs between
networks at the same setting, so a completion rate measured under an opened cordon is not
comparable across study areas unless the ceiling is reported beside it.

Inputs are the published TNTP link files and node coordinates, through the same reader and the
same border definition exposure_ratio_real_networks.py uses.

    python3 arrival_ceiling_networks.py
"""
import json
import os
import sys
from pathlib import Path
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
# The reader and the network files sit at the root of the package, one level above this
# directory, which is where every script of this directory resolves them.
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import exposure_ratio_real_networks as X   # noqa: E402

# load_coords dispatches on the suffix, so the paths stay as Path objects
NETS = [("Anaheim", Path(ROOT) / "networks" / "Anaheim_net.tntp",
         Path(ROOT) / "networks" / "anaheim_nodes.geojson"),
        ("SiouxFalls", Path(ROOT) / "networks" / "SiouxFalls_net.tntp",
         Path(ROOT) / "networks" / "SiouxFalls_node.tntp")]
# Band 0 is the convex hull alone, the one border the coordinates fix without a free
# parameter. It anchors the range: a ceiling quoted only at chosen bands reports the choice
# as much as the network.
BANDS = (0.0, 0.05, 0.10, 0.20)
STUDY_BAND = 0.10          # the setting Section IV-A adopts


def reach_avoiding(dest, radj, blocked):
    """Nodes with a directed route to dest that enters no blocked node before it."""
    seen, q = {dest}, deque([dest])
    while q:
        u = q.popleft()
        for v in radj.get(u, ()):
            if v in seen or (v in blocked and v != dest):
                continue
            seen.add(v)
            q.append(v)
    return seen


def ceiling(net_path, coord_path, band):
    zones, nodes, links = X.read_tntp(net_path)
    coords = X.load_coords(coord_path)
    radj = {}
    for a, b in links:
        radj.setdefault(b, []).append(a)
    border = X.perimeter_nodes(coords, band=band)
    origins = [n for n in range(1, nodes + 1) if n not in border]
    pairs = free = 0
    for z in range(1, zones + 1):
        rf = reach_avoiding(z, radj, set())
        rb = reach_avoiding(z, radj, border)
        for o in origins:
            if o != z and o in rf:
                pairs += 1
                free += o in rb
    return {"band": band, "border_nodes": len(border), "zones": zones, "nodes": nodes,
            "pairs": pairs, "arrival_reachable": free,
            "ceiling_pct": round(100.0 * free / pairs, 1) if pairs else None}


def main():
    out = {}
    print("%-11s %-6s %-8s %-8s %-12s %s" %
          ("network", "band", "border", "pairs", "can arrive", "ceiling"))
    for name, netp, coordp in NETS:
        out[name] = []
        for band in BANDS:
            r = ceiling(netp, coordp, band)
            out[name].append(r)
            print("%-11s %-6.2f %-8d %-8d %-12d %.1f%%" %
                  (name, band, r["border_nodes"], r["pairs"],
                   r["arrival_reachable"], r["ceiling_pct"]))

    # regression witness: the study's own setting on Anaheim is the pair count and the ceiling
    # Section V-C reports, computed here from the published files rather than restated.
    a = next(r for r in out["Anaheim"] if r["band"] == STUDY_BAND)
    assert (a["pairs"], a["arrival_reachable"]) == (12183, 5692), \
        "Anaheim at the study band no longer gives 5,692 of 12,183: %s" % a
    print("\n  regression witness: Anaheim at band %.2f gives %d of %d, the pair set Section V-C "
          "scores" % (STUDY_BAND, a["arrival_reachable"], a["pairs"]))

    with open(os.path.join(HERE, "arrival_ceiling_networks.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("  wrote arrival_ceiling_networks.json")


if __name__ == "__main__":
    main()
