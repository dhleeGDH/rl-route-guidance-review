# -*- coding: utf-8 -*-
"""The attainable maximum of Sioux Falls at the border its own publication draws.

WHY THIS EXISTS. The manuscript calibrates the coordinate proxy for a border set against the one
published network carrying a drawn outer face, and quotes the calibration as a 13-point
overstatement. No script produced the two figures behind it. This computes both from the
published files through the same reader, the same border-free pair enumeration and the same
reachability test attainable_maximum_networks.py uses, so the proxy row and the outer-face row
differ in the border set alone.

THE OUTER FACE. It is derived, not listed. Sioux Falls is published as a planar straight-line
drawing, and the face of such a drawing containing infinity is traced by starting at the lowest
node and taking the most clockwise neighbour at every step. The trace is accepted only where the
polygon it closes contains every remaining node, which is the defining property of the outer face
and fails on any mistraced walk.

    python3 outer_face_sioux.py
"""
import json
import math
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
# The reader and the network files sit at the root of the package, one level above this
# directory, which is where every script of this directory resolves them.
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import exposure_ratio_real_networks as X            # noqa: E402
import attainable_maximum_networks as A             # noqa: E402

NET = Path(ROOT) / "networks" / "SiouxFalls_net.tntp"
COORD = Path(ROOT) / "networks" / "SiouxFalls_node.tntp"


def outer_face(coords, links):
    """The outer face of the published planar drawing, from its rotation system.

    Every face is traced by the standard rule: from half-edge (u, v) the next half-edge is (v, w)
    with w the neighbour clockwise-adjacent to u in the angular order around v. The trace is
    accepted only where the face count satisfies Euler's formula, which fails on any drawing the
    published coordinates do not embed planarly, and the outer face is then the unique face whose
    signed area runs against every other. An earlier greedy walk over turn angles returned the
    same node set in an order crossing itself, and no check caught it.
    """
    und = set()
    for a, b in links:
        und.add((min(a, b), max(a, b)))
    adj = {}
    for a, b in und:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    rot = {v: sorted(ns, key=lambda w: math.atan2(coords[w][1] - coords[v][1],
                                                  coords[w][0] - coords[v][0]))
           for v, ns in adj.items()}
    seen, faces = set(), []
    for a, b in list(und) + [(b, a) for a, b in und]:
        if (a, b) in seen:
            continue
        face, cur = [], (a, b)
        while cur not in seen:
            seen.add(cur)
            face.append(cur[0])
            u, v = cur
            r = rot[v]
            cur = (v, r[(r.index(u) - 1) % len(r)])
        faces.append(face)
    euler = 2 - len(coords) + len(und)
    assert len(faces) == euler, \
        "the drawing is not planar under its published coordinates: %d faces against %d" \
        % (len(faces), euler)

    def area(f):
        p = [coords[k] for k in f]
        m = len(p)
        return sum(p[i][0] * p[(i + 1) % m][1] - p[(i + 1) % m][0] * p[i][1] for i in range(m)) / 2
    outer = [f for f in faces if area(f) < 0]
    assert len(outer) == 1, "the orientation does not single out one face: %d candidates" % len(outer)
    return outer[0]


def inside(poly, p):
    """True where p lies inside or on the closed polygon."""
    x, y = p
    n, hit = len(poly), False
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if min(x0, x1) - 1e-9 <= x <= max(x0, x1) + 1e-9 and \
           min(y0, y1) - 1e-9 <= y <= max(y0, y1) + 1e-9 and \
           abs((x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)) < 1e-6:
            return True
        if (y0 > y) != (y1 > y):
            xi = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x < xi:
                hit = not hit
    return hit


def attainable(border):
    """A.ceiling's computation at an explicitly given border set."""
    zones, nodes, links = X.read_tntp(NET)
    radj = {}
    for a, b in links:
        radj.setdefault(b, []).append(a)
    origins = [n for n in range(1, nodes + 1) if n not in border]
    pairs = free = 0
    for z in range(1, zones + 1):
        rf = A.reach_avoiding(z, radj, set())
        rb = A.reach_avoiding(z, radj, border)
        for o in origins:
            if o != z and o in rf:
                pairs += 1
                free += o in rb
    return {"border_nodes": len(border), "pairs": pairs, "arrival_reachable": free,
            "pct": round(100.0 * free / pairs, 1) if pairs else None,
            "border": sorted(border)}


def main():
    zones, nodes, links = X.read_tntp(NET)
    coords = X.load_coords(COORD)
    face = outer_face(coords, links)
    poly = [coords[n] for n in face]
    outside = [n for n in coords if n not in face and not inside(poly, coords[n])]
    print("outer face traced: %d nodes %s" % (len(face), sorted(face)))
    print("  nodes left outside the traced polygon: %s" % (outside or "none"))
    assert not outside, "the walk is not the outer face; %s lie outside it" % outside

    hull = X.perimeter_nodes(coords, band=0.0)
    print("  coordinate hull: %d nodes %s" % (len(hull), sorted(hull)))
    assert set(hull) <= set(face), \
        "the hull is not a subset of the outer face: %s" % sorted(set(hull) - set(face))

    res = {"hull": attainable(set(hull)), "outer_face": attainable(set(face))}
    print()
    print("%-12s %-8s %-8s %-12s %s" % ("border", "nodes", "pairs", "can arrive", "attainable"))
    for k in ("hull", "outer_face"):
        r = res[k]
        print("%-12s %-8d %-8d %-12d %.1f%%" %
              (k, r["border_nodes"], r["pairs"], r["arrival_reachable"], r["pct"]))
    gap = round(res["hull"]["pct"] - res["outer_face"]["pct"], 1)
    res["overstatement_points"] = gap
    print("\n  the proxy overstates by %.1f points at its own pair set" % gap)

    # The two shares stand over different pair sets, since the wider border removes origins.
    # The overstatement is therefore also reported over the pair set the wider border leaves,
    # where the two borders are compared on one population.
    zones, nodes, links = X.read_tntp(NET)
    radj = {}
    for a, b in links:
        radj.setdefault(b, []).append(a)
    common = []
    for z in range(1, zones + 1):
        rf = A.reach_avoiding(z, radj, set())
        for o in range(1, nodes + 1):
            if o != z and o not in set(face) and o in rf:
                common.append((o, z))
    for k, border in (("hull", set(hull)), ("outer_face", set(face))):
        free = 0
        for z in set(d for _, d in common):
            rb = A.reach_avoiding(z, radj, border)
            free += sum(1 for o, d in common if d == z and o in rb)
        res[k]["pct_common_set"] = round(100.0 * free / len(common), 1)
        print("  %-10s on the %d pairs the wider border leaves: %.1f%%"
              % (k, len(common), res[k]["pct_common_set"]))
    res["overstatement_points_common_set"] = round(
        res["hull"]["pct_common_set"] - res["outer_face"]["pct_common_set"], 1)
    print("  overstatement on one pair set: %.1f points" % res["overstatement_points_common_set"])

    with open(os.path.join(HERE, "outer_face_sioux.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("\n  wrote outer_face_sioux.json")


if __name__ == "__main__":
    main()
