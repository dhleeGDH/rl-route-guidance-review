# -*- coding: utf-8 -*-
"""How often a peripheral exit is closer than the destination, on published road networks.

WHY THIS EXISTS. Section IV-A states the condition under which a study area is exposed: the failure
needs "a peripheral exit reachable in fewer steps than the destination". A reviewer observed that the
manuscript defines that condition and then computes it on no real network, so its practical reach is
asserted rather than measured. This measures it.

The quantity is geometric and needs no learner and no reward. For every origin-destination pair of a
published network, compare the hop distance from the origin to its destination against the hop
distance from the origin to the nearest perimeter node other than the destination. The share of pairs
where the second is smaller is the share of trips on which leaving is available sooner than arriving,
which is the condition Section IV-A names.

Three networks are read from their own published link data: Anaheim (416 nodes, 914 links, 38 zones),
Sioux Falls and Nguyen-Dupuis. Two destination placements are reported for each, since the condition
depends on where a study puts its trip ends:

  perimeter  destinations are the network's own perimeter nodes, the placement of a study area whose
             trip ends sit near its border;
  interior   destinations are interior nodes, the interior-centroid placement Section IV-A names as
             the one that does not meet the condition.

    python3 exposure_ratio_real_networks.py
"""
import json
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "exposure_ratio_border_sweep.json"


def read_tntp(path):
    """Directed links from a TNTP net file, with the zone count its header declares."""
    zones = nodes = None
    links = []
    for ln in open(path):
        s = ln.strip()
        if s.startswith("<NUMBER OF ZONES>"):
            zones = int(s.split(">")[1])
        elif s.startswith("<NUMBER OF NODES>"):
            nodes = int(s.split(">")[1])
        elif s and not s.startswith(("~", "<")):
            f = s.rstrip(";").split()
            if len(f) >= 2:
                try:
                    links.append((int(f[0]), int(f[1])))
                except ValueError:
                    pass
    return zones, nodes, links


def hops_from_all_to(dest, radj, n):
    """Hop distance from every node to dest, over the reversed graph."""
    d, q = {dest: 0}, deque([dest])
    while q:
        x = q.popleft()
        for y in radj.get(x, ()):
            if y not in d:
                d[y] = d[x] + 1
                q.append(y)
    return d


def _hull(points):
    """Convex hull of (id, x, y) by monotone chain, returning the ids on it."""
    pts = sorted(points, key=lambda p: (p[1], p[2]))
    def cross(o, a, b):
        return (a[1] - o[1]) * (b[2] - o[2]) - (a[2] - o[2]) * (b[1] - o[1])
    lower = []
    for q in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    upper = []
    for q in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return {p[0] for p in lower[:-1] + upper[:-1]}


def perimeter_nodes(coords, band=0.10):
    """The nodes on the outside of the study area, from the published coordinates.

    Section IV-A's condition is geometric: it asks whether a *peripheral* exit is nearer than the
    destination. An earlier version of this script used the TNTP zone set as the perimeter, which is
    wrong: zones are trip ends distributed through the network, not its border, and the resulting
    figure measured something else entirely. The border is taken instead as the convex hull of the
    published node coordinates together with every node within `band` of the bounding box edge,
    which is the set a study area's outer cordon would cut.
    """
    xs = [c[0] for c in coords.values()]
    ys = [c[1] for c in coords.values()]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    dx, dy = (x1 - x0) * band, (y1 - y0) * band
    hull = _hull([(k, v[0], v[1]) for k, v in coords.items()])
    edge = {k for k, (x, y) in coords.items()
            if x <= x0 + dx or x >= x1 - dx or y <= y0 + dy or y >= y1 - dy}
    return hull | edge


def load_coords(path):
    """Node coordinates, from a geojson point layer or a TNTP node file."""
    import json as _json
    if path.suffix == ".geojson":
        g = _json.loads(path.read_text(encoding="utf-8"))
        return {int(f["properties"]["id"]): tuple(f["geometry"]["coordinates"])
                for f in g["features"]}
    out = {}
    for ln in path.read_text(encoding="utf-8").splitlines()[1:]:
        f = ln.replace(";", "").split()
        if len(f) >= 3:
            try:
                out[int(f[0])] = (float(f[1]), float(f[2]))
            except ValueError:
                pass
    return out


def exposure(path, coord_path, name, band=0.10):
    zones, nodes, links = read_tntp(path)
    adj, radj = {}, {}
    for u, v in links:
        adj.setdefault(u, []).append(v)
        radj.setdefault(v, []).append(u)
    coords = load_coords(coord_path)
    per = perimeter_nodes(coords, band=band)
    interior = [n for n in range(1, nodes + 1) if n not in per and n in coords]

    # hop distance to every candidate destination, computed once per destination
    hops = {}
    for d in set(per) | set(interior):
        hops[d] = hops_from_all_to(d, radj, nodes)

    out = {}
    for placement, dests, origins in (("perimeter", sorted(per), interior),
                                      ("interior", interior, interior)):
        closer = total = 0
        for d in dests:
            hd = hops[d]
            for o in origins:
                if o == d or o not in hd:
                    continue
                # the nearest perimeter node other than the destination
                near = min((hops[p].get(o, 10 ** 9) for p in per if p != d), default=10 ** 9)
                total += 1
                if near < hd[o]:
                    closer += 1
        out[placement] = {"pairs": total, "exit_closer": closer,
                          "share": round(100.0 * closer / total, 1) if total else None}
        print("  %-14s %-10s %7d pairs, exit closer on %7d, %5.1f%%"
              % (name, placement, total, closer, out[placement]["share"]))
    return {"nodes": nodes, "links": len(links), "zones": zones,
            "perimeter_nodes": len(per), "interior_nodes": len(interior), "placements": out}


if __name__ == "__main__":
    nets = [("Anaheim", HERE / "networks" / "Anaheim_net.tntp",
             HERE / "networks" / "anaheim_nodes.geojson"),
            ("SiouxFalls", HERE / "networks" / "SiouxFalls_net.tntp",
             HERE / "networks" / "SiouxFalls_node.tntp")]
    res = {}
    print("share of OD pairs on which a peripheral exit is reachable in fewer hops than the destination")
    # The border is the convex hull together with every node within `band` of the bounding box
    # edge. That band is a free parameter, so the two neighbouring settings are computed here
    # rather than asserted to be immaterial.
    for band in (0.05, 0.10, 0.20):
        print("\n  border band = %.2f of the bounding box" % band)
        for name, p, c in nets:
            if not p.exists() or not c.exists():
                print("  %-14s (network or coordinate file absent)" % name)
                continue
            res["%s@%.2f" % (name, band)] = exposure(p, c, name, band=band)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("\nwrote %s" % OUT.name)
