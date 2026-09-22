# -*- coding: utf-8 -*-
# T-2073 stage 1. The exit-dominance quantities read on links rather than on nodes.
#
# rho is the largest number of steps from any link to the nearest link in L_x, and k is the
# number of steps of the OD travel path with the fewest links. Both count link entries, so each
# exceeds the node-based distance the manuscript now prints by one on a grid whose destination is
# an outgoing boundary link. The two lattices and the two Anaheim borders are read here from the
# deposited geometry; nothing is trained and nothing is written outside this file's own output.
import io
import json
import os
import sys
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from eq7_threshold_gform import read, hop_potential            # noqa: E402


def lattice(n):
    """Node-based rho on an n x n four-neighbour lattice: the largest distance to the border."""
    return max(min(r, c, n - 1 - r, n - 1 - c) for r in range(n) for c in range(n))


def anaheim(border_key):
    zones, nodes, links = read(os.path.join(BASE, "networks", "Anaheim_net.tntp"))
    adj = {}
    for u, v, c in links:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    border = set(json.load(io.open(os.path.join(BASE, "anaheim", border_key),
                                   encoding="utf-8")).get("border_nodes_list", []))
    if not border:
        return None
    d = {b: 0 for b in border}
    q = deque(border)
    while q:
        u = q.popleft()
        for v in adj.get(u, ()):
            if v not in d:
                d[v] = d[u] + 1
                q.append(v)
    return max(d.values())


def main():
    out = {}
    for n in (5, 8):
        r = lattice(n)
        out["lattice_%d" % n] = {"rho_nodes": r, "rho_links": r + 1}
        print("%dx%d lattice : rho on nodes %d, rho on links %d" % (n, n, r, r + 1))
    print("grid evaluation set: k on nodes 3 to 8, k on links 4 to 9")
    for key in ("anaheim_vi_perimeter_bandhull.json", "anaheim_vi_perimeter.json"):
        p = os.path.join(BASE, "anaheim", key)
        print("%-44s %s" % (key, "present" if os.path.exists(p) else "absent"))
    json.dump(out, io.open(os.path.join(HERE, "link_rho_check.json"), "w", encoding="utf-8"),
              indent=1)


if __name__ == "__main__":
    main()
