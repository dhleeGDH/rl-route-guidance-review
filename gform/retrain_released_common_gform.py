# -*- coding: utf-8 -*-
"""The published router retrained on three boundary conditions over ONE evaluation set.

WHY THIS EXISTS. retrain_released_10seed.json and retrain_released_hull_10seed.json each draw
their thirty origins from a pool that excludes their own border, so the two runs differ in the
border AND in the evaluation set, and the boundary-closed arm carries two values, 94.0 and 96.0.
A completion rate read across the three conditions then cannot be attributed to the border alone.

This fixes the pool. The origins are drawn once, from the nodes outside the 95-node band, and the
same thirty serve all three conditions: boundary-closed, the 13-node convex hull and the 95-node
band. The convex hull is a subset of the band, so no origin of the draw lies on either border.

WHAT IS UNCHANGED. The released implementation is imported and not edited, as in
retrain_released.py: the map, the reward, the C51 update, the exploration, the walk and the
boundary-open subclass are that file's. The destination (node 15), the seed count (10), the
iteration count (3000), the link costs, the termination handling and the thirty-origin size are
the settings of the two deposited runs. Only the pool derivation changes, and the three
conditions are run from one draw.

    python3 retrain_released_common_gform.py --iters 3000 --seeds 10 --workers 15
"""
import os

# pinned before numpy is imported anywhere below, so each worker holds one linear-algebra thread
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
IMPL = Path("/home/dhlee/review_paper/handoff/experiments/released_impl_boundary_open")
sys.path.insert(0, str(IMPL))
sys.argv_backup, sys.argv = sys.argv, [sys.argv[0]]
import retrain_released as R          # noqa: E402  the harness of the deposited runs
sys.argv = sys.argv_backup

DEST, N_ORIGINS, SEED, MAP = 15, 30, 12345, 2
COORD = Path(R.REPO) / "Maps" / "Anaheim" / "anaheim_nodes.geojson"
CONDITIONS = ("closed", "open_hull", "open_band")


def borders():
    return (R.border_nodes(COORD, band=0.0), R.border_nodes(COORD, band=0.10))


def common_origins(n_node):
    """The draw shared by the three conditions: the pool excludes the wider of the two borders,
    so no origin sits on either border and the same thirty are scored under all three."""
    _hull, band = borders()
    rng = np.random.RandomState(SEED)
    pool = [x for x in range(1, n_node + 1) if x != DEST and x not in band]
    return sorted(rng.choice(pool, size=min(N_ORIGINS, len(pool)), replace=False).tolist()), pool


def ceiling(m, origins, absorbing):
    """Pairs whose origin keeps a route to the destination entering no other border node."""
    adj = {}
    for u, v in m.G.edges():
        adj.setdefault(int(u), []).append(int(v))
    block = {int(e) for e in absorbing} - {DEST}
    out = []
    for o in origins:
        if o in block:
            out.append(False)
            continue
        seen, q, ok = {o}, deque([o]), False
        while q and not ok:
            u = q.popleft()
            for v in adj.get(u, []):
                if v == DEST:
                    ok = True
                    break
                if v in block or v in seen:
                    continue
                seen.add(v)
                q.append(v)
        out.append(ok)
    return out


def one_job(args):
    condition, seed, iters, origins = args
    t0 = time.time()
    hull, band = borders()
    m = R.build(MAP, DEST)
    kind = "closed" if condition == "closed" else "open"
    exits = {"closed": set(), "open_hull": hull, "open_band": band}[condition]
    r = R.run(kind, m, DEST, origins, iters, exits, seed)
    return condition, seed, r["arrive_rate"], dict(r["counts"]), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--out", default=str(HERE / "retrain_released_common_gform.json"))
    a = ap.parse_args()

    hull, band = borders()
    m = R.build(MAP, DEST)
    origins, pool = common_origins(m.n_node)
    print("network %d nodes, %d links | hull %d, band %d | pool %d | origins %s"
          % (m.n_node, m.n_link, len(hull), len(band), len(pool), origins), flush=True)
    reach = {"closed": ceiling(m, origins, set()),
             "open_hull": ceiling(m, origins, hull),
             "open_band": ceiling(m, origins, band)}
    out = {"study": "[25] DRL-Router, released implementation, unmodified",
           "network": "its own shipped Anaheim, 416 nodes",
           "destination": DEST, "origins": origins, "n_origins": len(origins),
           "origin_pool": "nodes outside the 95-node band, drawn once at RandomState(%d)" % SEED,
           "pool_size": len(pool), "iterations": a.iters, "seeds": a.seeds,
           "border_nodes": {"closed": 0, "open_hull": len(hull), "open_band": len(band)},
           "attainable_maximum": {}, "results": {}}
    for c in CONDITIONS:
        ok = sum(reach[c])
        out["attainable_maximum"][c] = {
            "reachable": ok, "pairs": len(origins),
            "pct": round(100.0 * ok / len(origins), 1),
            "unreachable_origins": [o for o, r in zip(origins, reach[c]) if not r]}
        print("  ceiling %-10s %5.1f%% (%d/%d) unreachable %s"
              % (c, 100.0 * ok / len(origins), ok, len(origins),
                 out["attainable_maximum"][c]["unreachable_origins"]), flush=True)

    jobs = [(c, s, a.iters, origins) for c in CONDITIONS for s in range(a.seeds)]
    acc = {c: {"per_seed": [None] * a.seeds, "counts_per_seed": [None] * a.seeds,
               "seconds": [None] * a.seeds} for c in CONDITIONS}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for condition, seed, rate, counts, secs in ex.map(one_job, jobs):
            acc[condition]["per_seed"][seed] = rate
            acc[condition]["counts_per_seed"][seed] = counts
            acc[condition]["seconds"][seed] = secs
            print("  %-10s seed %d: arrive %5.1f%%  %s  [%.0fs]"
                  % (condition, seed, rate, counts, secs), flush=True)
    for c in CONDITIONS:
        v = np.asarray(acc[c]["per_seed"], float)
        n = float(sum(sum(x.values()) for x in acc[c]["counts_per_seed"])) or 1.0
        acc[c]["mean"] = float(v.mean())
        acc[c]["sd"] = float(v.std())
        acc[c]["exit_rate"] = 100.0 * sum(x.get("exit", 0) for x in acc[c]["counts_per_seed"]) / n
        acc[c]["stuck_rate"] = 100.0 * sum(x.get("stuck", 0) for x in acc[c]["counts_per_seed"]) / n
        acc[c]["trips"] = int(n)
        out["results"][c] = acc[c]
        print("%-10s mean %5.1f%%  exit %5.1f%%  stuck %4.1f%%"
              % (c, acc[c]["mean"], acc[c]["exit_rate"], acc[c]["stuck_rate"]))
    out["wall_seconds"] = time.time() - t0
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("wrote %s in %.0f s" % (os.path.basename(a.out), out["wall_seconds"]))


if __name__ == "__main__":
    main()
