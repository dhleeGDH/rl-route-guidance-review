# -*- coding: utf-8 -*-
"""Seed dispersion for the Section V-B planner comparison.

state_condition_demo.py runs one draw of 200 OD pairs and one congestion realization, so the
figures it prints carry no dispersion. The manuscript reported them bare. This repeats the same
computation over independent seeds and reports the spread, so the state-cost numbers are stated
on the same footing as the completion rates elsewhere in Section V.

Nothing about the comparison changes: same grid, same cost model, same two planners, same
minimum OD separation. Only the seed varies.
"""
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from state_condition_demo import (N, cost, snapshot_route, forecast_route,   # noqa: E402
                                  realized, planned_snapshot)

N_OD = 200
MIN_SEP = 3
SEEDS = list(range(12345, 12355))          # ten seeds, the first is the published run


def one_seed(seed):
    rng = np.random.RandomState(seed)
    snap, fc, eta, slower, n = [], [], [], 0, 0
    while n < N_OD:
        o = (rng.randint(N), rng.randint(N))
        d = (rng.randint(N), rng.randint(N))
        if abs(o[0] - d[0]) + abs(o[1] - d[1]) < MIN_SEP:
            continue
        field = rng.rand(N, N)
        phase = rng.uniform(0, 2 * np.pi, size=(N, N))
        rs = snapshot_route(o, d, field, phase)
        rf = forecast_route(o, d, field, phase)
        sr, fr = realized(rs, field, phase), realized(rf, field, phase)
        sp = planned_snapshot(rs, field, phase)
        snap.append(sr); fc.append(fr); eta.append(abs(sr - sp))
        slower += sr > fr + 1e-6
        n += 1
    snap, fc, eta = np.array(snap), np.array(fc), np.array(eta)
    per_trip = 100.0 * eta / snap          # per-trip ETA error, % of the trip's realized time
    return (100 * (snap - fc).mean() / fc.mean(),      # snapshot excess, %
            100 * eta.mean() / snap.mean(),            # snapshot ETA error, %
            100 * slower / N_OD,                       # strictly slower, %
            per_trip)


def main():
    out = [one_seed(s) for s in SEEDS]
    rows = np.array([r[:3] for r in out])
    trips = np.concatenate([r[3] for r in out])
    names = ["snapshot excess (%)", "snapshot ETA error (%)", "strictly slower (%)"]
    print("Section V-B planner comparison over %d seeds, %d OD pairs each\n" % (len(SEEDS), N_OD))
    for i, name in enumerate(names):
        v = rows[:, i]
        print("  %-26s mean %5.1f   sd %4.1f   min %5.1f   max %5.1f"
              % (name, v.mean(), v.std(), v.min(), v.max()))
    print("\n  seed 12345 (the published run): " +
          ", ".join("%.1f" % x for x in rows[0]))
    # The manuscript needs the distribution, not the mean. A snapshot planner commits to a route
    # and quotes its cost, so the quantity a reliability-sensitive service cares about is the tail
    # of that quoted cost's error, not its average.
    q = np.percentile(trips, [50, 75, 90, 95, 99])
    print("\n  per-trip ETA error over %d trips (%d seeds x %d OD pairs)"
          % (trips.size, len(SEEDS), N_OD))
    print("    mean %5.1f%%   median %5.1f%%   p75 %5.1f%%   p90 %5.1f%%   p95 %5.1f%%   p99 %5.1f%%"
          % (trips.mean(), q[0], q[1], q[2], q[3], q[4]))
    print("    share of trips above 10%%: %5.1f%%      above 25%%: %5.1f%%"
          % (100.0 * (trips > 10).mean(), 100.0 * (trips > 25).mean()))
    import json
    with open(os.path.join(HERE, "state_condition_seeds.json"), "w", encoding="utf-8") as f:
        json.dump({"seeds": len(SEEDS), "od_pairs_per_seed": N_OD, "trips": int(trips.size),
                   "per_seed_snapshot_excess_pct": rows[:, 0].tolist(),
                   "per_seed_eta_error_pct": rows[:, 1].tolist(),
                   "per_seed_strictly_slower_pct": rows[:, 2].tolist(),
                   "eta_error_pct": {"mean": float(trips.mean()), "median": float(q[0]),
                                     "p75": float(q[1]), "p90": float(q[2]),
                                     "p95": float(q[3]), "p99": float(q[4]),
                                     "share_above_10pct": float(100.0 * (trips > 10).mean()),
                                     "share_above_25pct": float(100.0 * (trips > 25).mean())}}, f, indent=2)
    print("  wrote state_condition_seeds.json")


main()


if __name__ == "__main__":
    main()
