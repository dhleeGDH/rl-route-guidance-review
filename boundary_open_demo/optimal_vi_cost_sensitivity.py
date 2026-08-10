# -*- coding: utf-8 -*-
"""Is the optimal-policy completion of Section V-B sensitive to the expected edge cost?

WHY. The value iteration solves a deterministic expected-cost surrogate of an environment whose
edge cost is stochastic and time-varying, at 1 + 0.6 * U * (0.5 + 0.5 * sin(t/6)) with U drawn
uniform on [0, 1] once per episode. That surrogate needs one constant, and the two value-iteration
scripts in this folder disagreed on it. optimal_vi.py uses 1.15, the average of the cost over a
full period of the sinusoid. optimal_vi_boundary_dest.py used 1.0 + 0.6 * 0.5 = 1.3, which drops
the factor E[U] = 0.5 and is an error.

The size of the error is small and its direction favours the paper's claim, since a higher in-grid
cost makes exiting relatively more attractive. That is exactly the combination a referee should
not be asked to take on trust, so the constant is swept here across the whole range the
environment can produce rather than argued about.

The empirical mean edge cost depends on the horizon it is averaged over, since a trip of 3 to 8
links samples only part of the sinusoid: 1.19 over 4 steps, 1.23 over 8, 1.25 over 12, and 1.15
once the average covers a full period. The sweep therefore spans 1.00 to 1.60, from the minimum
edge cost to the maximum.

    python3 optimal_vi_cost_sensitivity.py

Writes optimal_vi_cost_sensitivity.csv next to this file.
"""
import csv
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import optimal_vi_boundary_dest as vi                      # noqa: E402
from train import make_eval_od                              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = [1.00, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40, 1.50, 1.60]


def completion(e_cost, ods, reward, boundary):
    vi.E_COST = e_cost                     # the module reads it at call time
    cache, ok = {}, 0
    for o, dl in ods:
        key = (dl, boundary)
        if key not in cache:
            cache[key] = vi.solve(dl, boundary, reward)
        V, phi, ol = cache[key]
        ok += vi.arrives(o, dl, V, phi, ol, reward)
    return 100.0 * ok / len(ods)


def main():
    ods = make_eval_od(n=200, seed=12345)
    rows = []
    print("Optimal-policy completion against the expected in-grid edge cost\n")
    print("  %-6s %10s %10s %10s %10s" % ("E_COST", "time/closed", "time/open",
                                          "algn/closed", "algn/open"))
    for e in GRID:
        vals = [completion(e, ods, r, b)
                for r in ("time_min", "aligned") for b in ("closed", "open")]
        rows.append([e] + vals)
        print("  %-6.2f %10.1f %10.1f %10.1f %10.1f" % (e, vals[0], vals[1], vals[2], vals[3]))

    out = os.path.join(HERE, "optimal_vi_cost_sensitivity.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["e_cost", "time_min_closed", "time_min_open",
                    "aligned_closed", "aligned_open"])
        w.writerows(rows)

    cols = list(zip(*[r[1:] for r in rows]))
    flat = all(len(set(c)) == 1 for c in cols)
    print("\n  every cell constant across the sweep: %s" % flat)
    print("  wrote %s" % os.path.basename(out))


main()
