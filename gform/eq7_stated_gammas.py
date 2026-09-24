# -*- coding: utf-8 -*-
# The Eq. (7) right-hand side of the bespoke grid at every discount the
# corpus states, under the A-2 definition: Delta is the smallest positive gap in the DISCOUNTED
# travel-time return, K and k_min are the extreme transition counts of the two paths, and the
# denominator carries no Phi_max term. Table I records 38 stated values; their multiplicities are
# read from gamma_todo.csv. gamma = 1 is the case of Proposition 1 and is counted there.
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from grid_gform import part_a                                      # noqa: E402

STATED = [(0.0, 1), (0.1, 2), (0.5, 1), (0.7, 1), (0.9, 5), (0.95, 5), (0.99, 21), (1.0, 1)]


def main():
    rows, met = [], 0
    print("%-8s %6s %10s %4s %6s %12s  %s"
          % ("gamma", "studies", "Delta", "K", "k_min", "Eq.(7) RHS", "gamma > RHS"))
    for g, n in STATED:
        if g == 1.0:
            rows.append({"gamma": g, "studies": n, "case": "Proposition 1, undiscounted"})
            met += n
            print("%-8s %6d %10s %4s %6s %12s  %s"
                  % (g, n, "-", "-", "-", "-", "Proposition 1"))
            continue
        r = part_a(g)
        r["studies"] = n
        rows.append(r)
        if r["delta"] is None:
            # no positive gap is measurable: at this discount the greedy travel-time route does
            # not complete from every node, so the comparison Eq. (7) bounds does not arise.
            print("%-8s %6d %10s %4s %6s %12s  %s"
                  % (g, n, "none", "-", "-", "-", "not measurable"))
            continue
        ok = g > r["eq7_rhs"]
        met += n if ok else 0
        print("%-8s %6d %10.6f %4d %6d %12.6f  %s"
              % (g, n, r["delta"], r["K"], r["k_min"], r["eq7_rhs"], ok))
    print("\nstated values meeting the condition: %d of %d" % (met, sum(n for _, n in STATED)))
    json.dump({"rows": rows, "met": met, "total": sum(n for _, n in STATED)},
              io.open(os.path.join(HERE, "eq7_stated_gammas.json"), "w", encoding="utf-8"),
              indent=1)


if __name__ == "__main__":
    main()
