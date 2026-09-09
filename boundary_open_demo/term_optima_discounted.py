# -*- coding: utf-8 -*-
"""Table IX's optimum column at the learner's discount.

WHY THIS EXISTS. Section V-C justifies reporting undiscounted optima by an argument that holds
for the pure travel-time reward alone: every step reward there is negative, so discounting works
against exit dominance. Eq. (4) is not of that form. It carries an arrival term of 10 at the
terminal, an exit charge of 5, and a signed shaping term, so the argument does not transfer to
the reward-term arms of Table IX. The exit-penalty row is the exposed one: 62.5% is a marginal
comparison between a charge paid within a few steps and a trip cost spread over up to eight, and
that margin is exactly what a discount moves.

This answers it by running the SAME exact value iteration at gamma = 0.99, through the same
solve() and arrives() the published column came from, with gamma threaded through both halves.

CONTROL. At gamma = 1.0 the script must reproduce the published Table IX optimum column exactly.
The run asserts it before any discounted figure is printed, so a discounted number can never be
read off a solver that has drifted from the one the manuscript reports.

    python3 term_optima_discounted.py
"""
import io, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from env import N_SIDE                                        # noqa: E402
from env_boundary_dest import make_eval_od                    # noqa: E402
from optimal_vi_boundary_dest import solve, arrives           # noqa: E402

# The rows of Table IX, by the weights each names. "time_min" is any reward that is not the
# aligned one, which is how solve() reads the travel-time objective.
ROWS = [
    ("Travel time alone",       dict(reward="time_min")),
    ("Potential shaping alone", dict(reward="aligned", beta=1.0, r_goal=0.0,  r_exit=0.0)),
    ("Exit penalty alone",      dict(reward="aligned", beta=0.0, r_goal=0.0,  r_exit=5.0)),
    ("Arrival term alone",      dict(reward="aligned", beta=0.0, r_goal=10.0, r_exit=0.0)),
    ("All three terms",         dict(reward="aligned", beta=1.0, r_goal=10.0, r_exit=5.0)),
]
# the published column, which the control must reproduce at gamma = 1.0
PUBLISHED = {"Travel time alone": 0.0, "Potential shaping alone": 4.0,
             "Exit penalty alone": 62.5, "Arrival term alone": 100.0,
             "All three terms": 100.0}
GAMMAS = [1.0, 0.99, 0.95, 0.90]


def completion(terms, gamma, eval_od):
    """The harness of arrival_magnitude_sweep.py, which produced the published column: the
    destination IS an outgoing boundary link in this eval set, not a node to be converted."""
    kw = {k: v for k, v in terms.items() if k != "reward"}
    cache, ok = {}, 0
    for o, dl in eval_od:
        if dl not in cache:
            cache[dl] = solve(dl, "open", terms["reward"], gamma=gamma, **kw)
        V, phi, open_links = cache[dl]
        ok += arrives(o, dl, V, phi, open_links, terms["reward"], gamma=gamma, **kw)
    return round(100.0 * ok / len(eval_od), 1)


def main():
    eval_od = make_eval_od(n=200, seed=12345)
    print("evaluation set: %d pairs on the %dx%d boundary-open grid\n" % (
        len(eval_od), N_SIDE, N_SIDE))

    out, failed = {}, []
    print("%-24s %s" % ("Reward", "  ".join("%7s" % ("gamma=%.2f" % g) for g in GAMMAS)))
    for name, terms in ROWS:
        vals = [completion(terms, g, eval_od) for g in GAMMAS]
        out[name] = dict(zip(["%.2f" % g for g in GAMMAS], vals))
        print("%-24s %s" % (name, "  ".join("%7.1f" % v for v in vals)))
        if abs(vals[0] - PUBLISHED[name]) > 0.05:
            failed.append("%s: gamma=1.0 gives %.1f where Table IX prints %.1f"
                          % (name, vals[0], PUBLISHED[name]))

    print("\n--- control: gamma = 1.0 against the published Table IX column ---")
    for name, _ in ROWS:
        v = out[name]["1.00"]
        print("  %-4s %-24s %5.1f against %5.1f" %
              ("OK" if abs(v - PUBLISHED[name]) <= 0.05 else "FAIL", name, v, PUBLISHED[name]))
    if failed:
        for f in failed:
            print("  - " + f)
        sys.exit("the solver no longer reproduces the published column; no discounted figure "
                 "read off it can be trusted")

    moved = [n for n, _ in ROWS if abs(out[n]["0.99"] - out[n]["1.00"]) > 0.05]
    print("\n--- the finding ---")
    print("  rows whose optimum moves between gamma = 1.00 and gamma = 0.99: %d" % len(moved))
    for n in moved:
        print("      %-24s %.1f -> %.1f" % (n, out[n]["1.00"], out[n]["0.99"]))
    if not moved:
        print("      none; every row of Table IX holds at the learner's discount")

    json.dump({"gammas": GAMMAS, "rows": out, "published": PUBLISHED,
               "moved_at_0.99": moved},
              io.open(os.path.join(HERE, "term_optima_discounted.json"), "w",
                      encoding="utf-8"), indent=1)
    print("\nwrote term_optima_discounted.json")


if __name__ == "__main__":
    main()
