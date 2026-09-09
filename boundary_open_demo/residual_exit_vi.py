# -*- coding: utf-8 -*-
"""Exact optimum under the residual-horizon exit charge (Section V-D control).

residual_exit_control.py reports a learned completion rate for a convention that charges an
exiting vehicle the rest of the step budget at the minimum in-grid cost. A learned rate alone
does not say whether a shortfall belongs to the objective or to the learner, which is the
separation Section VI-C asks every report to make, and the optimum for this convention was
missing from the record.

The charge depends on the step index, so the problem is no longer stationary and the value
function carries the step count: V[t] is the optimal return-to-go from a cell with t steps
already taken. Backward induction over the 120-step budget solves it exactly. Everything else
matches optimal_vi_boundary_dest.py: the same expected in-grid cost, the same aligned weights,
the destination as one outgoing boundary link, and the evaluation set of residual_exit_control.

    python3 residual_exit_vi.py
"""
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# Rewrapped only when this file is the program. check_exact_rows.py imports it after two other
# modules have already replaced sys.stdout, and a third rewrap closes the buffer underneath.
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from env import ACTIONS, N_SIDE, perimeter_links      # noqa: E402
from residual_exit_control import eval_set            # noqa: E402

N = N_SIDE
E_COST = 1.0 + 0.6 * 0.5 * 0.5                        # 1.15, as optimal_vi_boundary_dest.py
R_GOAL, R_EXIT, BETA = 10.0, 5.0, 1.0
RETURN_COST = 1.0                                     # detour cost of the non-terminal exit
T = 120                                               # max_steps of the four cells
NEG = -1e12


def in_grid(r, c):
    return 0 <= r < N and 0 <= c < N


def exit_value(convention, t, reward, V, r, c, return_cost=None):
    """Return-to-go of taking an open perimeter link that is not the destination link.

    removal      the published semantics: the exit ends the trip at an ordinary traversal cost.
    residual     the exit ends the trip and pays the rest of the budget at the minimum cost.
    nonterminal  the exit does not end the trip: a fixed detour cost returns the vehicle to the
                 node it left from, which adds a strictly costly self-loop to the closed graph.
    """
    pen = R_EXIT if reward == "aligned" else 0.0
    if convention == "nonterminal":
        # Round 351: the detour cost was a module constant read here while solve() and arrives()
        # took none, so a caller sweeping it would have solved one charge and acted under another.
        # It is threaded through all three, the constant remaining the default of every caller.
        return -(RETURN_COST if return_cost is None else return_cost) + V[t + 1, r, c]
    charge = 1.0 + (float(max(0, T - t - 1)) if convention == "residual" else 0.0)
    return -charge - pen


def solve(dst_link, boundary, reward, convention, return_cost=None):
    """Backward induction. Returns V of shape (T + 1, N, N)."""
    (dr, dc), _ = dst_link

    def phi(r, c):
        return abs(r - dr) + abs(c - dc)

    open_links = set() if boundary == "closed" else {
        l for l in perimeter_links(N) if l != dst_link}
    V = np.zeros((T + 1, N, N))
    for t in range(T - 1, -1, -1):
        for r in range(N):
            for c in range(N):
                best = NEG
                for a, (ar, ac) in enumerate(ACTIONS):
                    nr, nc = r + ar, c + ac
                    if in_grid(nr, nc):
                        rw = -E_COST + (BETA * (phi(r, c) - phi(nr, nc))
                                        if reward == "aligned" else 0.0)
                        q = rw + V[t + 1, nr, nc]
                    elif ((r, c), a) == dst_link:
                        q = -1.0 + (R_GOAL + BETA * phi(r, c) if reward == "aligned" else 0.0)
                    elif ((r, c), a) in open_links:
                        q = exit_value(convention, t, reward, V, r, c, return_cost)
                    else:
                        continue
                    best = max(best, q)
                V[t, r, c] = best
    return V, phi, open_links


def arrives(o, dst_link, V, phi, open_links, reward, convention, return_cost=None):
    r, c = o
    for t in range(T):
        best, arg = NEG, None
        for a, (ar, ac) in enumerate(ACTIONS):
            nr, nc = r + ar, c + ac
            if in_grid(nr, nc):
                rw = -E_COST + (BETA * (phi(r, c) - phi(nr, nc))
                                if reward == "aligned" else 0.0)
                q, tgt = rw + V[t + 1, nr, nc], (nr, nc)
            elif ((r, c), a) == dst_link:
                q = -1.0 + (R_GOAL + BETA * phi(r, c) if reward == "aligned" else 0.0)
                tgt = "ARR"
            elif ((r, c), a) in open_links:
                q = exit_value(convention, t, reward, V, r, c, return_cost)
                tgt = (r, c) if convention == "nonterminal" else "EXIT"
            else:
                continue
            if q > best:
                best, arg = q, tgt
        if arg == "ARR":
            return True
        if arg in ("EXIT", None):
            return False
        r, c = arg
    return False


def main():
    ods = eval_set()
    out = {"budget_steps": T, "n_eval": len(ods), "cells": {}}
    print("Optimal-policy completion, destination as an outgoing boundary link")
    print("cell                                   optimum (%)")
    for convention in ("removal", "residual", "nonterminal"):
        for boundary in ("closed", "open"):
            for reward in ("time_min", "aligned"):
                if convention != "removal" and boundary == "closed":
                    continue          # no exit exists, so the convention cannot apply
                cache, ok = {}, 0
                for o, dl in ods:
                    key = (dl, boundary, reward, convention)
                    if key not in cache:
                        cache[key] = solve(dl, boundary, reward, convention)
                    V, phi, ol = cache[key]
                    ok += arrives(o, dl, V, phi, ol, reward, convention)
                lab = "%s %-8s %s" % (boundary, reward, convention)
                pct = 100.0 * ok / len(ods)
                out["cells"][lab] = pct
                print("  %-38s %5.1f" % (lab, pct), flush=True)
    # The non-terminal cell above is solved at one traversal unit. Section S-IV reports a learned
    # rate at three detour costs, and a learned rate carries no reading without the optimum beside
    # it, so the charge is swept over the same three values here.
    print("\nnon-terminal exit, detour cost swept")
    sweep = {}
    for rc in (1.0, 3.0, 5.0):
        for reward in ("time_min", "aligned"):
            cache, ok = {}, 0
            for o, dl in ods:
                key = (dl, reward, rc)
                if key not in cache:
                    cache[key] = solve(dl, "open", reward, "nonterminal", rc)
                V, phi, ol = cache[key]
                ok += arrives(o, dl, V, phi, ol, reward, "nonterminal", rc)
            pct = 100.0 * ok / len(ods)
            sweep["%s detour %.1f" % (reward, rc)] = pct
            print("  %-38s %5.1f" % ("%s detour %.1f" % (reward, rc), pct), flush=True)
    assert abs(sweep["time_min detour 1.0"] - out["cells"]["open time_min nonterminal"]) < 1e-9, \
        "the swept solver disagrees with the published cell at the charge they share"
    out["nonterminal_detour_sweep"] = sweep

    with io.open(os.path.join(HERE, "residual_exit_vi.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nwrote residual_exit_vi.json")


if __name__ == "__main__":
    main()
