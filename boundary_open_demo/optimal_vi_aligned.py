# -*- coding: utf-8 -*-
"""Is arrival the optimum under the destination-aligned reward on the open grid?

A learner that reaches 0% completion on a cell can be failing in two different ways. Either
the objective it was given makes leaving optimal, which is the failure Section V-C establishes
for the travel-time reward, or the objective prefers arrival and the learner did not find it.
The two are told apart by computing the optimum directly.

This runs exact value iteration on the boundary-open 5x5 grid under the aligned reward of
Eq. (4) with the weights the experiment uses (shaping 1, arrival bonus 10, exit penalty 5),
on the expected edge cost, and reports the optimal policy's completion rate. Reward terms are
taken from the environment: an in-grid move pays -c + beta*(phi_before - phi_after), arrival
adds the goal bonus, and an exit at a boundary cell pays -1 - r_exit and terminates.
"""
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from optimal_vi import E_COST, EXIT_COST, N, in_grid, make_eval_od   # noqa: E402
from env import ACTIONS                                              # noqa: E402

BETA, R_GOAL, R_EXIT = 1.0, 10.0, 5.0
NEG = -1e9


def phi(r, c, dst):
    return abs(r - dst[0]) + abs(c - dst[1])


def value_iteration_aligned(dst, boundary="open", iters=4000, tol=1e-10):
    V = np.zeros((N, N))
    for _ in range(iters):
        newV = np.full((N, N), NEG)
        for r in range(N):
            for c in range(N):
                if (r, c) == dst:
                    newV[r, c] = 0.0
                    continue
                best = NEG
                for dr, dc in ACTIONS:
                    nr, nc = r + dr, c + dc
                    if in_grid(nr, nc):
                        rew = -E_COST + BETA * (phi(r, c, dst) - phi(nr, nc, dst))
                        if (nr, nc) == dst:
                            best = max(best, rew + R_GOAL)
                        else:
                            best = max(best, rew + V[nr, nc])
                    elif boundary == "open":
                        best = max(best, -EXIT_COST - R_EXIT)      # absorbing exit
                newV[r, c] = best
        if np.max(np.abs(newV - V)) < tol:
            V = newV
            break
        V = newV
    return V


def optimal_arrives(od, dst_V, dst):
    """Follow the greedy optimal policy and report whether it reaches the destination."""
    r, c = od
    for _ in range(4 * N * N):
        if (r, c) == dst:
            return True
        best, arg = NEG, None
        for dr, dc in ACTIONS:
            nr, nc = r + dr, c + dc
            if in_grid(nr, nc):
                rew = -E_COST + BETA * (phi(r, c, dst) - phi(nr, nc, dst))
                q = rew + (R_GOAL if (nr, nc) == dst else dst_V[nr, nc])
            else:
                q = -EXIT_COST - R_EXIT
                nr, nc = None, None
            if q > best:
                best, arg = q, (nr, nc)
        if arg == (None, None):
            return False                      # the optimal action is to leave
        r, c = arg
    return False


def main():
    ods = make_eval_od(n=200, seed=12345, min_sep=3)
    by_k, arrived = {}, 0
    cache = {}
    for o, d in ods:
        if d not in cache:
            cache[d] = value_iteration_aligned(d)
        ok = optimal_arrives(o, cache[d], d)
        arrived += ok
        k = abs(o[0] - d[0]) + abs(o[1] - d[1])
        by_k.setdefault(k, []).append(ok)
    print("Optimal policy under the destination-aligned reward, boundary-open 5x5 grid")
    print("  overall completion of the optimal policy: %.1f%%" % (100.0 * arrived / len(ods)))
    for k in sorted(by_k):
        v = by_k[k]
        print("    k=%d  %5.1f%%  (%d pairs)" % (k, 100.0 * np.mean(v), len(v)))
    print()
    print("  A completion near 100%% means arrival is the optimum, so a learner that reaches")
    print("  0%% on this cell has failed to find it rather than been directed away from it.")


if __name__ == "__main__":
    main()
