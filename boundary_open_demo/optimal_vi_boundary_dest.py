# -*- coding: utf-8 -*-
"""Exact optimal-policy completion under the destination-as-boundary-link model.

Value iteration on the expected edge cost, with the destination expressed as one outgoing
boundary link and arrival defined as taking that link. Run for both rewards and both boundary
conditions, on the same 200-pair evaluation set the learned experiment uses.

This separates the objective layer from the learning layer. Where the optimal policy completes
near 100% the objective prefers arrival, so a learner that falls short has failed to find an
optimum that exists. Where it completes near 0% the objective itself prefers leaving.
"""
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from env import ACTIONS, N_SIDE, perimeter_links   # noqa: E402
from train import make_eval_od                      # noqa: E402

N = N_SIDE
E_COST = 1.0 + 0.6 * 0.5                             # expected in-grid edge cost
R_GOAL, R_EXIT, BETA = 10.0, 5.0, 1.0
NEG = -1e9


def in_grid(r, c):
    return 0 <= r < N and 0 <= c < N


def solve(dst_link, boundary, reward, iters=4000, tol=1e-10):
    (dr, dc), _ = dst_link

    def phi(r, c):
        return abs(r - dr) + abs(c - dc)

    open_links = set() if boundary == "closed" else {
        l for l in perimeter_links(N) if l != dst_link}
    V = np.zeros((N, N))
    for _ in range(iters):
        nv = np.full((N, N), NEG)
        for r in range(N):
            for c in range(N):
                best = NEG
                for a, (ar, ac) in enumerate(ACTIONS):
                    nr, nc = r + ar, c + ac
                    if in_grid(nr, nc):
                        rw = -E_COST + (BETA * (phi(r, c) - phi(nr, nc))
                                        if reward == "aligned" else 0.0)
                        best = max(best, rw + V[nr, nc])
                    elif ((r, c), a) == dst_link:
                        rw = -1.0 + (R_GOAL + BETA * phi(r, c) if reward == "aligned" else 0.0)
                        best = max(best, rw)
                    elif ((r, c), a) in open_links:
                        rw = -1.0 - (R_EXIT if reward == "aligned" else 0.0)
                        best = max(best, rw)
                nv[r, c] = best
        if np.max(np.abs(nv - V)) < tol:
            V = nv
            break
        V = nv
    return V, phi, open_links


def arrives(o, dst_link, V, phi, open_links, reward):
    r, c = o
    for _ in range(4 * N * N):
        best, arg = NEG, None
        for a, (ar, ac) in enumerate(ACTIONS):
            nr, nc = r + ar, c + ac
            if in_grid(nr, nc):
                rw = -E_COST + (BETA * (phi(r, c) - phi(nr, nc)) if reward == "aligned" else 0.0)
                q, tgt = rw + V[nr, nc], (nr, nc)
            elif ((r, c), a) == dst_link:
                rw = -1.0 + (R_GOAL + BETA * phi(r, c) if reward == "aligned" else 0.0)
                q, tgt = rw, "ARR"
            elif ((r, c), a) in open_links:
                rw = -1.0 - (R_EXIT if reward == "aligned" else 0.0)
                q, tgt = rw, "EXIT"
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
    ods = make_eval_od(n=200, seed=12345)
    print("Optimal-policy completion, destination as an outgoing boundary link")
    for reward in ("time_min", "aligned"):
        for boundary in ("closed", "open"):
            cache, ok = {}, 0
            for o, dl in ods:
                key = (dl, boundary)
                if key not in cache:
                    cache[key] = solve(dl, boundary, reward)
                V, phi, ol = cache[key]
                ok += arrives(o, dl, V, phi, ol, reward)
            print("  %-9s %-6s : %5.1f%%" % (reward, boundary, 100.0 * ok / len(ods)))


if __name__ == "__main__":
    main()
