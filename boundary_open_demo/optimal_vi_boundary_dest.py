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
# Rewrapped only when this file is the program. Importers that print first find the buffer
# closed underneath them otherwise, which is the failure residual_exit_vi.py records.
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from env import ACTIONS, N_SIDE, perimeter_links   # noqa: E402
from train import make_eval_od                      # noqa: E402

N = N_SIDE
# Expected in-grid edge cost of 1 + 0.6 * U * (0.5 + 0.5 * sin(t/6)) with U uniform on [0, 1],
# averaged over a full period of the sinusoid: 1 + 0.6 * 0.5 * 0.5. This read 1.0 + 0.6 * 0.5 =
# 1.3 until 2026-08-09, dropping the factor E[U] and overstating the in-grid cost, which biases a
# deterministic surrogate toward exiting. optimal_vi.py always used 1.15 and the two disagreed.
# The correction changes no reported figure: optimal_vi_cost_sensitivity.py sweeps the constant
# from the minimum edge cost to the maximum and every cell holds at 100.0 / 0.0 / 100.0 / 100.0.
E_COST = 1.0 + 0.6 * 0.5 * 0.5                       # 1.15
R_GOAL, R_EXIT, BETA = 10.0, 5.0, 1.0
NEG = -1e9


def in_grid(r, c):
    return 0 <= r < N and 0 <= c < N


def solve(dst_link, boundary, reward, iters=4000, tol=1e-10,
          r_goal=None, r_exit=None, beta=None, gamma=1.0, n=None):
    """The three weights default to the published aligned reward. A caller passing them can ask
    the same computation what a single-term variant prefers, which is what an arrival-term arm
    needs before its learned figure can be read as a learner falling short."""
    R_GOAL_ = R_GOAL if r_goal is None else r_goal
    R_EXIT_ = R_EXIT if r_exit is None else r_exit
    BETA_ = BETA if beta is None else beta
    # Round 352: the lattice side was a module global read by solve(), arrives() and in_grid(),
    # so a caller sweeping the interior depth would have solved one lattice and acted on another.
    # It is a parameter of both functions now, the module value remaining the default.
    N = N_SIDE if n is None else n

    def in_grid(r, c):
        return 0 <= r < N and 0 <= c < N

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
                        rw = -E_COST + (BETA_ * (phi(r, c) - phi(nr, nc))
                                        if reward == "aligned" else 0.0)
                        best = max(best, rw + gamma * V[nr, nc])
                    elif ((r, c), a) == dst_link:
                        rw = -1.0 + (R_GOAL_ + BETA_ * phi(r, c) if reward == "aligned" else 0.0)
                        best = max(best, rw)
                    elif ((r, c), a) in open_links:
                        rw = -1.0 - (R_EXIT_ if reward == "aligned" else 0.0)
                        best = max(best, rw)
                nv[r, c] = best
        if np.max(np.abs(nv - V)) < tol:
            V = nv
            break
        V = nv
    return V, phi, open_links


def arrives(o, dst_link, V, phi, open_links, reward,
            r_goal=None, r_exit=None, beta=None, gamma=1.0, n=None):
    """The greedy policy of V, read under the same weights V was solved with.

    Round 61: the three weights were module constants here while solve() took them as
    arguments, and gamma is threaded through both for the same reason, so a caller asking for a single-term reward got a value function under its own
    weights and a greedy policy under the published ones. Every default caller is unaffected,
    since the defaults are the module constants.
    """
    R_GOAL_ = R_GOAL if r_goal is None else r_goal
    R_EXIT_ = R_EXIT if r_exit is None else r_exit
    BETA_ = BETA if beta is None else beta
    N = N_SIDE if n is None else n

    def in_grid(r, c):
        return 0 <= r < N and 0 <= c < N

    r, c = o
    for _ in range(4 * N * N):
        best, arg = NEG, None
        for a, (ar, ac) in enumerate(ACTIONS):
            nr, nc = r + ar, c + ac
            if in_grid(nr, nc):
                rw = -E_COST + (BETA_ * (phi(r, c) - phi(nr, nc)) if reward == "aligned" else 0.0)
                q, tgt = rw + gamma * V[nr, nc], (nr, nc)
            elif ((r, c), a) == dst_link:
                rw = -1.0 + (R_GOAL_ + BETA_ * phi(r, c) if reward == "aligned" else 0.0)
                q, tgt = rw, "ARR"
            elif ((r, c), a) in open_links:
                rw = -1.0 - (R_EXIT_ if reward == "aligned" else 0.0)
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
