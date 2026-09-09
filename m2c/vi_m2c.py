# -*- coding: utf-8 -*-
"""M2-C step 3: exact value iteration under the published and the discount-consistent shaping.

The solver is the one that produced the printed values, archive/.../optimal_vi_boundary_dest.py,
with one switch added:

    shaping="published" : beta*(Phi(n_t) - Phi(n_{t+1})), no increment on an exit
    shaping="m2"        : beta*(Phi(n_t) - gamma*Phi(n_{t+1})), Phi = 0 at both terminals (T-1901)
    shaping="c"         : beta*(Phi(n_t) - gamma*Phi(n_{t+1})), Phi(exit) = Phi(n_t)     (T-1903)

Nothing in archive/ or repo_v11/ is modified; both are read only.
"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARC = os.path.join(ROOT, "archive/experiments_dup/handoff_experiments/boundary_open_demo")
sys.path.insert(0, ARC)

from optimal_vi_boundary_dest import (ACTIONS, E_COST, NEG, N_SIDE, R_GOAL, R_EXIT, BETA,
                                      perimeter_links)  # noqa: E402


def solve(dst_link, boundary, reward, shaping="published", gamma=1.0, iters=4000, tol=1e-10,
          r_goal=None, r_exit=None, beta=None, n=None):
    R_GOAL_ = R_GOAL if r_goal is None else r_goal
    R_EXIT_ = R_EXIT if r_exit is None else r_exit
    BETA_ = BETA if beta is None else beta
    N = N_SIDE if n is None else n
    g = gamma if shaping in ("m2", "c") else 1.0   # the potential's own discount

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
                        rw = -E_COST + (BETA_ * (phi(r, c) - g * phi(nr, nc))
                                        if reward == "aligned" else 0.0)
                        best = max(best, rw + gamma * V[nr, nc])
                    elif ((r, c), a) == dst_link:
                        # arrival: Phi(terminal) = 0 under both forms
                        rw = -1.0 + (R_GOAL_ + BETA_ * phi(r, c) if reward == "aligned" else 0.0)
                        best = max(best, rw)
                    elif ((r, c), a) in open_links:
                        # exit: the potential vanishes under m2 and is uncollected under published
                        rw = -1.0 - (R_EXIT_ if reward == "aligned" else 0.0)
                        if reward == "aligned" and shaping == "m2":
                            rw += BETA_ * phi(r, c)               # Phi(exit) = 0
                        elif reward == "aligned" and shaping == "c":
                            rw += BETA_ * (1.0 - gamma) * phi(r, c)  # Phi(exit) = Phi(n_t)
                        best = max(best, rw)
                nv[r, c] = best
        if np.max(np.abs(nv - V)) < tol:
            V = nv
            break
        V = nv
    return V, phi, open_links


def arrives(o, dst_link, V, phi, open_links, reward, shaping="published", gamma=1.0,
            r_goal=None, r_exit=None, beta=None, n=None, cap=200):
    """Does the greedy policy of V reach the destination link from o?"""
    R_GOAL_ = R_GOAL if r_goal is None else r_goal
    R_EXIT_ = R_EXIT if r_exit is None else r_exit
    BETA_ = BETA if beta is None else beta
    N = N_SIDE if n is None else n
    g = gamma if shaping in ("m2", "c") else 1.0

    def in_grid(r, c):
        return 0 <= r < N and 0 <= c < N

    r, c = o
    for _ in range(cap):
        best, tgt = NEG, None
        for a, (ar, ac) in enumerate(ACTIONS):
            nr, nc = r + ar, c + ac
            if in_grid(nr, nc):
                rw = -E_COST + (BETA_ * (phi(r, c) - g * phi(nr, nc)) if reward == "aligned" else 0.0)
                q, t = rw + gamma * V[nr, nc], (nr, nc)
            elif ((r, c), a) == dst_link:
                rw = -1.0 + (R_GOAL_ + BETA_ * phi(r, c) if reward == "aligned" else 0.0)
                q, t = rw, "arrived"
            elif ((r, c), a) in open_links:
                rw = -1.0 - (R_EXIT_ if reward == "aligned" else 0.0)
                if reward == "aligned" and shaping == "m2":
                    rw += BETA_ * phi(r, c)
                elif reward == "aligned" and shaping == "c":
                    rw += BETA_ * (1.0 - gamma) * phi(r, c)
                q, t = rw, "exited"
            else:
                continue
            if q > best:
                best, tgt = q, t
        if tgt == "arrived":
            return True
        if tgt == "exited" or tgt is None:
            return False
        r, c = tgt
    return False
