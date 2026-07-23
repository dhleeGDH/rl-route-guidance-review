"""Does exit dominance survive the agent's discount? (Section V-C qualification)

The bound and the value iteration of Section V-C are undiscounted, while the DQN learns with
gamma = 0.99. Discounting shrinks the later per-step costs of a completed trip more than the
one or two costs of an exit, so it works AGAINST exit dominance. This re-runs the same exact
value iteration at gamma = 0.99 and reports the optimal policy's completion rate stratified by
OD graph distance, so the "directionally preserved" claim is checked rather than asserted.
"""
import numpy as np

from env import N_SIDE, ACTIONS
from optimal_vi import E_COST, EXIT_COST, in_grid

N = N_SIDE


def value_iteration(dst, boundary="open", gamma=1.0):
    V = np.zeros((N, N))
    for _ in range(20000):
        Vn = V.copy()
        for r in range(N):
            for c in range(N):
                if (r, c) == dst:
                    Vn[r, c] = 0.0
                    continue
                best = -1e18
                for (dr, dc) in ACTIONS:
                    nr, nc = r + dr, c + dc
                    if in_grid(nr, nc):
                        q = -E_COST + gamma * V[nr, nc]
                    else:
                        if boundary == "closed":
                            continue
                        q = -EXIT_COST      # exit terminal, no bootstrap past it
                    best = max(best, q)
                Vn[r, c] = best
        if np.max(np.abs(Vn - V)) < 1e-12:
            V = Vn
            break
        V = Vn
    return V


def completes(od, dst, V, gamma):
    """Follow the greedy policy from od; True if it reaches dst instead of exiting."""
    r, c = od
    for _ in range(4 * N * N):
        if (r, c) == dst:
            return True
        best, arg = -1e18, None
        for (dr, dc) in ACTIONS:
            nr, nc = r + dr, c + dc
            q = (-E_COST + gamma * V[nr, nc]) if in_grid(nr, nc) else -EXIT_COST
            if q > best:
                best, arg = q, (nr, nc)
        if not in_grid(*arg):
            return False        # the optimal policy exits
        r, c = arg
    return False


def sweep(gamma):
    cells = [(r, c) for r in range(N) for c in range(N)]
    by_k = {}
    for dst in cells:
        V = value_iteration(dst, "open", gamma)
        for o in cells:
            k = abs(o[0] - dst[0]) + abs(o[1] - dst[1])
            if k == 0:
                continue
            by_k.setdefault(k, []).append(completes(o, dst, V, gamma))
    return by_k


print("optimal travel-time policy on the BOUNDARY-OPEN 5x5 grid")
print("completion rate of the optimal policy, by OD graph distance k\n")
print("%-6s %s" % ("k", "  ".join("g=%.2f" % g for g in (1.0, 0.99, 0.95, 0.90))))
rows = {g: sweep(g) for g in (1.0, 0.99, 0.95, 0.90)}
ks = sorted(rows[1.0])
for k in ks:
    print("%-6d %s" % (k, "  ".join("%5.1f%%" % (100.0 * np.mean(rows[g][k])) for g in (1.0, 0.99, 0.95, 0.90))))
print()
for g in (1.0, 0.99, 0.95, 0.90):
    allv = [v for k in ks for v in rows[g][k]]
    print("gamma=%.2f  overall optimal-policy completion: %.1f%%" % (g, 100.0 * np.mean(allv)))
