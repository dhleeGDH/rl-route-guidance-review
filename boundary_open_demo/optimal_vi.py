"""Exact value iteration on the 5x5 grid (Section V-C).

Computes the optimal policy under the travel-time reward on the boundary-open network, using
the expected edge cost, and reports the optimal policy's OD trip completion rate stratified by
OD graph distance. This establishes the boundary of the objective failure directly: a
held-out pair on which the optimal policy itself exits is an objective failure, not a learner
failure. Compared against the DQN's held-out completion (also stratified by distance), it
separates the two mechanisms.
"""
import numpy as np
from env import N_SIDE, ACTIONS

N = N_SIDE
# expected edge cost: base 1 + E[congestion term]. congestion=0.6, field ~U[0,1] (E=0.5),
# temporal factor 0.5+0.5*sin (E over t ~ 0.5). E[cost] = 1 + 0.6*0.5*0.5 = 1.15.
E_COST = 1.15
EXIT_COST = 1.0


def in_grid(r, c):
    return 0 <= r < N and 0 <= c < N


def value_iteration(dst, boundary="open"):
    """Return optimal value V[r,c] and greedy policy for the time-min reward.
    Terminals: destination (V=0) and OUT (exit). Undiscounted; costs are positive so VI
    converges to the min-cost-to-terminal (max return = least negative)."""
    V = np.zeros((N, N))
    for _ in range(500):
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
                        q = -E_COST + V[nr, nc]
                    else:
                        if boundary == "closed":
                            continue  # unavailable
                        q = -EXIT_COST + 0.0  # exit terminal
                    best = max(best, q)
                Vn[r, c] = best
        if np.max(np.abs(Vn - V)) < 1e-9:
            V = Vn
            break
        V = Vn
    return V


def optimal_completes(od, boundary="open"):
    """Does the optimal time-min policy from o reach dst rather than exiting?"""
    o, dst = od
    V = value_iteration(dst, boundary)
    r, c = o
    for _ in range(4 * N * N):
        if (r, c) == dst:
            return True
        best, ba = -1e18, None
        for a, (dr, dc) in enumerate(ACTIONS):
            nr, nc = r + dr, c + dc
            if in_grid(nr, nc):
                q = -E_COST + V[nr, nc]
            else:
                if boundary == "closed":
                    continue
                q = -EXIT_COST
            if q > best:
                best, ba = q, a
        dr, dc = ACTIONS[ba]
        nr, nc = r + dr, c + dc
        if not in_grid(nr, nc):
            return False  # optimal policy exits
        r, c = nr, nc
    return False


def make_eval_od(n=200, seed=12345, min_sep=3):
    rng = np.random.RandomState(seed)
    ods = []
    while len(ods) < n:
        o = (rng.randint(N), rng.randint(N))
        d = (rng.randint(N), rng.randint(N))
        if abs(o[0] - d[0]) + abs(o[1] - d[1]) >= min_sep:
            ods.append((o, d))
    return ods


if __name__ == "__main__":
    eval_od = make_eval_od()
    from collections import defaultdict
    by_k = defaultdict(list)
    for od in eval_od:
        (o, d) = od
        k = abs(o[0] - d[0]) + abs(o[1] - d[1])
        by_k[k].append(optimal_completes(od, "open"))
    print("Optimal (expected-cost VI) time-min policy on the BOUNDARY-OPEN 5x5 grid:")
    print("OD dist k |  n  | optimal completion")
    tot_n = tot_c = 0
    for k in sorted(by_k):
        n = len(by_k[k]); c = sum(by_k[k]); tot_n += n; tot_c += c
        print(f"   {k:2d}     | {n:3d} | {c/n:6.1%}")
    print(f"  ALL     | {tot_n:3d} | {tot_c/tot_n:6.1%}")
    # closed reference: optimal always completes (no exit)
    cc = sum(optimal_completes(od, "closed") for od in eval_od[:50])
    print(f"\n[closed, first 50] optimal completion = {cc}/50 (expect 50/50)")
