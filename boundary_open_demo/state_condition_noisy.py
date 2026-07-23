"""State-condition demonstration with a noisy (realizable) forecaster and a scale sweep.

The Section V-A forecaster is a perfect oracle, so the 12.5% ETA error it "removes" is true
by construction. This script (a) inserts a noisy forecaster whose future-cost
estimates carry a multiplicative error swept over a plausible range, so the reported gap
reflects what a realizable forecaster recovers, and (b) sweeps the grid size to test the claim
that the cost grows with network scale. Cost model, snapshot planner, and perfect forecaster
match state_condition_demo.py; only the noisy forecaster and the size sweep are added.
"""
import heapq
import numpy as np

A = 1.5
P = 6.0
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def cost(cell, t, field, phase):
    r, c = cell
    return 1.0 + max(0.0, A * field[r, c] * (0.5 + 0.5 * np.sin(2 * np.pi * t / P + phase[r, c])))


def neighbors(cell, N):
    r, c = cell
    for dr, dc in MOVES:
        nr, nc = r + dr, c + dc
        if 0 <= nr < N and 0 <= nc < N:
            yield (nr, nc)


def snapshot_route(o, d, field, phase, N):
    dist = {o: 0.0}; prev = {}; pq = [(0.0, o)]
    while pq:
        du, u = heapq.heappop(pq)
        if u == d:
            break
        if du > dist.get(u, 1e18):
            continue
        for v in neighbors(u, N):
            nd = du + cost(v, 0, field, phase)
            if nd < dist.get(v, 1e18):
                dist[v] = nd; prev[v] = u; heapq.heappush(pq, (nd, v))
    path = [d]
    while path[-1] != o:
        path.append(prev[path[-1]])
    return path[::-1]


def forecast_route(o, d, field, phase, N, noise=0.0, err=None, max_steps=80):
    """Time-expanded Dijkstra minimizing the planner's cost estimate. noise=0 is the perfect
    oracle; noise>0 multiplies each future cost estimate by (1 + err[cell,k]), err a fixed draw
    of standard deviation `noise`, so the planner commits against imperfect predictions while
    realized travel time uses the true cost."""
    start = (o, 0); dist = {start: 0.0}; prev = {}; pq = [(0.0, start)]
    best_goal = None
    while pq:
        du, (u, k) = heapq.heappop(pq)
        if u == d:
            best_goal = (u, k); break
        if k >= max_steps or du > dist.get((u, k), 1e18):
            continue
        for v in neighbors(u, N):
            c_est = cost(v, k, field, phase)
            if noise > 0.0:
                c_est *= max(0.05, 1.0 + err[v[0], v[1], k % err.shape[2]])
            nd = du + c_est
            if nd < dist.get((v, k + 1), 1e18):
                dist[(v, k + 1)] = nd; prev[(v, k + 1)] = (u, k)
                heapq.heappush(pq, (nd, (v, k + 1)))
    node = best_goal; path = [node[0]]
    while node in prev:
        node = prev[node]; path.append(node[0])
    return path[::-1]


def realized(route, field, phase):
    return sum(cost(route[k + 1], k, field, phase) for k in range(len(route) - 1))


def run(N, noise, n_od=200, seed=12345, min_sep=3):
    rng = np.random.RandomState(seed)
    snap_ex, fc_ex, subopt, n = [], [], 0, 0
    while n < n_od:
        o = (rng.randint(N), rng.randint(N)); d = (rng.randint(N), rng.randint(N))
        if abs(o[0] - d[0]) + abs(o[1] - d[1]) < max(min_sep, 1):
            continue
        field = rng.rand(N, N); phase = rng.uniform(0, 2 * np.pi, size=(N, N))
        err = rng.normal(0, noise, size=(N, N, int(P) + 1)) if noise > 0 else None
        rs = snapshot_route(o, d, field, phase, N)
        rf = forecast_route(o, d, field, phase, N, noise, err)
        sr, fr = realized(rs, field, phase), realized(rf, field, phase)
        snap_ex.append(sr); fc_ex.append(fr)
        if sr > fr + 1e-6:
            subopt += 1
        n += 1
    snap_ex, fc_ex = np.array(snap_ex), np.array(fc_ex)
    excess = 100 * (snap_ex - fc_ex).mean() / fc_ex.mean()  # % snapshot is worse than this forecaster
    return excess, 100 * subopt / n_od


if __name__ == "__main__":
    print("=== M6a: noisy forecaster on the 5x5 grid (snapshot excess over the forecaster) ===")
    print("noise=multiplicative SD on future-cost estimates; 0 = perfect oracle")
    for noise in [0.0, 0.1, 0.2, 0.3, 0.5]:
        ex, sub = run(5, noise)
        print(f"  noise {noise:.2f}: snapshot travel time {ex:5.1f}% above the forecaster's, "
              f"snapshot slower on {sub:.0f}% of OD", flush=True)
    print("=== M6b: scale sweep (perfect forecaster, snapshot excess grows with grid size) ===")
    for N in [5, 8, 11]:
        ex, sub = run(N, 0.0, min_sep=max(3, N // 2))
        print(f"  {N}x{N} grid: snapshot travel time {ex:5.1f}% above forecast, "
              f"slower on {sub:.0f}% of OD", flush=True)
