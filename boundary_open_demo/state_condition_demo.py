"""An empirical footing for the state condition, parallel to the reward condition.

On the same time-varying 5x5 grid used in Section V, a committed OD route is
planned at departure and then traversed under the true time-varying link costs. Two planners
differ only in the information their state carries at commitment:

  snapshot-extrapolated : sees the current-time link costs c(.,0) and commits the shortest
                          route assuming they persist (what a non-predictive dynamic state
                          supports).
  forecast-conditioned  : knows the future link costs c(.,t) each link will present when
                          traversed and commits the time-consistent (time-dependent) shortest
                          route (what a forecast-conditioned state supports; a perfect
                          forecast, the best case).

Both commit a full route before departure. The state condition predicts the snapshot route is
planned against values that will have changed, so its planned ETA and its realized travel time
diverge and its realized travel time exceeds the time-consistent route's. The experiment
measures that gap. Writes state_condition_summary.csv.

Cost model. Section V's grid uses a single global time factor sin(t/6), under which every
link congests in phase, so the least-cost route is nearly time-invariant and snapshot
extrapolation costs almost nothing (that experiment isolates the reward, not the state). Real
networks instead congest asynchronously: arterials peak at different times of day. This
demonstration therefore uses a per-link phase, the analogue for the state condition of the
corpus-typical reward the boundary demonstration adopts:

  c(cell, t) = 1 + max(0, A * field[cell] * (0.5 + 0.5 sin(2*pi*t/P + phase[cell]))),

with amplitude A = 1.5, period P = 6 steps, field ~ U[0,1] and phase ~ U[0, 2*pi] drawn per
episode. Under it the least-cost route genuinely depends on when each link is reached, so a
route committed from the departure snapshot is planned against values that will have changed.
"""
import heapq
import numpy as np

N = 5
A = 1.5
P = 6.0
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def cost(cell, t, field, phase):
    r, c = cell
    return 1.0 + max(0.0, A * field[r, c] * (0.5 + 0.5 * np.sin(2 * np.pi * t / P + phase[r, c])))


def neighbors(cell):
    r, c = cell
    for dr, dc in MOVES:
        nr, nc = r + dr, c + dc
        if 0 <= nr < N and 0 <= nc < N:
            yield (nr, nc)


def snapshot_route(o, d, field, phase):
    """Static Dijkstra with the departure-time snapshot cost c(., 0)."""
    dist = {o: 0.0}; prev = {}; pq = [(0.0, o)]
    while pq:
        du, u = heapq.heappop(pq)
        if u == d:
            break
        if du > dist.get(u, 1e18):
            continue
        for v in neighbors(u):
            nd = du + cost(v, 0, field, phase)
            if nd < dist.get(v, 1e18):
                dist[v] = nd; prev[v] = u; heapq.heappush(pq, (nd, v))
    path = [d]
    while path[-1] != o:
        path.append(prev[path[-1]])
    return path[::-1]


def forecast_route(o, d, field, phase, max_steps=60):
    """Time-expanded Dijkstra: minimize sum c(next, k) over the route, k the step index."""
    start = (o, 0); dist = {start: 0.0}; prev = {}; pq = [(0.0, start)]
    best_goal = None
    while pq:
        du, (u, k) = heapq.heappop(pq)
        if u == d:
            best_goal = (u, k); break
        if k >= max_steps or du > dist.get((u, k), 1e18):
            continue
        for v in neighbors(u):
            nd = du + cost(v, k, field, phase)
            if nd < dist.get((v, k + 1), 1e18):
                dist[(v, k + 1)] = nd; prev[(v, k + 1)] = (u, k)
                heapq.heappush(pq, (nd, (v, k + 1)))
    node = best_goal; path = [node[0]]
    while node in prev:
        node = prev[node]; path.append(node[0])
    return path[::-1]


def realized(route, field, phase):
    """True travel time traversing a committed route: cost of step k is c(route[k+1], k)."""
    return sum(cost(route[k + 1], k, field, phase) for k in range(len(route) - 1))


def planned_snapshot(route, field, phase):
    """What the snapshot planner predicted: every link at its departure-time cost c(., 0)."""
    return sum(cost(route[k + 1], 0, field, phase) for k in range(len(route) - 1))


def main(n_od=200, seed=12345, min_sep=3):
    rng = np.random.RandomState(seed)
    snap_real, fc_real, snap_plan, snap_eta_err, suboptimal = [], [], [], [], 0
    n = 0
    while n < n_od:
        o = (rng.randint(N), rng.randint(N)); d = (rng.randint(N), rng.randint(N))
        if abs(o[0] - d[0]) + abs(o[1] - d[1]) < min_sep:
            continue
        field = rng.rand(N, N)
        phase = rng.uniform(0, 2 * np.pi, size=(N, N))
        rs = snapshot_route(o, d, field, phase)
        rf = forecast_route(o, d, field, phase)
        sr = realized(rs, field, phase); fr = realized(rf, field, phase); sp = planned_snapshot(rs, field, phase)
        snap_real.append(sr); fc_real.append(fr); snap_plan.append(sp)
        snap_eta_err.append(abs(sr - sp))
        if sr > fr + 1e-6:
            suboptimal += 1
        n += 1
    snap_real = np.array(snap_real); fc_real = np.array(fc_real)
    snap_plan = np.array(snap_plan); snap_eta_err = np.array(snap_eta_err)
    excess = (snap_real - fc_real)
    print("=== state-condition demonstration (200 OD pairs, 5x5 time-varying grid) ===")
    print(f"mean realized travel time:  snapshot {snap_real.mean():.3f}  vs forecast {fc_real.mean():.3f}")
    print(f"snapshot excess over forecast: {excess.mean():.3f} ({100*excess.mean()/fc_real.mean():.1f}% higher)")
    print(f"snapshot ETA error |realized-planned|: mean {snap_eta_err.mean():.3f} "
          f"({100*snap_eta_err.mean()/snap_real.mean():.1f}% of realized); "
          f"forecast ETA error is 0 by construction")
    print(f"snapshot route strictly slower than forecast on {100*suboptimal/n_od:.1f}% of OD pairs")
    with open("state_condition_summary.csv", "w") as f:
        f.write("metric,value\n")
        f.write(f"snapshot_mean_realized,{snap_real.mean():.4f}\n")
        f.write(f"forecast_mean_realized,{fc_real.mean():.4f}\n")
        f.write(f"snapshot_excess_pct,{100*excess.mean()/fc_real.mean():.2f}\n")
        f.write(f"snapshot_eta_error_mean,{snap_eta_err.mean():.4f}\n")
        f.write(f"snapshot_eta_error_pct,{100*snap_eta_err.mean()/snap_real.mean():.2f}\n")
        f.write(f"snapshot_suboptimal_pct,{100*suboptimal/n_od:.2f}\n")
    print("wrote state_condition_summary.csv")


if __name__ == "__main__":
    main()
