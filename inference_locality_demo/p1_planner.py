"""Inference-locality (P1) demonstration: the value of a real-time forecast that a local
state cannot obtain.

Three planners are compared on the incident-bearing, boundary-closed grid (see env_p1.py):

  predictive-optimal : time-expanded shortest path with full knowledge of the non-recurrent
                       incidents (a real-time forecast). Upper bound for a predictive state.
  reactive-local     : replans each step on the current congestion snapshot, assuming it
                       persists; sees an incident only after its onset. A fair upper bound
                       for a local, non-predictive state, since no local policy can observe
                       an incident before it starts.
  historical-mean    : routes on the recurrent background only. Confirms the incidents are
                       not recoverable from history, so training on the realization
                       distribution cannot substitute for the forecast.

The gap between predictive-optimal and the better of the two local baselines is the value of
information a local state cannot hold, which is the inference-locality cost.
"""

import heapq
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env_p1 import GridP1Env, ACTIONS

ENV_KW = dict(recur_amp=0.6, n_incidents=3, inc_mag=6.0, inc_dur=(4, 10),
              horizon_max=60, max_steps=60)


def _true_cost(env, r, c, t):
    return 1.0 + env._cell_congestion(r, c, t)


def _recur_cost(env, r, c):
    return 1.0 + env._cell_recurrent(r, c)


def predictive_optimal(env, od):
    """Min-cost OD path over the time-expanded graph using true (forecast) congestion."""
    (o, d) = od
    T = env.max_steps
    dist = {(o, 0): 0.0}
    pq = [(0.0, o[0], o[1], 0)]
    while pq:
        g, r, c, t = heapq.heappop(pq)
        if (r, c) == d:
            return g
        if t >= T or g > dist.get(((r, c), t), 1e9):
            continue
        for dr, dc in ACTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < env.n and 0 <= nc < env.n:
                ng = g + _true_cost(env, nr, nc, t)
                if ng < dist.get(((nr, nc), t + 1), 1e9):
                    dist[((nr, nc), t + 1)] = ng
                    heapq.heappush(pq, (ng, nr, nc, t + 1))
    return None


def _first_move(env, cost_fn, r, c, d):
    """Dijkstra on a static cost snapshot; return the first action toward d."""
    dist = {(r, c): 0.0}; prev = {}; pq = [(0.0, r, c)]
    while pq:
        g, cr, cc = heapq.heappop(pq)
        if (cr, cc) == d:
            break
        if g > dist.get((cr, cc), 1e9):
            continue
        for a, (dr, dc) in enumerate(ACTIONS):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < env.n and 0 <= nc < env.n:
                ng = g + cost_fn(nr, nc)
                if ng < dist.get((nr, nc), 1e9):
                    dist[(nr, nc)] = ng; prev[(nr, nc)] = (cr, cc, a)
                    heapq.heappush(pq, (ng, nr, nc))
    cur = d
    while cur in prev and prev[cur][:2] != (r, c):
        cur = prev[cur][:2]
    if cur in prev:
        return prev[cur][2]
    for a, (dr, dc) in enumerate(ACTIONS):
        nr, nc = r + dr, c + dc
        if 0 <= nr < env.n and 0 <= nc < env.n and env._phi(nr, nc) < env._phi(r, c):
            return a
    return int(np.argmax(env.available_actions()))


def execute(env, od, mode):
    """Run a replanning local planner on the true environment; return realized travel time.
    mode='reactive' uses the current snapshot; mode='historical' uses recurrent-only."""
    env.reset(od=od)
    r, c = od[0]; d = od[1]
    for _ in range(env.max_steps):
        if (r, c) == d:
            break
        if mode == "reactive":
            tnow = env._t
            cost_fn = lambda nr, nc: _true_cost(env, nr, nc, tnow)
        else:  # historical mean: recurrent background only
            cost_fn = lambda nr, nc: _recur_cost(env, nr, nc)
        a = _first_move(env, cost_fn, r, c, d)
        _, _, done, info = env.step(a)
        r, c = env.pos
        if done:
            return info["travel_time"]
    return env.travel_time


def make_eval_od(n=200, n_side=5, seed=12345):
    rng = np.random.RandomState(seed)
    ods = []
    while len(ods) < n:
        o = (rng.randint(n_side), rng.randint(n_side))
        d = (rng.randint(n_side), rng.randint(n_side))
        if abs(o[0] - d[0]) + abs(o[1] - d[1]) >= 3:
            ods.append((o, d))
    return ods


def run(n_seeds=20):
    eval_od = make_eval_od()
    pred, reac, hist = [], [], []
    for seed in range(n_seeds):
        for od in eval_od:
            e1 = GridP1Env(state_mode="predictive", seed=seed, **ENV_KW)
            e1.reset(od=od); pred.append(predictive_optimal(e1, od))
            e2 = GridP1Env(state_mode="predictive", seed=seed, **ENV_KW)
            reac.append(execute(e2, od, "reactive"))
            e3 = GridP1Env(state_mode="predictive", seed=seed, **ENV_KW)
            hist.append(execute(e3, od, "historical"))
    pred = np.array([x for x in pred if x is not None]); reac = np.array(reac); hist = np.array(hist)
    pm, rm, hm = pred.mean(), reac.mean(), hist.mean()
    ps, rs, hs = pred.std(), reac.std(), hist.std()
    print(f"predictive-optimal : {pm:.2f} (std {ps:.2f})")
    print(f"reactive-local     : {rm:.2f} (std {rs:.2f})   gap vs predictive {100*(rm-pm)/rm:.1f}%")
    print(f"historical-mean    : {hm:.2f} (std {hs:.2f})   gap vs predictive {100*(hm-pm)/hm:.1f}%")
    best_local = min(rm, hm)
    print(f"predictive vs best local: mean {100*(best_local-pm)/best_local:.1f}% lower, "
          f"std {ps:.2f} vs {min(rs,hs):.2f}")

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    bins = np.linspace(min(pred.min(), reac.min(), hist.min()),
                       np.percentile(np.concatenate([pred, reac, hist]), 99), 40)
    ax.hist(reac, bins=bins, alpha=0.55, color="#a50f15", hatch="xx", edgecolor="black",
            linewidth=0.4, label=f"Reactive local (mean {rm:.1f})")
    ax.hist(pred, bins=bins, alpha=0.55, color="#238b45", hatch="..", edgecolor="black",
            linewidth=0.4, label=f"Predictive optimal (mean {pm:.1f})")
    ax.axvline(rm, color="#a50f15", ls="--", lw=1.5); ax.axvline(pm, color="#238b45", ls="-.", lw=1.5)
    ax.set_xlabel("Realized OD travel time"); ax.set_ylabel("Count (OD x realizations)")
    ax.legend(fontsize=9); fig.tight_layout()
    fig.savefig("fig_v3_inference_locality.png", dpi=300)
    print("wrote fig_v3_inference_locality.png")
    with open("summary_p1.csv", "w") as f:
        f.write("planner,mean_travel_time,std_travel_time\n")
        f.write(f"predictive_optimal,{pm:.4f},{ps:.4f}\n")
        f.write(f"reactive_local,{rm:.4f},{rs:.4f}\n")
        f.write(f"historical_mean,{hm:.4f},{hs:.4f}\n")


if __name__ == "__main__":
    run()
