# -*- coding: utf-8 -*-
"""What does the destination-aligned reward cost in travel time?

Every outcome table in Section V reports OD trip completion. The aligned reward of Eq. (4)
deliberately changes the objective, so a reader is entitled to ask whether it restores
completion by sending vehicles on longer routes. Completion alone cannot answer that.

This measures, for each of the four cells, the realized travel time of the trips that
complete, against the time-consistent optimum for the same origin-destination pair under the
same congestion realization. The ratio of the two is the detour a policy takes.

Travel time is accumulated from the environment's own link costs during evaluation,
independently of whichever reward the policy was trained on, so the two rewards are compared
in the same units.

The optimum is exact rather than estimated. Link cost depends on the step at which a vehicle
enters a cell, so the shortest time-consistent route is a shortest path over (cell, step)
states, computed here by Dijkstra on that product graph with the environment's own cost
function.

Design fixed before running: four cells, 10 seeds, 3000 episodes, 200 evaluation OD pairs and
a 120-step budget, all as in Section V-A. Only the measurement is added.
"""
import argparse
import heapq
import io
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from env import GridRouteEnv, ACTIONS, N_SIDE            # noqa: E402
from dqn import DQNAgent                                 # noqa: E402
from train import make_eval_od                           # noqa: E402


def step_cost(cong, congestion, nr, nc, t):
    """The environment's own link cost, with the step index passed explicitly."""
    return 1.0 + max(0.0, congestion * cong[nr, nc] * (0.5 + 0.5 * np.sin(t / 6.0)))


def optimal_time(cong, congestion, origin, dst, n, max_steps):
    """Least realized travel time from origin to dst, over (cell, step) states."""
    best = {}
    pq = [(0.0, origin[0], origin[1], 0)]
    while pq:
        c, r, cc, t = heapq.heappop(pq)
        if (r, cc) == dst:
            return c
        if t >= max_steps or best.get((r, cc, t), 1e18) < c:
            continue
        for dr, dc in ACTIONS:
            nr, nc = r + dr, cc + dc
            if 0 <= nr < n and 0 <= nc < n:
                nc_cost = c + step_cost(cong, congestion, nr, nc, t)
                if nc_cost < best.get((nr, nc, t + 1), 1e18):
                    best[(nr, nc, t + 1)] = nc_cost
                    heapq.heappush(pq, (nc_cost, nr, nc, t + 1))
    return None


def evaluate(agent, boundary, reward, eval_od, seed=777, max_steps=120):
    env = GridRouteEnv(boundary=boundary, reward=reward, seed=seed, max_steps=max_steps)
    realized, ratios, arrived = [], [], 0
    for od in eval_od:
        env.reset(od=od)
        tt = 0.0
        for _ in range(env.max_steps):
            r, c = env.pos
            a = agent.act(env._obs(), env.available_actions(), eps=0.0)
            dr, dc = ACTIONS[a]
            ec = env._edge_cost(r, c, a)          # None on an off-grid move
            _, _, done, info = env.step(a)
            if ec is not None:
                tt += ec
            if done:
                if info["outcome"] == "arrived":
                    arrived += 1
                    opt = optimal_time(env._cong, env.congestion, od[0], od[1][0],
                                       env.n, env.max_steps)
                    realized.append(tt)
                    if opt:
                        ratios.append(tt / opt)
                break
    return (arrived / len(eval_od),
            float(np.mean(realized)) if realized else float("nan"),
            float(np.mean(ratios)) if ratios else float("nan"))


def run_cell(boundary, reward, seed, episodes, eval_od, max_steps=120):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = GridRouteEnv(boundary=boundary, reward=reward, seed=1000 + seed, max_steps=max_steps)
    agent = DQNAgent(env.state_dim, env.n_actions, seed=seed)
    eps0, eps1 = 1.0, 0.05
    decay = int(0.6 * episodes)
    for ep in range(1, episodes + 1):
        eps = max(eps1, eps0 - (eps0 - eps1) * ep / decay)
        env.reset()
        for _ in range(env.max_steps):
            s = env._obs()
            mask = env.available_actions()
            a = agent.act(s, mask, eps)
            s2, r, done, _ = env.step(a)
            agent.buf.add(s, a, r, s2, float(done),
                          env.available_actions().astype(np.float32))
            agent.learn()
            if done:
                break
    return evaluate(agent, boundary, reward, eval_od, max_steps=max_steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="travel_time_cost.json")
    a = ap.parse_args()

    eval_od = make_eval_od(n=200, n_side=N_SIDE, seed=12345)
    out = {}
    for boundary in ("closed", "open"):
        for reward in ("time_min", "aligned"):
            comps, tts, rats = [], [], []
            for s in range(a.seeds):
                c, tt, ratio = run_cell(boundary, reward, s, a.episodes, eval_od)
                comps.append(100 * c); tts.append(tt); rats.append(ratio)
                print("  %-6s %-9s seed %d: completion %5.1f%%  travel time %6.2f  detour %.3f"
                      % (boundary, reward, s, 100 * c, tt, ratio), flush=True)
            key = "%s_%s" % (boundary, reward)
            out[key] = {"completion_mean": float(np.nanmean(comps)),
                        "travel_time_mean": float(np.nanmean(tts)),
                        "travel_time_sd": float(np.nanstd(tts)),
                        "detour_mean": float(np.nanmean(rats)),
                        "detour_sd": float(np.nanstd(rats))}
            print("== %-6s %-9s completion %5.1f%%  travel time %6.2f (sd %.2f)  "
                  "detour %.3f (sd %.3f)"
                  % (boundary, reward, out[key]["completion_mean"],
                     out[key]["travel_time_mean"], out[key]["travel_time_sd"],
                     out[key]["detour_mean"], out[key]["detour_sd"]), flush=True)
    with open(os.path.join(HERE, a.out), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
