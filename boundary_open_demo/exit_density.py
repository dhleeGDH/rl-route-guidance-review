# -*- coding: utf-8 -*-
"""How far does the boundary-open failure depend on how many perimeter links offer an exit?

The main experiment opens every non-destination perimeter link, which is the densest exit
geometry a study area can have. A real study area is usually sparser. This varies the exit
density and leaves everything else at the Section V-A configuration, so the completion rate
is read against exit geometry alone.

The analytical bound of Section V-C predicts the direction. Exit dominance needs
k > rho * c_max + 1, where rho is the largest distance from any cell to the nearest exit.
Opening fewer links raises rho, which raises the threshold k, so the exit
dominates for fewer origin-destination pairs. A monotone recovery as density falls therefore confirms the bound rather than
contradicting the collapse.

Design fixed before running:
  density d      : 100%, 50%, 25% of the non-destination perimeter links, drawn afresh at
                   random for each seed and each episode
  reward         : time_min (the corpus mode) and aligned (control)
  seeds          : 10, matching the main grid experiment
  everything else: identical to Section V-A (3000 episodes, 200 OD pairs, 120-step budget)
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from env import GridRouteEnv, N_SIDE                   # noqa: E402
from dqn import DQNAgent                               # noqa: E402
from train import make_eval_od                         # noqa: E402


def rho_of(open_cells, n=N_SIDE):
    """Largest Manhattan distance from any cell to the nearest cell that owns an exit."""
    if not open_cells:
        return n
    return max(min(abs(r - er) + abs(c - ec) for er, ec in open_cells)
               for r in range(n) for c in range(n))


def evaluate(agent, density, reward, eval_od, seed=777, max_steps=120):
    env = GridRouteEnv(boundary="open", reward=reward, seed=seed, max_steps=max_steps,
                       exit_density=density)
    by_k, arrived, rhos = {}, 0, []
    for od in eval_od:
        env.reset(od=od)
        rhos.append(rho_of({cell for cell, _ in env._open_links}))
        k = abs(od[0][0] - od[1][0][0]) + abs(od[0][1] - od[1][0][1])
        ok = 0
        for _ in range(env.max_steps):
            s = env._obs()
            a = agent.act(s, env.available_actions(), eps=0.0)
            _, _, done, info = env.step(a)
            if done:
                ok = int(info["outcome"] == "arrived")
                break
        arrived += ok
        by_k.setdefault(k, []).append(ok)
    by_k = {k: float(np.mean(v)) for k, v in sorted(by_k.items())}
    return arrived / len(eval_od), by_k, float(np.mean(rhos))


def run_cell(density, reward, seed, episodes, eval_od, max_steps=120):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = GridRouteEnv(boundary="open", reward=reward, seed=1000 + seed,
                       max_steps=max_steps, exit_density=density)
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
    return evaluate(agent, density, reward, eval_od, max_steps=max_steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="exit_density_results.json")
    a = ap.parse_args()

    eval_od = make_eval_od(n=200, n_side=N_SIDE, seed=12345)
    levels = [("100%", 1.0), ("50%", 0.5), ("25%", 0.25)]
    out = {}
    for label, density in levels:
        for reward in ("time_min", "aligned"):
            comps, rhos, ks = [], [], []
            for s in range(a.seeds):
                comp, by_k, rho = run_cell(density, reward, s, a.episodes, eval_od)
                comps.append(comp); rhos.append(rho); ks.append(by_k)
                print("  d=%-5s %-9s seed %d: completion %5.1f%%  rho=%.1f"
                      % (label, reward, s, 100 * comp, rho), flush=True)
            m = 100 * float(np.mean(comps))
            sd = 100 * float(np.std(comps))
            allk = sorted({k for d in ks for k in d})
            bykm = {k: 100 * float(np.mean([d[k] for d in ks if k in d])) for k in allk}
            out["%s_%s" % (label, reward)] = {
                "density": density, "completion_mean": m, "completion_std": sd,
                "completion_by_seed": [100 * c for c in comps],
                "rho_mean": float(np.mean(rhos)), "by_k": bykm,
            }
            print("== d=%-5s %-9s completion %5.1f%% (sd %4.1f)  rho=%.1f  by_k=%s"
                  % (label, reward, m, sd, np.mean(rhos),
                     {k: round(v, 1) for k, v in bykm.items()}), flush=True)
    with open(os.path.join(HERE, a.out), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
