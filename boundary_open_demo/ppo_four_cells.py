# -*- coding: utf-8 -*-
"""The four boundary-and-reward cells again, with a policy-gradient learner.

Section V reports the four cells for a value-based learner. The 0% collapse under the
travel-time reward is algorithm-free by the bound and the value iteration of Section V-C, but
the size of the recovery under the destination-aligned reward is not, and neither is the
ablation. This runs the same four cells with PPO in place of the Deep Q-Network and changes
nothing else: same grid, same cost model, same reward definitions, same 200-pair evaluation
set, same 120-step budget.

Design fixed before running: 4 cells, 5 seeds, 3000 episodes, rollout of 2048 transitions
between updates. A learner that fails to learn would tell us nothing about the reward, so the
closed destination-aligned cell is the sanity check: it must reach high completion for the run
to be informative, and that is reported alongside the rest rather than checked privately.
"""
import argparse
import io
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from env import GridRouteEnv, N_SIDE          # noqa: E402
from ppo import PPOAgent                      # noqa: E402
from train import make_eval_od                # noqa: E402

ROLLOUT = 2048


def evaluate(agent, boundary, reward, eval_od, seed=777, max_steps=120):
    env = GridRouteEnv(boundary=boundary, reward=reward, seed=seed, max_steps=max_steps)
    arrived = 0
    for od in eval_od:
        env.reset(od=od)
        for _ in range(env.max_steps):
            a, _, _ = agent.act(env._obs(), env.available_actions(), greedy=True)
            _, _, done, info = env.step(a)
            if done:
                arrived += info["outcome"] == "arrived"
                break
    return arrived / len(eval_od)


def run_cell(boundary, reward, seed, episodes, eval_od, max_steps=120):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = GridRouteEnv(boundary=boundary, reward=reward, seed=1000 + seed, max_steps=max_steps)
    agent = PPOAgent(env.state_dim, env.n_actions, seed=seed)
    steps = 0
    for _ in range(episodes):
        env.reset()
        for _ in range(env.max_steps):
            s = env._obs()
            m = env.available_actions()
            a, lp, val = agent.act(s, m)
            _, r, done, _ = env.step(a)
            agent.store(s, a, lp, val, r, done, m)
            steps += 1
            if steps % ROLLOUT == 0:
                agent.update()
            if done:
                break
    agent.update()
    return evaluate(agent, boundary, reward, eval_od, max_steps=max_steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="ppo_four_cells.json")
    a = ap.parse_args()

    eval_od = make_eval_od(n=200, n_side=N_SIDE, seed=12345)
    out = {}
    for boundary in ("closed", "open"):
        for reward in ("time_min", "aligned"):
            comps = []
            for s in range(a.seeds):
                c = 100 * run_cell(boundary, reward, s, a.episodes, eval_od)
                comps.append(c)
                print("  %-6s %-9s seed %d: completion %5.1f%%" % (boundary, reward, s, c),
                      flush=True)
            out["%s_%s" % (boundary, reward)] = {
                "completion_mean": float(np.mean(comps)),
                "completion_sd": float(np.std(comps)),
                "per_seed": comps}
            print("== %-6s %-9s completion %5.1f%% (sd %.1f)"
                  % (boundary, reward, np.mean(comps), np.std(comps)), flush=True)
    with open(os.path.join(HERE, a.out), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
