# -*- coding: utf-8 -*-
"""Is the PPO shortfall on the boundary-open aligned cell a tuning artifact?

WHY THIS EXISTS. The manuscript reports the policy-gradient learner at 2.7% on the boundary-open
aligned cell at 8000 episodes, against 96.7% for the value-based learner, and concludes that an
optimal-arrival objective is necessary but not sufficient for a learner. A collapse this severe could instead be an exploration or step-size artifact of the single
hyperparameter setting used, in which case the insufficiency claim would rest on one point of a
sensitive surface. The four-cell protocol deliberately runs untuned defaults; the question is
whether any ordinary setting of the usual knobs recovers completion.

One factor moves at a time from the shipped defaults (lr 3e-4, entropy 0.01, GAE lambda 0.95),
which answers the artifact question directly: an artifact of a single mis-set knob would recover
under some neighbouring value of that knob. Every run uses the boundary-open aligned cell, the
budget at which the value-based learner recovers, and the evaluation OD set of the four-cell
protocol.

    python3 ppo_sensitivity.py               # 7 settings x 3 seeds, 8000 episodes each
    python3 ppo_sensitivity.py --smoke       # 400 episodes, 1 seed, to check the harness

Writes ppo_sensitivity.json next to this file.
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

from env_boundary_dest import BoundaryDestEnv, make_eval_od      # noqa: E402
from ppo import PPOAgent                                         # noqa: E402

ROLLOUT = 2048   # matches ppo_four_cells.py

# One factor at a time around the shipped defaults. The default row is run once.
SETTINGS = [
    ("default",        {}),
    ("ent 0.0",        {"ent_coef": 0.0}),
    ("ent 0.05",       {"ent_coef": 0.05}),
    ("lr 1e-4",        {"lr": 1e-4}),
    ("lr 1e-3",        {"lr": 1e-3}),
    ("lambda 0.90",    {"lam": 0.90}),
    ("lambda 0.99",    {"lam": 0.99}),
]


def evaluate(agent, eval_od, max_steps=120, seed=777):
    env = BoundaryDestEnv(boundary="open", reward="aligned", seed=seed, max_steps=max_steps)
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


def run(seed, episodes, eval_od, overrides, max_steps=120):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = BoundaryDestEnv(boundary="open", reward="aligned", seed=1000 + seed,
                          max_steps=max_steps)
    agent = PPOAgent(env.state_dim, env.n_actions, seed=seed, **overrides)
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
    return evaluate(agent, eval_od, max_steps=max_steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.episodes, a.seeds = 400, 1

    od = make_eval_od(n=200, seed=12345)
    out = {"episodes": a.episodes, "seeds": a.seeds, "cell": "open aligned", "settings": {}}
    for label, ov in SETTINGS:
        comps = []
        for s in range(a.seeds):
            c = 100 * run(s, a.episodes, od, ov)
            comps.append(c)
            print("  %-12s seed %d: completion %5.1f%%" % (label, s, c), flush=True)
        out["settings"][label] = {"overrides": ov,
                                  "completion_mean": float(np.mean(comps)),
                                  "completion_sd": float(np.std(comps)),
                                  "per_seed": comps}
        print("== %-12s %d ep: completion %5.1f%% (sd %.1f)"
              % (label, a.episodes, np.mean(comps), np.std(comps)), flush=True)
    with io.open(os.path.join(HERE, "ppo_sensitivity.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote ppo_sensitivity.json")


if __name__ == "__main__":
    main()
