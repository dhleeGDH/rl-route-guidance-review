# -*- coding: utf-8 -*-
"""Component ablation of the aligned reward on the boundary-destination geometry, per seed.

WHY THIS EXISTS. Section V-B reports the ablation of the aligned reward's three terms on the
boundary-destination geometry: shaping alone 4.8%, arrival bonus and exit penalty 0.0% each,
and 29.4% (standard deviation 25.0) for the three together. Those figures were produced on an earlier
setup and only their summary statistics survive. The per-seed values behind the high-dispersion
combined cell are needed, on the precedent S-I.E2 set for the 48.0%
(31.8) cell. This reruns the four conditions with the four-cell protocol and records every
seed, so the supplementary material can print the values instead of a mean that hides a
near-bimodal spread.

    python3 ablation_bdest.py            # 4 conditions x 5 seeds, 3000 episodes
    python3 ablation_bdest.py --smoke    # 300 episodes, 2 seeds

Writes ablation_bdest.json next to this file.
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
from dqn import DQNAgent                                         # noqa: E402

CONDS = [
    ("shaping only",       dict(beta=1.0, r_goal=0.0,  r_exit=0.0)),
    ("shaping vanishing",  dict(beta=1.0, r_goal=0.0,  r_exit=0.0, vanish_potential=True)),
    ("arrival bonus only", dict(beta=0.0, r_goal=10.0, r_exit=0.0)),
    ("exit penalty only",  dict(beta=0.0, r_goal=0.0,  r_exit=5.0)),
    ("full aligned",       dict(beta=1.0, r_goal=10.0, r_exit=5.0)),
]


def evaluate(agent, eval_od, terms, max_steps=120, seed=777):
    env = BoundaryDestEnv(boundary="open", reward="aligned", seed=seed,
                          max_steps=max_steps, **terms)
    arrived = 0
    for od in eval_od:
        env.reset(od=od)
        for _ in range(env.max_steps):
            a = agent.act(env._obs(), env.available_actions(), eps=0.0)
            _, _, done, info = env.step(a)
            if done:
                arrived += info["outcome"] == "arrived"
                break
    return 100.0 * arrived / len(eval_od)


def run(seed, episodes, eval_od, terms, max_steps=120):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = BoundaryDestEnv(boundary="open", reward="aligned", seed=1000 + seed,
                          max_steps=max_steps, **terms)
    agent = DQNAgent(env.state_dim, env.n_actions, seed=seed)
    eps0, eps1 = 1.0, 0.05
    decay = int(0.6 * episodes)
    for ep in range(1, episodes + 1):
        eps = max(eps1, eps0 - (eps0 - eps1) * ep / decay)
        env.reset()
        for _ in range(env.max_steps):
            s = env._obs()
            m = env.available_actions()
            a = agent.act(s, m, eps)
            s2, r, done, _ = env.step(a)
            agent.buf.add(s, a, r, s2, float(done),
                          env.available_actions().astype(np.float32))
            agent.learn()
            if done:
                break
    return evaluate(agent, eval_od, terms, max_steps=max_steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.episodes, a.seeds = 300, 2

    od = make_eval_od(n=200, seed=12345)
    out = {"episodes": a.episodes, "seeds": a.seeds, "conditions": {}}
    for name, terms in CONDS:
        comps = []
        for s in range(a.seeds):
            c = run(s, a.episodes, od, terms)
            comps.append(c)
            print("  %-18s seed %d: completion %5.1f%%" % (name, s, c), flush=True)
        out["conditions"][name] = {"terms": terms,
                                   "completion_mean": float(np.mean(comps)),
                                   "completion_sd": float(np.std(comps)),
                                   "per_seed": comps}
        print("== %-18s completion %5.1f%% (sd %.1f)  seeds %s"
              % (name, np.mean(comps), np.std(comps),
                 [round(c, 1) for c in comps]), flush=True)
    with io.open(os.path.join(HERE, "ablation_bdest.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote ablation_bdest.json")


if __name__ == "__main__":
    main()
