# -*- coding: utf-8 -*-
"""The four cells with the destination as an outgoing boundary link.

Same learner, same budget, same evaluation protocol as Section V-A. The only change is the
one env_boundary_dest.py describes: the destination is a link that leaves the lattice, the
closed condition opens no other perimeter link, and the open condition opens the rest.

Run to compare against the recorded values 85.0 / 0.0 closed and open travel-time, and
98.8 / 96.8 closed and open destination-aligned.
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

from env_boundary_dest import BoundaryDestEnv, make_eval_od   # noqa: E402
from dqn import DQNAgent                                       # noqa: E402


def evaluate(agent, boundary, reward, eval_od, max_steps, seed=777):
    env = BoundaryDestEnv(boundary=boundary, reward=reward, seed=seed, max_steps=max_steps)
    arrived = 0
    for od in eval_od:
        env.reset(od=od)
        for _ in range(env.max_steps):
            s = env._obs()
            a = agent.act(s, env.available_actions(), eps=0.0)
            _, _, done, info = env.step(a)
            if done:
                arrived += info["outcome"] == "arrived"
                break
    return 100.0 * arrived / len(eval_od)


def run_cell(boundary, reward, seed, episodes, eval_od, max_steps=120):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = BoundaryDestEnv(boundary=boundary, reward=reward, seed=1000 + seed,
                          max_steps=max_steps)
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
    return evaluate(agent, boundary, reward, eval_od, max_steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="four_cells_boundary_dest.json")
    a = ap.parse_args()

    eval_od = make_eval_od(n=200, seed=12345)
    recorded = {("closed", "time_min"): 85.0, ("open", "time_min"): 0.0,
                ("closed", "aligned"): 98.8, ("open", "aligned"): 96.8}
    out = {}
    for boundary in ("closed", "open"):
        for reward in ("time_min", "aligned"):
            comps = [run_cell(boundary, reward, s, a.episodes, eval_od)
                     for s in range(a.seeds)]
            key = "%s_%s" % (boundary, reward)
            out[key] = {"mean": float(np.mean(comps)), "sd": float(np.std(comps)),
                        "per_seed": comps, "recorded": recorded[(boundary, reward)]}
            print("== %-6s %-9s %5.1f%% (sd %4.1f)   recorded %5.1f%%"
                  % (boundary, reward, np.mean(comps), np.std(comps),
                     recorded[(boundary, reward)]), flush=True)
    with open(os.path.join(HERE, a.out), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
