# -*- coding: utf-8 -*-
"""How often does training under the arrival term alone reach the destination at all?

Table IX reports an optimum at arrival on every pair under an arrival term alone against a learner
completing 0.0% on every seed, and rounds 61 and 64 excluded the magnitude, the budget, the state
sentinel and the encoding of arrival by running each. The explanation left standing is that the
learner never meets the arriving transition: with rho = 2 an epsilon-greedy walk reaches the
perimeter in a couple of steps, and an episode ending there carries no arrival to learn from.

That is a counting question, not an opinion. This runs the same environment and the same learner
as the published arm and records, per seed, how many training episodes end in arrival and how many
end at a wrong exit. The four-cell aligned reward is run beside it as the reference, since that
reward reaches 35.1% and must therefore meet arrival during training.

    OMP_NUM_THREADS=1 python3 arrival_visits.py --seeds 10 --episodes 3000
"""
import argparse
import io
import json
import os
from collections import Counter

import numpy as np
import torch

from env_boundary_dest import BoundaryDestEnv, make_eval_od
from dqn import DQNAgent

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))

ARMS = [("arrival term alone", dict(beta=0.0, r_goal=10.0, r_exit=0.0)),
        ("all three terms", dict(beta=1.0, r_goal=10.0, r_exit=5.0))]


def train_and_count(seed, episodes, terms, max_steps=120):
    """The training loop of four_cells_boundary_dest.run_cell, with the terminal of every
    training episode tallied. Nothing else differs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = BoundaryDestEnv(boundary="open", reward="aligned", seed=1000 + seed,
                          max_steps=max_steps, **terms)
    agent = DQNAgent(env.state_dim, env.n_actions, seed=seed)
    eps0, eps1, decay = 1.0, 0.05, int(0.6 * episodes)
    tally = Counter()
    for ep in range(1, episodes + 1):
        eps = max(eps1, eps0 - (eps0 - eps1) * ep / decay)
        env.reset()
        for _ in range(env.max_steps):
            s, m = env._obs(), env.available_actions()
            a = agent.act(s, m, eps)
            s2, r, done, info = env.step(a)
            agent.buf.add(s, a, r, s2, float(done),
                          env.available_actions().astype(np.float32))
            agent.learn()
            if done:
                tally[info["outcome"]] += 1
                break
        else:
            tally["timeout"] += 1
    return tally


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--out", default=os.path.join(HERE, "arrival_visits.json"))
    a = ap.parse_args()
    res = {}
    print("training terminals on the boundary-open grid, %d seeds x %d episodes"
          % (a.seeds, a.episodes))
    for name, terms in ARMS:
        per_seed = [train_and_count(s, a.episodes, terms) for s in range(a.seeds)]
        arr = [t["arrived"] for t in per_seed]
        ex = [t["exited"] for t in per_seed]
        res[name] = {"terms": terms,
                     "arrived_per_seed": arr, "exited_per_seed": ex,
                     "arrived_mean": float(np.mean(arr)),
                     "arrived_share_pct": round(100.0 * float(np.mean(arr)) / a.episodes, 3),
                     "exited_share_pct": round(100.0 * float(np.mean(ex)) / a.episodes, 1)}
        print("  %-20s arrivals %7.1f of %d episodes (%.3f%%), exits %.1f%%"
              % (name, res[name]["arrived_mean"], a.episodes,
                 res[name]["arrived_share_pct"], res[name]["exited_share_pct"]), flush=True)
        print("     per seed: %s" % arr, flush=True)
    json.dump({"seeds": a.seeds, "episodes": a.episodes, "arms": res},
              io.open(a.out, "w", encoding="utf-8"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
