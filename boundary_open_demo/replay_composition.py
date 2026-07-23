# -*- coding: utf-8 -*-
"""Does replay composition explain the lattice learning gap? (Section VI-C hypothesis 3)

The open lattice under the destination-aligned reward makes arrival the optimum and the
value-based learner reaches 30.8% of it, against 94.2% and 100.0% on the two benchmark road
networks. Exit geometry and cost homogeneity are already excluded: thinning the exits raises
completion, and breaking the cost symmetry lowers it. Section VI-C names three candidates that
remain, and this script runs the one that is cheapest to separate.

The hypothesis: on a lattice whose whole perimeter absorbs, most episodes end by leaving after
a few steps, so the replay buffer fills with short exit trajectories and holds very few
transitions from trips that arrived. The learner then has almost no samples of the event the
reward is built to prefer.

The intervention keeps the environment, the reward, the network and the learner fixed and
changes only which transitions the learner sees. Transitions from episodes that ended in
arrival are written to a second buffer, and each minibatch draws a fixed share from it. If
replay composition drives the gap, completion rises with that share. If it does not, the
hypothesis is excluded the way exit geometry and cost homogeneity were.

Reports the arrival share of the ordinary buffer as well, which is the quantity the hypothesis
is about and which the manuscript has never measured.
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
from dqn import DQNAgent, ReplayBuffer        # noqa: E402
from train import make_eval_od                # noqa: E402


def evaluate(agent, eval_od, max_steps, seed=777):
    env = GridRouteEnv(boundary="open", reward="aligned", seed=seed, max_steps=max_steps)
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


def run(seed, episodes, eval_od, arrival_share, max_steps=120):
    """arrival_share = 0.0 reproduces the ordinary agent exactly."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = GridRouteEnv(boundary="open", reward="aligned", seed=1000 + seed, max_steps=max_steps)
    agent = DQNAgent(env.state_dim, env.n_actions, seed=seed)
    arr_buf = ReplayBuffer(20000, env.state_dim)

    eps0, eps1 = 1.0, 0.05
    decay = int(0.6 * episodes)
    n_arrived = 0
    n_ep = 0
    for ep in range(1, episodes + 1):
        env.reset()
        traj = []
        outcome = "timeout"
        for _ in range(env.max_steps):
            s = env._obs()
            m = env.available_actions()
            eps = max(eps1, eps0 - (eps0 - eps1) * ep / decay)
            a = agent.act(s, m, eps)
            _, r, done, info = env.step(a)
            tr = (s, a, r, env._obs(), float(done), env.available_actions().astype(np.float32))
            agent.buf.add(*tr)
            traj.append(tr)
            # mixed minibatch: the learner's own step, then one extra step drawn from the
            # arrival buffer at the requested share
            agent.learn()
            if arrival_share > 0 and len(arr_buf) >= agent.batch \
                    and np.random.rand() < arrival_share:
                main, agent.buf = agent.buf, arr_buf
                agent.learn()
                agent.buf = main
            if done:
                outcome = info["outcome"]
                break
        n_ep += 1
        if outcome == "arrived":
            n_arrived += 1
            for tr in traj:
                arr_buf.add(*tr)

    comp = evaluate(agent, eval_od, max_steps)
    return comp, 100.0 * n_arrived / n_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--shares", default="0.0,0.25,0.5")
    ap.add_argument("--out", default="replay_composition.json")
    a = ap.parse_args()

    eval_od = make_eval_od(n=200, n_side=N_SIDE, seed=12345)
    out = {}
    for sh in [float(x) for x in a.shares.split(",")]:
        comps, arr = [], []
        for s in range(a.seeds):
            c, ar = run(s, a.episodes, eval_od, sh)
            comps.append(c)
            arr.append(ar)
            print("  share %.2f seed %d: completion %5.1f%%  training episodes arriving %4.1f%%"
                  % (sh, s, c, ar), flush=True)
        out["%.2f" % sh] = {"arrival_share": sh, "seeds": a.seeds,
                            "completion_mean": float(np.mean(comps)),
                            "completion_sd": float(np.std(comps)),
                            "train_arrival_rate_mean": float(np.mean(arr)),
                            "per_seed_completion": comps}
        print("== share %.2f: completion %5.1f%% (sd %.1f), training arrivals %.1f%%"
              % (sh, np.mean(comps), np.std(comps), np.mean(arr)), flush=True)
    with open(os.path.join(HERE, a.out), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
