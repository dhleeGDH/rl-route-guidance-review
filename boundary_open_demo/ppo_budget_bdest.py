# -*- coding: utf-8 -*-
"""Does a longer budget recover the policy-gradient learner on the current geometry?

WHY THIS EXISTS. Section V attributes the value-based learner's shortfall on the boundary-open
aligned cell to training budget: 30.8% at 3000 episodes and 96.7% at 8000. A reviewer observed
that the attribution rests on that learner alone, since the policy-gradient replication is
reported only at 3000 episodes, where it reaches 1.8%. If PPO also recovers at the longer budget,
the budget explanation is general. If it does not, the explanation is specific to the value-based
learner and the manuscript has to say so.

`ppo_budget.py` already answers a version of this at 9000 episodes, but on the earlier geometry,
in which the destination is an interior node. Every figure Section V now reports comes from the
boundary-destination geometry, where arrival is the event of leaving on the destination's own
outgoing link. This runs the same budget question on that geometry, so the answer is comparable
with the 30.8% and 96.7% the manuscript prints.

Only the budget changes against the four-cell PPO replication. Both aligned cells are re-run, so
the boundary-closed cell is measured on the same footing.

    python3 ppo_budget_bdest.py                  # 5 seeds, 8000 episodes, both aligned cells
    python3 ppo_budget_bdest.py --episodes 3000  # reproduce the four-cell figure

Writes ppo_budget_bdest.json next to this file.
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

# The rollout length is fixed in ppo_four_cells.py rather than in ppo.py, and the four-cell
# replication this extends uses 2048 transitions. It is repeated here so the two runs match.
ROLLOUT = 2048


def evaluate(agent, boundary, reward, eval_od, max_steps, seed=777):
    env = BoundaryDestEnv(boundary=boundary, reward=reward, seed=seed, max_steps=max_steps)
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
    env = BoundaryDestEnv(boundary=boundary, reward=reward, seed=1000 + seed,
                          max_steps=max_steps)
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
    ap.add_argument("--episodes", type=int, default=8000)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()

    od = make_eval_od(n=200, seed=12345)
    out = {"episodes": a.episodes, "seeds": a.seeds}
    for boundary in ("closed", "open"):
        comps = []
        for s in range(a.seeds):
            c = 100 * run_cell(boundary, "aligned", s, a.episodes, od)
            comps.append(c)
            print("  %-6s aligned seed %d: completion %5.1f%%" % (boundary, s, c), flush=True)
        out["%s_aligned" % boundary] = {"completion_mean": float(np.mean(comps)),
                                        "completion_sd": float(np.std(comps)),
                                        "per_seed": comps}
        print("== %-6s aligned %d ep: completion %5.1f%% (sd %.1f)"
              % (boundary, a.episodes, np.mean(comps), np.std(comps)), flush=True)
    with io.open(os.path.join(HERE, "ppo_budget_bdest.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote ppo_budget_bdest.json")


if __name__ == "__main__":
    main()
