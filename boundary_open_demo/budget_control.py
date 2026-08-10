# -*- coding: utf-8 -*-
"""Training-budget control for the boundary-open destination-aligned cell.

WHY. Section V-C reports that the learning-level shortfall appears on the bespoke grid and on
neither benchmark network, and Section VI-C builds a research direction on that contrast. The
grid cell was trained for 3000 episodes and the Sioux Falls cell for 8000, so training length is
a rival explanation for exactly the difference being explained: the proposed mechanism is replay
composition, a quantity that accumulates with training. Sioux Falls is also the network closest
to the grid in size, at 24 nodes and 76 directed links against 25 and 80, which is what makes the
contrast load-bearing.

WHAT THIS RUNS. The boundary-open destination-aligned cell only, since the other three cells are
either at the optimum already (closed, both rewards) or analytically forced to zero (open,
travel-time). Each seed is trained once to 8000 episodes and evaluated at 3000 and at 8000, so
the two budgets are paired within a seed rather than compared across independent runs. Everything
else is the configuration of four_cells_boundary_dest.py, unchanged and imported from it.

    python3 budget_control.py                 # 10 seeds, checkpoints at 3000 and 8000
    python3 budget_control.py --seeds 3       # a shorter probe

Writes budget_control.json next to this file.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# four_cells_boundary_dest re-wraps sys.stdout at import time. Wrapping it here as well left
# two wrappers over one buffer, and collecting the first closed the buffer under the second.

from env_boundary_dest import BoundaryDestEnv, make_eval_od      # noqa: E402
from dqn import DQNAgent                                          # noqa: E402
from four_cells_boundary_dest import evaluate                     # noqa: E402


def run_with_checkpoints(boundary, reward, seed, checkpoints, eval_od, max_steps=120):
    """Train once to max(checkpoints), evaluating at each checkpoint along the way.

    The epsilon schedule decays over 0.6 of the FULL budget, exactly as run_cell does for its own
    budget. The 3000-episode checkpoint of an 8000-episode run is therefore not identical to a
    3000-episode run; it is the same agent earlier in training. The 3000-episode figure the
    manuscript reports comes from four_cells_boundary_dest.py and is quoted alongside for that
    reason.
    """
    episodes = max(checkpoints)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = BoundaryDestEnv(boundary=boundary, reward=reward, seed=1000 + seed, max_steps=max_steps)
    agent = DQNAgent(env.state_dim, env.n_actions, seed=seed)
    eps0, eps1 = 1.0, 0.05
    decay = int(0.6 * episodes)
    arrivals, out = 0, {}
    for ep in range(1, episodes + 1):
        eps = max(eps1, eps0 - (eps0 - eps1) * ep / decay)
        env.reset()
        for _ in range(env.max_steps):
            s = env._obs()
            m = env.available_actions()
            a = agent.act(s, m, eps)
            s2, r, done, info = env.step(a)
            agent.buf.add(s, a, r, s2, float(done),
                          env.available_actions().astype(np.float32))
            agent.learn()
            if done:
                arrivals += info.get("outcome") == "arrived"
                break
        if ep in checkpoints:
            out[ep] = {"completion": evaluate(agent, boundary, reward, eval_od, max_steps),
                       "arriving_episodes_pct": 100.0 * arrivals / ep}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--checkpoints", type=int, nargs="+", default=[3000, 8000])
    ap.add_argument("--out", default="budget_control.json")
    a = ap.parse_args()

    eval_od = make_eval_od(n=200, seed=12345)
    cps = sorted(a.checkpoints)
    per_seed = {c: [] for c in cps}
    arr = {c: [] for c in cps}
    t0 = time.time()
    for s in range(a.seeds):
        r = run_with_checkpoints("open", "aligned", s, set(cps), eval_od)
        for c in cps:
            per_seed[c].append(r[c]["completion"])
            arr[c].append(r[c]["arriving_episodes_pct"])
        print("seed %d  %s   [%.0fs]" % (s, "  ".join("%d ep: %5.1f%%" % (c, r[c]["completion"])
                                                      for c in cps), time.time() - t0), flush=True)

    res = {"cell": "open_aligned", "seeds": a.seeds,
           "manuscript_3000_episode_value": 30.75,
           "checkpoints": {str(c): {"mean": float(np.mean(per_seed[c])),
                                    "sd": float(np.std(per_seed[c])),
                                    "per_seed": per_seed[c],
                                    "arriving_episodes_pct_mean": float(np.mean(arr[c]))}
                           for c in cps}}
    with open(os.path.join(HERE, a.out), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    for c in cps:
        print("== %5d episodes  %5.1f%% (sd %4.1f)   arriving episodes %4.1f%%"
              % (c, np.mean(per_seed[c]), np.std(per_seed[c]), np.mean(arr[c])))
    print("wrote", a.out)


main()
