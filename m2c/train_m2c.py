# -*- coding: utf-8 -*-
"""M2-C step 5: retrain the destination-aligned cells, exit-terminal potential retained.

The learner, the seeds, the budgets and the evaluation set are the published ones; only the reward
changes, through the mixin of shaping_m2.py. Travel-time cells are not retrained: their reward has
no shaping term.

    python3 train_m2.py <group>        groups: grid, terms, depth
"""
import json, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "repo_v11", "boundary_open_demo"))
sys.path.insert(0, HERE)

import numpy as np
import torch
torch.set_num_threads(1)

from env_boundary_dest import BoundaryDestEnv, make_eval_od     # noqa: E402
from dqn import DQNAgent                                        # noqa: E402
from shaping_m2c import DiscountConsistentShapingC                # noqa: E402

GAMMA = 0.99


class M2CBoundaryDestEnv(DiscountConsistentShapingC, BoundaryDestEnv):
    gamma_shaping = GAMMA


def evaluate(agent, boundary, reward, eval_od, max_steps, terms, seed=777):
    """The published harness of four_cells_curves.evaluate, with the M2 environment."""
    env = M2CBoundaryDestEnv(boundary=boundary, reward=reward, seed=seed, max_steps=max_steps, **terms)
    arrived = 0
    for od in eval_od:
        env.reset(od)
        for _ in range(max_steps):
            s = env._obs()
            a = agent.act(s, env.available_actions(), 0.0)
            _, r, done, info = env.step(a)
            if done:
                arrived += info["outcome"] == "arrived"
                break
    return 100.0 * arrived / len(eval_od)


def run_cell(boundary, reward, seed, episodes, eval_od, max_steps=120, terms=None):
    terms = terms or {}
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = M2CBoundaryDestEnv(boundary=boundary, reward=reward, seed=1000 + seed,
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
            agent.buf.add(s, a, r, s2, float(done), env.available_actions().astype(np.float32))
            agent.learn()
            if done:
                break
    return agent, env


def cell_completion(boundary, reward, seed, episodes, eval_od, max_steps=120, terms=None):
    agent, _ = run_cell(boundary, reward, seed, episodes, eval_od, max_steps, terms)
    return evaluate(agent, boundary, reward, eval_od, max_steps, terms or {})


GROUPS = {
    # four groups, one per process, so the wall clock is set by the slowest
    "g1": [("open", "aligned", {}, b) for b in (3000, 8000)],
    "g2": [("closed", "aligned", {}, b) for b in (3000, 8000)],
    "g3": [("open", "aligned", dict(beta=1.0, r_goal=0.0, r_exit=0.0), b) for b in (3000, 8000)]
          + [("open", "aligned", dict(beta=0.0, r_goal=10.0, r_exit=0.0), b) for b in (3000, 8000)],
    "g4": [("open", "aligned", dict(beta=0.0, r_goal=0.0, r_exit=5.0), b) for b in (3000, 8000)]
          + [("open", "aligned", dict(beta=1.0, r_goal=10.0, r_exit=0.0), b) for b in (3000, 8000)],
}


def main():
    group = sys.argv[1]
    eval_od = make_eval_od(n=200, seed=12345)
    out = {}
    for boundary, reward, terms, episodes in GROUPS[group]:
        key = "%s|%s|%s|%d" % (boundary, reward, json.dumps(terms, sort_keys=True), episodes)
        t0 = time.time()
        vals = [cell_completion(boundary, reward, s, episodes, eval_od, terms=terms)
                for s in range(10)]
        out[key] = dict(mean=round(float(np.mean(vals)), 1), values=[round(v, 1) for v in vals],
                        seconds=round(time.time() - t0, 1))
        print("%-58s mean %5.1f  (%.0fs)" % (key, out[key]["mean"], out[key]["seconds"]), flush=True)
    json.dump(out, open(os.path.join(HERE, "train_m2c_%s.json" % group), "w"), indent=1)


if __name__ == "__main__":
    main()
