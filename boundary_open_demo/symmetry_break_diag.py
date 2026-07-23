# -*- coding: utf-8 -*-
"""MC2 diagnostic: is the lattice learning gap (open aligned ~30.8%) a property of the
grid's homogeneity (identical degree, identical link lengths) rather than of the boundary?

The benchmark road networks are irregular; the lattice is uniform. If value aliasing among
symmetric states drives the shortfall, giving each cell a fixed, distinct static traversal
cost, so symmetric cells become distinguishable by their local cost features, should raise
open-aligned completion toward the benchmark level. The heterogeneity is static (identical
every episode, shared across seeds) so it is a property of the network, not added noise.

Runs open aligned at hetero in {0.0 (homogeneous baseline), 0.5, 1.0} over 10 seeds.
Everything else, eval set, reward, learner, budget, matches four_cells_boundary_dest.py.
"""
import numpy as np
import torch

from env_boundary_dest import BoundaryDestEnv, make_eval_od
from dqn import DQNAgent

_H = np.random.RandomState(7).rand(5, 5)  # fixed heterogeneous cost field, shared by all runs


class HeteroGridEnv(BoundaryDestEnv):
    def __init__(self, hetero=0.0, **kw):
        self.hetero = hetero
        super().__init__(**kw)

    def _edge_cost(self, r, c, a):
        from env import ACTIONS
        dr, dc = ACTIONS[a]
        nr, nc = r + dr, c + dc
        if not self._in_grid(nr, nc):
            return None
        base = 1.0 + self.hetero * _H[nr, nc]
        cong = self.congestion * self._cong[nr, nc] * (0.5 + 0.5 * np.sin(self._t / 6.0))
        return base + max(0.0, cong)


def evaluate(agent, hetero, eval_od, max_steps=120, seed=777):
    env = HeteroGridEnv(hetero=hetero, boundary="open", reward="aligned",
                        seed=seed, max_steps=max_steps)
    arrived = 0
    for od in eval_od:
        env.reset(od=od)
        for _ in range(env.max_steps):
            s = env._obs()
            with torch.no_grad():
                a = agent.act(s, env.available_actions(), eps=0.0)
            _, _, done, info = env.step(a)
            if done:
                arrived += info["outcome"] == "arrived"
                break
    return 100.0 * arrived / len(eval_od)


def run_cell(hetero, seed, episodes, eval_od, max_steps=120):
    np.random.seed(seed); torch.manual_seed(seed)
    env = HeteroGridEnv(hetero=hetero, boundary="open", reward="aligned",
                        seed=1000 + seed, max_steps=max_steps)
    agent = DQNAgent(env.state_dim, env.n_actions, seed=seed)
    eps0, eps1 = 1.0, 0.05; decay = int(0.6 * episodes)
    for ep in range(1, episodes + 1):
        eps = max(eps1, eps0 - (eps0 - eps1) * ep / decay)
        env.reset()
        for _ in range(env.max_steps):
            s = env._obs(); m = env.available_actions()
            a = agent.act(s, m, eps); s2, r, done, _ = env.step(a)
            agent.buf.add(s, a, r, s2, float(done), env.available_actions().astype(np.float32))
            agent.learn()
            if done:
                break
    return evaluate(agent, hetero, eval_od)


def main(seeds=10, episodes=3000):
    eval_od = make_eval_od(n=200, seed=12345)
    for hetero in (0.0, 0.5, 1.0):
        comps = [run_cell(hetero, s, episodes, eval_od) for s in range(seeds)]
        m, sd = np.mean(comps), np.std(comps)
        print("hetero %.1f  open aligned %.1f (sd %.1f)  seeds %s"
              % (hetero, m, sd, [round(x, 1) for x in comps]), flush=True)


if __name__ == "__main__":
    main()
