# -*- coding: utf-8 -*-
"""The interior-destination control on the SUMO replication, not only on the bespoke grid.

WHY THIS EXISTS. Section V-A defines arrival as leaving on the destination's own outgoing boundary
link, which makes arrival and a wrong exit the same kind of event differing only in the link taken.
That equality is what isolates the reward as the cause. The bespoke grid already carries a control
in which the destination sits inside the network. A reviewer observed that the control runs on the
bespoke grid alone, so a reader cannot tell whether the result survives a microscopic simulator
whose car-following, junction control and lane changing the bespoke grid abstracts away.

This runs the same control on the SUMO 5x5 replication. The destination is drawn from the nine
interior cells, arrival is entering that cell, and every other element is the released environment's
own: the congestion field, the edge costs, the learner, the budget and the evaluation protocol.
On a boundary-open network the perimeter still offers exits, which is the contrast under test.

    OMP_NUM_THREADS=1 python3 sumo_interior_dest.py --seeds 10 --episodes 3000
"""
import argparse
import json
import os
import sys
import time
from itertools import product

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "boundary_open_demo"))
from sumo_env import SumoGridEnv, N, _boundary_cells   # noqa: E402
from dqn import DQNAgent                               # noqa: E402

torch.set_num_threads(1)

# an action index no real action can equal, since the four moves are 0, 1, 2 and 3; it divides
# into the observation slot as 3.0, outside the 0-to-1 range every real exit action occupies
NO_EXIT_INDEX = 9
INTERIOR = [cell for cell in product(range(N), range(N)) if cell not in set(_boundary_cells())]


class InteriorDestSumoEnv(SumoGridEnv):
    """Destination is an interior cell of the SUMO grid; arrival is entering it."""

    # The base class stores the destination's own exit action and both reads it into the
    # observation and compares an action against it. An interior cell carries no such action and
    # the helper returns None there, which the observation cannot divide. The slot instead holds
    # a sentinel no action index can equal, so the observation keeps its width and no exit is
    # ever opened at the destination.
    _dst_exit = NO_EXIT_INDEX

    @property
    def dst_exit(self):
        return self._dst_exit

    @dst_exit.setter
    def dst_exit(self, value):
        self._dst_exit = NO_EXIT_INDEX if value is None else value

    def reset(self, od=None):
        if od is None:
            while True:
                o = (self.rng.randint(N), self.rng.randint(N))
                d = INTERIOR[self.rng.randint(len(INTERIOR))]
                if abs(o[0] - d[0]) + abs(o[1] - d[1]) >= 3:
                    break
            od = (o, d)
        return super().reset(od=od)

    def step(self, a):
        obs, reward, done, info = super().step(a)
        if not self.done and self.pos == self.dst:
            if self.reward == "aligned":
                reward += self.align_strength * self.r_goal
            self.done = True
            self.outcome = "arrived"
            self._cleanup()
            return self._obs(), reward, True, {"outcome": self.outcome}
        return obs, reward, done, info


def make_eval_od(n=200, seed=12345):
    rng = np.random.RandomState(seed)
    ods = []
    while len(ods) < n:
        o = (rng.randint(N), rng.randint(N))
        d = INTERIOR[rng.randint(len(INTERIOR))]
        if abs(o[0] - d[0]) + abs(o[1] - d[1]) >= 3:
            ods.append((o, d))
    return ods


def evaluate(agent, boundary, reward, eval_od, tag):
    env = InteriorDestSumoEnv(boundary=boundary, reward=reward, seed=777, label="ev" + tag)
    arrived = 0
    for od in eval_od:
        s = env.reset(od=od)
        for _ in range(env.max_steps):
            a = agent.act(s, env.available_actions(), eps=0.0)
            s, _, done, info = env.step(a)
            if done:
                arrived += info["outcome"] == "arrived"
                break
    env.close()
    return arrived / len(eval_od)


def train_condition(boundary, reward, seed, episodes, eval_od, tag):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = InteriorDestSumoEnv(boundary=boundary, reward=reward, seed=1000 + seed, label="tr" + tag)
    agent = DQNAgent(env.state_dim, env.n_actions, seed=seed)
    eps0, eps1, decay = 1.0, 0.05, int(0.6 * episodes)
    for ep in range(1, episodes + 1):
        eps = max(eps1, eps0 - (eps0 - eps1) * ep / decay)
        s = env.reset()
        for _ in range(env.max_steps):
            m = env.available_actions()
            a = agent.act(s, m, eps)
            s2, r, done, _ = env.step(a)
            agent.buf.add(s, a, r, s2, float(done), env.available_actions().astype(np.float32))
            agent.learn()
            s = s2
            if done:
                break
    env.close()
    return evaluate(agent, boundary, reward, eval_od, tag + "e")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="sumo_interior_dest.json")
    ap.add_argument("--only", default=None,
                    help="run one condition, e.g. open_aligned, so the four run as four processes")
    a = ap.parse_args()
    if a.smoke:
        a.episodes, a.seeds = 120, 1

    # the control is worthless if the destination can sit on the perimeter after all
    assert not (set(INTERIOR) & set(_boundary_cells())), "interior set touches the perimeter"
    assert len(INTERIOR) == (N - 2) ** 2, "interior set is not the (N-2) square"
    print("SUMO %dx%d interior destination: %d interior cells, seeds=%d, episodes=%d"
          % (N, N, len(INTERIOR), a.seeds, a.episodes), flush=True)

    eval_od = make_eval_od()
    assert all(d in set(INTERIOR) for _, d in eval_od), "an evaluation destination is not interior"

    t0 = time.time()
    res = {}
    CONDS = [("closed", "time_min"), ("open", "time_min"),
             ("closed", "aligned"), ("open", "aligned")]
    if a.only:
        CONDS = [c for c in CONDS if "%s_%s" % c == a.only]
        assert CONDS, "no condition named %s" % a.only
    for boundary, reward in CONDS:
        comps = []
        for s in range(a.seeds):
            tag = "%s%s%d" % (boundary[0], reward[0], s)
            comps.append(train_condition(boundary, reward, s, a.episodes, eval_od, tag))
            print("  %-6s %-9s seed %d: %5.1f%%  [%ds]"
                  % (boundary, reward, s, 100 * comps[-1], time.time() - t0), flush=True)
        v = 100 * np.asarray(comps)
        res["%s_%s" % (boundary, reward)] = {
            "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "per_seed": [round(float(x), 1) for x in v],
            "seeds": a.seeds, "episodes": a.episodes}
        print("== %-6s %-9s completion %5.1f%% (sd %4.1f)"
              % (boundary, reward, v.mean(), v.std(ddof=1)), flush=True)

    json.dump({"network": "SUMO 5x5 replication, interior destination",
               "interior_cells": len(INTERIOR), "eval_pairs": len(eval_od),
               "episodes": a.episodes, "seeds": a.seeds, "cells": res},
              open(a.out, "w"), indent=1)
    print("wrote %s  [%ds total]" % (a.out, time.time() - t0))
