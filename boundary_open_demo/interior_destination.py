# -*- coding: utf-8 -*-
"""Do the four cells hold when the destination is an interior node? (Section V-A abstraction)

The main experiment defines arrival as leaving on the destination's own outgoing boundary
link, which makes arrival and a wrong exit the same kind of event and differing only in which
link is taken. That equality is what isolates the reward as the cause. Real trips, however,
mostly end inside the network, so whether the collapse and the recovery survive the change is
an open question.

This wraps the released environment and moves the destination inside. Arrival is entering a
designated interior cell, exits keep their meaning on the open boundary, and everything else
(the congestion model, the costs, the learner, the budget, the evaluation protocol) is the
environment's own. The travel-time reward is unchanged. The aligned reward keeps the same
three terms with the potential measured to the interior cell.

The analytical bound of Section V-C does not depend on where the destination sits: it bounds a
completing route from above by its own graph distance and an exiting route from below by the
distance to the nearest boundary. This script is the empirical counterpart of that argument.
"""
import argparse
import io
import json
import os
import sys

import numpy as np
import torch

# Floating-point addition is not associative, so the thread count of the linear-algebra library
# changes the order of summation and moves a training trajectory. Supplementary S-I states that
# the count is pinned to one for every cell of the study; this runner was the one that was not.
torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from env import GridRouteEnv, N_SIDE, perimeter_links   # noqa: E402
from dqn import DQNAgent                                # noqa: E402

BETA, R_G, R_X = 1.0, 10.0, 5.0


class InteriorDestEnv(GridRouteEnv):
    """Destination is an interior cell; arrival is entering it."""

    def reset(self, od=None):
        # od is (origin, boundary_link, goal_cell) here; the base class only understands the
        # first two, so the goal is split off before delegating.
        goal = od[2] if od is not None else None
        base_od = (od[0], od[1]) if od is not None else None
        # the base reset ends by calling _obs(), which needs a goal, so it is seeded first
        self.goal = goal if goal is not None else getattr(self, "goal", (1, 1))
        super().reset(od=base_od)
        # place the destination inside, away from the perimeter, and keep the origin separated
        inner = [(r, c) for r in range(1, self.n - 1) for c in range(1, self.n - 1)]
        if od is None:
            # The interior of a 5x5 grid is its central 3x3, so an origin near the middle has
            # no interior goal at separation 3 and sampling the goal alone never terminates.
            # The origin is resampled with it, over the whole grid, as training does.
            for _ in range(500):
                o = (self.rng.randint(self.n), self.rng.randint(self.n))
                g = inner[self.rng.randint(len(inner))]
                if abs(o[0] - g[0]) + abs(o[1] - g[1]) >= 3:
                    self.pos, self.goal = o, g
                    break
            else:
                self.pos, self.goal = (0, 0), (self.n - 2, self.n - 2)
        else:
            self.goal = goal
        # the base builds _open_links as every perimeter link EXCEPT its own destination link,
        # which it holds open through _is_dst_link. That predicate is disowned above, so the
        # link belongs in the set on the open condition and nowhere on the closed one.
        if self.boundary != "closed" and self._dst_link is not None:
            self._open_links = set(self._open_links) | {self._dst_link}
        return self._obs()

    def _obs(self):
        o = super()._obs().copy()
        o[2] = self.goal[0] / (self.n - 1)      # destination cell replaces the boundary cell
        o[3] = self.goal[1] / (self.n - 1)
        o[4] = 0.0                              # no destination-link direction to carry
        return o

    def _phi(self, r, c):
        return abs(r - self.goal[0]) + abs(c - self.goal[1])

    def _is_dst_link(self, r, c, a):
        """No link is the destination here: arrival is entering an interior cell.

        The base class treats its destination link as arrival AND as permanently open, which
        left one takeable exit on the condition this control calls boundary-closed. Exact value
        iteration on the shipped construction returns 56.5% completion for the travel-time
        optimum, against 100.0% once that link closes with the rest of the perimeter: the cell
        was boundary-open and the 80.2% it reported was an optimum leaving, not a learner
        falling short. Disowning the link here closes the condition and hands the same link to
        the open condition as an ordinary exit, charged like every other.
        """
        return False

    def step(self, action):
        obs, r, done, info = super().step(action)
        # The base already applies the aligned shaping through the overridden _phi, so the
        # subclass adding it again doubled beta on every in-grid move: one step measured
        # 0.8737 where single shaping gives -0.1263. Arrival keeps its own terminal bonus.
        if self.pos == self.goal and info["outcome"] != "exited":
            done, self.done = True, True
            info["outcome"] = self.outcome = "arrived"
            if self.reward == "aligned":
                r = r + R_G
        return obs, r, done, info


def make_od(n_pairs, n_side, seed):
    rng = np.random.RandomState(seed)
    inner = [(r, c) for r in range(1, n_side - 1) for c in range(1, n_side - 1)]
    links = perimeter_links(n_side)
    out = []
    while len(out) < n_pairs:
        o = (rng.randint(n_side), rng.randint(n_side))
        g = inner[rng.randint(len(inner))]
        if abs(o[0] - g[0]) + abs(o[1] - g[1]) < 3:
            continue
        out.append((o, links[0], g))
    return out


def evaluate(agent, boundary, reward, eval_od, max_steps, seed=777):
    env = InteriorDestEnv(boundary=boundary, reward=reward, seed=seed, max_steps=max_steps)
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


def run_cell(boundary, reward, seed, episodes, eval_od, max_steps=120, **reward_kw):
    """reward_kw passes the reward weights through to the environment, so a single-term arm can
    be run under this construction. Passing none reproduces the published cells exactly."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = InteriorDestEnv(boundary=boundary, reward=reward, seed=1000 + seed,
                          max_steps=max_steps, **reward_kw)
    agent = DQNAgent(env.state_dim, env.n_actions, seed=seed)
    eps0, eps1 = 1.0, 0.05
    decay = int(0.6 * episodes)
    for ep in range(1, episodes + 1):
        env.reset()
        for _ in range(env.max_steps):
            s = env._obs()
            m = env.available_actions()
            a = agent.act(s, m, max(eps1, eps0 - (eps0 - eps1) * ep / decay))
            _, r, done, _ = env.step(a)
            agent.buf.add(s, a, r, env._obs(), float(done),
                          env.available_actions().astype(np.float32))
            agent.learn()
            if done:
                break
    return evaluate(agent, boundary, reward, eval_od, max_steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="interior_destination.json")
    a = ap.parse_args()
    eval_od = make_od(200, N_SIDE, seed=12345)
    out = {}
    for boundary in ("closed", "open"):
        for reward in ("time_min", "aligned"):
            comps = [run_cell(boundary, reward, s, a.episodes, eval_od)
                     for s in range(a.seeds)]
            out["%s_%s" % (boundary, reward)] = {
                "mean": float(np.mean(comps)), "sd": float(np.std(comps)),
                "per_seed": comps}
            print("== %-6s %-9s completion %5.1f%% (sd %.1f)"
                  % (boundary, reward, np.mean(comps), np.std(comps)), flush=True)
    with open(os.path.join(HERE, a.out), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
