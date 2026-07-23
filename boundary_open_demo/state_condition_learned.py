# -*- coding: utf-8 -*-
"""Does the state requirement cost anything to a learned policy, not just to a planner?

Section V-B compares two planners that differ in the information their state has. The reward
requirement, by contrast, is demonstrated on trained policies. The two requirements are
therefore supported by different kinds of evidence, and the review says so. This closes that
asymmetry by running the state contrast through the same learner, reward, network, and budget
that the reward cells use, so the only thing that differs is the state.

The contrast is the review's own operational definition. A dynamic state is forecast-conditioned
when it gives the conditions each candidate link will present at the time it is traversed, and
instantaneous when it assumes present conditions persist. Both variants here see exactly the
same links over exactly the same horizon and carry exactly the same number of state dimensions.
Only the time index at which the downstream costs are read differs: the forecast variant reads
the cost of a link at the step the vehicle would reach it, and the instantaneous variant reads
every one of them at the departure step.

The cost model is the one the planner comparison uses, not the one the reward cells use.
env.py drives every link with a single global sinusoid, so at any step all links carry the
same temporal factor and the spatial field is fixed within an episode. A forecast of that
tells a vehicle nothing about which link to prefer, which makes the state requirement vacuous
by construction. state_condition_demo.py already records this and gives each cell its own
phase for exactly this reason. That model is adopted here unchanged, at amplitude 1.5 and
period 6, so that there is something for a forecast to be right about.

Design fixed before running: closed boundary, travel-time reward, 10 seeds, 3000 episodes,
horizon 3, the 200-pair evaluation set of Section V-A. The boundary is held closed on purpose.
An open boundary would let the reward requirement drive the outcome, and the point here is to
isolate the state. Under a travel-time reward on a closed network both variants complete trips,
so the quantity that can separate them is the realized travel time of the trips they complete,
which is what Section V-B measures for the planners.
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

from env import GridRouteEnv, ACTIONS, UNAVAIL_COST, N_SIDE   # noqa: E402
from dqn import DQNAgent                                      # noqa: E402
from train import make_eval_od                                # noqa: E402

HORIZON = 3
AMP = 1.5           # amplitude, as in state_condition_demo
PERIOD = 6.0        # steps per congestion cycle, as in state_condition_demo


class StateVariantEnv(GridRouteEnv):
    """The grid of Section V-A with the downstream lookahead added to the state.

    state_variant='forecast' reads each downstream link at the step it would be traversed.
    state_variant='instant' reads all of them at the current step, which is the review's
    definition of an instantaneous dynamic state applied to the same lookahead.
    """

    def __init__(self, state_variant="forecast", horizon=HORIZON, **kw):
        self.state_variant = state_variant
        self.horizon = horizon
        super().__init__(**kw)
        self.state_dim = 8 + 4

    def reset(self, od=None):
        # drawn before the base reset, which builds an observation on the way out
        self._phase = self.rng.rand(self.n, self.n) * 2.0 * np.pi
        return super().reset(od=od)

    def _cell_cost(self, r, c, t):
        """Cost of entering cell (r, c) at step t, under the planner comparison's model."""
        return 1.0 + max(0.0, AMP * self._cong[r, c]
                         * (0.5 + 0.5 * np.sin(2.0 * np.pi * t / PERIOD + self._phase[r, c])))

    def _edge_cost_at(self, r, c, a, t):
        """Traversal cost of moving from (r,c) by action a, evaluated at step t."""
        dr, dc = ACTIONS[a]
        nr, nc = r + dr, c + dc
        if not self._in_grid(nr, nc):
            return None
        return self._cell_cost(nr, nc, t)

    def _edge_cost(self, r, c, a):
        """The environment's own reward path uses the same model at the current step."""
        return self._edge_cost_at(r, c, a, self._t)

    def _lookahead(self):
        """Cost of continuing straight for `horizon` steps in each of the four directions.

        The forecast variant advances the clock with the vehicle. The instantaneous variant
        holds it at the current step. Off-grid continuation contributes UNAVAIL_COST, which
        is what the base observation already does for an unavailable move.
        """
        r0, c0 = self.pos
        out = []
        for a in range(4):
            dr, dc = ACTIONS[a]
            r, c, total = r0, c0, 0.0
            for h in range(self.horizon):
                t = self._t + h if self.state_variant == "forecast" else self._t
                ec = self._edge_cost_at(r, c, a, t)
                if ec is None:
                    total += UNAVAIL_COST
                    break
                total += ec
                r, c = r + dr, c + dc
            out.append(total / self.horizon)
        return out

    def _obs(self):
        base = super()._obs()
        return np.concatenate([base, np.array(self._lookahead(), dtype=np.float32)])


def run_cell(state_variant, seed, episodes, eval_od, max_steps=120):
    """Same epsilon schedule, buffer, and update cadence as train.train_condition."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = StateVariantEnv(state_variant=state_variant, boundary="closed", reward="time_min",
                          seed=1000 + seed, max_steps=max_steps)
    agent = DQNAgent(env.state_dim, env.n_actions, seed=seed)
    eps0, eps1 = 1.0, 0.05
    decay_eps = int(0.6 * episodes)
    for ep in range(1, episodes + 1):
        eps = max(eps1, eps0 - (eps0 - eps1) * ep / decay_eps)
        env.reset()
        for _ in range(env.max_steps):
            s = env._obs()
            mask = env.available_actions()
            a = agent.act(s, mask, eps)
            s2, r, done, _ = env.step(a)
            mask2 = env.available_actions()
            agent.buf.add(s, a, r, s2, float(done), mask2.astype(np.float32))
            agent.learn()
            if done:
                break
    return evaluate(agent, state_variant, eval_od, max_steps=max_steps)


def evaluate(agent, state_variant, eval_od, seed=777, max_steps=120):
    """Realized travel time is accumulated from the environment's own link costs.

    The reward is not read here. A completed trip contributes the sum of the costs of the
    links it actually traversed, at the steps it traversed them, so the measure is the same
    quantity for both variants regardless of what either was trained on.
    """
    env = StateVariantEnv(state_variant=state_variant, boundary="closed", reward="time_min",
                          seed=seed, max_steps=max_steps)
    arrived, times = 0, []
    for od in eval_od:
        env.reset(od=od)
        total = 0.0
        for _ in range(env.max_steps):
            s = env._obs()
            a = agent.act(s, env.available_actions(), eps=0.0)
            step_cost = env._edge_cost_at(env.pos[0], env.pos[1], a, env._t)
            _, _, done, info = env.step(a)
            if step_cost is not None:
                total += step_cost
            if done:
                if info["outcome"] == "arrived":
                    arrived += 1
                    times.append(total)
                break
    return 100.0 * arrived / len(eval_od), float(np.mean(times)) if times else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="state_condition_learned.json")
    a = ap.parse_args()

    eval_od = make_eval_od(n=200, n_side=N_SIDE, seed=12345)
    out = {}
    for variant in ("forecast", "instant"):
        comps, times = [], []
        for s in range(a.seeds):
            c, tm = run_cell(variant, s, a.episodes, eval_od)
            comps.append(c)
            times.append(tm)
            print("  %-8s seed %d: completion %5.1f%%  realized travel time %.3f"
                  % (variant, s, c, tm), flush=True)
        out[variant] = {"completion_mean": float(np.mean(comps)),
                        "completion_sd": float(np.std(comps)),
                        "time_mean": float(np.mean(times)),
                        "time_sd": float(np.std(times)),
                        "per_seed_completion": comps,
                        "per_seed_time": times}
        print("== %-8s completion %5.1f%% (sd %.1f)  time %.3f (sd %.3f)"
              % (variant, np.mean(comps), np.std(comps), np.mean(times), np.std(times)),
              flush=True)

    f, i = out["forecast"], out["instant"]
    excess = 100.0 * (i["time_mean"] - f["time_mean"]) / f["time_mean"]
    out["instant_excess_pct"] = excess
    print("\nrealized travel time of the instantaneous state, over the forecast-conditioned "
          "state: %+.1f%%" % excess)
    with open(os.path.join(HERE, a.out), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
