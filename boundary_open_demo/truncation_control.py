# -*- coding: utf-8 -*-
"""Does bootstrapping at the step cap, rather than treating it as a terminal, move the cells?

Rounds 48b, 49b and 50b all raised the same point: Section V-A trains "with the step budget treated
as a terminal", which is the practice [79] documents as biasing a value function in a state that
carries no time index. Under an all-negative reward a timeout bootstrapped at zero is a value-0
absorbing option, so the reviewers offered it as an alternative account of the boundary-open aligned
cell's 35.1%.

This runs the four cells twice under one function: once exactly as published, and once with the
truncation bootstrapped, which is the single line `float(done)` becomes `float(done and not
timeout)`. Everything else, including the seeds, is the published configuration.

Control: the published arm must reproduce Table VIII, at 99.9 / 100.0 / 0.0 / 35.1. If it does not,
this file is measuring something other than the published cells and nothing here may be used.
"""
import argparse, io, json, os, sys

import numpy as np
import torch

torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from env_boundary_dest import BoundaryDestEnv, make_eval_od      # noqa: E402
from dqn import DQNAgent                                          # noqa: E402
import four_cells_boundary_dest as F                              # noqa: E402


def run_cell(boundary, reward, seed, episodes, eval_od, bootstrap_truncation, max_steps=120):
    """The published run_cell, with one line switched by bootstrap_truncation."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = BoundaryDestEnv(boundary=boundary, reward=reward, seed=1000 + seed,
                          max_steps=max_steps, state_sentinel="high")
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
            s2, r, done, info = env.step(a)
            timeout = done and info.get("outcome") == "timeout"
            flag = float(done and not timeout) if bootstrap_truncation else float(done)
            agent.buf.add(s, a, r, s2, flag, env.available_actions().astype(np.float32))
            agent.learn()
            if done:
                break
    return F.evaluate(agent, boundary, reward, eval_od, max_steps, state_sentinel="high")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    eval_od = make_eval_od(n=200, seed=12345)
    published = {("closed", "time_min"): 99.9, ("closed", "aligned"): 100.0,
                 ("open", "time_min"): 0.0, ("open", "aligned"): 35.1}
    out = {}
    for boundary in ("closed", "open"):
        for reward in ("time_min", "aligned"):
            row = {}
            for label, boot in (("terminal", False), ("bootstrapped", True)):
                v = [run_cell(boundary, reward, s, a.episodes, eval_od, boot)
                     for s in range(a.seeds)]
                row[label] = {"mean": float(np.mean(v)), "per_seed": v}
                print("  %-6s %-9s %-12s %5.1f%%" % (boundary, reward, label, np.mean(v)),
                      flush=True)
            ref = published[(boundary, reward)]
            got = round(row["terminal"]["mean"], 1)
            if (a.episodes, a.seeds) == (3000, 10):
                assert abs(got - ref) < 0.05, (
                    "control failed: published arm gives %.1f against Table VIII's %.1f"
                    % (got, ref))
            else:
                print("     (control skipped: %d episodes x %d seeds is not the published "
                      "configuration)" % (a.episodes, a.seeds), flush=True)
            row["published"] = ref
            row["delta"] = round(row["bootstrapped"]["mean"] - row["terminal"]["mean"], 1)
            out["%s_%s" % (boundary, reward)] = row
            print("     control: terminal arm reproduces Table VIII at %.1f%%, delta %+.1f"
                  % (ref, row["delta"]), flush=True)
    json.dump(out, io.open(os.path.join(HERE, "truncation_control.json"), "w",
                           encoding="utf-8"), indent=1)
    print("wrote truncation_control.json")


if __name__ == "__main__":
    main()
