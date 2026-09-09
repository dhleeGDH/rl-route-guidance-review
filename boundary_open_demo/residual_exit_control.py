# -*- coding: utf-8 -*-
"""Is the collapse the objective, or the terminal convention?

A reviewer put the sharpest form of the objection. Under Eq. (3) leaving costs one step and ends
the episode, so an exiting return is about -1 while a completed trip pays for every link. That
gap is what the exit-dominance bound formalises. But the same paper already charges the rest of
the step budget at the minimum traversal cost when a vehicle reaches the stranded node of
Nguyen-Dupuis, and that is the standard stochastic-shortest-path handling of an improper policy.
Apply the same convention to a wrong exit and the arithmetic changes: an exit at step t pays
1 + (T - t - 1), which is at least what a completed trip of the same length would pay.

If completion returns under the plain travel-time reward once the convention changes, the
diagnosis "the fault lies in the objective" does not survive as stated. If it does not return,
the diagnosis is separated from the convention and the paper is stronger. Either way the cell
belongs in the record, and it was missing.

Nothing but the exit charge changes: same learner, same state, same reward functions, same
budget, same evaluation set, same seeds as the four headline cells.

    python3 residual_exit_control.py            # 10 seeds, 3000 episodes
    python3 residual_exit_control.py --smoke
"""
import argparse, io, json, os, sys
import numpy as np, torch

# Pinned for the same reason four_cells_boundary_dest.py pins: the thread count of the
# linear-algebra library sets the order of summation in the gradient, and Section V-A
# states the count as one. Leaving the runner unpinned moved these cells between samples.
torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
# Rewrapped only when this file is the program, for the reason residual_exit_vi.py records.
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from env_boundary_dest import BoundaryDestEnv          # noqa: E402
from dqn import DQNAgent                               # noqa: E402


def eval_set(n=200, seed=999):
    rng = np.random.RandomState(seed)
    e = BoundaryDestEnv(boundary="open", reward="time_min", seed=1, max_steps=120)
    from env_boundary_dest import perimeter_links
    links = perimeter_links(e.n)
    out = []
    while len(out) < n:
        dl = links[rng.randint(len(links))]
        o = (rng.randint(e.n), rng.randint(e.n))
        d = abs(o[0] - dl[0][0]) + abs(o[1] - dl[0][1])
        if 3 <= d <= 8:
            out.append((o, dl))
    return out


def run(boundary, reward, seed, episodes, residual, ods, max_steps=120):
    np.random.seed(seed); torch.manual_seed(seed)
    env = BoundaryDestEnv(boundary=boundary, reward=reward, seed=1000 + seed, max_steps=max_steps)
    env.residual_on_exit = residual
    agent = DQNAgent(env.state_dim, env.n_actions, seed=seed)
    eps0, eps1, decay = 1.0, 0.05, int(0.6 * episodes)
    for ep in range(1, episodes + 1):
        eps = max(eps1, eps0 - (eps0 - eps1) * ep / decay)
        env.reset()
        for _ in range(env.max_steps):
            s, m = env._obs(), env.available_actions()
            a = agent.act(s, m, eps)
            s2, r, done, _ = env.step(a)
            agent.buf.add(s, a, r, s2, float(done), env.available_actions().astype(np.float32))
            agent.learn()
            if done:
                break
    ee = BoundaryDestEnv(boundary=boundary, reward=reward, seed=777, max_steps=max_steps)
    ee.residual_on_exit = residual
    arrived = 0
    for od in ods:
        ee.reset(od=od)
        for _ in range(ee.max_steps):
            s, m = ee._obs(), ee.available_actions()
            _, _, done, info = ee.step(agent.act(s, m, eps=0.0))
            if done:
                arrived += info["outcome"] == "arrived"
                break
    return 100.0 * arrived / len(ods)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.episodes, a.seeds = 300, 2
    ods, out = eval_set(), {"episodes": a.episodes, "seeds": a.seeds, "cells": {}}
    print("cell                                    completion (%)")
    for residual in (False, True):
        for boundary in ("closed", "open"):
            for reward in ("time_min", "aligned"):
                c = [run(boundary, reward, s, a.episodes, residual, ods) for s in range(a.seeds)]
                lab = "%s %-8s residual=%s" % (boundary, reward, residual)
                out["cells"][lab] = {"mean": float(np.mean(c)), "sd": float(np.std(c, ddof=1)),
                                     "per_seed": c}
                print("  %-38s %5.1f  (sd %4.1f)  %s"
                      % (lab, np.mean(c), np.std(c, ddof=1), [round(x, 1) for x in c]), flush=True)
    with io.open(os.path.join(HERE, "residual_exit_control.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nwrote residual_exit_control.json")


if __name__ == "__main__":
    main()
