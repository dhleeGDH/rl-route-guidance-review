# -*- coding: utf-8 -*-
"""Training-budget control for the 7x7 SUMO reconstruction of [50]'s formulation.

WHY. sumo_recon.py runs 1500 episodes on a 49-node network and reports the boundary-closed
travel-time cell at 5.17%. Read on its own that cell contradicts Section IV-B, which holds a
travel-time objective adequate where no exit exists. The bespoke 25-node grid of Section V-B needs
8000 episodes to reach 96.7% on a cell whose optimum is 100.0%, and reads 30.8% at 3000, so 1500
episodes on a network twice that size is a rival explanation for the whole result. This separates
the two the same way boundary_open_demo/budget_control.py does for the grid.

WHAT THIS RUNS. All four cells. The two boundary-closed cells and the boundary-open aligned cell
are learning-limited and are the reason for the run. The boundary-open travel-time cell is
included as the control: its optimum is leaving, so a longer budget must NOT lift it, and a rise
there would mean the budget was doing something other than letting the learner converge.

Each seed is trained once to the largest checkpoint and evaluated at every checkpoint along the
way, so the budgets are paired within a seed rather than compared across independent runs.
Everything else is the configuration of sumo_recon.py, imported from it unchanged.

CAVEAT, the same one budget_control.py carries. Epsilon decays over 0.6 of the FULL budget, so the
1500-episode checkpoint of an 8000-episode run is not a 1500-episode run; it is the same agent
earlier in training under a slower schedule. sumo_recon.py's own 1500-episode figures are printed
alongside for that reason.

    python3 sumo_recon_budget.py                      # 5 seeds, checkpoints 1500 / 4000 / 8000
    python3 sumo_recon_budget.py --seeds 2 --checkpoints 500 1500

Writes sumo_recon_budget.json and sumo_recon_budget.csv next to this file.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "boundary_open_demo"))

import sumo_recon as recon                                    # noqa: E402
from dqn import DQNAgent                                      # noqa: E402

CELLS = [("closed", "time_min"), ("open", "time_min"),
         ("open", "aligned"), ("closed", "aligned")]

# sumo_recon.py's pre-correction 1500-episode result. It is INVALID and is carried only so the
# output shows what it is being replaced by. Two defects, either fatal. The 7x7 run used 5x5
# boundary geometry, since sumo_env._boundary_cells and _exit_action_at bound N as a default
# argument at import and never saw sumo_recon's `sumo_env.N = 7`. And its evaluation set drew
# 103 of 200 destinations at interior cells offering no arrival action, so half the set could
# not be completed by any policy. Both are fixed as of 2026-08-09.
RECON_1500 = {("closed", "time_min"): 5.17, ("open", "time_min"): 0.00,
              ("open", "aligned"): 28.00, ("closed", "aligned"): 100.00}


def evaluate_at(agent, boundary, reward, eval_od, tag):
    """Completion over the fixed evaluation set, on a fresh env so training state cannot leak."""
    env = recon.SumoGridEnv(boundary=boundary, reward=reward, seed=777, label="ev" + tag)
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
    return 100.0 * arrived / len(eval_od)


def run_with_checkpoints(boundary, reward, seed, checkpoints, eval_od, tag):
    episodes = max(checkpoints)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = recon.SumoGridEnv(boundary=boundary, reward=reward, seed=1000 + seed,
                            label="tr" + tag)
    agent = DQNAgent(env.state_dim, env.n_actions, seed=seed)
    eps0, eps1 = 1.0, 0.05
    decay = int(0.6 * episodes)
    arrivals, out = 0, {}
    for ep in range(1, episodes + 1):
        eps = max(eps1, eps0 - (eps0 - eps1) * ep / decay)
        s = env.reset()
        for _ in range(env.max_steps):
            m = env.available_actions()
            a = agent.act(s, m, eps)
            s2, r, done, info = env.step(a)
            agent.buf.add(s, a, r, s2, float(done),
                          env.available_actions().astype(np.float32))
            agent.learn()
            s = s2
            if done:
                arrivals += info.get("outcome") == "arrived"
                break
        if ep in checkpoints:
            out[ep] = {"completion": evaluate_at(agent, boundary, reward, eval_od,
                                                 "%s_%d" % (tag, ep)),
                       "arriving_episodes_pct": 100.0 * arrivals / ep}
    env.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--checkpoints", type=int, nargs="+", default=[1500, 4000, 8000])
    ap.add_argument("--out", default="sumo_recon_budget")
    a = ap.parse_args()

    cps = sorted(a.checkpoints)
    eval_od = recon.make_eval_od()
    t0 = time.time()
    result = {"network": "SUMO 7x7 reconstruction of [50]", "seeds": a.seeds,
              "checkpoints": cps, "recon_1500_episode_published": {},
              "cells": {}}

    for boundary, reward in CELLS:
        key = "%s_%s" % (boundary, reward)
        result["recon_1500_episode_published"][key] = RECON_1500[(boundary, reward)]
        per = {c: [] for c in cps}
        arr = {c: [] for c in cps}
        for s in range(a.seeds):
            tag = "%s%s%d" % (boundary[0], reward[0], s)
            r = run_with_checkpoints(boundary, reward, s, set(cps), eval_od, tag)
            for c in cps:
                per[c].append(r[c]["completion"])
                arr[c].append(r[c]["arriving_episodes_pct"])
            print("  %-8s %-9s seed %d  %s   [%.0fs]"
                  % (boundary, reward, s,
                     "  ".join("%d ep: %5.1f%%" % (c, r[c]["completion"]) for c in cps),
                     time.time() - t0), flush=True)
        result["cells"][key] = {
            str(c): {"mean": float(np.mean(per[c])), "sd": float(np.std(per[c])),
                     "per_seed": per[c],
                     "arriving_episodes_pct_mean": float(np.mean(arr[c]))}
            for c in cps}
        print("== %-8s %-9s %s" % (boundary, reward,
                                   "  ".join("%d ep: %5.1f%% (sd %4.1f)"
                                             % (c, np.mean(per[c]), np.std(per[c]))
                                             for c in cps)), flush=True)

    with open(os.path.join(HERE, a.out + ".json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(HERE, a.out + ".csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["boundary", "reward", "episodes", "completion_mean", "completion_sd",
                    "arriving_episodes_pct_mean", "recon_1500_published"])
        for boundary, reward in CELLS:
            key = "%s_%s" % (boundary, reward)
            for c in cps:
                d = result["cells"][key][str(c)]
                w.writerow([boundary, reward, c, "%.2f" % d["mean"], "%.2f" % d["sd"],
                            "%.2f" % d["arriving_episodes_pct_mean"],
                            "%.2f" % RECON_1500[(boundary, reward)]])
    print("\nwrote %s.json and %s.csv  [%.0fs total]" % (a.out, a.out, time.time() - t0))


main()
