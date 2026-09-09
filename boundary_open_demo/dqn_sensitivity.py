# -*- coding: utf-8 -*-
"""Is the value-based recovery on the boundary-open aligned cell a tuning artifact?

WHY THIS EXISTS. Cold review of v809, M9: "하이퍼파라미터 민감도 분석은 PPO에 대해서만 존재하고,
논문의 헤드라인을 만드는 DQN에 대해서는 없습니다 ... 97.7%라는 회복 수치가 학습자 아티팩트가
아님을 배제할 수 없습니다." The observation is correct. ppo_sensitivity.py sweeps the on-policy
learner because its 1.8% was the value under attack; the value-based learner, which produces the
headline 0.0% against 97.7%, was run at one setting.

Two cells are swept, not one. The 97.7% recovery is the number the comment names. The 0.0%
collapse is swept as well, at no extra argument, because a collapse that moved under any knob
would be the more damaging finding and exact value iteration alone does not rule out that a
learner reaches something the optimum does not.

One factor moves at a time from the shipped defaults (lr 1e-3, buffer 20000, target_every 200,
batch 64, epsilon floor 0.05 decayed over the first 60% of episodes), which answers the artifact
question directly: an artifact of a single mis-set knob recovers under some neighbouring value of
that knob. Every run calls run_cell from four_cells_boundary_dest.py, the function that produced
the published cells, so the sweep cannot differ from the headline in anything but the knob.

    python3 dqn_sensitivity.py               # 9 settings x 2 cells x 3 seeds, 8000 episodes
    python3 dqn_sensitivity.py --smoke       # 300 episodes, 1 seed, to check the harness

Writes dqn_sensitivity.json next to this file. A run at a different seed or episode count must
pass --out, since a published file is never overwritten by a variant.
"""
import argparse
import io
import json
import os
import sys

import numpy as np
import torch

torch.set_num_threads(1)          # 32 cores were driven to a load of 52 by unpinned runners

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from env_boundary_dest import make_eval_od          # noqa: E402
from four_cells_boundary_dest import run_cell       # noqa: E402

# No stdout wrapper here. four_cells_boundary_dest rebinds sys.stdout on import; wrapping the same
# buffer a second time closes it under the first wrapper and every print in this module fails.

# One factor at a time around the shipped defaults. The default row is run once and must
# reproduce the published cell, which is the control on the harness.
SETTINGS = [
    ("default",         {}, {}),
    ("lr 3e-4",         {"lr": 3e-4}, {}),
    ("lr 3e-3",         {"lr": 3e-3}, {}),
    ("buffer 5000",     {"buffer": 5000}, {}),
    ("buffer 50000",    {"buffer": 50000}, {}),
    ("target 50",       {"target_every": 50}, {}),
    ("target 500",      {"target_every": 500}, {}),
    ("batch 128",       {"batch": 128}, {}),
    ("eps floor 0.20",  {}, {"eps_floor": 0.20}),
    ("eps decay 0.30",  {}, {"eps_decay_frac": 0.30}),
]

CELLS = [("open aligned", "open", "aligned"),
         ("open time_min", "open", "time_min")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="dqn_sensitivity.json")
    a = ap.parse_args()
    if a.smoke:
        a.episodes, a.seeds, a.out = 300, 1, "dqn_sensitivity_smoke.json"

    eval_od = make_eval_od(n=200, seed=12345)
    out = {"episodes": a.episodes, "seeds": a.seeds, "settings": {}}
    for label, agent_kw, sched in SETTINGS:
        out["settings"][label] = {"agent_kw": agent_kw, "schedule": sched, "cells": {}}
        for cell_label, boundary, reward in CELLS:
            comps = []
            for s in range(a.seeds):
                # run_cell already returns a percentage. ppo_sensitivity.py multiplies by 100
                # because its own evaluate() returns a fraction; copying that line here reported
                # 1600.0% in the smoke run.
                c = run_cell(boundary, reward, s, a.episodes, eval_od,
                             agent_kw=agent_kw, **sched)
                comps.append(c)
                print("  %-16s %-14s seed %d: %5.1f%%" % (label, cell_label, s, c), flush=True)
            out["settings"][label]["cells"][cell_label] = {
                "completion_mean": float(np.mean(comps)),
                "completion_sd": float(np.std(comps)),
                "per_seed": comps}
            print("== %-16s %-14s %d ep: %5.1f%% (sd %.1f)"
                  % (label, cell_label, a.episodes, np.mean(comps), np.std(comps)), flush=True)
        with io.open(os.path.join(HERE, a.out), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)     # written after every setting, so a kill keeps the rest
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
