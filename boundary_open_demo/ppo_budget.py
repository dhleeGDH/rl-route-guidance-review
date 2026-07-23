# -*- coding: utf-8 -*-
"""Does the destination-aligned recovery return to PPO at a larger budget?

The four-cell PPO replication reaches 0.5% on the boundary-open aligned cell at 3000 episodes,
against 96.8% for the value-based learner. Exact value iteration under the same aligned reward
puts the optimal policy's completion at 100%, so that 0.5% is a learner failing to find the
optimum rather than an objective directing it away. The same pattern was resolved by budget on
SUMO (3000 to 6000) and on Sioux Falls (8000 to 16000). This tests it here.

Only the budget changes. Both aligned cells are re-run so the closed cell, which had not
converged either at 89.5%, is measured on the same footing.
"""
import io, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from ppo_four_cells import run_cell           # noqa: E402
from train import make_eval_od                # noqa: E402
from env import N_SIDE                        # noqa: E402

EPISODES, SEEDS = 9000, 5
od = make_eval_od(n=200, n_side=N_SIDE, seed=12345)
out = {}
for boundary in ("closed", "open"):
    comps = []
    for s in range(SEEDS):
        c = 100 * run_cell(boundary, "aligned", s, EPISODES, od)
        comps.append(c)
        print("  %-6s aligned  seed %d: completion %5.1f%%" % (boundary, s, c), flush=True)
    out["%s_aligned_9000" % boundary] = {"completion_mean": float(np.mean(comps)),
                                         "completion_sd": float(np.std(comps)),
                                         "per_seed": comps}
    print("== %-6s aligned  9000 ep: completion %5.1f%% (sd %.1f)"
          % (boundary, np.mean(comps), np.std(comps)), flush=True)
json.dump(out, open(os.path.join(HERE, "ppo_budget.json"), "w", encoding="utf-8"), indent=2)
print("wrote ppo_budget.json")
