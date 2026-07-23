"""Is the boundary-open collapse a property of a terminal (absorbing) exit, or of open
topology itself? (Section V robustness check.)

The main experiment models a vehicle that leaves the modeled area as an absorbing
exit -- the vehicle-removal semantics of the corpus's dominant microscopic
simulator. A vehicle leaving a real study area is not annihilated, however: it drives on
unmodeled exterior roads and can return at some cost.

This script adds a non-terminal "costly_return" boundary: leaving costs a fixed
detour penalty return_cost and re-enters the vehicle at the same node, the episode
continuing. Under the corpus-typical travel-time-minimizing reward we compare:

    open           (terminal exit)     -> expected ~0%   (main result)
    costly_return  (non-terminal exit) -> expected ~closed rate

for a range of return_cost. If completion recovers once the exit is made
non-terminal, the collapse is localized to terminal-exit + unaligned-reward rather
than to open topology, which is exactly the configuration the corpus's dominant
evaluation (SUMO removal + travel-time reward) embodies. A costly return does NOT
route the vehicle to the destination through the exterior, so trip completion stays
a faithful measure of in-network navigation (no external-shortcut confound).
"""

import numpy as np
import torch

from env import GridRouteEnv
from dqn import DQNAgent
from sweep_extra import make_eval_od, evaluate, train_eval


def study_costly_return(seeds=5, episodes=3000, return_costs=(1.0, 3.0, 5.0),
                        max_steps=50, out_csv="costly_return_summary.csv"):
    eval_od = make_eval_od(5)
    out = []

    # reference: the two main-experiment cells recomputed here for a like-for-like
    # comparison (same seeds, eval set, and budget as the costly_return cells).
    for boundary, reward, rc in [("closed", "time_min", None),
                                 ("open", "time_min", None)]:
        env_kw = dict(boundary=boundary, reward=reward, n_side=5, max_steps=max_steps)
        comps = [train_eval(env_kw, s, episodes, eval_od) for s in range(seeds)]
        out.append((f"{boundary}_{reward}", np.nan, np.mean(comps), np.std(comps)))
        print(f"{boundary:13} {reward:9} return_cost=  --  "
              f"completion={np.mean(comps):.3f} +/- {np.std(comps):.3f}", flush=True)

    # costly_return under the corpus-typical travel-time reward, swept over the
    # detour penalty. Non-terminal exit -> leaving is pure loss -> recovery expected.
    for rc in return_costs:
        env_kw = dict(boundary="costly_return", reward="time_min", n_side=5,
                      max_steps=max_steps, return_cost=rc)
        comps = [train_eval(env_kw, s, episodes, eval_od) for s in range(seeds)]
        out.append(("costly_return_time_min", rc, np.mean(comps), np.std(comps)))
        print(f"{'costly_return':13} {'time_min':9} return_cost={rc:4.1f}  "
              f"completion={np.mean(comps):.3f} +/- {np.std(comps):.3f}", flush=True)

    # sanity: the aligned reward remains a complete solution under costly_return too.
    env_kw = dict(boundary="costly_return", reward="aligned", n_side=5,
                  max_steps=max_steps, return_cost=1.0)
    comps = [train_eval(env_kw, s, episodes, eval_od) for s in range(seeds)]
    out.append(("costly_return_aligned", 1.0, np.mean(comps), np.std(comps)))
    print(f"{'costly_return':13} {'aligned':9} return_cost= 1.0  "
          f"completion={np.mean(comps):.3f} +/- {np.std(comps):.3f}", flush=True)

    # persist
    with open(out_csv, "w") as f:
        f.write("condition,return_cost,final_completion_mean,final_completion_std\n")
        for cond, rc, m, s in out:
            f.write(f"{cond},{'' if rc!=rc else rc},{m:.4f},{s:.4f}\n")
    print("\nwrote costly_return_summary.csv")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max_steps", type=int, default=50)
    ap.add_argument("--out", default="costly_return_summary.csv")
    args = ap.parse_args()
    if args.smoke:
        study_costly_return(seeds=1, episodes=400, return_costs=(1.0,))
    else:
        study_costly_return(max_steps=args.max_steps, out_csv=args.out)
