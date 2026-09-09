# -*- coding: utf-8 -*-
"""Mean travel time of the completing trips, for the four published cells.

Round 42 M3: every outcome table in Section V reports OD trip completion, while the metric this
literature actually reports is travel time. The claim that a closed cordon "returns one verdict
for two opposed objectives" was demonstrated on completion alone, so a reader could ask whether
the destination-aligned reward buys completion with longer routes.

Training is NOT reimplemented here. This calls four_cells_boundary_dest.run_cell unchanged, so
every trained policy is the published one seed for seed, and intercepts the evaluate() call that
run_cell makes at the end to capture the trained agent. The added pass replays the same
evaluation, in the same order, on an environment built with the same seed, and accumulates the
environment's own link cost.

An older travel_time_cost_bdest.json exists from 2026-07-21. It is NOT used: its completion
column reads 99.45 and 30.75 against the published 99.9 and 35.1, so it predates the current
protocol. Derived artifacts outlive their sources.

Two controls, both asserted at run time:
  1. the completion this pass measures equals the completion run_cell returns, seed for seed;
  2. on the travel-time cells the reward IS the negative cost, so the accumulated travel time
     must equal the negated sum of rewards to within floating-point tolerance. That validates
     the cost accounting against the environment itself rather than against a second copy of it.
"""
import argparse
import io
import json
import os
import sys

import numpy as np
import torch

torch.set_num_threads(1)          # see four_cells_boundary_dest.py: threads move the trajectory

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# four_cells_boundary_dest wraps sys.stdout on import; wrapping it again here closed the file
# under the first wrapper and every print after the import raised.
import four_cells_boundary_dest as F                          # noqa: E402
from env_boundary_dest import BoundaryDestEnv, make_eval_od   # noqa: E402
from env_boundary_dest import ACTIONS                         # noqa: E402


def travel_eval(agent, boundary, reward, eval_od, max_steps=120, seed=777):
    """Replay the published evaluation and accumulate the environment's own link cost."""
    env = BoundaryDestEnv(boundary=boundary, reward=reward, seed=seed, max_steps=max_steps,
                          state_sentinel="high")
    arrived, times, reward_sums, cost_sums = 0, [], [], []
    for od in eval_od:
        env.reset(od=od)
        cost = 0.0
        rsum = 0.0
        for _ in range(env.max_steps):
            s = env._obs()
            a = agent.act(s, env.available_actions(), eps=0.0)
            r, c = env.pos
            # Round 49b: Section V-B accumulates the environment's in-grid link costs and nothing
            # else, since _edge_cost returns None off the grid. Charging 1.0 for the arrival link
            # here put two travel-time figures of one paper on two accounting bases. This follows
            # the established convention: in-grid edge costs only.
            ec = env._edge_cost(r, c, a)
            step_cost = 0.0 if ec is None else float(ec)
            _, rw, done, info = env.step(a)
            cost += step_cost
            rsum += rw
            if done:
                if info["outcome"] == "arrived":
                    arrived += 1
                    times.append(cost)
                    reward_sums.append(rsum)
                    cost_sums.append(cost)
                break
    comp = 100.0 * arrived / len(eval_od)
    return comp, times, reward_sums, cost_sums


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="travel_time_bdest_ingrid.json")
    a = ap.parse_args()

    eval_od = make_eval_od(n=200, seed=12345)
    captured = {}
    _orig = F.evaluate

    def spy(agent, *args, **kw):
        captured["agent"] = agent
        return _orig(agent, *args, **kw)

    F.evaluate = spy

    out = {}
    for boundary in ("closed", "open"):
        for reward in ("time_min", "aligned"):
            comps, means = [], []
            for s in range(a.seeds):
                comp_pub = F.run_cell(boundary, reward, s, a.episodes, eval_od)
                comp, times, rsums, csums = travel_eval(captured["agent"], boundary, reward,
                                                        eval_od)
                assert abs(comp - comp_pub) < 1e-9, (
                    "control 1 failed: replay completion %.4f against run_cell %.4f"
                    % (comp, comp_pub))
                if reward == "time_min" and times:
                    # The reward charges 1.0 for the arrival link; in-grid accounting does not, so
                    # the two now differ by exactly one unit per completed trip. That exact gap is
                    # itself the control.
                    worst = max(abs((-rs) - (cs + 1.0)) for rs, cs in zip(rsums, csums))
                    assert worst < 1e-6, (
                        "control 2 failed: in-grid cost departs from reward minus one by %.3g"
                        % worst)
                comps.append(comp_pub)
                means.append(float(np.mean(times)) if times else float("nan"))
                print("  %-6s %-9s seed %d: completion %5.1f%%  travel time %s"
                      % (boundary, reward, s, comp_pub,
                         "%6.3f" % means[-1] if times else "  n/a"), flush=True)
            key = "%s_%s" % (boundary, reward)
            out[key] = {"completion_mean": float(np.mean(comps)),
                        "completion_per_seed": comps,
                        "travel_time_mean": float(np.nanmean(means)) if means else None,
                        "travel_time_sd": float(np.nanstd(means)) if means else None,
                        "travel_time_per_seed": means}
            print("== %-6s %-9s completion %5.1f%%  travel time %s" %
                  (boundary, reward, out[key]["completion_mean"],
                   "%6.3f (sd %.3f)" % (out[key]["travel_time_mean"], out[key]["travel_time_sd"])
                   if np.isfinite(out[key]["travel_time_mean"]) else "n/a (no completing trip)"),
                  flush=True)
    with open(os.path.join(HERE, a.out), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
