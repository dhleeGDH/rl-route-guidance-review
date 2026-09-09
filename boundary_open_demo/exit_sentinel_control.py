# -*- coding: utf-8 -*-
"""Is the transfer collapse a property of the boundary, or of the state encoding?

M9 of the 2026-09-02 cold review. Section V-D scores a policy trained under closure on an opened boundary and
reports 17.1% and 40.4%. A reviewer put the alternative explanation: an off-grid move carries
a cost slot of 9.0, above every real cost, and the exit moves the opened boundary makes
available are exactly the moves whose slot holds that value, so the two numbers may measure
what the network does with an input it never met rather than what the boundary does.

Two measurements answer that.

PREMISE. The sentinel is a function of the cell alone: _edge_cost returns None for any move
leaving the grid, on either boundary. A closed-trained policy therefore does meet the value,
at every border cell, throughout training. The first block counts the cells presenting it and
measures the share of exploration steps whose observation carries it.

ENCODING. The second block reruns the transfer cell under both encodings, nothing else
differing:

  high      the published behavior, an off-grid slot of 9.0
  matched   an off-grid slot of 1.0, the step cost the reward pays for that move

Both arms go through four_cells_boundary_dest.run_cell, the function behind the published
cells, so the control and the published cell cannot diverge in anything but the encoding.

NEGATIVE CONTROL. The high arm must reproduce zero_shot_transfer.json seed for seed on both
scoring columns. A departure means this script is not running the published configuration and
the comparison is void.

    OMP_NUM_THREADS=1 python3 exit_sentinel_control.py --seeds 10 --workers 16
"""
import argparse
import io
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from decimal import Decimal, ROUND_HALF_UP

torch.set_num_threads(1)          # 32 cores were driven to a load of 52 by unpinned runners

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from env import UNAVAIL_COST, N_SIDE                            # noqa: E402
from env_boundary_dest import BoundaryDestEnv, make_eval_od      # noqa: E402
from four_cells_boundary_dest import run_cell                    # noqa: E402

PUBLISHED = "zero_shot_transfer.json"
OUT = "exit_sentinel_control.json"


def occupancy(eval_od, seed=0, max_steps=120):
    """Share of closed-boundary exploration steps whose observation carries the sentinel."""
    env = BoundaryDestEnv(boundary="closed", reward="time_min", seed=1000 + seed,
                          max_steps=max_steps)
    rng = np.random.RandomState(seed)
    steps = carrying = 0
    for od in eval_od:
        env.reset(od=od)
        for _ in range(max_steps):
            s = env._obs()
            m = env.available_actions()
            steps += 1
            if np.any(np.asarray(s[5:9]) == UNAVAIL_COST):
                carrying += 1
            acts = np.flatnonzero(m)
            _, _, done, _ = env.step(int(acts[rng.randint(len(acts))]))
            if done:
                break
    cells = sum(1 for r in range(N_SIDE) for c in range(N_SIDE)
                if r in (0, N_SIDE - 1) or c in (0, N_SIDE - 1))
    return {"steps": steps, "steps_carrying_sentinel": carrying,
            "share_of_steps": round(100.0 * carrying / steps, 1),
            "cells_presenting_sentinel": cells, "cells_total": N_SIDE * N_SIDE,
            "share_of_cells": round(100.0 * cells / (N_SIDE * N_SIDE), 1)}


def _one(job):
    sent, reward, seed, episodes = job
    od = make_eval_od(n=200, seed=12345)
    return (sent, reward, seed,
            run_cell("closed", reward, seed, episodes, od, state_sentinel=sent),
            run_cell("closed", reward, seed, episodes, od, state_sentinel=sent,
                     eval_boundary="open"))


def p1(x):
    """One decimal, half up, the rounding the manuscript prints every cell at."""
    return float(Decimal(repr(float(x))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def boot(x, n=10000, seed=7):
    r = np.random.default_rng(seed)
    m = r.choice(np.asarray(x, float), size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    eval_od = make_eval_od(n=200, seed=12345)
    occ = occupancy(eval_od)
    print("premise: %d of %d cells present the sentinel; %.1f%% of %d exploration steps "
          "carry it" % (occ["cells_presenting_sentinel"], occ["cells_total"],
                        occ["share_of_steps"], occ["steps"]), flush=True)

    jobs = [(sent, rw, s, a.episodes)
            for sent in ("high", "matched") for rw in ("time_min", "aligned")
            for s in range(a.seeds)]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(_one, jobs))

    cells = {}
    for sent, rw, seed, closed, opened in res:
        d = cells.setdefault("%s_%s" % (sent, rw), {"scored_closed": [None] * a.seeds,
                                                    "scored_open": [None] * a.seeds})
        d["scored_closed"][seed] = closed
        d["scored_open"][seed] = opened
    for k, d in cells.items():
        d["closed_mean"] = float(np.mean(d["scored_closed"]))
        d["open_mean"] = float(np.mean(d["scored_open"]))
        d["open_ci"] = list(boot(d["scored_open"]))

    pub = json.load(io.open(os.path.join(HERE, PUBLISHED), encoding="utf-8"))
    drift = []
    for rw in ("time_min", "aligned"):
        p, h = pub["cells"][rw], cells["high_%s" % rw]
        for col in ("scored_closed", "scored_open"):
            if p[col] != h[col]:
                drift.append((rw, col, p[col], h[col]))
    print("negative control against %s: %s"
          % (PUBLISHED, "REPRODUCES" if not drift else "DRIFTED"), flush=True)
    for rw, col, p, h in drift:
        print("   %s %s\n     published %s\n     here      %s" % (rw, col, p, h), flush=True)

    out = {"episodes": a.episodes, "seeds": a.seeds, "trained_on": "closed",
           "premise": occ, "cells": cells,
           "reproduces_published": not drift, "paired": {}}
    # The closed column is the control's own negative control: the two rewards must stay
    # together on the boundary-closed network under either encoding, which is the paper's
    # claim. Derived here rather than by hand, since a hand-derived value has reached a built
    # PDF before.
    for sent in ("high", "matched"):
        ct = np.array(cells["%s_time_min" % sent]["scored_closed"], float)
        cg = np.array(cells["%s_aligned" % sent]["scored_closed"], float)
        clo, chi = boot(cg - ct)
        out.setdefault("paired_closed", {})[sent] = {
            "mean": float((cg - ct).mean()), "ci": [clo, chi],
            "excludes_zero": bool(clo > 0 or chi < 0),
            "printed_difference": p1((cg - ct).mean()),
            "printed_ci": [p1(clo), p1(chi)]}
        print("%-8s scored closed: paired %5.2f [%.2f-%.2f]  excludes zero %s"
              % (sent, (cg - ct).mean(), clo, chi,
                 out["paired_closed"][sent]["excludes_zero"]), flush=True)

    for sent in ("high", "matched"):
        t = np.array(cells["%s_time_min" % sent]["scored_open"], float)
        g = np.array(cells["%s_aligned" % sent]["scored_open"], float)
        lo, hi = boot(g - t)
        # The manuscript prints each cell at one decimal, so a difference a reader can take
        # from the printed cells is the difference of the printed cells. 40.55 rounds to 40.6
        # while the printed 63.2 and 22.7 differ by 40.5, and the smaller of the two is the
        # one stated, per the rule check_table_arithmetic.py enforces.
        pg, pt = p1(g.mean()), p1(t.mean())
        out["paired"][sent] = {"mean": float((g - t).mean()), "ci": [lo, hi],
                               "excludes_zero": bool(lo > 0 or hi < 0),
                               "printed_cells": [pt, pg],
                               "printed_difference": p1(pg - pt),
                               "printed_ci": [p1(lo), p1(hi)]}
        print("%-8s scored open: time_min %5.1f%% [%.1f-%.1f]   aligned %5.1f%% [%.1f-%.1f]   "
              "paired %5.1f [%.1f-%.1f]"
              % (sent, cells["%s_time_min" % sent]["open_mean"],
                 cells["%s_time_min" % sent]["open_ci"][0],
                 cells["%s_time_min" % sent]["open_ci"][1],
                 cells["%s_aligned" % sent]["open_mean"],
                 cells["%s_aligned" % sent]["open_ci"][0],
                 cells["%s_aligned" % sent]["open_ci"][1],
                 out["paired"][sent]["mean"], lo, hi), flush=True)
    json.dump(out, io.open(os.path.join(HERE, a.out), "w", encoding="utf-8"), indent=1)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
