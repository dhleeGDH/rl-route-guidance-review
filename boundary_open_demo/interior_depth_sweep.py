# -*- coding: utf-8 -*-
"""Does the separation hold as the interior of the study area deepens? (Section V-B control.)

The learned cells of the experiment sit on lattices whose deepest node lies 2 or 3 links from the
nearest border node, and on Anaheim, where the retraining reaches 12 at the band of a tenth and 14
at the hull. The lattice is swept here so the depth axis carries the study's own learner at every
step of it, from 2 to 6, under one evaluation protocol.

    n_side   5   7   9  11  13
    depth    2   3   4   5   6

TWO BUDGETS, NOT ONE. The travel-time arm exits at once and its rate does not move with training,
whereas the destination-aligned arm has to learn a route and its rate does: on the 5x5 grid the
same arm read 28.9% at 3000 episodes and 96.7% at 8000, a gap traced to the budget in 2026-08-09.
An aligned rate quoted at one budget therefore states a level of training as much as a level of
depth, and both budgets are run here so the two are separable. The measurement below settles which
of them moves: the budget sets the level and the depth leaves it where the budget put it.

THE OPTIMUM ACCOMPANIES EVERY RATE. Exact value iteration is run at each depth for both rewards,
which is what Section VI-C asks of every reported rate.

    python3 interior_depth_sweep.py [--seeds 10] [--jobs 14]
"""
import argparse
import io
import json
import os
import subprocess
import sys
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
SIDES = (5, 7, 9, 11, 13)
BUDGETS = (3000, 8000)
REWARDS = ("time_min", "aligned")
MAX_STEPS = 220
MIN_SEP = 5


def depth(n):
    """Largest number of links from a node to the nearest border node, by breadth-first search.
    Every perimeter cell of the lattice is a border node, an exit being an action on one of its
    outward links."""
    border = {(r, c) for r in range(n) for c in range(n)
              if r in (0, n - 1) or c in (0, n - 1)}
    dist = {b: 0 for b in border}
    q = deque(border)
    while q:
        r, c = q.popleft()
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in dist:
                dist[(nr, nc)] = dist[(r, c)] + 1
                q.append((nr, nc))
    return max(dist.values())


CELL = r'''
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch; torch.set_num_threads(1)
sys.path.insert(0, %r)
from sweep_extra import make_eval_od, train_eval
n, reward, ep, seed = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
od = make_eval_od(n, min_sep=%d)
kw = dict(boundary="open", reward=reward, n_side=n, max_steps=%d)
print(json.dumps({"n": n, "reward": reward, "ep": ep, "seed": seed,
                  "completion": round(100.0 * train_eval(kw, seed, ep, od), 1)}))
''' % (HERE, MIN_SEP, MAX_STEPS)


def optima():
    """Exact optimum at each depth, through the solver of the four main cells."""
    sys.path.insert(0, HERE)
    from optimal_vi_boundary_dest import solve, arrives
    from sweep_extra import make_eval_od
    out = {}
    for n in SIDES:
        ods = make_eval_od(n, min_sep=MIN_SEP)
        for reward in REWARDS:
            cache, ok = {}, 0
            for o, dl in ods:
                if dl not in cache:
                    cache[dl] = solve(dl, "open", reward, n=n)
                V, phi, ol = cache[dl]
                ok += arrives(o, dl, V, phi, ol, reward, n=n)
            out["%d|%s" % (n, reward)] = round(100.0 * ok / len(ods), 1)
            print("  optimum  n=%-3d %-8s %5.1f%%" % (n, reward, out["%d|%s" % (n, reward)]),
                  flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=14)
    a = ap.parse_args()

    print("interior depth of each lattice")
    for n in SIDES:
        print("  n_side %-3d depth %d" % (n, depth(n)))
    assert [depth(n) for n in SIDES] == [2, 3, 4, 5, 6], "the depth axis is not 2 to 6"

    print("\nexact optimum, boundary open")
    opt = optima()

    script = os.path.join(HERE, "_depth_cell.py")
    io.open(script, "w", encoding="utf-8").write(CELL)
    jobs = [(n, r, ep, s) for n in SIDES for r in REWARDS
            for ep in BUDGETS for s in range(a.seeds)]
    print("\n%d learned cells, %d at a time" % (len(jobs), a.jobs), flush=True)
    res, run = {}, []
    for i, j in enumerate(jobs):
        run.append((j, subprocess.Popen([sys.executable, script] + [str(x) for x in j],
                                        stdout=subprocess.PIPE, text=True)))
        if len(run) == a.jobs or i == len(jobs) - 1:
            for j2, p in run:
                d = json.loads(p.communicate()[0].strip())
                res.setdefault("%d|%s|%d" % (d["n"], d["reward"], d["ep"]), []).append(
                    d["completion"])
            run = []
            print("  %d/%d done" % (i + 1, len(jobs)), flush=True)
    os.remove(script)

    cells = {}
    for k, v in res.items():
        v = sorted(v)
        cells[k] = {"mean": round(sum(v) / len(v), 1), "per_seed": v, "n_seeds": len(v)}
    print("\n%-8s %-6s %-9s %8s %8s %9s" % ("n_side", "depth", "reward", "3000", "8000",
                                            "optimum"))
    for n in SIDES:
        for reward in REWARDS:
            print("%-8d %-6d %-9s %8.1f %8.1f %9.1f"
                  % (n, depth(n), reward, cells["%d|%s|3000" % (n, reward)]["mean"],
                     cells["%d|%s|8000" % (n, reward)]["mean"], opt["%d|%s" % (n, reward)]))
    json.dump({"sides": list(SIDES), "depths": {str(n): depth(n) for n in SIDES},
               "budgets": list(BUDGETS), "max_steps": MAX_STEPS, "min_sep": MIN_SEP,
               "eval_pairs": 200, "seeds": a.seeds, "optimum": opt, "cells": cells},
              io.open(os.path.join(HERE, "interior_depth_sweep.json"), "w", encoding="utf-8"),
              indent=1, sort_keys=True)
    print("\nwrote interior_depth_sweep.json")


if __name__ == "__main__":
    main()
