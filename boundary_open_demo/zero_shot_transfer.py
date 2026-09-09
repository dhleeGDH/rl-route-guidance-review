# -*- coding: utf-8 -*-
"""A policy trained under closure, scored on a boundary it never saw.

Round 75B M2, filed independently by all three reviewers of rounds 73 to 75: the collapse the
paper demonstrates is produced by training under an opened boundary, and every reviewed
evaluation whose geometry can be determined is closed. The step the argument actually needs is
the transfer one, stated in Section IV-D as "The two come apart in service" and never measured:
a policy trained where no exit exists, then placed on a network with one.

This trains the two rewards under closure and scores each agent twice, on the boundary it trained
on and on the opened boundary. Nothing else changes: the learner, the budget, the seeds, the
evaluation set and the environment are those of the four cells, through the same run_cell.

NEGATIVE CONTROL. The closed column must reproduce four_cells_boundary_dest.json seed for seed.
It is the same computation, so any drift means the eval_boundary parameter changed behaviour and
the transfer column cannot be trusted.
"""
import argparse
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from env_boundary_dest import make_eval_od                      # noqa: E402
from four_cells_boundary_dest import run_cell                   # noqa: E402

PUBLISHED = "four_cells_boundary_dest.json"
OUT = "zero_shot_transfer.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    eval_od = make_eval_od(n=200, seed=12345)
    pub = json.load(io.open(os.path.join(HERE, PUBLISHED), encoding="utf-8"))
    out = {"episodes": a.episodes, "seeds": a.seeds, "trained_on": "closed", "cells": {}}

    for reward in ("time_min", "aligned"):
        held, moved = [], []
        for s in range(a.seeds):
            held.append(run_cell("closed", reward, s, a.episodes, eval_od))
            moved.append(run_cell("closed", reward, s, a.episodes, eval_od,
                                  eval_boundary="open"))
        ref = pub["closed_%s" % reward]["per_seed"]
        same = held == ref
        out["cells"][reward] = {"scored_closed": held, "scored_open": moved,
                                "closed_mean": float(np.mean(held)),
                                "open_mean": float(np.mean(moved)),
                                "reproduces_published_closed": bool(same)}
        print("%-9s trained closed:  scored closed %5.1f%%   scored open %5.1f%%"
              % (reward, np.mean(held), np.mean(moved)), flush=True)
        print("%-9s negative control, closed column against %s: %s"
              % ("", PUBLISHED, "REPRODUCES" if same else "DRIFTED"), flush=True)
        if not same:
            print("   published %s" % ref, flush=True)
            print("   here      %s" % held, flush=True)

    # The paired difference of Section V-A, over the shared seeds, on each scoring.
    def boot(x, n=10000, seed=7):
        r = np.random.default_rng(seed)
        m = r.choice(np.asarray(x, float), size=(n, len(x)), replace=True).mean(axis=1)
        return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

    t = np.array(out["cells"]["time_min"]["scored_open"], float)
    g = np.array(out["cells"]["aligned"]["scored_open"], float)
    ct = np.array(out["cells"]["time_min"]["scored_closed"], float)
    cg = np.array(out["cells"]["aligned"]["scored_closed"], float)
    for label, diff in (("open", g - t), ("closed", cg - ct)):
        lo, hi = boot(diff)
        out.setdefault("paired", {})[label] = {"mean": float(diff.mean()), "ci": [lo, hi],
                                               "excludes_zero": bool(lo > 0 or hi < 0)}
        print("paired difference, scored %-6s %6.2f [%.2f-%.2f]  excludes zero %s"
              % (label, diff.mean(), lo, hi, out["paired"][label]["excludes_zero"]), flush=True)
    json.dump(out, io.open(os.path.join(HERE, a.out), "w", encoding="utf-8"), indent=1)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
