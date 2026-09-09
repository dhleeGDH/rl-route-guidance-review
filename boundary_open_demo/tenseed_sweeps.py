# -*- coding: utf-8 -*-
"""The alignment sweep and the 8x8 grid at the seed count every other learned cell now uses.

WHY THIS EXISTS. A reviewer objected that Table VI could not carry a within-table comparison while
its rows differed in seed count and dispersion statistic. Two cells outside that table sit in the
same position and are compared against it in the body text:

  - the alignment sweep of Fig. 6(a), whose full-alignment endpoint the body compares directly
    against Table VI ("29.4% at full alignment over its five seeds, against the 30.8% Table VI
    reports for the same cell over ten"), a sentence that states the protocol difference rather
    than removing it;
  - the 8x8 grid, reported at five seeds and compared against the 5x5 interval computed over ten.

Both are rerun here at ten seeds, and the per-seed values are recorded so the same percentile
bootstrap applies to them as to every row of Table VI.

    OMP_NUM_THREADS=1 python3 tenseed_sweeps.py --only dose
    OMP_NUM_THREADS=1 python3 tenseed_sweeps.py --only grid8
"""
import argparse
import json

import numpy as np
import torch

from sweep_extra import make_eval_od, train_eval

torch.set_num_threads(1)

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--episodes", type=int, default=3000)
ap.add_argument("--only", choices=("dose", "grid8"), required=True)
ap.add_argument("--out", default=None)
a = ap.parse_args()
out = a.out or ("dose_response_10seed.json" if a.only == "dose" else "grid8_10seed.json")

res = {}
if a.only == "dose":
    eval_od = make_eval_od(5)
    print("alignment sweep on the boundary-open 5x5, %d seeds, %d episodes"
          % (a.seeds, a.episodes), flush=True)
    for lam in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        env_kw = dict(boundary="open", reward="aligned", n_side=5,
                      max_steps=120, align_strength=lam)
        v = 100 * np.asarray([train_eval(env_kw, s, a.episodes, eval_od)
                              for s in range(a.seeds)], dtype=float)
        res["lambda_%.2f" % lam] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                                    "per_seed": [round(float(x), 1) for x in v],
                                    "seeds": a.seeds, "episodes": a.episodes}
        print("  lambda=%.2f  completion %5.1f%% (sd %4.1f)" % (lam, v.mean(), v.std(ddof=1)),
              flush=True)
else:
    eval_od = make_eval_od(8, min_sep=5)
    print("8x8 boundary-open grid, %d seeds, %d episodes" % (a.seeds, a.episodes), flush=True)
    for reward in ("time_min", "aligned"):
        env_kw = dict(boundary="open", reward=reward, n_side=8, max_steps=220)
        v = 100 * np.asarray([train_eval(env_kw, s, a.episodes, eval_od)
                              for s in range(a.seeds)], dtype=float)
        res["open_%s" % reward] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                                   "per_seed": [round(float(x), 1) for x in v],
                                   "seeds": a.seeds, "episodes": a.episodes}
        print("  open %-9s completion %5.1f%% (sd %4.1f)" % (reward, v.mean(), v.std(ddof=1)),
              flush=True)

json.dump({"what": a.only, "episodes": a.episodes, "seeds": a.seeds,
           "eval_pairs": len(eval_od), "cells": res}, open(out, "w"), indent=1)
print("wrote %s" % out)
