# -*- coding: utf-8 -*-
"""95% percentile bootstrap intervals for the interior-depth sweep.

The convention of ppo_terms_intervals.py is used unchanged: 10,000 resamples of the ten seeds at
seed 7, reported as the 2.5th and 97.5th percentiles of the resampled mean. A cell whose seeds are
all identical returns a degenerate interval, which is the correct reading of ten equal values.

    python3 interior_depth_intervals.py
"""
import io
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "interior_depth_sweep.json")


def boot(x, n=10000, seed=7):
    r = np.random.default_rng(seed)
    m = r.choice(np.asarray(x, dtype=float), size=(n, len(x)), replace=True).mean(axis=1)
    return round(float(np.percentile(m, 2.5)), 1), round(float(np.percentile(m, 97.5)), 1)


def main():
    d = json.load(io.open(SRC, encoding="utf-8"))
    # negative control: a spread sample must return a non-degenerate interval, else the resampler
    # is not resampling and every printed bound is the mean.
    lo, hi = boot([0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 10.0, 30.0, 50.0, 70.0])
    assert hi - lo > 10.0, "the bootstrap returned a degenerate interval on a spread sample"
    print("negative control: spread sample gives [%.1f-%.1f]" % (lo, hi))

    out = {"resamples": 10000, "seed": 7, "seeds": d["seeds"], "cells": {}}
    print("\n%-8s %-6s %-9s %-8s %s" % ("n_side", "depth", "reward", "episodes", "completion"))
    for n in d["sides"]:
        for reward in ("time_min", "aligned"):
            for ep in d["budgets"]:
                k = "%d|%s|%d" % (n, reward, ep)
                lo, hi = boot(d["cells"][k]["per_seed"])
                out["cells"][k] = {"mean": d["cells"][k]["mean"], "lo": lo, "hi": hi}
                print("%-8d %-6s %-9s %-8d %5.1f%% [%.1f-%.1f]"
                      % (n, d["depths"][str(n)], reward, ep, d["cells"][k]["mean"], lo, hi))
    json.dump(out, io.open(os.path.join(HERE, "interior_depth_intervals.json"), "w",
                           encoding="utf-8"), indent=1, sort_keys=True)
    print("\nwrote interior_depth_intervals.json")


if __name__ == "__main__":
    main()
