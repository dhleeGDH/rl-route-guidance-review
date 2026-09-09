# -*- coding: utf-8 -*-
"""Bootstrap intervals for the Sioux Falls cells at the drawn outer face of 12 nodes.

WHY THIS EXISTS. benchmark_demo.py prints a standard deviation, and Section V-A reports a
bootstrap interval for every learned cell. The published Sioux Falls cells were run at 11 of the
12 nodes on the outer face of the drawing, a border outer_face_sioux.py derives from the rotation
system of the published coordinates. This reports the cells at the full face through the same
resampler ppo_terms_intervals.py uses, at the same resample count and the same seed.

    python3 outerface12_intervals.py
"""
import io
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "benchmark_results_outerface12.npz")


def boot(x, n=10000, seed=7):
    r = np.random.default_rng(seed)
    m = r.choice(np.asarray(x, float), size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    z = np.load(NPZ)
    out = {"resamples": 10000, "seed": 7, "border_nodes": 12,
           "border": [1, 2, 3, 6, 7, 8, 12, 13, 18, 20, 21, 24], "cells": {}}
    print("%-30s %6s %-16s %s" % ("cell", "mean", "interval", "seeds"))
    for k in sorted(z.files):
        v = np.asarray(z[k], float).ravel() * 100.0
        if v.size < 2:
            continue
        lo, hi = boot(v)
        out["cells"][k] = {"mean": round(float(v.mean()), 1),
                           "ci": [round(lo, 1), round(hi, 1)],
                           "seeds": [round(float(x), 1) for x in v]}
        print("%-30s %6.1f [%5.1f-%5.1f]  %s"
              % (k, v.mean(), lo, hi, " ".join("%.0f" % x for x in v)))
    # the paired difference between the two rewards on the opened border, over shared seeds
    ot = next((k for k in z.files if "open" in k and "time" in k), None)
    oa = next((k for k in z.files if "open" in k and "align" in k), None)
    if ot and oa:
        a = np.asarray(z[ot], float).ravel() * 100.0
        b = np.asarray(z[oa], float).ravel() * 100.0
        d = b - a
        lo, hi = boot(d)
        out["paired_open"] = {"mean": round(float(d.mean()), 1), "ci": [round(lo, 1), round(hi, 1)]}
        print("\npaired difference on the opened border: %.1f points [%.1f-%.1f]"
              % (d.mean(), lo, hi))
    with io.open(os.path.join(HERE, "outerface12_intervals.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("  wrote outerface12_intervals.json")


if __name__ == "__main__":
    main()
