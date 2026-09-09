# -*- coding: utf-8 -*-
"""Percentile bootstrap intervals for the non-terminal-exit cells of Section V-C.

WHY THIS EXISTS. Section V-C printed 46.2% (39.0 to 54.9) for the non-terminal exit, and the
supplement derived the residual-charge convention beside it but not this one, though S-I.B promises
both. Three read-throughs on 2026-08-26 found the missing half. Rebuilding the interval from
costly_return_10seed_v3_per_seed.json under the convention of ppo_terms_intervals.py (10,000
resamples, seed 7) returns 54.3 rather than 54.9: the published bound came from a bootstrap at a
different seed, and 0.6 points is the Monte-Carlo spread of the endpoint itself. One convention is
used across both documents from here.

    python3 costly_return_intervals.py
"""
import io
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE / "costly_return_10seed_v3_per_seed.json"
OUT = HERE / "costly_return_intervals.json"


def boot(x, n=10000, seed=7):
    r = np.random.default_rng(seed)
    m = r.choice(np.asarray(x, float), size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


d = json.load(io.open(SRC, encoding="utf-8"))
out = {"resamples": 10000, "seed": 7, "convention": "Section V-A",
       "seeds": d["seeds"], "episodes": d["episodes"], "max_steps": d["max_steps"], "cells": {}}
for name, series in d["cells"].items():
    lo, hi = boot(series)
    out["cells"][name] = {"mean": sum(series) / len(series), "lo": lo, "hi": hi, "n": len(series)}
    print("  %-32s %.1f [%.1f-%.1f]" % (name, out["cells"][name]["mean"], lo, hi))
json.dump(out, io.open(OUT, "w", encoding="utf-8"), indent=1)
print("wrote", OUT.name)
