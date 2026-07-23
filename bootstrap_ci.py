# -*- coding: utf-8 -*-
"""Interval estimates for the completion rates reported in Section V.

Completion is a bounded proportion, and a t-interval does not know that. On the aligned cells
it returned upper limits of 100.7 and 101.1, and on the Nguyen-Dupuis open travel-time cell a
lower limit of -0.5. Those limits were being truncated to the bound when written into the
manuscript, which hid the problem rather than fixing it.

A percentile bootstrap over the seed-level results respects the bound by construction, since
every resampled mean is itself an average of observed completion rates. It is also the
interval Agarwal et al. recommend for small-sample reinforcement-learning comparisons, which
this manuscript already cites.

The point estimate stays the mean. The interquartile mean those authors also recommend is
printed alongside for reference and is not adopted: on these cells it sits above the mean, and
raising the reported figures is not the purpose of changing the interval.

Run from experiments/. Prints every cell the manuscript reports and writes nothing.
"""
import io
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
B = 10000
SEED = 20260719


def iqm(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return float(np.mean(x[int(np.floor(n * 0.25)):int(np.ceil(n * 0.75))]))


def bootstrap_ci(x, rng, b=B):
    x = np.asarray(x, float)
    means = np.mean(rng.choice(x, size=(b, len(x)), replace=True), axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def line(label, v, rng):
    lo, hi = bootstrap_ci(v, rng)
    print("  %-34s n=%2d  mean %5.1f  95%% bootstrap [%5.1f, %5.1f]   (IQM %5.1f)"
          % (label, len(v), np.mean(v), lo, hi, iqm(v)))


def main():
    rng = np.random.RandomState(SEED)

    grid = np.load(os.path.join(HERE, "boundary_open_demo", "results_10seed.npz"))
    print("Bespoke 5x5 lattice, final evaluation point, 10 seeds")
    for cell in ("closed_time_min", "open_time_min", "closed_aligned", "open_aligned"):
        key = cell + "__comp"
        if key in grid:
            line(cell, np.asarray(grid[key])[:, -1] * 100.0, rng)

    bench = np.load(os.path.join(HERE, "benchmark_network", "benchmark_results.npz"))
    print("\nBenchmark road networks, 5 seeds")
    for key in bench.files:
        v = np.asarray(bench[key], float)
        line(key, v * 100.0 if v.max() <= 1.0 else v, rng)

    sing = np.load(os.path.join(HERE, "boundary_open_demo", "singleton_env_results.npz"))
    print("\nTraining-context generalization, 5 seeds")
    for key in sing.files:
        v = np.asarray(sing[key], float)
        line(key, v * 100.0 if v.max() <= 1.0 else v, rng)


if __name__ == "__main__":
    main()
