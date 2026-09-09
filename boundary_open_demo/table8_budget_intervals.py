# -*- coding: utf-8 -*-
"""Table VIII at both training budgets, derived from the per-seed series of each arm.

WHY THIS EXISTS. Table VIII printed the learned column at 3000 episodes alone. Two reviewers of
round 82 read the gap between the optimum column and that one column as the evidence behind
Recommendation 2, and both observed that the supplement moves the three-term arm from 35.1% to
97.7% over the same budgets. A single-budget column cannot separate an arm whose shortfall is a
property of the objective from an arm whose shortfall is an unfinished learner, which is the very
distinction the recommendation asks a study to report.

Every value below is read from the per-seed series of the run that produced it. Nothing is
restated from the manuscript. The resampler is the one of bootstrap_ci.py, at the same B and seed,
so an interval printed here is comparable with every other interval in the paper.

The 3000-episode column is NOT recomputed into the table. A percentile bootstrap at B = 10,000
carries a Monte-Carlo error of about a quarter of a point on these seed counts: the three-term
lower bound lands anywhere in 26.05 to 26.40 depending on where in the resampler's stream the cell
is drawn, and converges to 26.20 at B = 400,000. The published 26.1 is one such draw and is not
wrong. Replacing it would move a printed value across three artifacts for Monte-Carlo noise, so
this script instead ASSERTS that each published 3000-episode figure sits inside the band, and
derives the 8000-episode column alone.

    python3 table8_budget_intervals.py
"""
import io
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
B = 10000
SEED = 20260719

# arm -> (budget -> (source file, path to the per-seed list))
SOURCES = {
    "Travel time alone": {
        3000: ("results_10seed.npz", "open_time_min__comp"),
        8000: ("matched_budget_grid.json", ("cells", "open_time_min", "per_seed")),
    },
    "Potential shaping alone": {
        3000: ("ablation_budget.json", ("arms", "shaping only @ 3000", "per_seed")),
        8000: ("ablation_budget.json", ("arms", "shaping only @ 8000", "per_seed")),
    },
    "Arrival term alone": {
        3000: ("arrival_term_budget.json", ("3000", "per_seed")),
        8000: ("arrival_term_budget.json", ("8000", "per_seed")),
    },
    "Exit penalty alone": {
        3000: ("ablation_budget.json", ("arms", "exit penalty only @ 3000", "per_seed")),
        8000: ("ablation_budget.json", ("arms", "exit penalty only @ 8000", "per_seed")),
    },
    "All three terms": {
        3000: ("ablation_bdest_matched.json", ("cells", "all three terms")),
        8000: ("matched_budget_grid.json", ("cells", "open_aligned", "per_seed")),
    },
}


def series(fname, path):
    p = os.path.join(HERE, fname)
    if fname.endswith(".npz"):
        return np.asarray(np.load(p)[path])[:, -1] * 100.0
    d = json.load(io.open(p, encoding="utf-8"))
    for k in path:
        d = d[k]
    return np.asarray(d, float)


def ci(x, rng):
    x = np.asarray(x, float)
    if x.min() == x.max():
        return None
    m = np.mean(rng.choice(x, size=(B, len(x)), replace=True), axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def cell(x, rng):
    x = np.asarray(x, float)
    iv = ci(x, rng)
    if iv is None:
        return "%.1f (all %d seeds)" % (np.mean(x), len(x))
    return "%.1f [%.1f-%.1f]" % (np.mean(x), iv[0], iv[1])


# what Table VIII prints today in the 3000-episode column, checked rather than regenerated
PUBLISHED_3000 = {
    "Travel time alone": (0.0, None),
    "Potential shaping alone": (4.3, (4.0, 4.8)),
    "Arrival term alone": (0.0, None),
    "Exit penalty alone": (0.0, None),
    "All three terms": (35.1, (26.1, 43.0)),
}
MC = 0.35  # the Monte-Carlo band of a B = 10,000 percentile bootstrap on ten seeds


def main():
    rng = np.random.RandomState(SEED)
    print("%-26s %-26s %-26s" % ("Reward", "3000 ep (published, checked)", "8000 ep (derived here)"))
    out, bad = {}, []
    for arm, by_budget in SOURCES.items():
        v3 = series(*by_budget[3000])
        v8 = series(*by_budget[8000])
        for b, v in ((3000, v3), (8000, v8)):
            assert len(v) == 10, "%s @ %d carries %d seeds, not 10" % (arm, b, len(v))
        pm, pi = PUBLISHED_3000[arm]
        if abs(np.mean(v3) - pm) > 0.05:
            bad.append("%s: mean %.2f against the printed %.1f" % (arm, np.mean(v3), pm))
        iv = ci(v3, rng)
        if (pi is None) != (iv is None):
            bad.append("%s: one of the two states an interval and the other does not" % arm)
        elif pi is not None:
            if abs(iv[0] - pi[0]) > MC or abs(iv[1] - pi[1]) > MC:
                bad.append("%s: [%.2f-%.2f] against the printed [%.1f-%.1f], beyond the "
                           "Monte-Carlo band" % (arm, iv[0], iv[1], pi[0], pi[1]))
        s8 = cell(v8, rng)
        out[arm] = s8
        p3 = "%.1f (all 10 seeds)" % pm if pi is None else "%.1f [%.1f-%.1f]" % (pm, pi[0], pi[1])
        print("%-26s %-26s %-26s" % (arm, p3, s8))
    if bad:
        raise SystemExit("TABLE VIII BUDGET CHECK FAILED: " + "; ".join(bad))
    json.dump(out, io.open(os.path.join(HERE, "table8_budget_intervals.json"), "w",
                           encoding="utf-8"), indent=1, sort_keys=True)
    print("\nthe published 3000-episode column reproduces from the per-seed series")


if __name__ == "__main__":
    main()
