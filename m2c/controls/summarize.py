# -*- coding: utf-8 -*-
"""T-1941: the printed form of each control cell, under the deposited interval convention.

The interval is bootstrap_travel_time_ci.py's boot(), which T-1940 fixed as the convention for
every M2-C value the manuscript prints: 10,000 separate resamples, the seed a checksum of the
path, the cell and the field, so a cell reproduces its own interval alone or in any order.

    python3 summarize.py
"""
import io, json, os, sys
from decimal import Decimal, ROUND_HALF_UP

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, "repo_v11"))
import numpy as np                                              # noqa: E402
from bootstrap_travel_time_ci import boot                       # noqa: E402


def load(name):
    return json.load(io.open(os.path.join(HERE, name), encoding="utf-8"))


def p1(x):
    """One decimal, half up, the rounding the manuscript prints every cell at.

    "%.1f" is not that rounding. 48.05 and 64.15 are stored just below the tie in binary and
    print a digit low under it, which is how T-1968 read four bounds of this file as
    disagreeing with the manuscript when they agree. T-1969 fixed the display.
    """
    return float(Decimal(repr(float(x))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def show(rel, cell, field, values):
    m, lo, hi = boot(values, "%s|%s|%s" % (rel, cell, field))
    print("  %-46s %5.1f (%.1f to %.1f)"
          % ("%s %s" % (cell, field), p1(m), p1(lo), p1(hi)))
    return m, lo, hi


def sentinel(variant):
    rel = "handoff/m2c/controls/exit_sentinel_control_%s.json" % variant
    d = load(os.path.basename(rel))
    print("--- exit sentinel, variant %s   (reproduces the published run: %s)"
          % (variant, d["reproduces_published"]))
    out = {}
    for sent in ("high", "matched"):
        for rw in ("time_min", "aligned"):
            c = d["cells"]["%s_%s" % (sent, rw)]
            out[(sent, rw, "open")] = show(rel, "%s_%s" % (sent, rw), "scored_open",
                                           c["scored_open"])
        t = np.array(d["cells"]["%s_time_min" % sent]["scored_open"], float)
        g = np.array(d["cells"]["%s_aligned" % sent]["scored_open"], float)
        out[(sent, "paired", "open")] = show(rel, sent, "paired_open", list(g - t))
        # The manuscript prints the difference a reader can take from the two printed cells,
        # which check_table_arithmetic.py enforces, so the paired mean above is not the printed
        # one where the two roundings part. 24.45 shows as 24.5 and prints as 41.5 - 17.1.
        pt = p1(d["cells"]["%s_time_min" % sent]["open_mean"])
        pg = p1(d["cells"]["%s_aligned" % sent]["open_mean"])
        print("  %-46s %5.1f  (= %.1f - %.1f)"
              % ("%s printed difference" % sent, p1(pg - pt), pg, pt))
        ct = np.array(d["cells"]["%s_time_min" % sent]["scored_closed"], float)
        cg = np.array(d["cells"]["%s_aligned" % sent]["scored_closed"], float)
        out[(sent, "paired", "closed")] = show(rel, sent, "paired_closed", list(cg - ct))
    return out


def replacement(variant):
    """The arrival-sentinel replacement of S-I.B, whose paired interval the supplement prints."""
    rel = "handoff/m2c/controls/sentinel_control_%s.json" % variant
    d = load(os.path.basename(rel))
    print("--- arrival-sentinel replacement, variant %s" % variant)
    for cell in ("high_open_aligned", "matched_open_aligned"):
        print("  %-46s %5.1f" % (cell, p1(d["cells"][cell]["mean"])))
    return show(rel, "paired_open_aligned", "per_seed", d["paired_open_aligned"]["per_seed"])


def truncation(variant):
    rel = "handoff/m2c/controls/truncation_control_%s.json" % variant
    d = load(os.path.basename(rel))
    print("--- truncation, variant %s" % variant)
    worst = 0.0
    for key in ("closed_time_min", "closed_aligned", "open_time_min", "open_aligned"):
        row = d[key]
        a, b = row["terminal"]["mean"], row["bootstrapped"]["mean"]
        print("  %-18s terminal %5.1f%%  bootstrapped %5.1f%%  delta %+.1f"
              % (key, a, b, row["delta"]))
        worst = max(worst, abs(row["delta"]))
    print("  largest move %.1f points" % worst)
    return d, worst


if __name__ == "__main__":
    for v in ("P", "C"):
        if os.path.exists(os.path.join(HERE, "exit_sentinel_control_%s.json" % v)):
            sentinel(v)
    for v in ("P", "C"):
        if os.path.exists(os.path.join(HERE, "sentinel_control_%s.json" % v)):
            replacement(v)
    for v in ("P", "C"):
        if os.path.exists(os.path.join(HERE, "truncation_control_%s.json" % v)):
            truncation(v)
