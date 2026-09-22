# -*- coding: utf-8 -*-
"""The four Anaheim cells printed against the optimum, with convention A intervals.

Convention A is table6_rows.py's ci(): a 95% percentile bootstrap of 10,000 resamples whose
generator is seeded from the constant 20260719 and a checksum of the cell key, so an interval
does not depend on what was computed before it. The key is the run's own name, never a caption.

    python3 anaheim_cells_summary_gform.py
"""
import json
import os
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
B, SEED = 10000, 20260719


def ci(per_seed, label):
    x = np.asarray([v for v in per_seed if v is not None], float)
    rng = np.random.RandomState((SEED + zlib.crc32(label.encode("utf-8"))) % (2 ** 32))
    means = np.mean(rng.choice(x, size=(B, len(x)), replace=True), axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fmt(per_seed, key, unit=""):
    vals = [v for v in per_seed if v is not None]
    if not vals:
        return "n/a"
    m, lo, hi = ci(vals, key)
    if abs(hi - lo) < 0.05 and abs(m - lo) < 0.05:
        return "%.1f%s (all %d seeds)" % (m, unit, len(vals))
    return "%.1f%s [%.1f-%.1f]" % (m, unit, lo, hi)


def main():
    run = json.load(open(os.path.join(HERE, "anaheim_cells_gform.json"), encoding="utf-8"))
    e200 = json.load(open(os.path.join(HERE, "anaheim_eval200_optimum_gform.json"),
                          encoding="utf-8"))
    full = json.load(open(os.path.join(HERE, "anaheim_shaped_vi_gform.json"),
                          encoding="utf-8"))["hull"]["cells"]
    FULL = {"closed_time_min": "closed time_min | beta 0 (deposited reward)",
            "open_time_min": "open time_min | beta 0 (deposited reward)",
            "closed_aligned": "closed aligned | beta 1, hop potential",
            "open_aligned": "open aligned | beta 1, hop potential"}
    print("%-16s %-24s %-26s %-10s %-10s" %
          ("cell", "trained, 200 draws", "travel time of arrivals",
           "optimum200", "TableIV"))
    for k in ("closed_time_min", "open_time_min", "closed_aligned", "open_aligned"):
        rec = run[k]
        rw = "aligned" if k.endswith("aligned") else "time_min"
        bd = k.split("_")[0]
        ckey = "anaheim|%s|8000|%s" % (rw, bd)
        tkey = ckey + "|travel_time"
        print("%-16s %-24s %-26s %-10s %-10s" %
              (k, fmt(rec["completion_per_seed"], ckey),
               fmt(rec["travel_time_per_seed"], tkey, " min"),
               "%.1f" % e200["cells"][k]["arrive_pct"],
               "%.1f" % full[FULL[k]]["arrive_pct"]))
    print("\nattainable maximum: %.1f%% on the 200 draws, %.1f%% on the 15,289 pairs"
          % (e200["attainable_maximum_pct"], 97.8))
    st = run["open_aligned"]["steps"]
    m = np.asarray(run["open_aligned"]["curves"]).mean(0)
    end = m[-1]
    for tol in (5.0, 2.0):
        idx = [i for i in range(len(m)) if all(abs(m[j] - end) <= tol for j in range(i, len(m)))]
        print("open_aligned stays within +-%.0f points of the last checkpoint (%.2f) from %d"
              % (tol, end, st[idx[0]]))
    print("open_aligned curve:", " ".join("%d:%.1f" % (s, v) for s, v in zip(st, m)))


if __name__ == "__main__":
    main()
