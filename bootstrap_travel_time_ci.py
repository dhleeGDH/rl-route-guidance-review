# -*- coding: utf-8 -*-
"""95% percentile bootstrap intervals for the travel-time records, seeded per cell.

Appendix E states one dispersion convention: a 95% percentile bootstrap from 10,000 resamples.
A single RNG drawn across cells makes each interval depend on the order the cells are visited,
which moved one printed bound by 0.01 between two runs of the same data. The seed is therefore
derived from the cell key, so any cell reproduces its own interval alone or in any order.

    python3 bootstrap_travel_time_ci.py
"""
import io, json, os, zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ["boundary_open_demo/travel_time_bdest_ingrid.json",
         "boundary_open_demo/travel_time_bdest_ingrid_8000.json",
         "benchmark_network/travel_time_benchmark.json",
         "sumo_corridor/travel_time_sumo_3000.json",
         "sumo_corridor/travel_time_sumo_8000.json"]
RESAMPLES = 10000


def boot(values, key):
    v = np.array([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return None
    rng = np.random.RandomState(zlib.crc32(key.encode("utf-8")) & 0x7FFFFFFF)
    s = [np.mean(rng.choice(v, v.size, replace=True)) for _ in range(RESAMPLES)]
    return float(np.mean(v)), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def main():
    for rel in FILES:
        path = os.path.join(HERE, rel)
        if not os.path.exists(path):
            print("absent:", rel)
            continue
        d = json.load(io.open(path, encoding="utf-8"))
        for cell, v in d.items():
            if not isinstance(v, dict):
                continue
            for src, dst, fmt in (("travel_time_per_seed", "travel_time_ci95", "%.2f (%.2f to %.2f)"),
                                  ("completion_per_seed", "completion_ci95", "%.1f (%.1f to %.1f)")):
                b = boot(v.get(src) or [], "%s|%s|%s" % (rel, cell, src))
                if b:
                    v[dst] = {"mean": b[0], "lo": b[1], "hi": b[2], "printed": fmt % b}
                else:
                    v.pop(dst, None)
        d["_dispersion"] = ("95%% percentile bootstrap over the per-seed values, %d resamples, "
                            "the seed derived from the cell key so an interval reproduces in any "
                            "order (Appendix E convention)" % RESAMPLES)
        io.open(path, "w", encoding="utf-8").write(json.dumps(d, ensure_ascii=False, indent=1))
        print(rel)
        for cell, v in d.items():
            if isinstance(v, dict) and "travel_time_ci95" in v:
                print("   %-34s completion %-24s travel time %s"
                      % (cell, v.get("completion_ci95", {}).get("printed", "-"),
                         v["travel_time_ci95"]["printed"]))


if __name__ == "__main__":
    main()
