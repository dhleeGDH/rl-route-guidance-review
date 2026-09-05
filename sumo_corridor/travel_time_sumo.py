# -*- coding: utf-8 -*-
"""Travel time of the completing trips on the SUMO replication of the grid.

BOND item 1 asks a study to report the OD trip completion rate together with the travel time of
the completing trips. The travel time was recorded for the bespoke-grid cells
alone, which leaves the item open on the SUMO cells. This measures it on the SUMO
cells of Table V, at both budgets, on the convention of Appendix E: the in-network traversal cost
of a completing trip, with the arriving move excluded.

sumo_train.train_condition is imported rather than reimplemented, so each policy is the published
one seed for seed and the completion rate returned beside the travel time is the published rate.

    OMP_NUM_THREADS=1 python3 travel_time_sumo.py --episodes 3000 --seeds 10
"""
import argparse, io, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch                              # noqa: E402
torch.set_num_threads(1)

import sumo_train as T                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.smoke:
        a.episodes, a.seeds = 250, 1
    out_name = a.out or ("travel_time_sumo_%d.json" % a.episodes)

    eval_od = T.make_eval_od()
    out = {}
    print("SUMO grid: %d evaluation draws, %d episodes, %d seeds" % (len(eval_od), a.episodes, a.seeds))
    for boundary, reward in T.CONDITIONS:
        rates, tts, ns = [], [], []
        for seed in range(a.seeds):
            tag = "%s%s%d" % (boundary[:1], reward[:1], seed)
            r, tt, n = T.train_condition(boundary, reward, seed, a.episodes, eval_od, tag,
                                         return_travel_time=True)
            rates.append(100.0 * r); tts.append(tt); ns.append(n)
        key = "%s_%s" % (boundary, reward)
        m = float(np.nanmean(tts))
        out[key] = {"completion_mean": float(np.mean(rates)), "completion_per_seed": rates,
                    "travel_time_mean": m, "travel_time_per_seed": tts,
                    "completing_trips_per_seed": ns,
                    "episodes": a.episodes, "seeds": a.seeds}
        print("  %-6s %-9s completion %5.1f%%  travel time %s"
              % (boundary, reward, np.mean(rates),
                 ("%6.2f" % m) if np.isfinite(m) else "   n/a"), flush=True)
    io.open(os.path.join(HERE, out_name), "w", encoding="utf-8").write(
        json.dumps(out, ensure_ascii=False, indent=1))
    print("\nwrote", out_name)


if __name__ == "__main__":
    main()
