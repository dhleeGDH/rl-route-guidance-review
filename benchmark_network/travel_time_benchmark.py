# -*- coding: utf-8 -*-
"""Travel time of the completing trips on the benchmark networks.

WHY THIS EXISTS. BOND item 1 asks a study to report the OD trip completion rate together with the
travel time of the completing trips. The travel time was recorded for the bespoke-grid cells
alone, which answers that item on those cells and leaves it open on the benchmark networks. This measures it on the benchmark networks, on the convention of
Appendix E: the in-network traversal cost of a completing trip, with the arriving move excluded.

Training is benchmark_demo.train_eval unchanged, seed for seed, so the completion rate returned
here is the published rate of the same cell and the travel time is measured on the same policy.

    python3 travel_time_benchmark.py [--nets sioux_falls] [--seeds 10] [--smoke]
"""
import argparse, io, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch                                  # noqa: E402
torch.set_num_threads(1)                      # feedback_pin_torch_threads

import benchmark_demo as B                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nets", nargs="+", default=["sioux_falls"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="travel_time_benchmark.json")
    a = ap.parse_args()

    out = {}
    for name in a.nets:
        net = B.NETWORKS[name]
        eps = a.episodes or (300 if a.smoke else net["episodes"])
        seeds = 2 if a.smoke else a.seeds
        od = B.make_eval_od(net)
        print("%s: %d evaluation draws, %d episodes, %d seeds" % (name, len(od), eps, seeds))
        for boundary in ("closed", "open"):
            for reward in ("time_min", "aligned"):
                rates, tts, ns = [], [], []
                for s in range(seeds):
                    r, tt, n = B.train_eval(net, boundary, reward, s, eps, od,
                                            return_travel_time=True)
                    rates.append(100.0 * r); tts.append(tt); ns.append(n)
                key = "%s_%s_%s" % (name, boundary, reward)
                out[key] = {"completion_mean": float(np.mean(rates)),
                            "completion_per_seed": rates,
                            "travel_time_mean": float(np.nanmean(tts)),
                            "travel_time_sd": float(np.nanstd(tts)),
                            "travel_time_per_seed": tts,
                            "completing_trips_per_seed": ns,
                            "episodes": eps, "seeds": seeds}
                print("  %-6s %-9s completion %5.1f%%  travel time %s"
                      % (boundary, reward, np.mean(rates),
                         ("%6.2f" % np.nanmean(tts)) if np.isfinite(np.nanmean(tts)) else "   n/a"),
                      flush=True)
    io.open(os.path.join(HERE, a.out), "w", encoding="utf-8").write(
        json.dumps(out, ensure_ascii=False, indent=1))
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
