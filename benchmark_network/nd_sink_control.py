"""How much the Nguyen-Dupuis stranded-node handling moves the reported completion.

Nguyen-Dupuis is directed and its two designated destinations have no outgoing arc. A vehicle
sent to the destination it was not given therefore reaches a node with no move and, on the
boundary-closed network, no exit. Two handlings of that state are possible:

  charged  the vehicle pays out the rest of its step budget at the minimum traversal cost, the
           handling benchmark_demo.py uses and the manuscript reports. The charge is a travel
           cost, not an exit penalty, and it enters both reward functions on the same terms.
  free     the episode ends there at no charge, the superseded handling. Stopping early is then
           the cheapest outcome under the travel-time reward.

The manuscript states the design choice. It does not state what the superseded handling changes,
which is what this measures: the same four cells, the same five seeds, the same episodes and
step budget, with STRANDED_CHARGE the only difference.

    python3 nd_sink_control.py                 # 5 seeds, both handlings, all four cells
    python3 nd_sink_control.py --seeds 2       # shorter check

Writes nd_sink_control.json next to this file.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import benchmark_demo as B


def run(handling, seeds, episodes, eval_od):
    B.STRANDED_CHARGE = (handling == "charged")
    net = B.NETWORKS["nguyen_dupuis"]
    out = {}
    for boundary in ("closed", "open"):
        for reward in ("time_min", "aligned"):
            rates = [B.train_eval(net, boundary, reward, s, episodes, eval_od)
                     for s in range(seeds)]
            out["%s_%s" % (boundary, reward)] = [float(x) for x in rates]
            print("[%-8s | %-6s | %-8s] completion %5.1f%% (sd %4.1f)  seeds=%s"
                  % (handling, boundary, reward, 100 * np.mean(rates), 100 * np.std(rates),
                     ["%.0f" % (100 * x) for x in rates]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=0)
    a = ap.parse_args()

    net = B.NETWORKS["nguyen_dupuis"]
    episodes = a.episodes or net["episodes"]
    eval_od = B.make_eval_od(net)
    res = {"seeds": a.seeds, "episodes": episodes, "network": "nguyen_dupuis"}
    for handling in ("charged", "free"):
        res[handling] = run(handling, a.seeds, episodes, eval_od)

    # the manuscript reports the charged handling, so the difference is stated in that direction
    res["difference"] = {
        k: round(100 * (np.mean(res["charged"][k]) - np.mean(res["free"][k])), 1)
        for k in res["charged"]}
    p = Path(__file__).parent / "nd_sink_control.json"
    p.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\ndifference (charged minus free, percentage points): %s" % res["difference"])
    print("wrote %s" % p)


if __name__ == "__main__":
    main()
