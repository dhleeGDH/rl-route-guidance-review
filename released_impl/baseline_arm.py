# -*- coding: utf-8 -*-
"""The study's own Dijkstra baseline under the two boundary conditions.

WHY THIS EXISTS. The 2026-09-02 cold review, M3, objects that no reviewed study is shown to have a
conclusion the boundary condition actually undermines, and reads the published retraining as
Proposition 2 restated on a larger network. What the retraining left out is the study's own
reference. [25] ships a Dijkstra baseline inside the released code, uses it to warm start the
learner, and compares against it. This walks that baseline on the same destination, the same 30
origins and the same border as the published learned arm, with the released files unmodified and
with no training, since a shortest-path plan needs none.

The learned arm is already published at 3000 iterations across ten seeds: 96.0% arrival with the
boundary closed and 40.0% once the hull is opened. This script supplies the missing column.

    python3 baseline_arm.py --dest 15 --origins 30 --samples 10
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "DRL-Router")); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import func                                                        # noqa: E402  released, unmodified
import DRL_C51                                                     # noqa: E402  released, unmodified
from retrain_released import build, border_nodes, _Quiet           # noqa: E402


def walk_dijkstra(agent, start, exits):
    state, cost, steps = start, 0.0, 0
    while True:
        if state == agent.termination:
            return "arrive", cost, steps
        if state in exits:
            return "exit", cost, steps
        flag, path, _ = func.dijkstra(agent.map.G, state - 1, agent.termination - 1)
        if flag == -1 or not path:
            return "stuck", cost, steps
        action = path[0] + 1
        if action not in agent.xtates[state - 1]['actions']:
            return "stuck", cost, steps
        idx = agent.xtates[state - 1]['actions'].index(action)
        state_prime = agent.xtates[state - 1]['state_prime'][idx]
        s2 = agent.map.sigma[action - 1][action - 1]; mu = agent.map.mu[action - 1]
        cov2 = np.log(s2 / mu ** 2 + 1); mul = np.log(mu ** 2 / np.sqrt(s2 + mu ** 2))
        cost += float(np.random.lognormal(mul, np.sqrt(cov2)))
        state, steps = state_prime, steps + 1
        if steps > agent.state_size:
            return "stuck", cost, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=int, default=15)
    ap.add_argument("--origins", type=int, default=30)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--band", type=float, default=0.0)
    ap.add_argument("--map", type=int, default=2)
    ap.add_argument("--out", default="baseline_arm_hull.json")
    a = ap.parse_args()
    m = build(a.map, a.dest)
    exits = set(border_nodes(HERE / "DRL-Router" / "Maps" / "Anaheim" / "anaheim_nodes.geojson",
                             band=a.band)) - {a.dest}
    rng = np.random.RandomState(12345)
    pool = [n for n in range(1, m.n_node + 1) if n != a.dest and n not in exits]
    origins = sorted(rng.choice(pool, size=min(a.origins, len(pool)), replace=False).tolist())
    with _Quiet():
        agent = DRL_C51.DRL_Agent(DRL_C51.Xtates(m, num_atoms=51), m, a.dest)
    res = {}
    for cond in ("closed", "open"):
        ex = exits if cond == "open" else set()
        per = []
        for s in range(a.samples):
            np.random.seed(s)
            c = {"arrive": 0, "exit": 0, "stuck": 0}
            for o in origins:
                r, _, _ = walk_dijkstra(agent, o, ex)
                c[r] += 1
            per.append(round(100.0 * c["arrive"] / len(origins), 1))
        res[cond] = {"per_sample": per, "mean": round(float(np.mean(per)), 1)}
        print("  %-7s baseline arrival %5.1f%%" % (cond, res[cond]["mean"]), flush=True)
    # The paper states each arm relative to what the border leaves attainable, which is the
    # quantity Section III-D defines. Deriving it here rather than in prose keeps every printed
    # percentage traceable to a run, which check_internal_contradiction requires.
    L = {"closed": 96.0, "open": 40.0}
    att = {"closed": 100.0, "open": res["open"]["mean"]}
    # 2026-09-02: gap_points took the difference of the RAW rates while sitting inside the
    # relative_to_attainable block, so the file reported 53.3 (93.3 - 40.0) beside a baseline of
    # 100.0 and a learned value of 42.9, whose difference is 57.1. A read-through caught the
    # mismatch. Both differences are kept, each named for the scale it belongs to.
    rel = {}
    for c in ("closed", "open"):
        b = round(100.0 * res[c]["mean"] / att[c], 1)
        l = round(100.0 * L[c] / att[c], 1)
        rel[c] = {"baseline": b, "learned": l,
                  "gap_points": round(b - l, 1),
                  "gap_points_raw": round(abs(res[c]["mean"] - L[c]), 1)}
    out = {"study": "[25] DRL-Router, released implementation, unmodified",
           "arm": "the study's own released Dijkstra baseline, no training",
           "destination": a.dest, "origins": len(origins), "border_nodes": len(exits),
           "band": a.band, "samples": a.samples, "results": res,
           "learned_arm_published": L, "attainable": att, "relative_to_attainable": rel}
    print("  relative to attainable:", json.dumps(rel))
    (HERE / a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
