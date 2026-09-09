# -*- coding: utf-8 -*-
"""Does the boundary condition change the comparison the study itself reports?

WHY THIS EXISTS. The 2026-09-02 cold review, M3, objects that no reviewed study is shown to have a
conclusion that the boundary condition actually undermines. The retraining already in the paper
opens the boundary of a published implementation and reports that the learned policy leaves. A
reviewer reads that as Proposition 2 restated on a larger network, and the objection stands: the
study's own claim is a comparison against a baseline, and that comparison was never re-measured.

WHAT THIS ADDS. [25] ships a Dijkstra baseline inside the released code and uses it both to warm
start the learner and as the reference its own text compares against. This script walks BOTH under
the same two boundary conditions, on the same origin set, with the released files unmodified:

    arm A  the learned distributional policy      (the study's method)
    arm B  the released Dijkstra baseline         (the study's own reference)

A deterministic shortest-path baseline plans to the destination and therefore arrives whichever
terminal set the environment carries. A policy that minimises travel time leaves once a nearer
terminal exists. The prediction under test is that the two arms agree with the boundary closed and
separate once it opens, so the advantage the study reports is a property of the condition rather
than of the method alone. A null result is equally reportable: if the baseline also fails, the
comparison survives the condition and M3 is answered by saying so.

    python3 baseline_vs_learned.py --dest 15 --origins 30 --iters 3000 --seeds 10
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "DRL-Router")); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import func                                        # noqa: E402  released, unmodified
import DRL_C51                                     # noqa: E402  released, unmodified
from retrain_released import BoundaryOpenC51, build, border_nodes, _Quiet   # noqa: E402


def walk_dijkstra(agent, start, exits):
    """The study's own baseline under the same terminal set, with its own cost sampling."""
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
        idx = agent.xtates[state - 1]['actions'].index(action)
        state_prime = agent.xtates[state - 1]['state_prime'][idx]
        sigma2 = agent.map.sigma[action - 1][action - 1]; mu = agent.map.mu[action - 1]
        cov2_log = np.log(sigma2 / mu ** 2 + 1); mu_log = np.log(mu ** 2 / (np.sqrt(sigma2 + mu ** 2)))
        cost += float(np.random.lognormal(mu_log, np.sqrt(cov2_log)))
        state, steps = state_prime, steps + 1
        if steps > agent.state_size:
            return "stuck", cost, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=int, default=15)
    ap.add_argument("--origins", type=int, default=30)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--band", type=float, default=0.0)
    ap.add_argument("--map", type=int, default=2)
    ap.add_argument("--out", default="baseline_vs_learned.json")
    a = ap.parse_args()

    exits = set(border_nodes(HERE / "DRL-Router" / "Maps" / "Anaheim" / "anaheim_nodes.geojson",
                             band=a.band)) - {a.dest}
    m = build(a.map, a.dest)
    rng = np.random.RandomState(12345)
    pool = [n for n in range(1, m.n_node + 1) if n != a.dest and n not in exits]
    origins = sorted(rng.choice(pool, size=min(a.origins, len(pool)), replace=False).tolist())

    res = {}
    for cond in ("closed", "open"):
        ex = exits if cond == "open" else set()
        learned, base = [], []
        for s in range(a.seeds):
            np.random.seed(s)
            X = DRL_C51.Xtates(m, num_atoms=51)
            cls = BoundaryOpenC51 if cond == "open" else DRL_C51.DRL_Agent
            agent = cls(X, m, a.dest)
            if cond == "open":
                agent.set_exits(ex)
            with _Quiet():
                agent.train_dijkstra(a.iters)
            cl = {"arrive": 0, "exit": 0, "stuck": 0}
            cb = {"arrive": 0, "exit": 0, "stuck": 0}
            for o in origins:
                with _Quiet():
                    if cond == "open":
                        r, _, _ = agent.walk(o)
                    else:
                        _, path, _ = agent.find_path(o)
                        r = "arrive" if path != -1 else "stuck"
                cl[r] += 1
                with _Quiet():
                    rb, _, _ = walk_dijkstra(agent, o, ex)
                cb[rb] += 1
            n = float(len(origins))
            learned.append(round(100.0 * cl["arrive"] / n, 1))
            base.append(round(100.0 * cb["arrive"] / n, 1))
            print("  %-6s seed %d  learned %5.1f%%  baseline %5.1f%%" % (cond, s, learned[-1], base[-1]), flush=True)
        res[cond] = {"learned_per_seed": learned, "baseline_per_seed": base,
                     "learned_mean": round(float(np.mean(learned)), 1),
                     "baseline_mean": round(float(np.mean(base)), 1)}
    out = {"study": "[25] DRL-Router, released implementation, unmodified",
           "arms": {"A": "the study's learned distributional policy",
                    "B": "the study's own released Dijkstra baseline"},
           "destination": a.dest, "origins": len(origins), "border_nodes": len(exits),
           "band": a.band, "iterations": a.iters, "seeds": a.seeds, "results": res}
    (HERE / a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
