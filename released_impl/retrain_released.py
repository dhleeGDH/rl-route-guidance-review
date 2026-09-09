# -*- coding: utf-8 -*-
"""Retrain a published implementation on a boundary-open variant of its own network.

WHY THIS EXISTS. Round 43's M1 objects that the failure this study demonstrates is not observed in
any reviewed study, and asks for one of the seven code-released studies among the 41 to be retrained
under a boundary-open condition. The adjudication in `code_boundary_adjudication.md` records six of
the seven as boundary-closed in their shipped environment: nothing in them offers a move that leaves
the modeled network, so the failure cannot arise as released. That is a statement about what the
released code does, not about what would happen if the boundary were opened.

This opens it. The target is [25], whose implementation ships its own Anaheim network, terminates an
episode only at the destination, and learns a distributional value over travel time. The released
files are not edited. The boundary-open variant is a subclass that adds a second terminal set, the
border nodes of the study's own network, and a walk that reports which terminal a trip reached.

The border is the one Section IV-A defines and `exposure_ratio_real_networks.py` computes: the convex
hull of the published node coordinates together with every node within a tenth of the bounding box of
an edge. The reward is the study's own, a lognormal travel-time sample per link, unchanged.

    python3 retrain_released.py --dest 15 --origins 40 --iters 4000
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path(sys.argv[sys.argv.index("--repo") + 1]) if "--repo" in sys.argv else HERE / "DRL-Router"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parent))

import func            # noqa: E402  the released module, unmodified
import DRL_C51         # noqa: E402  the released module, unmodified
from exposure_ratio_real_networks import _hull, load_coords   # noqa: E402


def border_nodes(coord_path, band=0.10):
    coords = load_coords(coord_path)
    xs = [c[0] for c in coords.values()]
    ys = [c[1] for c in coords.values()]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    dx, dy = (x1 - x0) * band, (y1 - y0) * band
    hull = _hull([(k, v[0], v[1]) for k, v in coords.items()])
    edge = {k for k, (x, y) in coords.items()
            if x <= x0 + dx or x >= x1 - dx or y <= y0 + dy or y >= y1 - dy}
    return hull | edge


class BoundaryOpenC51(DRL_C51.DRL_Agent):
    """The released learner on a network whose border ends a trip.

    Only the terminal set changes. Where the released class ends an episode at `termination` alone,
    this one ends it at `termination` or at any border node, which is the boundary-open condition of
    Section IV-A. The reward, the update, the action space and the exploration are the released ones.
    """

    def set_exits(self, exits):
        self.exits = set(int(e) for e in exits) - {int(self.termination)}

    def _terminal(self, state):
        return state == self.termination or state in getattr(self, "exits", ())

    # the three updates and the walk differ from the released ones only in the terminal test
    def C51_with_dijkstra(self, state, action_idx, reward, state_prime):
        if self._terminal(state_prime) and state_prime != self.termination:
            state_prime = self.termination        # collapse to the terminal branch
        return super().C51_with_dijkstra(state, action_idx, reward, state_prime)

    def C51_on_policy(self, state, action_idx, reward, state_prime, parameter=None, obj="LET"):
        if self._terminal(state_prime) and state_prime != self.termination:
            state_prime = self.termination
        return super().C51_on_policy(state, action_idx, reward, state_prime, parameter, obj)

    def walk(self, start, parameter=0.1, obj="LET"):
        """Greedy walk under the learned distributions, reporting the terminal reached."""
        state, cost, steps = start, 0.0, 0
        while True:
            if state == self.termination:
                return "arrive", cost, steps
            if state in getattr(self, "exits", ()):
                return "exit", cost, steps
            if len(self.xtates[state - 1]["actions"]) == 0:
                return "stuck", cost, steps
            _, action, state_prime = self.get_optimal_action(
                self.xtates, state, parameter=parameter, obj=obj)
            sigma2 = self.map.sigma[action - 1][action - 1]
            mu = self.map.mu[action - 1]
            cov2_log = np.log(sigma2 / mu ** 2 + 1)
            mu_log = np.log(mu ** 2 / (np.sqrt(sigma2 + mu ** 2)))
            cost += float(np.random.lognormal(mu_log, np.sqrt(cov2_log)))
            state, steps = state_prime, steps + 1
            if steps > self.state_size:
                return "stuck", cost, steps


def build(map_index, dest):
    # extract_map resolves its table paths relative to the working directory, so the released
    # repository is entered rather than copied out of.
    import os
    os.chdir(str(REPO))
    m = func.Map()
    m.extract_map(map_index=map_index, r_0=1, r_s=dest)
    m.G = func.convert_map2graph(m)
    return m


class _Quiet:
    """The released trainer prints a full node path per episode, which is megabytes per run."""

    def __enter__(self):
        import io as _io
        self._old, sys.stdout = sys.stdout, _io.StringIO()
        return self

    def __exit__(self, *a):
        sys.stdout = self._old
        return False


def run(condition, m, dest, origins, iters, exits, seed):
    np.random.seed(seed)
    X = DRL_C51.Xtates(m, num_atoms=51)
    cls = BoundaryOpenC51 if condition == "open" else DRL_C51.DRL_Agent
    agent = cls(X, m, dest)
    if condition == "open":
        agent.set_exits(exits)
    with _Quiet():
        agent.train_dijkstra(iters)
    out = {"arrive": 0, "exit": 0, "stuck": 0}
    for o in origins:
      with _Quiet():
        if condition == "open":
            r, _, _ = agent.walk(o)
        else:
            cost, path, _ = agent.find_path(o)
            r = "arrive" if path != -1 else "stuck"
      out[r] += 1
    n = sum(out.values())
    return {"counts": out, "pairs": n, "arrive_rate": round(100.0 * out["arrive"] / n, 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(HERE / "DRL-Router"))
    ap.add_argument("--map", type=int, default=2, help="2 is the shipped Anaheim network")
    ap.add_argument("--dest", type=int, default=15)
    ap.add_argument("--origins", type=int, default=40)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="retrain_released.json")
    ap.add_argument("--band", type=float, default=0.10,
                    help="border band as a fraction of the bounding box; 0 is the convex hull alone")
    a = ap.parse_args()

    m = build(a.map, a.dest)
    exits = border_nodes(Path(a.repo) / "Maps" / "Anaheim" / "anaheim_nodes.geojson", band=a.band)
    rng = np.random.RandomState(12345)
    pool = [n for n in range(1, m.n_node + 1) if n != a.dest and n not in exits]
    origins = sorted(rng.choice(pool, size=min(a.origins, len(pool)), replace=False).tolist())
    print("network: %d nodes, %d links; border %d; destination %d; %d origins; %d iterations"
          % (m.n_node, m.n_link, len(exits), a.dest, len(origins), a.iters), flush=True)

    res = {}
    t0 = time.time()
    for condition in ("closed", "open"):
        per_seed, per_seed_counts = [], []
        for s in range(a.seeds):
            r = run(condition, m, a.dest, origins, a.iters, exits, seed=s)
            # The exit rate, not the arrival rate, is what separates the two conditions: a walk
            # that neither arrives nor exits is a greedy cycle cut at the step cap, which belongs
            # to training convergence rather than to the boundary. Both are recorded.
            per_seed.append(r["arrive_rate"])
            per_seed_counts.append(dict(r["counts"]))
            print("  %-6s seed %d: arrive %5.1f%%  %s  [%ds]"
                  % (condition, s, r["arrive_rate"], r["counts"], time.time() - t0), flush=True)
        v = np.asarray(per_seed, float)
        _n = float(sum(sum(c.values()) for c in per_seed_counts)) or 1.0
        res[condition] = {"per_seed": per_seed, "mean": float(v.mean()),
                          "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                          "counts_per_seed": per_seed_counts,
                          "exit_rate": 100.0 * sum(c.get("exit", 0) for c in per_seed_counts) / _n,
                          "stuck_rate": 100.0 * sum(c.get("stuck", 0) for c in per_seed_counts) / _n}
        print("== %-6s arrive %5.1f%% (sd %4.1f)" % (condition, v.mean(), res[condition]["sd"]),
              flush=True)

    json.dump({"study": "[25] DRL-Router, released implementation, unmodified",
               "network": "its own shipped Anaheim, %d nodes" % m.n_node,
               "border_nodes": len(exits), "band": a.band, "destination": a.dest, "origins": len(origins),
               "iterations": a.iters, "seeds": a.seeds, "results": res},
              open(HERE / a.out, "w"), indent=1)
    print("wrote %s" % a.out)
