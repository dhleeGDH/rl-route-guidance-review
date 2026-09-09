# -*- coding: utf-8 -*-
"""T-1904: the remaining destination-aligned learned cells under the M2-C reward.

Each published driver is used unchanged; only the environment class it instantiates is replaced by
a subclass carrying the M2-C increment. Travel-time cells are not retrained: their reward has no
shaping term, so the change cannot reach them.

    python3 train_rest.py <group>     groups: sioux, interior, depth, exitconv
"""
import json, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in ("boundary_open_demo", "benchmark_network"):
    sys.path.insert(0, os.path.join(ROOT, "repo_v11", p))
sys.path.insert(0, HERE)

import numpy as np
import torch
torch.set_num_threads(1)
from shaping_m2c import DiscountConsistentShapingC      # noqa: E402

GAMMA = 0.99


def patched(module, attr, base):
    """Replace module.attr with an M2-C subclass of the published class."""
    sub = type("M2C" + base.__name__, (DiscountConsistentShapingC, base), {"gamma_shaping": GAMMA})
    setattr(module, attr, sub)
    return sub


def sioux():
    """Sioux Falls, 30 seeds at 8000 episodes, the destination-aligned cells."""
    import benchmark_demo as B
    patched(B, "GraphRouteEnv", B.GraphRouteEnv)
    net = dict(B.NETWORKS["sioux_falls"])
    od = B.make_eval_od(net, n=200, seed=999)
    out = {}
    for boundary in ("closed", "open"):
        t0 = time.time()
        vals = [100.0 * B.train_eval(net, boundary, "aligned", s, net["episodes"], od)
                for s in range(30)]
        out["sioux|%s|aligned|8000" % boundary] = dict(mean=round(float(np.mean(vals)), 1),
                                                       values=[round(v, 1) for v in vals],
                                                       seconds=round(time.time() - t0, 1))
        print("sioux %-7s aligned 8000ep 30 seeds: mean %5.1f (%.0fs)"
              % (boundary, np.mean(vals), time.time() - t0), flush=True)
    return out


def interior():
    """The interior-destination control on the bespoke grid, ten seeds at both budgets."""
    import interior_destination as I
    patched(I, "InteriorDestEnv", I.InteriorDestEnv)
    eval_od = I.make_od(200, I.N_SIDE, seed=12345)     # the published draw of this control
    out = {}
    for episodes in (3000, 8000):
        for boundary in ("closed", "open"):
            t0 = time.time()
            vals = [I.run_cell(boundary, "aligned", s, episodes, eval_od) for s in range(10)]
            vals = [v[0] if isinstance(v, tuple) else v for v in vals]
            out["interior_grid|%s|aligned|%d" % (boundary, episodes)] = dict(
                mean=round(float(np.mean(vals)), 1), values=[round(float(v), 1) for v in vals],
                seconds=round(time.time() - t0, 1))
            print("interior grid %-7s aligned %dep: mean %5.1f (%.0fs)"
                  % (boundary, episodes, np.mean(vals), time.time() - t0), flush=True)
    return out


def depth():
    """The depth-sweep learned cells, five sides, ten seeds at 8000 episodes."""
    import interior_deep_control as D
    patched(D, "DeepInteriorEnv", D.DeepInteriorEnv)
    out = {}
    for n_side in (5, 7, 9, 11, 13):
        D.DeepInteriorEnv.MARGIN = 4 if n_side >= 13 else max(1, (n_side - 3) // 2)
        od = D.eval_od(200, n_side, D.DeepInteriorEnv.MARGIN, seed=12345)
        for boundary in ("closed", "open"):
            t0 = time.time()
            vals = [D.run_cell(boundary, "aligned", s, 8000, od, n_side) for s in range(10)]
            out["depth%d|%s|aligned|8000" % (n_side, boundary)] = dict(
                mean=round(float(np.mean(vals)), 1), values=[round(float(v), 1) for v in vals],
                seconds=round(time.time() - t0, 1))
            print("depth side %-3d %-7s aligned 8000ep: mean %5.1f (%.0fs)"
                  % (n_side, boundary, np.mean(vals), time.time() - t0), flush=True)
    return out


def exitconv():
    """The two alternative exit conventions, ten seeds at 3000 episodes."""
    import env as E
    from env_boundary_dest import make_eval_od
    import four_cells_curves as F
    patched(F, "BoundaryDestEnv", F.BoundaryDestEnv)
    eval_od = make_eval_od(n=200, seed=12345)
    out = {}
    for boundary in ("costly_return",):
        t0 = time.time()
        vals = [F.run_cell(boundary, "aligned", s, 3000, eval_od, 3000)[1][-1] for s in range(10)]
        out["exitconv|%s|aligned|3000" % boundary] = dict(
            mean=round(float(np.mean(vals)), 1), values=[round(float(v), 1) for v in vals],
            seconds=round(time.time() - t0, 1))
        print("exit convention %-14s aligned 3000ep: mean %5.1f (%.0fs)"
              % (boundary, np.mean(vals), time.time() - t0), flush=True)
    return out


if __name__ == "__main__":
    g = sys.argv[1]
    res = {"sioux": sioux, "interior": interior, "depth": depth, "exitconv": exitconv}[g]()
    json.dump(res, open(os.path.join(HERE, "train_rest_%s.json" % g), "w"), indent=1)
