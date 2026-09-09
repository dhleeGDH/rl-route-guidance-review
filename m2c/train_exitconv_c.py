# -*- coding: utf-8 -*-
"""T-1904: the two alternative exit conventions under M2-C, on the published construction.

The published driver is repo_v11/boundary_open_demo/costly_return.py, which trains on GridRouteEnv
through sweep_extra.train_eval at n_side 5, max_steps 50, ten seeds. Only the environment class is
replaced. The residual-charge convention is the boundary "open" cell of the same construction under
the residual charge documented in S-I.B, which the published env exposes as return_cost.
"""
import json, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "repo_v11", "boundary_open_demo"))
sys.path.insert(0, HERE)
import numpy as np, torch
torch.set_num_threads(1)
import sweep_extra as S
import env as E
from shaping_m2c import DiscountConsistentShapingC

M2C = type("M2CGridRouteEnv", (DiscountConsistentShapingC, E.GridRouteEnv), {"gamma_shaping": 0.99})
S.GridRouteEnv = M2C          # the driver builds its env through this name
E.GridRouteEnv = M2C

if __name__ == "__main__":
    eval_od = S.make_eval_od(5)
    out = {}
    cells = [("costly_return, detour 1.0, aligned",
              dict(boundary="costly_return", reward="aligned", n_side=5, max_steps=50, return_cost=1.0)),
             ("costly_return, detour 1.0, travel-time",
              dict(boundary="costly_return", reward="time_min", n_side=5, max_steps=50, return_cost=1.0)),
             ("costly_return, detour 3.0, travel-time",
              dict(boundary="costly_return", reward="time_min", n_side=5, max_steps=50, return_cost=3.0)),
             ("costly_return, detour 5.0, travel-time",
              dict(boundary="costly_return", reward="time_min", n_side=5, max_steps=50, return_cost=5.0)),
             ("open, aligned (reference)",
              dict(boundary="open", reward="aligned", n_side=5, max_steps=50)),
             ("closed, aligned (reference)",
              dict(boundary="closed", reward="aligned", n_side=5, max_steps=50))]
    for label, kw in cells:
        t0 = time.time()
        vals = [100.0 * S.train_eval(kw, s, 3000, eval_od) for s in range(10)]
        out[label] = dict(mean=round(float(np.mean(vals)), 1),
                          values=[round(v, 1) for v in vals], seconds=round(time.time() - t0, 1))
        print("%-42s mean %5.1f  (%.0fs)" % (label, np.mean(vals), time.time() - t0), flush=True)
    json.dump(out, open(os.path.join(HERE, "train_exitconv_c.json"), "w"), indent=1)
