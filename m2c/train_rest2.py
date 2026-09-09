# -*- coding: utf-8 -*-
"""T-1905: the second batch of destination-aligned learned cells under the M2-C reward.

Same discipline as train_rest.py of T-1904. Each published driver is used unchanged and only the
environment class it instantiates is replaced by an M2-C subclass. Travel-time cells carry no
shaping term, so the mixin is inert on them; where one is run here it is the driver-swap control
of the ticket's stop condition (a).

One cell per process, so that the cells run four at a time and the 5x5 and 7x7 SUMO drivers, which
rebind sumo_env.N and sumo_env.NET at import, never share an interpreter.

    python3 train_rest2.py <cell>          one cell, writes cells/<cell>.json
    python3 train_rest2.py --list          the cell names
"""
import json, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.join(ROOT, "repo_v11")
for p in ("boundary_open_demo", "benchmark_network", "sumo_corridor"):
    sys.path.insert(0, os.path.join(REPO, p))
sys.path.insert(0, HERE)

import numpy as np
import torch
torch.set_num_threads(1)
from shaping_m2c import DiscountConsistentShapingC      # noqa: E402

GAMMA = 0.99
EPIS = os.environ.get("T1905_EPISODES")                  # smoke override
OUT = os.path.join(HERE, "cells")


def patched(module, attr):
    base = getattr(module, attr)
    sub = type("M2C" + base.__name__, (DiscountConsistentShapingC, base), {"gamma_shaping": GAMMA})
    setattr(module, attr, sub)
    return sub


def _ep(n):
    return int(EPIS) if EPIS else n


# ---------------------------------------------------------------- the SUMO 5x5 grid
def _sumo5(boundary, reward, episodes, seeds, variant):
    import sumo_train as T
    if variant == "C":
        patched(T, "SumoGridEnv")
    eval_od = T.make_eval_od()
    return [100.0 * T.train_condition(boundary, reward, s, episodes, eval_od,
                                      "%s%s%s%d" % (variant, boundary[0], reward[0], s))
            for s in seeds]


# ---------------------------------------------------------------- the SUMO 7x7 grid
def _sumo7(boundary, reward, episodes, seeds, variant):
    import sumo_recon as R
    if variant == "C":
        patched(R, "SumoGridEnv")
    eval_od = R.make_eval_od()
    return [100.0 * R.train_condition(boundary, reward, s, episodes, eval_od,
                                      "%s%s%s%d" % (variant, boundary[0], reward[0], s))
            for s in seeds]


# ---------------------------------------------------------------- Sioux Falls, interior destination
def _siouxint(boundary, reward, episodes, seeds, variant):
    import benchmark_demo as B
    if variant == "C":
        patched(B, "GraphRouteEnv")
    net = dict(B.NETWORKS["sioux_falls"])
    net["od_mode"] = "dest_interior"
    od = B.make_eval_od(net)
    return [100.0 * B.train_eval(net, boundary, reward, s, episodes, od) for s in seeds]


# ---------------------------------------------------------------- the 8x8 lattice
def _grid8(boundary, reward, episodes, seeds, variant):
    """The 8x8 lattice replication (S-I.E).

    The published ten-seed driver is not in the released tree. The construction is rebuilt here
    from the released env and the released evaluation draw: sweep_extra.make_eval_od(8) at its
    default separation gives 200 pairs and 191 distinct, which is the lattice row of Table S-4,
    and Table S-1 fixes the budget at 120 steps. It reproduces the published cell closely but not
    seed for seed, so the published reward is run beside the M2-C one and both are reported.
    """
    import env as E
    import sweep_extra as X
    if variant == "C":
        patched(E, "GridRouteEnv")
        patched(X, "GridRouteEnv")
    od = X.make_eval_od(8)
    kw = dict(boundary=boundary, reward=reward, n_side=8, max_steps=120)
    return [100.0 * X.train_eval(kw, s, episodes, od) for s in seeds]


R10 = list(range(10))
CELLS = {
    # the driver-swap control: the mixin is inert on a travel-time reward
    "s5_p_closed_tm3000": (_sumo5, "closed", "time_min", 3000, R10, "P"),
    "s5_p_open_tm3000":   (_sumo5, "open",   "time_min", 3000, R10, "P"),
    "s5_c_closed_al3000": (_sumo5, "closed", "aligned",  3000, R10, "C"),
    "s5_c_open_al3000":   (_sumo5, "open",   "aligned",  3000, R10, "C"),
    "s5_c_closed_al8000": (_sumo5, "closed", "aligned",  8000, R10, "C"),
    "s5_c_open_al8000":   (_sumo5, "open",   "aligned",  8000, R10, "C"),
    # the 7x7 published ten-seed run is not in the released tree, so it is re-run beside C
    "s7_p_open_al8000":   (_sumo7, "open",   "aligned",  8000, R10, "P"),
    "s7_p_closed_al8000": (_sumo7, "closed", "aligned",  8000, R10, "P"),
    "s7_p_open_tm8000":   (_sumo7, "open",   "time_min", 8000, R10, "P"),
    "s7_c_open_al8000":   (_sumo7, "open",   "aligned",  8000, R10, "C"),
    "s7_c_closed_al8000": (_sumo7, "closed", "aligned",  8000, R10, "C"),
    # 30 seeds, split in two so that no single cell holds up the batch
    "si_c_closed_al8000_a": (_siouxint, "closed", "aligned", 8000, list(range(15)), "C"),
    "si_c_closed_al8000_b": (_siouxint, "closed", "aligned", 8000, list(range(15, 30)), "C"),
    "si_c_open_al8000_a":   (_siouxint, "open",   "aligned", 8000, list(range(15)), "C"),
    "si_c_open_al8000_b":   (_siouxint, "open",   "aligned", 8000, list(range(15, 30)), "C"),
    # the 8x8 lattice, reconstructed driver, published beside C
    "g8_p_open_al3000":   (_grid8, "open",   "aligned",  3000, R10, "P"),
    "g8_c_open_al3000":   (_grid8, "open",   "aligned",  3000, R10, "C"),
    "g8_p_open_al8000":   (_grid8, "open",   "aligned",  8000, R10, "P"),
    "g8_c_open_al8000":   (_grid8, "open",   "aligned",  8000, R10, "C"),
    "g8_p_closed_al3000": (_grid8, "closed", "aligned",  3000, R10, "P"),
    "g8_c_closed_al3000": (_grid8, "closed", "aligned",  3000, R10, "C"),
    "g8_p_closed_al8000": (_grid8, "closed", "aligned",  8000, R10, "P"),
    "g8_c_closed_al8000": (_grid8, "closed", "aligned",  8000, R10, "C"),
    "g8_p_open_tm3000":   (_grid8, "open",   "time_min", 3000, R10, "P"),
    "g8_p_open_tm8000":   (_grid8, "open",   "time_min", 8000, R10, "P"),
}


if __name__ == "__main__":
    if sys.argv[1] == "--list":
        print("\n".join(CELLS))
        sys.exit(0)
    name = sys.argv[1]
    fn, boundary, reward, episodes, seeds, variant = CELLS[name]
    episodes = _ep(episodes)
    t0 = time.time()
    vals = fn(boundary, reward, episodes, seeds, variant)
    rec = dict(cell=name, boundary=boundary, reward=reward, episodes=episodes, variant=variant,
               seeds=seeds, mean=round(float(np.mean(vals)), 1),
               values=[round(float(v), 1) for v in vals],
               seconds=round(time.time() - t0, 1))
    os.makedirs(OUT, exist_ok=True)
    json.dump(rec, open(os.path.join(OUT, name + ".json"), "w"), indent=1)
    print("%-24s %-6s %-8s %5dep %s  mean %5.1f  (%.0fs)"
          % (name, boundary, reward, episodes, variant, rec["mean"], rec["seconds"]), flush=True)
