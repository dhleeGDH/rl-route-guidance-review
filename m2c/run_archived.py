# -*- coding: utf-8 -*-
"""T-1919: run the three archived drivers T-1905 could not reach, published and M2-C.

The drivers live in the archived experiment tree and are reachable at handoff/experiments/ since
T-1921. Each is imported and its main() called, so the published code path runs unchanged; for the
M2-C form only the environment class the driver instantiates is replaced by a subclass carrying the
discount-consistent increment, exactly as train_rest.py and train_rest2.py do.

No archived file is written. Every driver's output is redirected into handoff/m2c/cells3/:
two take --out, and residual_exit_control.py writes to its own HERE, which is rebound here.

    python3 run_archived.py <driver> <P|C> [extra argv]
"""
import importlib, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EXP = os.path.join(ROOT, "handoff", "experiments")
for p in ("boundary_open_demo", "benchmark_network", "sumo_corridor"):
    sys.path.insert(0, os.path.join(EXP, p))
sys.path.insert(0, HERE)

import torch
torch.set_num_threads(1)
from shaping_m2c import DiscountConsistentShapingC      # noqa: E402

GAMMA = 0.99
OUT = os.path.join(HERE, "cells3")

# driver module -> (module holding the env class, attribute name, default out flag)
# driver -> (module holding the env class, attribute, out flag, how the driver is entered)
# sumo_interior_dest.py has no main(); its body sits under __main__, so it is entered with runpy
# and the env class is patched at its source module before the driver imports it.
DRIVERS = {
    "dqn_sensitivity":      ("four_cells_boundary_dest", "BoundaryDestEnv", "--out", "main"),
    "residual_exit_control": ("residual_exit_control", "BoundaryDestEnv", None, "main"),
    "sumo_interior_dest":   ("sumo_env", "SumoGridEnv", "--out", "runpy"),
}


def patch(module, attr):
    base = getattr(module, attr)
    sub = type("M2C" + base.__name__, (DiscountConsistentShapingC, base), {"gamma_shaping": GAMMA})
    setattr(module, attr, sub)
    return sub


if __name__ == "__main__":
    name, variant = sys.argv[1], sys.argv[2]
    extra = sys.argv[3:]
    envmod, envattr, outflag, entry = DRIVERS[name]
    os.makedirs(OUT, exist_ok=True)
    tag = os.environ.get("T1919_TAG", "")
    dest = os.path.join(OUT, "%s_%s%s.json" % (name, variant, tag))
    t0 = time.time()
    if entry == "runpy":
        if variant == "C":
            patch(importlib.import_module(envmod), envattr)
        sys.argv = [name] + extra + [outflag, dest]
        import runpy
        runpy.run_module(name, run_name="__main__")
    else:
        drv = importlib.import_module(name)
        if variant == "C":
            patch(importlib.import_module(envmod), envattr)
        if outflag:
            sys.argv = [name] + extra + [outflag, dest]
        else:
            drv.HERE = OUT                   # residual_exit_control writes HERE/<name>.json
            sys.argv = [name] + extra
        drv.main()
        if not outflag:
            os.rename(os.path.join(OUT, name + ".json"), dest)
    print("%s %s -> %s  (%.0fs)" % (name, variant, os.path.basename(dest), time.time() - t0),
          flush=True)
