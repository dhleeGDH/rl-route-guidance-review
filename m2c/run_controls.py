# -*- coding: utf-8 -*-
"""T-1941: the sentinel-encoding control and the truncation control, published and M2-C.

Both drivers live in the archived experiment tree and are reachable at handoff/experiments/ since
T-1921. Neither archived file is read for anything but import, and neither is written: every output
of this runner goes to handoff/m2c/controls/.

The archived copies of env.py, env_boundary_dest.py and four_cells_boundary_dest.py are supersets
of the released ones (state_sentinel, residual_on_exit, eval_boundary, reward_kw), identical at
their defaults, and the drivers call them with keywords the released copies do not take. The
archive directory is therefore put ahead of repo_v11/ on sys.path.

For the M2-C form only the environment class the driver instantiates is replaced by a subclass
carrying the discount-consistent increment, exactly as train_m2c.py and run_archived.py do. The
class is patched at env_boundary_dest before either driver imports it, so the driver's own
reference and four_cells_boundary_dest's both hold the subclass.

    python3 run_controls.py exit_sentinel P|C [--workers 16]
    python3 run_controls.py truncation    P|C [--workers 16]
    python3 run_controls.py sentinel      P|C [--workers 16]
    python3 run_controls.py arrival       P|C
"""
import io, json, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ARCH = os.path.join(ROOT, "handoff", "experiments", "boundary_open_demo")
sys.path.insert(0, HERE)

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402
torch.set_num_threads(1)
from shaping_m2c import DiscountConsistentShapingC              # noqa: E402

sys.path.insert(0, ARCH)                    # ahead of the repo_v11 paths shaping_m2c inserts

GAMMA = 0.99
OUT = os.path.join(HERE, "controls")


def patch(module, attr):
    base = getattr(module, attr)
    sub = type("M2C" + base.__name__, (DiscountConsistentShapingC, base), {"gamma_shaping": GAMMA})
    setattr(module, attr, sub)
    return sub


def prepare(variant):
    import env_boundary_dest
    if variant == "C":
        patch(env_boundary_dest, "BoundaryDestEnv")
    return env_boundary_dest


# --------------------------------------------------------------- the sentinel-encoding control
def exit_sentinel(variant, workers, seeds):
    prepare(variant)
    import exit_sentinel_control as drv
    dest = os.path.join(OUT, "exit_sentinel_control_%s.json" % variant)
    sys.argv = ["exit_sentinel_control", "--seeds", str(seeds), "--workers", str(workers),
                "--out", dest]
    drv.main()
    return dest


# --------------------------------------------------------------- the truncation control
_JOB = {}


def _one(job):
    boundary, reward, arm, seed = job
    v = _JOB["drv"].run_cell(boundary, reward, seed, _JOB["episodes"], _JOB["od"], arm == "bootstrapped")
    return job, v


def truncation(variant, workers, seeds, episodes=3000):
    """The driver's own run_cell over the four cells and the two arms.

    main() is not called: it asserts the terminal arm against the published four cells, which is
    the control for P and is false by construction for C, whose terminal arm is the M2-C cell.
    The measurement is the driver's, seed for seed; only the loop around it is here, so that the
    80 cells run on the pool instead of in series. run_cell seeds numpy and torch from its own
    seed argument, so a cell is independent of the order it is run in.
    """
    from concurrent.futures import ProcessPoolExecutor
    prepare(variant)
    import truncation_control as drv
    from env_boundary_dest import make_eval_od
    _JOB.update(drv=drv, od=make_eval_od(n=200, seed=12345), episodes=episodes)
    jobs = [(b, r, arm, s) for b in ("closed", "open") for r in ("time_min", "aligned")
            for arm in ("terminal", "bootstrapped") for s in range(seeds)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_one, jobs))
    out = {}
    for (b, r, arm, seed), v in res:
        out.setdefault("%s_%s" % (b, r), {}).setdefault(arm, [None] * seeds)[seed] = v
    for key, row in out.items():
        for arm in ("terminal", "bootstrapped"):
            row[arm] = {"mean": float(np.mean(row[arm])), "per_seed": row[arm]}
        row["delta"] = round(row["bootstrapped"]["mean"] - row["terminal"]["mean"], 1)
        print("  %-18s terminal %5.1f%%  bootstrapped %5.1f%%  delta %+.1f"
              % (key, row["terminal"]["mean"], row["bootstrapped"]["mean"], row["delta"]),
              flush=True)
    out["_meta"] = {"variant": variant, "episodes": episodes, "seeds": seeds,
                    "gamma_shaping": GAMMA if variant == "C" else None,
                    "seconds": round(time.time() - t0, 1)}
    dest = os.path.join(OUT, "truncation_control_%s.json" % variant)
    json.dump(out, io.open(dest, "w", encoding="utf-8"), indent=1)
    print("wrote", dest, flush=True)
    return dest


# --------------------------------------------------------------- the state-sentinel replacement
def _one_sentinel(job):
    sent, boundary, reward, seed = job
    return job, _JOB["drv"].run_cell(boundary, reward, seed, _JOB["episodes"], _JOB["od"],
                                     state_sentinel=sent)


def sentinel(variant, workers, seeds, episodes=3000):
    """The driver's own run_cell over the four cells under both encodings.

    main() is not called for two reasons. Its negative control opens the published cells at a
    path relative to the working directory, so it passes silently from anywhere but the archive
    directory; and it runs the eighty cells in series. Both are answered here: the published
    file is opened at its absolute path and every cell is compared to it seed for seed, which is
    the same comparison main() makes and is stricter than the mean it prints.
    """
    from concurrent.futures import ProcessPoolExecutor
    prepare(variant)
    import sentinel_control as drv
    from env_boundary_dest import make_eval_od
    _JOB.update(drv=drv, od=make_eval_od(n=200, seed=12345), episodes=episodes)
    jobs = [(sent, b, r, s) for sent in ("high", "matched") for b, r in drv.CELLS
            for s in range(seeds)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_one_sentinel, jobs))
    cells = {}
    for (sent, b, r, seed), v in res:
        cells.setdefault("%s_%s_%s" % (sent, b, r), [None] * seeds)[seed] = v
    pub = json.load(io.open(os.path.join(ARCH, "four_cells_boundary_dest.json"), encoding="utf-8"))
    out = {"episodes": episodes, "seeds": seeds, "cells": {}}
    for key, per_seed in sorted(cells.items()):
        row = {"per_seed": per_seed, "mean": float(np.mean(per_seed))}
        if key.startswith("high_"):
            ref = pub[key[len("high_"):]]["per_seed"]
            row["reproduces_published"] = bool(np.allclose(ref, per_seed, atol=1e-6))
            row["published_per_seed"] = ref
        out["cells"][key] = row
        print("  %-24s mean %5.1f%%   %s" % (key, row["mean"],
              "" if "reproduces_published" not in row else
              "negative control: %s" % ("MATCH" if row["reproduces_published"] else "MISMATCH")),
              flush=True)
    h = np.array(out["cells"]["high_open_aligned"]["per_seed"], float)
    m = np.array(out["cells"]["matched_open_aligned"]["per_seed"], float)
    out["paired_open_aligned"] = {"per_seed": list(h - m), "mean": float((h - m).mean())}
    print("  paired high - matched, open aligned: %.1f points" % (h - m).mean(), flush=True)
    out["_meta"] = {"variant": variant, "gamma_shaping": GAMMA if variant == "C" else None,
                    "seconds": round(time.time() - t0, 1)}
    dest = os.path.join(OUT, "sentinel_control_%s.json" % variant)
    json.dump(out, io.open(dest, "w", encoding="utf-8"), indent=1)
    print("wrote", dest, flush=True)
    return dest


# --------------------------------------------------------------- the arrival-term encoding control
def arrival(variant, seeds):
    """The driver unchanged, through its own main(), with its output redirected.

    Forty cells in series is a few minutes, and the driver carries its own negative control
    against arrival_term_budget.json at an absolute path, so nothing is gained by replacing the
    loop. The arrival-term condition sets beta = 0, so the M2-C increment is zero by construction
    and the C arm is expected to reproduce the P arm exactly.
    """
    prepare(variant)
    import arrival_sentinel_control as drv
    dest = os.path.join(OUT, "arrival_sentinel_control_%s.json" % variant)
    sys.argv = ["arrival_sentinel_control", "--seeds", str(seeds), "--out", dest]
    drv.main()
    return dest


if __name__ == "__main__":
    which, variant = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 16
    seeds = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else 10
    episodes = int(sys.argv[sys.argv.index("--episodes") + 1]) if "--episodes" in sys.argv else 3000
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    if which == "exit_sentinel":
        exit_sentinel(variant, workers, seeds)
    elif which == "truncation":
        truncation(variant, workers, seeds, episodes)
    elif which == "sentinel":
        sentinel(variant, workers, seeds, episodes)
    elif which == "arrival":
        arrival(variant, seeds)
    else:
        raise SystemExit("unknown control %r" % which)
    print("%s %s done (%.0fs)" % (which, variant, time.time() - t0), flush=True)
