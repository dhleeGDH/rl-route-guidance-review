# -*- coding: utf-8 -*-
"""Derive the Table VI rows, and the body figures that quote them, from the run outputs.

WHY THIS EXISTS. Table VI reported its bespoke-grid rows over ten seeds with a bootstrap interval,
its extended-budget row over five seeds with a standard deviation, and its SUMO rows over five seeds
with a standard deviation. A reviewer objected that rows differing in seed count, budget and
dispersion statistic cannot carry a within-table comparison, since any difference between two rows is
confounded with the protocol behind them. Every learned cell was therefore rerun at ten seeds, and
the budget moved from the caption into a column of its own.

This prints the rows rather than leaving them to be copied. The builder's constants and the
manuscript's prose have twice drifted from the runs they describe, and a row transcribed by hand is
the same failure waiting to happen. Every figure below is read from the run's own output file.

    python3 table6_rows.py
"""
import json
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
B, SEED = 10000, 20260719          # the resample count and constant bootstrap_ci.py already uses


def ci(per_seed, label):
    """A percentile bootstrap whose interval does not depend on what was computed before it.

    bootstrap_ci.py draws every cell from one generator, so an interval moves when a cell is
    inserted ahead of it: the 5x5 open aligned cell printed 19.9 to 42.6 there and 19.9 to 42.4
    here purely from the draw order. The generator is therefore reseeded per cell, from the same
    constant and a checksum of the cell's own name. Python's hash() is unusable for this, since it
    is salted per process and would give one cell two intervals on two runs.
    """
    x = np.asarray(per_seed, float)
    rng = np.random.RandomState((SEED + zlib.crc32(label.encode("utf-8"))) % (2 ** 32))
    means = np.mean(rng.choice(x, size=(B, len(x)), replace=True), axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cell_key(net, reward, episodes, boundary):
    """The identity of a cell, independent of how any table chooses to label it.

    Seeding the bootstrap from a display label gave one cell two intervals: the SUMO open aligned
    cell printed 18.6 to 36.6 under the label this file uses and 18.6 to 36.2 under the label the
    builder prints. The key below names the run, never the caption.
    """
    return "%s|%s|%d|%s" % (net, reward, int(episodes), boundary)


def fmt(per_seed, key):
    """The printed cell.

    A cell whose seeds all agree produces an interval of zero width, and "0.0 [0.0-0.0]" carries no
    information a reader can use. Such a cell prints the agreement instead, which is the fact the
    interval was standing in for. Every other cell keeps the percentile interval.
    """
    if per_seed is None:
        return "PENDING", 0
    m, lo, hi = ci(per_seed, key)
    n = len(per_seed)
    if abs(hi - lo) < 0.05 and abs(m - lo) < 0.05:
        return "%.1f (all %d seeds)" % (m, n), n
    return "%.1f [%.1f-%.1f]" % (m, lo, hi), n


def load(path, key):
    """Per-seed values from a run output, or None when that run has not landed."""
    p = HERE / path
    if not p.exists():
        return None
    if p.suffix == ".json":
        d = json.loads(p.read_text(encoding="utf-8"))
        d = d.get("cells", d)
        v = d.get(key)
        return None if v is None else v["per_seed"]
    z = np.load(str(p))
    return None if key not in z.files else list(100 * np.asarray(z[key], float))


def grid(reward, episodes, boundary):
    cell = "%s_%s" % (boundary, reward)
    if episodes == 3000:
        return load("boundary_open_demo/four_cells_boundary_dest.json", cell)
    v = load("boundary_open_demo/matched_budget_grid.json", cell)
    return v if v is not None else load("boundary_open_demo/matched_%s.json" % cell, cell)


def sumo(reward, boundary, interior=False, episodes=3000):
    cell = "%s_%s" % (boundary, reward)
    stem = ("sumo_interior" if interior else "sumo_bdest") + ("8k" if episodes == 8000 else "")
    return load("sumo_corridor/%s_%s.json" % (stem, cell), cell)


def interior_grid(reward, episodes, boundary):
    path = ("boundary_open_demo/interior_destination_3000_10seed.json" if episodes == 3000
            else "boundary_open_demo/interior_destination_10seed.json")
    return load(path, "%s_%s" % (boundary, reward))


def bench(net, reward, boundary, interior=False):
    f = ("benchmark_network/benchmark_results_dest_interior.npz" if interior else
         ("benchmark_network/benchmark_results_10seed.npz" if net == "sioux_falls"
          else "benchmark_network/bench10_nguyen_dupuis.npz"))
    return load(f, "%s_%s_%s" % (net, boundary, reward))


PENDING = 0


def row(prefix, label, eps, getter, net=None, reward=None):
    global PENDING
    cells = []
    for boundary in ("closed", "open"):
        cells.append(fmt(getter(boundary), cell_key(net or prefix, reward or label, eps, boundary)))
    ns = {n for _, n in cells if n}
    PENDING += sum(1 for c, _ in cells if c == "PENDING")
    print("  %-6s %-30s %5d %4s  %-22s %-22s"
          % (prefix, label, eps, (ns.pop() if len(ns) == 1 else "?"), cells[0][0], cells[1][0]))


def main():
    print("TABLE VI, every learned row at ten seeds and one dispersion convention")
    print("  %-6s %-30s %5s %4s  %-22s %-22s"
          % ("Net", "Reward", "Ep", "n", "Boundary-closed", "Boundary-open"))
    for reward, label in (("time_min", "Travel-time-minimizing"),
                          ("aligned", "Destination-aligned")):
        for eps in (3000, 8000):
            row("grid", label, eps, lambda b, r=reward, e=eps: grid(r, e, b), "grid", reward)
    for reward, label in (("time_min", "Travel-time-minimizing"),
                          ("aligned", "Destination-aligned")):
        for eps in (3000, 8000):
            row("SUMO", label, eps, lambda b, r=reward, e=eps: sumo(r, b, episodes=e),
                "sumo", reward)

    print("\nInterior-destination control, same convention")
    for reward, label in (("time_min", "Travel-time-minimizing"),
                          ("aligned", "Destination-aligned")):
        for eps in (3000, 8000):
            row("grid", label, eps, lambda b, r=reward, e=eps: interior_grid(r, e, b), "grid_interior", reward)
        for eps in (3000, 8000):
            row("SUMO", label, eps,
                lambda b, r=reward, e=eps: sumo(r, b, interior=True, episodes=e),
                "sumo_interior", reward)
        row("sioux", label, 8000, lambda b, r=reward: bench("sioux_falls", r, b, interior=True), "sioux_interior", reward)

    print("\nBenchmark networks at ten seeds, boundary destination")
    for net, eps in (("sioux_falls", 8000), ("nguyen_dupuis", 3000)):
        for reward, label in (("time_min", "Travel-time-minimizing"),
                              ("aligned", "Destination-aligned")):
            row(net[:5], label, eps, lambda b, n=net, r=reward: bench(n, r, b), net, reward)

    print("\n%d cell(s) pending" % PENDING)
    return 0 if PENDING == 0 else 2


if __name__ == "__main__":
    sys.exit(main())

def load_json(rel):
    p = HERE / rel
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def convention(label, boundary):
    """Per-seed values for the two alternative scoring conventions of the boundary-open cell.

    Table VI's other rows score a trip that leaves the network as unarrived. A reviewer asked
    for the same cell under the other two conventions in the same table, so that the reader
    sees the dependence directly rather than inferring it from the supplement. Both cells are
    read from their own run outputs, never restated.
    """
    if label == "residual horizon":
        d = load_json("boundary_open_demo/residual_exit_control.json")
        if d is None:
            return None
        key = "closed time_min residual=True" if boundary == "closed" else "open time_min residual=True"
        cell = d["cells"].get(key)
        return cell["per_seed"] if cell else None
    if label == "non-terminal exit":
        d = load_json("boundary_open_demo/costly_return_10seed_v3_per_seed.json")
        if d is None:
            return None
        key = "closed_time_min@--" if boundary == "closed" else "costly_return_time_min@1.0"
        return d["cells"].get(key)
    if label == "non-terminal exit aligned":
        # The aligned reward under the same convention. Table VIII carried the travel-time row
        # alone, so the only aligned figure beside it was the terminal-exit cell and a reader
        # compared 46.2% against 35.1% across two conventions. Within this one the order is the
        # other way. The closed reference was not in the original job and has its own file.
        if boundary == "closed":
            d = load_json("boundary_open_demo/costly_return_aligned_closed.json")
            return d["cells"].get("closed_aligned@--") if d else None
        d = load_json("boundary_open_demo/costly_return_10seed_v3_per_seed.json")
        return d["cells"].get("costly_return_aligned@1.0") if d else None
    return None
