#!/usr/bin/env python3
"""Gate: every placed figure is written by its generator into handoff/manuscript/figures/, and the
PNG on disk is the PNG its generator produces.

WHY THIS EXISTS. T-1984 found plot_assembly.py writing Fig. 1 into archive/experiments_dup/,
because Path(__file__).resolve() follows the handoff/experiments symlink and parents[2] then lands
beside the archive rather than beside the manuscript. T-1995's audit found the same defect in four
more generators: plot_family_reporting.py resolved to a directory outside the repository, and the
three experiment generators resolved to directories that do not exist in this tree. The visible
symptom was fig_family_reporting.png still drawing the corpus of 94 six weeks after T-1993 moved
its ROWS to 93: the generator changed and the PNG could not follow.

Two checks, each per figure, from one rendering of the generator:

  (a) PATH. The generator is executed with Figure.savefig redirected, so the path it would write
      is captured and the image lands in a temporary file instead. The captured path must lie
      inside handoff/manuscript/figures and carry the file name the builder places.
  (b) FRESHNESS. The temporary image is compared pixel by pixel with the PNG on disk. They must
      agree in size and differ in at most FRESH_TOL of their pixels by more than 16/255, which
      absorbs rasteriser noise and catches a stale figure outright: the stale
      fig_family_reporting.png differed from its generator's output in 55,494 pixels.

      A first version compared git commit times and mtimes instead. Both failed the same way: a
      path-only fix to a generator changed its commit time and left an unchanged PNG looking stale
      forever, and a fresh clone sets every mtime to checkout time. What "no older than its
      generator" means for a figure is that regenerating it changes nothing, which is what is
      tested here.

The generators are the files that actually produced the placed PNGs, established at T-1996 by
regenerating each and finding it pixel-identical. The repo_v11 copies of the three experiment
generators are older revisions with other figure sizes and are not the sources.

    python3 check_figure_generators.py
"""
import io
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MS = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(MS))
FIG = os.path.realpath(os.path.join(MS, "figures"))
EXP = os.path.join(ROOT, "archive", "experiments_dup", "handoff_experiments")
FRESH_TOL = 0.0001          # share of pixels allowed to differ by more than 16/255

# placed PNG -> the generator that draws it
GENERATORS = {
    "fig_flow.png":              os.path.join(EXP, "screening_trail", "plot_assembly.py"),
    "fig_family_grid.png":       os.path.join(ROOT, "scripts", "plot_family_merged.py"),
    "fig_experiment_design.png": os.path.join(EXP, "boundary_open_demo", "plot_experiment_design.py"),
    "fig_networks.png":          os.path.join(EXP, "benchmark_network", "plot_networks.py"),
    "fig_reward_condition.png":  os.path.join(EXP, "boundary_open_demo", "plot_reward_condition.py"),
    # not placed, but kept in the figures directory and read as a record by
    # check_appendix_consistency, so it is held to the same rule
    "fig_family_reporting.png":  os.path.join(ROOT, "scripts", "plot_family_reporting.py"),
}


# Each generator runs in its own interpreter. A first version ran them in one process through
# runpy, and the rcParams one generator sets (serif font, 7 pt) bled into the next: the reward
# figure then differed from its committed PNG in 0.30% of pixels while reproducing it exactly when
# run alone. The bootstrap below redirects Figure.savefig inside the child and reports the path the
# generator meant to write beside the temporary file that received the image.
_BOOTSTRAP = r"""
import io, os, runpy, sys
gen, tmpdir, report = sys.argv[1], sys.argv[2], sys.argv[3]
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
orig = Figure.savefig
seen = []
def redirected(self, fname, *a, **k):
    target = os.path.realpath(str(fname))
    tmp = os.path.join(tmpdir, "%d_%s" % (len(seen), os.path.basename(target)))
    seen.append((target, tmp))
    return orig(self, tmp, *a, **k)
Figure.savefig = redirected
sys.stdout = sys.stderr = io.StringIO()
try:
    runpy.run_path(gen, run_name="__main__")
finally:
    with open(report, "w") as f:
        for target, tmp in seen:
            f.write(target + "\t" + tmp + "\n")
"""


def render(gen, tmpdir):
    """Run a generator in a fresh interpreter; return (path it meant to write, image it wrote)."""
    import subprocess
    report = os.path.join(tmpdir, os.path.basename(gen) + ".report")
    subprocess.run([sys.executable, "-c", _BOOTSTRAP, gen, tmpdir, report],
                   cwd=os.path.dirname(gen), capture_output=True, timeout=300)
    if not os.path.exists(report):
        return None, None
    lines = [l.rstrip("\n").split("\t") for l in io.open(report, encoding="utf-8") if "\t" in l]
    return tuple(lines[0]) if lines else (None, None)


def pixel_gap(a_path, b_path):
    """(size_equal, share of pixels differing by more than 16/255)."""
    from PIL import Image, ImageChops
    a = Image.open(a_path).convert("RGB")
    b = Image.open(b_path).convert("RGB")
    if a.size != b.size:
        return False, 1.0, a.size, b.size
    diff = ImageChops.difference(a, b).convert("L")
    bad = sum(1 for v in diff.getdata() if v > 16)
    return True, bad / float(a.size[0] * a.size[1]), a.size, b.size


def main():
    fails = []
    tmpdir = tempfile.mkdtemp(prefix="figgate_")
    rendered = {}
    print("--- (a) where each generator writes ---")
    for png, gen in GENERATORS.items():
        if not os.path.exists(gen):
            print("  FAIL %-26s generator missing: %s" % (png, os.path.relpath(gen, ROOT)))
            fails.append(png)
            continue
        target, tmp = render(gen, tmpdir)
        rendered[png] = tmp
        ok = target is not None and os.path.dirname(target) == FIG and os.path.basename(target) == png
        print("  %-4s %-26s -> %s" % ("OK" if ok else "FAIL", png,
                                     os.path.relpath(target, ROOT) if target else "no savefig reached"))
        if not ok:
            fails.append(png)

    print("--- (b) the PNG on disk is what its generator produces ---")
    for png in GENERATORS:
        p = os.path.join(FIG, png)
        tmp = rendered.get(png)
        if not os.path.exists(p) or not tmp:
            print("  FAIL %-26s missing" % png)
            fails.append(png)
            continue
        same_size, share, sa, sb = pixel_gap(p, tmp)
        ok = same_size and share <= FRESH_TOL
        note = ("size %sx%s on disk against %sx%s regenerated" % (sa + sb)) if not same_size \
            else "%.4f%% of pixels differ" % (100 * share)
        print("  %-4s %-26s %s" % ("OK" if ok else "FAIL", png, note))
        if not ok:
            fails.append(png)

    # negative control: a path resolving into the archive must be rejected by rule (a)
    print("--- negative control ---")
    probe = os.path.realpath(os.path.join(ROOT, "archive", "experiments_dup", "manuscript", "figures", "x.png"))
    caught = os.path.dirname(probe) != FIG
    print("  %-4s a path resolving into archive/experiments_dup/ is rejected" % ("OK" if caught else "FAIL"))
    if not caught:
        fails.append("negative control")

    print()
    if fails:
        print("FIGURE GENERATOR CHECK FAILED: %s" % ", ".join(sorted(set(fails))))
        sys.exit(1)
    print("FIGURE GENERATOR CHECK PASS: %d figures written into figures/ and identical to their generators' output"
          % len(GENERATORS))


if __name__ == "__main__":
    main()
