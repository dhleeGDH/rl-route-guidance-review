"""Separate the shaping term's two roles on the boundary-open network.

Section V shows that shaping-alone recovers completion to 28.9% because the exit terminal's
potential does not vanish, so the shaping is not policy-invariant and its accumulated return
already tilts the optimum toward arrival (an objective-alignment effect), on top of its
learning-aid effect. Making the potential vanish at the exit terminal (vanish_potential=True)
restores policy invariance: the shaping can then only aid learning and cannot move the
optimum, which remains to exit. Comparing shaping-alone under the two potentials isolates the
two roles.

  non-vanishing shaping-alone : reproduces the 28.9% of the main ablation (Phi(exit) > 0)
  vanishing shaping-alone     : Phi(exit) = 0 -> policy-invariant -> expected near 0%

Writes shaping_decomp_summary.csv.
"""
import numpy as np
from sweep_extra import make_eval_od, train_eval

EVAL = make_eval_od(5)
CONDS = [
    ("shaping-only, non-vanishing Phi(exit)", dict(vanish_potential=False)),
    ("shaping-only, vanishing Phi(exit)",     dict(vanish_potential=True)),
]

if __name__ == "__main__":
    print("=== shaping decomposition, boundary-open 5x5, 5 seeds ===", flush=True)
    rows = []
    for label, extra in CONDS:
        env_kw = dict(boundary="open", reward="aligned", n_side=5, max_steps=120,
                      beta=1.0, r_goal=0.0, r_exit=0.0, **extra)
        comps = [train_eval(env_kw, s, 3000, EVAL) for s in range(5)]
        m, sd = float(np.mean(comps)), float(np.std(comps, ddof=1))
        rows.append((label, m, sd, comps))
        print(f"{label}: {100*m:.1f}% +/- {100*sd:.1f}  seeds={[round(100*c,1) for c in comps]}",
              flush=True)
    with open("shaping_decomp_summary.csv", "w") as f:
        f.write("condition,completion_mean,completion_sd_sample\n")
        for label, m, sd, _ in rows:
            f.write(f"{label},{m:.4f},{sd:.4f}\n")
    print("wrote shaping_decomp_summary.csv", flush=True)
