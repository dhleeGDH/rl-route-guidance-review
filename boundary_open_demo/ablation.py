"""Component ablation of the destination-aligned reward on the boundary-open network.

The aligned objective adds three terms at once: potential-based shaping (beta), an arrival
bonus (R_goal), and an exit penalty (R_exit). This isolates each term (others zeroed) at its
full scale, on the open 5x5 network, to attribute the completion recovery to specific terms.
"""
import numpy as np
from sweep_extra import train_eval, make_eval_od

EVAL = make_eval_od(5)
CONDS = [
    ("shaping only",     dict(beta=1.0, r_goal=0.0,  r_exit=0.0)),
    ("arrival bonus only", dict(beta=0.0, r_goal=10.0, r_exit=0.0)),
    ("exit penalty only", dict(beta=0.0, r_goal=0.0,  r_exit=5.0)),
    ("full aligned",     dict(beta=1.0, r_goal=10.0, r_exit=5.0)),
]

print("=== aligned-reward component ablation, boundary-open 5x5, 5 seeds ===")
for name, params in CONDS:
    env_kw = dict(boundary="open", reward="aligned", n_side=5, max_steps=120, **params)
    comps = [train_eval(env_kw, s, 3000, EVAL) for s in range(5)]
    print(f"{name:20} completion = {100*np.mean(comps):.1f}% +/- {100*np.std(comps):.1f} "
          f"(seeds {[round(100*c) for c in comps]})", flush=True)
