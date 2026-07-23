"""Re-run the boundary-open alignment sweep (main-text Fig. 6) at five seeds to match the
main experiment, removing the seed asymmetry between the sweep endpoint and the main aligned
result. Writes dose_response_5seed.npy.
"""
import numpy as np
import torch
from env import GridRouteEnv
from dqn import DQNAgent
from sweep_extra import make_eval_od, train_eval


def main(seeds=5, episodes=3000):
    print("=== reward dose-response on boundary-open 5x5, 5 seeds ===", flush=True)
    eval_od = make_eval_od(5)
    rows = []
    for lam in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
        env_kw = dict(boundary="open", reward="aligned", n_side=5,
                      max_steps=120, align_strength=lam)
        comps = [train_eval(env_kw, s, episodes, eval_od) for s in range(seeds)]
        m, sd = float(np.mean(comps)), float(np.std(comps, ddof=1))
        rows.append((lam, m, sd))
        print(f"lambda={lam:.2f}  completion={m:.3f} +/- {sd:.3f}  "
              f"seeds={[f'{x:.3f}' for x in comps]}", flush=True)
    np.save("dose_response_5seed.npy", np.array([(r[0], r[1], r[2]) for r in rows]))
    print("wrote dose_response_5seed.npy", flush=True)


if __name__ == "__main__":
    main()
