"""Train the DQN across the four (boundary, reward) conditions and record how the
OD trip completion rate evolves during training.

Conditions
  (closed, time_min) : the typical setup evaluated on a closed network (converges)
  (open,   time_min) : the typical setup on an open network (early-exit failure)
  (open,   aligned)  : reward reformulated to remove the early-exit incentive
                       (causal control isolating reward design as the cause)
  (closed, aligned)  : reformulated reward on a closed network (completeness)

Evaluation uses one fixed held-out set of OD pairs, shared across every condition
and seed, scored greedily. Results are written to results.npz and summary.csv.
"""

import argparse
import numpy as np
import torch

from env import GridRouteEnv
from dqn import DQNAgent

CONDITIONS = [
    ("closed", "time_min"),
    ("open", "time_min"),
    ("open", "aligned"),
    ("closed", "aligned"),
]


def make_eval_od(n=200, n_side=5, seed=12345, min_sep=3):
    """Origin cells anywhere, destinations as outgoing boundary links."""
    from env import perimeter_links
    rng = np.random.RandomState(seed)
    links = perimeter_links(n_side)
    ods = []
    while len(ods) < n:
        o = (rng.randint(n_side), rng.randint(n_side))
        dl = links[rng.randint(len(links))]
        if abs(o[0] - dl[0][0]) + abs(o[1] - dl[0][1]) >= min_sep:
            ods.append((o, dl))
    return ods


def evaluate(agent, boundary, reward, eval_od, seed=777, max_steps=50):
    env = GridRouteEnv(boundary=boundary, reward=reward, seed=seed, max_steps=max_steps)
    arrived, returns = 0, []
    for od in eval_od:
        env.reset(od=od)
        total = 0.0
        for _ in range(env.max_steps):
            s = env._obs()
            a = agent.act(s, env.available_actions(), eps=0.0)
            _, r, done, info = env.step(a)
            total += r
            if done:
                arrived += info["outcome"] == "arrived"
                break
        returns.append(total)
    return arrived / len(eval_od), float(np.mean(returns))


def train_condition(boundary, reward, seed, n_episodes, eval_every, eval_od, max_steps=50):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = GridRouteEnv(boundary=boundary, reward=reward, seed=1000 + seed, max_steps=max_steps)
    agent = DQNAgent(env.state_dim, env.n_actions, seed=seed)

    eps0, eps1 = 1.0, 0.05
    decay_eps = int(0.6 * n_episodes)
    steps_eval, comp_curve, ret_curve = [], [], []

    for ep in range(1, n_episodes + 1):
        eps = max(eps1, eps0 - (eps0 - eps1) * ep / decay_eps)
        env.reset()
        for _ in range(env.max_steps):
            s = env._obs()
            mask = env.available_actions()
            a = agent.act(s, mask, eps)
            s2, r, done, _ = env.step(a)
            mask2 = env.available_actions()
            agent.buf.add(s, a, r, s2, float(done), mask2.astype(np.float32))
            agent.learn()
            if done:
                break
        if ep % eval_every == 0 or ep == 1:
            comp, ret = evaluate(agent, boundary, reward, eval_od, max_steps=max_steps)
            steps_eval.append(ep)
            comp_curve.append(comp)
            ret_curve.append(ret)
    return np.array(steps_eval), np.array(comp_curve), np.array(ret_curve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="quick single-seed test")
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--eval_every", type=int, default=150)
    ap.add_argument("--max_steps", type=int, default=120)
    ap.add_argument("--out", default="results.npz")
    args = ap.parse_args()

    if args.smoke:
        args.episodes, args.seeds, args.eval_every = 400, 1, 50

    eval_od = make_eval_od()
    results = {}
    summary_rows = []
    for (boundary, reward) in CONDITIONS:
        key = f"{boundary}_{reward}"
        comps, rets, steps = [], [], None
        for seed in range(args.seeds):
            st, comp, ret = train_condition(boundary, reward, seed,
                                            args.episodes, args.eval_every, eval_od,
                                            max_steps=args.max_steps)
            steps = st
            comps.append(comp)
            rets.append(ret)
            print(f"{key:18} seed{seed}  final completion={comp[-1]:.3f}  "
                  f"return={ret[-1]:.2f}", flush=True)
        comps = np.array(comps); rets = np.array(rets)
        results[f"{key}__steps"] = steps
        results[f"{key}__comp"] = comps
        results[f"{key}__ret"] = rets
        summary_rows.append((key, comps[:, -1].mean(), comps[:, -1].std(),
                             rets[:, -1].mean(), rets[:, -1].std()))

    np.savez(args.out, **results)
    with open("summary.csv", "w") as f:
        f.write("condition,final_completion_mean,final_completion_std,"
                "final_return_mean,final_return_std\n")
        for row in summary_rows:
            f.write(f"{row[0]},{row[1]:.4f},{row[2]:.4f},{row[3]:.4f},{row[4]:.4f}\n")
    print("\n=== SUMMARY (final trip completion rate) ===")
    for row in summary_rows:
        print(f"{row[0]:18} completion={row[1]:.3f} +/- {row[2]:.3f}   "
              f"return={row[3]:.2f} +/- {row[4]:.2f}")


if __name__ == "__main__":
    main()
