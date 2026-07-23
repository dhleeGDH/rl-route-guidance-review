"""Probe the variability of the aligned-reward recovery on the SUMO substrate (Section V-E).

The main 5-seed SUMO run recovers the boundary-open aligned condition to a mean of 76.2%
with a large seed spread (sample SD 24.5; seeds at 41.0/69.5/71.5/99.0/100.0%), a bimodal
mix of partial and near-full recovery. The question is whether the spread reflects the
decision or the training budget, and where the partial-recovery seeds stall.

This script isolates the open-aligned cell and re-trains it under three budget conditions,
logging a per-seed learning curve (checkpoint completion during training) so the stall point
of each seed is visible:

  (i)   b50_e3000  : 50-step decision budget, 3000 episodes  (baseline replication)
  (ii)  b100_e3000 : 100-step decision budget, 3000 episodes (larger decision budget)
  (iii) b50_e6000  : 50-step decision budget, 6000 episodes  (larger training budget)

Outputs sumo_budget_curves.csv (condition, seed, episode, completion) and sumo_budget_summary.csv
(condition, seed, final_completion) plus a printed summary.
"""
import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "boundary_open_demo"))
from sumo_env import SumoGridEnv, N
from dqn import DQNAgent


def make_eval_od(n=200, seed=12345, min_sep=3):
    rng = np.random.RandomState(seed)
    ods = []
    while len(ods) < n:
        o = (rng.randint(N), rng.randint(N))
        d = (rng.randint(N), rng.randint(N))
        if abs(o[0] - d[0]) + abs(o[1] - d[1]) >= min_sep:
            ods.append((o, d))
    return ods


def evaluate(agent, boundary, reward, eval_od, tag, max_steps):
    env = SumoGridEnv(boundary=boundary, reward=reward, seed=777,
                      max_steps=max_steps, label="ev" + tag)
    arrived = 0
    for od in eval_od:
        s = env.reset(od=od)
        for _ in range(env.max_steps):
            a = agent.act(s, env.available_actions(), eps=0.0)
            s, _, done, info = env.step(a)
            if done:
                arrived += info["outcome"] == "arrived"
                break
    env.close()
    return arrived / len(eval_od)


def train_condition(seed, episodes, max_steps, eval_full, eval_ckpt, checkpoints, tag):
    """Train open-aligned; return (final_completion, [(ep, comp), ...] curve)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = SumoGridEnv(boundary="open", reward="aligned", seed=1000 + seed,
                      max_steps=max_steps, label="tr" + tag)
    agent = DQNAgent(env.state_dim, env.n_actions, seed=seed)
    eps0, eps1, decay = 1.0, 0.05, int(0.6 * episodes)
    curve = []
    for ep in range(1, episodes + 1):
        eps = max(eps1, eps0 - (eps0 - eps1) * ep / decay)
        s = env.reset()
        for _ in range(env.max_steps):
            m = env.available_actions()
            a = agent.act(s, m, eps)
            s2, r, done, _ = env.step(a)
            agent.buf.add(s, a, r, s2, float(done),
                          env.available_actions().astype(np.float32))
            agent.learn()
            s = s2
            if done:
                break
        if ep in checkpoints:
            c = evaluate(agent, "open", "aligned", eval_ckpt, tag + f"c{ep}", max_steps)
            curve.append((ep, c))
            print(f"    {tag} seed{seed} ep{ep} ckpt_completion={c:.3f}", flush=True)
    env.close()
    final = evaluate(agent, "open", "aligned", eval_full, tag + "f", max_steps)
    return final, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    eval_full = make_eval_od(200)
    eval_ckpt = make_eval_od(60, seed=999)

    conditions = [
        ("b50_e3000", 50, 3000),
        ("b100_e3000", 100, 3000),
        ("b50_e6000", 50, 6000),
    ]
    if args.smoke:
        conditions = [("smoke_b50", 50, 200)]
        eval_full = make_eval_od(30)
        eval_ckpt = make_eval_od(20, seed=999)
        args.seeds = 1

    curve_rows = []
    summ_rows = []
    for name, max_steps, episodes in conditions:
        ckpts = set(np.linspace(episodes // 6, episodes, 6, dtype=int).tolist())
        finals = []
        print(f"== condition {name} (budget={max_steps}, episodes={episodes}) ==",
              flush=True)
        for seed in range(args.seeds):
            tag = f"{name[:4]}{seed}"
            final, curve = train_condition(seed, episodes, max_steps, eval_full,
                                           eval_ckpt, ckpts, tag)
            finals.append(final)
            summ_rows.append((name, seed, final))
            for ep, c in curve:
                curve_rows.append((name, seed, ep, c))
            print(f"  {name} seed{seed} FINAL completion={final:.3f}", flush=True)
        m, sd = float(np.mean(finals)), float(np.std(finals, ddof=1)) if len(finals) > 1 else 0.0
        print(f"== {name}: mean={m:.3f} sampleSD={sd:.3f} seeds={[f'{x:.3f}' for x in finals]}",
              flush=True)

    with open("sumo_budget_summary.csv", "w") as f:
        f.write("condition,seed,final_completion\n")
        for name, seed, final in summ_rows:
            f.write(f"{name},{seed},{final:.4f}\n")
    with open("sumo_budget_curves.csv", "w") as f:
        f.write("condition,seed,episode,completion\n")
        for name, seed, ep, c in curve_rows:
            f.write(f"{name},{seed},{ep},{c:.4f}\n")
    print("wrote sumo_budget_summary.csv and sumo_budget_curves.csv", flush=True)


if __name__ == "__main__":
    main()
