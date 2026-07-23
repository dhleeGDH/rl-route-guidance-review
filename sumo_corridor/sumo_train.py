"""Train the same DQN protocol as Section V on the SUMO 5x5 grid (Section V-B).

Ports the bespoke-grid demonstration onto SUMO to show the boundary-open failure on the
field's own simulator: under the corpus-typical travel-time reward, an open boundary (where
SUMO removes a vehicle that leaves the network) collapses OD completion, and a
destination-aligned reward restores it -- the same result as the bespoke grid.
"""
import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "boundary_open_demo"))
from sumo_env import SumoGridEnv, N
from dqn import DQNAgent

CONDITIONS = [("closed", "time_min"), ("open", "time_min"),
              ("open", "aligned"), ("closed", "aligned")]


def make_eval_od(n=200, seed=12345, min_sep=3):
    """Origins anywhere, destinations on the boundary, as training draws them.

    Destinations were drawn from every cell here, which does not match SumoGridEnv.reset:
    an interior cell has no outgoing boundary link, so _exit_action_at returns None and the
    arrival action never becomes available. 55 of the 200 pairs were unreachable by
    construction, and the completion ceiling was the 72.5% of pairs that were reachable.
    """
    from sumo_env import _boundary_cells
    rng = np.random.RandomState(seed)
    bcells = _boundary_cells()
    ods = []
    while len(ods) < n:
        o = (rng.randint(N), rng.randint(N))
        d = bcells[rng.randint(len(bcells))]
        if abs(o[0] - d[0]) + abs(o[1] - d[1]) >= min_sep:
            ods.append((o, d))
    return ods


def evaluate(agent, boundary, reward, eval_od, tag):
    env = SumoGridEnv(boundary=boundary, reward=reward, seed=777, label="ev" + tag)
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


def train_condition(boundary, reward, seed, episodes, eval_od, tag):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = SumoGridEnv(boundary=boundary, reward=reward, seed=1000 + seed, label="tr" + tag)
    agent = DQNAgent(env.state_dim, env.n_actions, seed=seed)
    eps0, eps1, decay = 1.0, 0.05, int(0.6 * episodes)
    for ep in range(1, episodes + 1):
        eps = max(eps1, eps0 - (eps0 - eps1) * ep / decay)
        s = env.reset()
        for _ in range(env.max_steps):
            m = env.available_actions()
            a = agent.act(s, m, eps)
            s2, r, done, _ = env.step(a)
            agent.buf.add(s, a, r, s2, float(done), env.available_actions().astype(np.float32))
            agent.learn()
            s = s2
            if done:
                break
    env.close()
    return evaluate(agent, boundary, reward, eval_od, tag + "e")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.episodes, args.seeds = 250, 1
    eval_od = make_eval_od()
    rows = []
    for (boundary, reward) in CONDITIONS:
        comps = []
        for seed in range(args.seeds):
            tag = f"{boundary[0]}{reward[0]}{seed}"
            c = train_condition(boundary, reward, seed, args.episodes, eval_od, tag)
            comps.append(c)
            print(f"  {boundary:6} {reward:9} seed{seed} completion={c:.3f}", flush=True)
        m, sd = float(np.mean(comps)), float(np.std(comps))
        rows.append((f"{boundary}_{reward}", m, sd))
        print(f"== {boundary:6} {reward:9} completion={m:.3f} +/- {sd:.3f}", flush=True)
    with open("sumo_summary.csv", "w") as f:
        f.write("condition,final_completion_mean,final_completion_std\n")
        for cond, m, sd in rows:
            f.write(f"{cond},{m:.4f},{sd:.4f}\n")
    print("wrote sumo_summary.csv")


if __name__ == "__main__":
    main()
