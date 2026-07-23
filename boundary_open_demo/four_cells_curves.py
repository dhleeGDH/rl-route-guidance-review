# -*- coding: utf-8 -*-
"""Training curves for the four bespoke-grid cells, from the same environment, evaluation
set, and learner as four_cells_boundary_dest.py, so the supplement curve figure (Fig. S-1)
and Table IV report one run rather than two. Writes results_10seed.npz with per-checkpoint
completion and return, keyed as <cell>__comp/__ret/__steps to match plot_reward_condition and
plot_10seed.
"""
import numpy as np
import torch

from env_boundary_dest import BoundaryDestEnv, make_eval_od
from dqn import DQNAgent

CONDITIONS = [("closed", "time_min"), ("open", "time_min"),
              ("open", "aligned"), ("closed", "aligned")]


def evaluate(agent, boundary, reward, eval_od, max_steps, seed=777):
    env = BoundaryDestEnv(boundary=boundary, reward=reward, seed=seed, max_steps=max_steps)
    arrived, ret = 0, 0.0
    for od in eval_od:
        env.reset(od=od)
        r_ep = 0.0
        for _ in range(env.max_steps):
            s = env._obs()
            with torch.no_grad():
                a = agent.act(s, env.available_actions(), eps=0.0)
            _, r, done, info = env.step(a)
            r_ep += r
            if done:
                arrived += info["outcome"] == "arrived"
                break
        ret += r_ep
    return 100.0 * arrived / len(eval_od), ret / len(eval_od)


def run_cell(boundary, reward, seed, episodes, eval_od, eval_every, max_steps=120):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = BoundaryDestEnv(boundary=boundary, reward=reward, seed=1000 + seed,
                          max_steps=max_steps)
    agent = DQNAgent(env.state_dim, env.n_actions, seed=seed)
    eps0, eps1 = 1.0, 0.05
    decay = int(0.6 * episodes)
    steps, comps, rets = [], [], []
    for ep in range(1, episodes + 1):
        eps = max(eps1, eps0 - (eps0 - eps1) * ep / decay)
        env.reset()
        for _ in range(env.max_steps):
            s = env._obs()
            m = env.available_actions()
            a = agent.act(s, m, eps)
            s2, r, done, _ = env.step(a)
            agent.buf.add(s, a, r, s2, float(done),
                          env.available_actions().astype(np.float32))
            agent.learn()
            if done:
                break
        if ep % eval_every == 0 or ep == 1:
            # Checkpoint evaluation must not perturb the training trajectory. Snapshot the
            # global RNGs and restore them after, so the final policy matches the
            # final-only run of four_cells_boundary_dest.py rather than diverging from it.
            np_state = np.random.get_state()
            torch_state = torch.random.get_rng_state()
            comp, ret = evaluate(agent, boundary, reward, eval_od, max_steps)
            np.random.set_state(np_state)
            torch.random.set_rng_state(torch_state)
            steps.append(ep); comps.append(comp); rets.append(ret)
    return np.array(steps), np.array(comps), np.array(rets)


def main(seeds=10, episodes=3000, eval_every=150):
    eval_od = make_eval_od(n=200, seed=12345)
    out = {}
    for boundary, reward in CONDITIONS:
        key = f"{boundary}_{reward}"
        C, R, steps = [], [], None
        for seed in range(seeds):
            st, comp, ret = run_cell(boundary, reward, seed, episodes, eval_od, eval_every)
            steps = st; C.append(comp); R.append(ret)
            print(f"{key:18} seed{seed} final completion={comp[-1]:.1f}", flush=True)
        out[f"{key}__steps"] = steps
        out[f"{key}__comp"] = np.array(C) / 100.0   # plot_* scale by 100
        out[f"{key}__ret"] = np.array(R)
        print(f"== {key:18} final {np.mean([c[-1] for c in C]):.1f} "
              f"(sd {np.std([c[-1] for c in C]):.1f})", flush=True)
    np.savez("results_10seed.npz", **out)
    print("wrote results_10seed.npz", flush=True)


if __name__ == "__main__":
    main()
