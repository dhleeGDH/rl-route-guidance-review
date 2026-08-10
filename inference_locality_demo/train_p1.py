"""Train local-state and predictive-state DQN agents on the moving-congestion grid
and compare realized OD travel time and its variability.

Both agents run on the same boundary-closed network with the same destination-aligned
reward, so both complete essentially all trips; the quantity under study is travel-time
quality, which is where a non-predictive (locally observed) dynamic state is expected to
fall short of a predictive one. This is the inference-locality (P1) counterpart to the
boundary-open (P2) demonstration.
"""

import argparse
import numpy as np
import torch

from env_p1 import GridP1Env
from dqn import DQNAgent

ENV_KW = dict(amp=8.0, width=1.0, wave_period=16.0, static_amp=0.3,
              horizon=3, max_steps=60)


def make_eval_od(n=200, n_side=5, seed=12345):
    rng = np.random.RandomState(seed)
    ods = []
    while len(ods) < n:
        o = (rng.randint(n_side), rng.randint(n_side))
        d = (rng.randint(n_side), rng.randint(n_side))
        if abs(o[0] - d[0]) + abs(o[1] - d[1]) >= 3:
            ods.append((o, d))
    return ods


def evaluate(agent, state_mode, eval_od, seed=777):
    env = GridP1Env(state_mode=state_mode, seed=seed, **ENV_KW)
    tts, arrived = [], 0
    for od in eval_od:
        env.reset(od=od)
        for _ in range(env.max_steps):
            s = env._obs()
            a = agent.act(s, env.available_actions(), eps=0.0)
            _, _, done, info = env.step(a)
            if done:
                if info["outcome"] == "arrived":
                    arrived += 1
                    tts.append(info["travel_time"])
                break
    tts = np.array(tts)
    return arrived / len(eval_od), float(tts.mean()), float(tts.std())


def train_condition(state_mode, seed, n_episodes, eval_every, eval_od):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = GridP1Env(state_mode=state_mode, seed=1000 + seed, **ENV_KW)
    agent = DQNAgent(env.state_dim, env.n_actions, seed=seed)
    eps0, eps1 = 1.0, 0.05
    decay = int(0.6 * n_episodes)
    steps, comp_c, ttm_c, tts_c = [], [], [], []
    for ep in range(1, n_episodes + 1):
        eps = max(eps1, eps0 - (eps0 - eps1) * ep / decay)
        env.reset()
        for _ in range(env.max_steps):
            s = env._obs()
            mask = env.available_actions()
            a = agent.act(s, mask, eps)
            s2, r, done, _ = env.step(a)
            agent.buf.add(s, a, r, s2, float(done),
                          env.available_actions().astype(np.float32))
            agent.learn()
            if done:
                break
        if ep % eval_every == 0 or ep == 1:
            comp, ttm, tts = evaluate(agent, state_mode, eval_od)
            steps.append(ep); comp_c.append(comp); ttm_c.append(ttm); tts_c.append(tts)
    return np.array(steps), np.array(comp_c), np.array(ttm_c), np.array(tts_c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--eval_every", type=int, default=150)
    ap.add_argument("--out", default="results_p1.npz")
    args = ap.parse_args()
    if args.smoke:
        args.episodes, args.seeds, args.eval_every = 600, 1, 100

    eval_od = make_eval_od()
    results, rows = {}, []
    for mode in ("local", "predictive"):
        comps, ttms, ttss, steps = [], [], [], None
        for seed in range(args.seeds):
            st, comp, ttm, tts = train_condition(mode, seed, args.episodes,
                                                 args.eval_every, eval_od)
            steps = st; comps.append(comp); ttms.append(ttm); ttss.append(tts)
            print(f"{mode:11} seed{seed} completion={comp[-1]:.3f} "
                  f"travel_time={ttm[-1]:.2f} (+/-{tts[-1]:.2f})", flush=True)
        comps, ttms, ttss = np.array(comps), np.array(ttms), np.array(ttss)
        results[f"{mode}__steps"] = steps
        results[f"{mode}__comp"] = comps
        results[f"{mode}__ttm"] = ttms
        results[f"{mode}__tts"] = ttss
        rows.append((mode, comps[:, -1].mean(), ttms[:, -1].mean(), ttms[:, -1].std(),
                     ttss[:, -1].mean()))
    np.savez(args.out, **results)
    print("\n=== SUMMARY (final) ===")
    for m, comp, ttm_mean, ttm_std, tts_mean in rows:
        print(f"{m:11} completion={comp:.3f}  mean_travel_time={ttm_mean:.2f} "
              f"(+/-{ttm_std:.2f} across seeds)  within-run_std={tts_mean:.2f}")
    if len(rows) == 2:
        gain = 100 * (rows[0][2] - rows[1][2]) / rows[0][2]
        print(f"\npredictive reduces mean travel time by {gain:.1f}% vs local")


if __name__ == "__main__":
    main()
