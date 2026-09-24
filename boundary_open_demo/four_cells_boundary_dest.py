# -*- coding: utf-8 -*-
"""The four cells with the destination as an outgoing boundary link.

Same learner, same budget, same evaluation protocol as Section V-A. The only change is the
one env_boundary_dest.py describes: the destination is a link that leaves the lattice, the
closed condition opens no other perimeter link, and the open condition opens the rest.

Run to compare against the recorded values 85.0 / 0.0 closed and open travel-time, and
98.8 / 96.8 closed and open destination-aligned.
"""
import argparse
import io
import json
import os
import sys

import numpy as np
import torch


# Floating-point addition is not associative, so the thread count of the linear-algebra
# library changes the order of summation in the gradient and moves the training
# trajectory: the same seed returns 42.5% on the boundary-open aligned cell at one
# thread and 33.5% at eight. The count is pinned so a seed determines a run.
torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from env_boundary_dest import BoundaryDestEnv, make_eval_od   # noqa: E402
from dqn import DQNAgent                                       # noqa: E402


def evaluate(agent, boundary, reward, eval_od, max_steps, seed=777,
             state_sentinel="high"):
    env = BoundaryDestEnv(boundary=boundary, reward=reward, seed=seed, max_steps=max_steps,
                          state_sentinel=state_sentinel)
    arrived = 0
    for od in eval_od:
        env.reset(od=od)
        for _ in range(env.max_steps):
            s = env._obs()
            a = agent.act(s, env.available_actions(), eps=0.0)
            _, _, done, info = env.step(a)
            if done:
                arrived += info["outcome"] == "arrived"
                break
    return 100.0 * arrived / len(eval_od)


def run_cell(boundary, reward, seed, episodes, eval_od, max_steps=120,
             state_sentinel="high", curve_every=None, curve=None, eval_boundary=None,
             agent_kw=None, eps_floor=0.05, eps_decay_frac=0.6,
             **reward_kw):
    """reward_kw passes beta, r_goal and r_exit through to the environment. With none supplied
    the cell is the published one, so an ablation calling this function with one term zeroed
    differs from the headline run in that term alone."""
    """state_sentinel defaults to the published behavior. sentinel_control.py passes
    "matched" to run the same cells through this same function, so the control and the
    published cells cannot diverge in anything other than the encoding under test."""
    """curve_every records a checkpoint every N episodes into the list passed as curve, as
    (episode, completion, mean episode return over the window). The training-curve figures were
    drawn from a separate run file whose final points read 30.8% and 99.5% against the 35.1%
    and 99.9% this function returns, which put two measurements of one cell in one document.
    The evaluation draws from the global generators, so their states are saved and restored
    around it and the trajectory is the one produced with curve_every left unset.
    curves_match_finals.py is the control: it asserts the last recorded checkpoint equals the
    returned value seed for seed, and fails where the two runs drift apart."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = BoundaryDestEnv(boundary=boundary, reward=reward, seed=1000 + seed,
                          max_steps=max_steps, state_sentinel=state_sentinel, **reward_kw)
    # agent_kw, eps_floor and eps_decay_frac exist for dqn_sensitivity.py, which moves one
    # learner knob at a time. Left at their defaults every call written before they existed
    # returns exactly what it returned, so the sweep and the headline cells cannot diverge in
    # anything but the knob under test.
    agent = DQNAgent(env.state_dim, env.n_actions, seed=seed, **(agent_kw or {}))
    eps0, eps1 = 1.0, eps_floor
    decay = int(eps_decay_frac * episodes)
    window = []
    for ep in range(1, episodes + 1):
        eps = max(eps1, eps0 - (eps0 - eps1) * ep / decay)
        env.reset()
        ep_ret = 0.0
        for _ in range(env.max_steps):
            s = env._obs()
            m = env.available_actions()
            a = agent.act(s, m, eps)
            s2, r, done, _ = env.step(a)
            ep_ret += r
            agent.buf.add(s, a, r, s2, float(done),
                          env.available_actions().astype(np.float32))
            agent.learn()
            if done:
                break
        window.append(ep_ret)
        if curve_every and ep % curve_every == 0:
            np_state, torch_state = np.random.get_state(), torch.get_rng_state()
            comp = evaluate(agent, boundary, reward, eval_od, max_steps,
                            state_sentinel=state_sentinel)
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)
            curve.append((ep, comp, float(np.mean(window))))
            window = []
    # eval_boundary lets a cell be scored on a boundary other than the one it trained on,
    # which is what zero_shot_transfer.py needs. Left unset it is the training boundary, so
    # every call written before this parameter existed returns exactly what it returned.
    return evaluate(agent, eval_boundary or boundary, reward, eval_od, max_steps,
                    state_sentinel=state_sentinel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="four_cells_boundary_dest.json")
    a = ap.parse_args()

    eval_od = make_eval_od(n=200, seed=12345)
    recorded = {("closed", "time_min"): 85.0, ("open", "time_min"): 0.0,
                ("closed", "aligned"): 98.8, ("open", "aligned"): 96.8}
    out = {}
    for boundary in ("closed", "open"):
        for reward in ("time_min", "aligned"):
            comps = [run_cell(boundary, reward, s, a.episodes, eval_od)
                     for s in range(a.seeds)]
            key = "%s_%s" % (boundary, reward)
            out[key] = {"mean": float(np.mean(comps)), "sd": float(np.std(comps)),
                        "per_seed": comps, "recorded": recorded[(boundary, reward)]}
            print("== %-6s %-9s %5.1f%% (sd %4.1f)   recorded %5.1f%%"
                  % (boundary, reward, np.mean(comps), np.std(comps),
                     recorded[(boundary, reward)]), flush=True)
    with open(os.path.join(HERE, a.out), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
