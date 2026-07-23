# -*- coding: utf-8 -*-
"""Does arrival encoded as a constraint recover completion where a weighted term does not?

Section VI-C prescribes encoding arrival as a constraint or a termination guarantee rather
than as a weighted reward term, on the ground that a weighted term is fragile to network
scale and to the learner. The manuscript demonstrates the weighted term and reports that
fragility. It has never demonstrated the prescription itself.

This runs the first instrument that section names, a constrained formulation with a non-exit
constraint, solved in its Lagrangian form:

    maximize  E[sum_t r_t]   subject to  E[sum_t d_t] <= 0
    r~_t = r_t - lam * d_t,  d_t = 1 on an exit transition and 0 otherwise
    lam  <- clip(lam + eta * e_hat, 0, LAM_MAX)

The environment is untouched. GridRouteEnv(boundary="open", reward="time_min") is the same
constructor that yields 0.0% completion, exits stay admissible and terminal, and everything
added lives in the agent. Masking the exit action was considered and rejected: the mask
available_actions() returns on a closed network is exactly the mask a hard exit-mask produces
on an open one, so that cell would restate the boundary-closed result rather than test the
prescription.

lam starts at 0, so the run begins from the unmodified travel-time objective and any recovery
is attributable to the dual ascent. LAM_MAX is a rail rather than a weight. On the 5x5 lattice
the largest separation is 8 and edge costs lie in [1, 1.6], so no in-network route costs more
than 12.8 and any lam above 11.8 makes exiting dominated at every evaluated pair. A converged
lam that settles strictly below the rail is the pre-registered evidence that the multiplier
found the threshold rather than being handed one. A run that saturates is reported as capped.

The exit rate is estimated from the greedy policy on a held-out probe set. The behaviour-policy
estimate is the textbook choice and fails here for a reason specific to this code: train.py
floors epsilon at 0.05, so a converged greedy policy that never exits still exits at random
under the behaviour policy, the estimate never reaches zero, and lam climbs to the rail on
every run. A greedy probe gives both learners the same estimation mechanism, which is required
when learner-independence is the property under test.
"""
import argparse
import io
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from env import GridRouteEnv, N_SIDE      # noqa: E402
from dqn import DQNAgent                  # noqa: E402
from ppo import PPOAgent                  # noqa: E402
from train import make_eval_od            # noqa: E402

LAM0 = 0.0
ETA = 0.05           # per episode
LAM_MAX = 50.0       # safety rail, not a tuned weight
PROBE_N = 50
PROBE_SEED = 24680
PROBE_EVERY = 50     # episodes between greedy re-estimates of the exit rate
SUFFICIENT_5X5 = 11.8   # lam above which exit is dominated at every evaluated pair


def make_probe_od(eval_od, n=PROBE_N, n_side=N_SIDE, seed=PROBE_SEED, min_sep=3):
    """Probe pairs disjoint from the evaluation set. Destinations are boundary links."""
    from env import perimeter_links
    rng = np.random.RandomState(seed)
    links = perimeter_links(n_side)
    taken = {(tuple(o), dl) for o, dl in eval_od}
    out = []
    while len(out) < n:
        o = (rng.randint(n_side), rng.randint(n_side))
        dl = links[rng.randint(len(links))]
        if abs(o[0] - dl[0][0]) + abs(o[1] - dl[0][1]) < min_sep:
            continue
        if (o, dl) in taken:
            continue
        taken.add((o, dl))
        out.append((o, dl))
    return out


def _greedy_action(agent, env, kind):
    s = env._obs()
    m = env.available_actions()
    if kind == "dqn":
        return agent.act(s, m, eps=0.0)
    a, _, _ = agent.act(s, m, greedy=True)
    return a


def rollout(agent, env, od, kind):
    """Run one greedy episode and report its outcome."""
    env.reset(od=od)
    for _ in range(env.max_steps):
        a = _greedy_action(agent, env, kind)
        _, _, done, info = env.step(a)
        if done:
            return info["outcome"]
    return "timeout"


def exit_rate(agent, boundary, probe_od, kind, max_steps, seed=555):
    env = GridRouteEnv(boundary=boundary, reward="time_min", seed=seed, max_steps=max_steps)
    exited = sum(rollout(agent, env, od, kind) == "exited" for od in probe_od)
    return exited / len(probe_od)


def evaluate(agent, boundary, eval_od, kind, max_steps, seed=777):
    env = GridRouteEnv(boundary=boundary, reward="time_min", seed=seed, max_steps=max_steps)
    arrived = sum(rollout(agent, env, od, kind) == "arrived" for od in eval_od)
    return 100.0 * arrived / len(eval_od)


def run_cell(boundary, learner, seed, episodes, eval_od, probe_od, max_steps=120,
             lam_fixed=None, shaped=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = GridRouteEnv(boundary=boundary, reward="time_min", seed=1000 + seed,
                       max_steps=max_steps)
    kind = "dqn" if learner == "dqn" else "ppo"
    if kind == "dqn":
        agent = DQNAgent(env.state_dim, env.n_actions, seed=seed)
    else:
        agent = PPOAgent(env.state_dim, env.n_actions, seed=seed)

    # e_hat starts at 0 and lam is held until the first measurement. Seeding it at 1.0 and
    # updating from episode 1 gave every cell a deterministic eta*PROBE_EVERY head start,
    # which showed up as an identical non-zero lam on the closed sanity cell.
    lam, e_hat, steps = (LAM0 if lam_fixed is None else float(lam_fixed)), 0.0, 0
    measured = False
    lam_trace = []
    eps0, eps1 = 1.0, 0.05
    decay_eps = int(0.6 * episodes)

    for ep in range(1, episodes + 1):
        env.reset()
        for _ in range(env.max_steps):
            s = env._obs()
            m = env.available_actions()
            if kind == "dqn":
                eps = max(eps1, eps0 - (eps0 - eps1) * ep / decay_eps)
                a = agent.act(s, m, eps)
            else:
                a, lp, val = agent.act(s, m)
            before = env._phi(*env.pos)
            _, r, done, info = env.step(a)
            # the constraint cost: 1 on an exit transition, 0 otherwise
            d = 1.0 if (done and info["outcome"] == "exited") else 0.0
            r_tilde = r - lam * d
            if shaped:
                # potential-based shaping on the same potential the aligned reward uses,
                # added to test whether the constraint's shortfall is one of learnability
                after = 0.0 if d else env._phi(*env.pos)
                r_tilde += env.beta * (before - after)
            if kind == "dqn":
                agent.buf.add(s, a, r_tilde, env._obs(), float(done),
                              env.available_actions().astype(np.float32))
                agent.learn()
            else:
                agent.store(s, a, lp, val, r_tilde, done, m)
                steps += 1
                if steps % 2048 == 0:
                    agent.update()
            if done:
                break
        if ep % PROBE_EVERY == 0:
            e_hat = exit_rate(agent, boundary, probe_od, kind, max_steps)
            measured = True
        # dual ascent, delta = 0. No update before the constraint has been measured once.
        # lam_fixed holds the multiplier at a stated level instead, which separates a failure
        # of the constrained objective from a failure caused by the size of the multiplier.
        if measured and lam_fixed is None:
            lam = float(np.clip(lam + ETA * e_hat, 0.0, LAM_MAX))
        if ep % 100 == 0:
            lam_trace.append((ep, lam, e_hat))
    if kind == "ppo":
        agent.update()

    comp = evaluate(agent, boundary, eval_od, kind, max_steps)
    return comp, lam, e_hat, lam_trace


def main():
    global LAM_MAX
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--cells", default="C1,C2,C3")
    ap.add_argument("--lam_fixed", type=float, default=None)
    ap.add_argument("--lam_max", type=float, default=LAM_MAX,
                    help="raise the safety rail to test whether the dual ascent converges "
                         "below it rather than resting on it")
    ap.add_argument("--shaped", action="store_true")
    ap.add_argument("--out", default="constrained_arrival.json")
    a = ap.parse_args()
    LAM_MAX = float(a.lam_max)

    eval_od = make_eval_od(n=200, n_side=N_SIDE, seed=12345)
    probe_od = make_probe_od(eval_od)

    CELLS = {
        "C1": ("open", "dqn", a.seeds),
        "C2": ("open", "ppo", a.seeds),
        "C3": ("closed", "dqn", 5),          # sanity: the constraint must never bind
    }
    out = {}
    for name in [c.strip() for c in a.cells.split(",") if c.strip()]:
        boundary, learner, nseeds = CELLS[name]
        comps, lams = [], []
        for s in range(nseeds):
            c, lam, e_hat, trace = run_cell(boundary, learner, s, a.episodes,
                                            eval_od, probe_od, lam_fixed=a.lam_fixed,
                                            shaped=a.shaped)
            comps.append(c)
            lams.append(lam)
            print("  %-3s %-4s %-6s seed %d: completion %5.1f%%  lambda %6.2f  exit_rate %.2f"
                  % (name, learner, boundary, s, c, lam, e_hat), flush=True)
        out[name] = {"boundary": boundary, "learner": learner, "seeds": nseeds,
                     "completion_mean": float(np.mean(comps)),
                     "completion_sd": float(np.std(comps)),
                     "lambda_mean": float(np.mean(lams)),
                     "lambda_max_seen": float(np.max(lams)),
                     "saturated": bool(np.max(lams) >= LAM_MAX - 1e-9),
                     "per_seed_completion": comps, "per_seed_lambda": lams}
        print("== %-3s %-4s %-6s completion %5.1f%% (sd %.1f)  lambda %.2f  saturated %s"
              % (name, learner, boundary, np.mean(comps), np.std(comps),
                 np.mean(lams), out[name]["saturated"]), flush=True)

    print("\npre-registered check: a converged lambda strictly below the rail of %.0f and above"
          % LAM_MAX)
    print("the analytic sufficiency level of %.1f means the multiplier found the threshold."
          % SUFFICIENT_5X5)
    with open(os.path.join(HERE, a.out), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
