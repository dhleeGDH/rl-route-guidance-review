# -*- coding: utf-8 -*-
"""T-1903 step 2: the three checks the ticket names, on the M2-C increment.

(a) a completing trajectory carries beta*Phi(o), path-independent at every discount;
(b) an exiting trajectory carries beta*Phi(o) - beta*gamma^h*Phi(n_exit), the forfeited potential of
    the node it leaves from, discounted to the step of departure;
(c) at gamma = 1.0 the wrapper reproduces the published environment at its default, reward by reward.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "repo_v11", "boundary_open_demo"))
from shaping_m2c import make            # noqa: E402
from env import GridRouteEnv            # noqa: E402


def replay(gamma, seed, boundary, actions):
    """Shaping sum of one action sequence, as the aligned reward minus the travel-time reward."""
    kw = dict(boundary=boundary, seed=seed, max_steps=120)
    a_env = make(GridRouteEnv, gamma=gamma, reward="aligned", **kw)
    t_env = make(GridRouteEnv, gamma=gamma, reward="time_min", **kw)
    a_env.reset(); t_env.reset()
    total, disc, k = 0.0, 1.0, 0
    phi_o = a_env._phi(*a_env.pos)
    outcome, phi_last = None, None
    for a in actions:
        phi_last = a_env._phi(*a_env.pos)
        _, ra, da, ia = a_env.step(a)
        _, rt, _, _ = t_env.step(a)
        bonus = 0.0
        if ia.get("outcome") == "arrived":
            bonus = a_env.r_goal
        elif ia.get("outcome") == "exited":
            bonus = -a_env.r_exit
        total += disc * (ra - rt - bonus)
        disc *= gamma
        k += 1
        outcome = ia.get("outcome")
        if da:
            break
    return total, outcome, k, phi_o, phi_last, a_env


def main():
    rng = np.random.RandomState(0)
    worst_a = worst_b = 0.0
    n_a = n_b = 0
    gaps = []
    for gamma in (1.0, 0.999, 0.99, 0.95):
        for seed in range(6):
            for boundary in ("closed", "open"):
                for _ in range(8):
                    acts = [rng.randint(4) for _ in range(40)]
                    s, outcome, k, phi_o, phi_last, env = replay(gamma, seed, boundary, acts)
                    if outcome == "arrived":
                        worst_a = max(worst_a, abs(s - env.beta * phi_o)); n_a += 1
                    elif outcome == "exited":
                        expect = env.beta * phi_o - env.beta * (gamma ** k) * phi_last
                        worst_b = max(worst_b, abs(s - expect)); n_b += 1
                        gaps.append((gamma, env.beta * (gamma ** k) * phi_last))
    print("(a) completing trajectories : %d, worst |sum - beta*Phi(o)|            = %.3e"
          % (n_a, worst_a))
    print("(b) exiting trajectories    : %d, worst |sum - (beta*Phi(o) - beta*gamma^h*Phi(n_exit))| = %.3e"
          % (n_b, worst_b))
    for g in (1.0, 0.99, 0.95):
        v = [x for gg, x in gaps if gg == g]
        if v:
            print("    gamma=%-6s forfeited potential: mean %.3f, max %.3f  (the gap from a completing route)"
                  % (g, sum(v) / len(v), max(v)))

    print("\n(c) regression at gamma = 1.0 against the published environment, reward by reward")
    worst_c, n_c = 0.0, 0
    for seed in range(8):
        for boundary in ("closed", "open"):
            for reward in ("aligned", "time_min"):
                acts = [rng.randint(4) for _ in range(60)]
                pub = GridRouteEnv(boundary=boundary, reward=reward, seed=seed, max_steps=120)
                new = make(GridRouteEnv, gamma=1.0, reward=reward, boundary=boundary,
                           seed=seed, max_steps=120)
                pub.reset(); new.reset()
                for a in acts:
                    _, rp, dp, _ = pub.step(a)
                    _, rn, dn, _ = new.step(a)
                    worst_c = max(worst_c, abs(rp - rn)); n_c += 1
                    if dp or dn:
                        break
    print("    transitions compared %d, worst |published - M2C| at gamma=1.0 = %.3e" % (n_c, worst_c))
    print("    identical at gamma = 1.0 : %s" % (worst_c < 1e-12))


if __name__ == "__main__":
    main()
