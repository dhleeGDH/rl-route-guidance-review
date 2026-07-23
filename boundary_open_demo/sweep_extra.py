"""Two robustness studies for Section V (P2):

(A) Reward dose-response: on the boundary-open network, sweep the alignment strength
    lambda from 0 (plain travel-time objective) to 1 (fully destination-aligned) and
    measure the OD trip completion rate. A monotonic rise answers the objection that
    the main result only contrasts two extremes.

(B) Larger network: repeat the open/time-min vs open/aligned contrast on an 8x8 grid
    to show the boundary-open failure is not an artifact of the 5x5 size.
"""

import numpy as np
import torch

from env import GridRouteEnv
from dqn import DQNAgent


def make_eval_od(n_side, n=200, min_sep=3, seed=12345):
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


def evaluate(agent, env_kw, eval_od):
    env = GridRouteEnv(seed=777, **env_kw)
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
    return arrived / len(eval_od)


def train_eval(env_kw, seed, n_episodes, eval_od):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = GridRouteEnv(seed=1000 + seed, **env_kw)
    agent = DQNAgent(env.state_dim, env.n_actions, seed=seed)
    eps0, eps1, decay = 1.0, 0.05, int(0.6 * n_episodes)
    for ep in range(1, n_episodes + 1):
        eps = max(eps1, eps0 - (eps0 - eps1) * ep / decay)
        env.reset()
        for _ in range(env.max_steps):
            s = env._obs(); mask = env.available_actions()
            a = agent.act(s, mask, eps)
            s2, r, done, _ = env.step(a)
            agent.buf.add(s, a, r, s2, float(done),
                          env.available_actions().astype(np.float32))
            agent.learn()
            if done:
                break
    return evaluate(agent, env_kw, eval_od)


def study_dose_response(seeds=3, episodes=3000):
    print("=== (A) reward dose-response on the boundary-open 5x5 network ===")
    eval_od = make_eval_od(5)
    rows = []
    for lam in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
        env_kw = dict(boundary="open", reward="aligned", n_side=5,
                      max_steps=120, align_strength=lam)
        comps = [train_eval(env_kw, s, episodes, eval_od) for s in range(seeds)]
        rows.append((lam, np.mean(comps), np.std(comps)))
        print(f"lambda={lam:.2f}  completion={np.mean(comps):.3f} +/- {np.std(comps):.3f}",
              flush=True)
    np.save("dose_response.npy", np.array(rows))


def study_larger(seeds=3, episodes=3000):
    print("\n=== (B) larger 8x8 network ===")
    eval_od = make_eval_od(8, min_sep=5)
    for reward in ["time_min", "aligned"]:
        env_kw = dict(boundary="open", reward=reward, n_side=8, max_steps=220)
        comps = [train_eval(env_kw, s, episodes, eval_od) for s in range(seeds)]
        print(f"8x8 open {reward:9} completion={np.mean(comps):.3f} +/- {np.std(comps):.3f}",
              flush=True)


if __name__ == "__main__":
    study_dose_response()
    study_larger()
