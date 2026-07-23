"""Does a boundary-closed network teach routing, or only the training context?

A boundary-closed network guarantees arrival: with no exit available, an episode ends only at
the destination or at the step budget, so even a wandering policy arrives. Completion is then
a property of the topology rather than of the policy, and nothing in training forces the
policy to become a routing rule. Kirk et al. call training and testing on the same instance a
singleton environment; the question here is what such training leaves behind.

The experiment holds the network, the reward, the state, the learner and the budget fixed at
the Section V settings (boundary-closed, travel-time reward) and varies only the set of
destinations seen in training:

  singleton : every training episode uses one fixed destination.
  multi     : every training episode draws a destination at random (the Section V setup).

Both are then evaluated on the same two held-out OD sets: the trained destination, and
destinations never seen in training. The destination is part of the state in both conditions
([cur_r, cur_c, dst_r, dst_c, four local costs]), so a collapse on unseen destinations is not
a missing input. It is the policy fixed to the context it was trained on.

The training loop mirrors train.py exactly, so the only difference is the destination draw.
Writes singleton_env_results.npz.
"""
import numpy as np
import torch

from dqn import DQNAgent
from env import GridRouteEnv, N_SIDE

MAX_STEPS = 120             # the Section V budget
from env import perimeter_links   # noqa: E402
TRAIN_DST = ((0, 4), 0)     # the one destination link the singleton condition ever sees
MIN_SEP = 3


def sample_origin(rng, dst_cell):
    while True:
        o = (rng.randint(N_SIDE), rng.randint(N_SIDE))
        if abs(o[0] - dst_cell[0]) + abs(o[1] - dst_cell[1]) >= MIN_SEP:
            return o


def make_eval(n=200, seed=4242):
    """Held-out OD sets: the trained destination link, and destination links never trained on."""
    rng = np.random.RandomState(seed)
    links = perimeter_links(N_SIDE)
    seen = [(sample_origin(rng, TRAIN_DST[0]), TRAIN_DST) for _ in range(n)]
    unseen = []
    while len(unseen) < n:
        dl = links[rng.randint(len(links))]
        o = (rng.randint(N_SIDE), rng.randint(N_SIDE))
        if dl == TRAIN_DST or abs(o[0] - dl[0][0]) + abs(o[1] - dl[0][1]) < MIN_SEP:
            continue
        unseen.append((o, dl))
    return seen, unseen


def evaluate(agent, eval_od, seed=777):
    env = GridRouteEnv(boundary="closed", reward="time_min", seed=seed, max_steps=MAX_STEPS)
    arrived = 0
    for od in eval_od:
        env.reset(od=od)
        for _ in range(env.max_steps):
            s = env._obs()
            a = agent.act(s, env.available_actions(), 0.0)
            env.step(a)
            if env.done:
                break
        arrived += int(env.outcome == "arrived")
    return arrived / len(eval_od)


def train(context, seed, n_episodes):
    """context: 'singleton' (one fixed destination) or 'multi' (a destination per episode)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = GridRouteEnv(boundary="closed", reward="time_min", seed=1000 + seed,
                       max_steps=MAX_STEPS)
    agent = DQNAgent(env.state_dim, env.n_actions, seed=seed)
    rng = np.random.RandomState(5000 + seed)
    eps0, eps1 = 1.0, 0.05
    decay_eps = int(0.6 * n_episodes)

    for ep in range(1, n_episodes + 1):
        eps = max(eps1, eps0 - (eps0 - eps1) * ep / decay_eps)
        if context == "singleton":
            env.reset(od=(sample_origin(rng, TRAIN_DST[0]), TRAIN_DST))
        else:
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
    return agent


def main(seeds=(0, 1, 2, 3, 4), n_episodes=3000):
    seen_od, unseen_od = make_eval()
    out = {}
    for context in ("singleton", "multi"):
        acc_s, acc_u = [], []
        for sd in seeds:
            agent = train(context, sd, n_episodes)
            a_s, a_u = evaluate(agent, seen_od), evaluate(agent, unseen_od)
            acc_s.append(a_s); acc_u.append(a_u)
            print(f"  [{context}] seed {sd}: trained destination {100*a_s:5.1f}%   "
                  f"unseen destinations {100*a_u:5.1f}%", flush=True)
        out[context + "__seen"] = np.array(acc_s)
        out[context + "__unseen"] = np.array(acc_u)

    np.savez("singleton_env_results.npz", **out)
    print()
    print(f"=== boundary-closed, travel-time reward: OD trip completion (%), "
          f"{len(seeds)} seeds, {n_episodes} episodes ===")
    print(f"{'training context':<20}{'trained destination':>22}{'unseen destinations':>22}")
    for context in ("singleton", "multi"):
        s, u = out[context + "__seen"] * 100, out[context + "__unseen"] * 100
        print(f"{context:<20}{s.mean():>14.1f} ({s.std():4.1f}){u.mean():>14.1f} ({u.std():4.1f})")


if __name__ == "__main__":
    main()
