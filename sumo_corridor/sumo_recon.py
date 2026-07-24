"""Reconstruction of a released-code corpus system's formulation on SUMO (Section V-G).

The strongest available evidence short of re-running an unmodified released system is one
demonstrated instance: a published system whose quoted reward matches the demonstration's
travel-time reward, evaluated on an open-boundary variant of its own environment. This
script reconstructs the formulation of [50] (XRouting), a released-code
corpus system that reroutes vehicles on a SUMO urban network under a travel-time objective with
no arrival term, recorded among the purely travel-time-reward studies in Section IV-C. The reconstruction
uses that system's action (next-link rerouting) and its quoted reward (travel-time, no arrival
bonus) on a larger SUMO grid (7x7) than the Section V-C replication, closer to an urban network,
with SUMO's own vehicle-removal semantics at the boundary. Only the boundary (closed vs open)
and the reward (the system's travel-time reward vs a destination-aligned control) are varied.
The manuscript states this is a reconstruction of the system's formulation, not a run of its
unmodified code.
"""
import os
import sys
import numpy as np
import torch

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "boundary_open_demo"))

# patch the SUMO grid to 7x7 before importing the env class
import sumo_env
sumo_env.N = 7
sumo_env.COL = "ABCDEFG"
sumo_env.NET = os.path.join(HERE, "grid7.net.xml")
from sumo_env import SumoGridEnv
from dqn import DQNAgent

# exit/fringe edges of the freshly generated 7x7 net are not in the base-speed table;
# treat their traversal cost as the free-flow unit, matching the bespoke grid's exit cost.
_orig_cost = SumoGridEnv._cost
def _patched_cost(self, edge_id):
    if edge_id not in self._base_speed:
        return 1.0
    return _orig_cost(self, edge_id)
SumoGridEnv._cost = _patched_cost

N = 7


def make_eval_od(n=200, seed=12345, min_sep=4):
    rng = np.random.RandomState(seed)
    ods = []
    while len(ods) < n:
        o = (rng.randint(N), rng.randint(N)); d = (rng.randint(N), rng.randint(N))
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
                arrived += info["outcome"] == "arrived"; break
    env.close()
    return arrived / len(eval_od)


def train_condition(boundary, reward, seed, episodes, eval_od, tag):
    np.random.seed(seed); torch.manual_seed(seed)
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
            agent.learn(); s = s2
            if done:
                break
    env.close()
    return evaluate(agent, boundary, reward, eval_od, tag + "e")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.episodes, a.seeds = 200, 1
    print(f"[50]-reconstruction on SUMO 7x7: seeds={a.seeds}, episodes={a.episodes}", flush=True)
    eval_od = make_eval_od()
    rows = []
    for (boundary, reward) in [("closed", "time_min"), ("open", "time_min"),
                               ("open", "aligned"), ("closed", "aligned")]:
        comps = [train_condition(boundary, reward, s, a.episodes, eval_od, f"{boundary[0]}{reward[0]}{s}")
                 for s in range(a.seeds)]
        m, sd = 100 * np.mean(comps), 100 * np.std(comps)
        rows.append((boundary, reward, m, sd))
        print(f"== {boundary:6} {reward:9} completion={m:.1f} +/- {sd:.1f}  seeds={[round(100*x) for x in comps]}", flush=True)
    with open(os.path.join(HERE, "sumo_recon_summary.csv"), "w") as f:
        f.write("boundary,reward,completion_mean,completion_std\n")
        for b, r, m, sd in rows:
            f.write(f"{b},{r},{m:.2f},{sd:.2f}\n")
    print("wrote sumo_recon_summary.csv", flush=True)
