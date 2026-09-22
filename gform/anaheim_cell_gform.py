# -*- coding: utf-8 -*-
"""The four cells of Section IV-C on Anaheim, at the construction Table IV is solved on.

WHY THIS EXISTS. Table V carries a trained cell on the bespoke grid and on Sioux Falls and none
on Anaheim, while Table IV carries the optimum of both rewards on all three networks. This trains
the same four cells on Anaheim, so the two tables have the same three rows.

WHAT IS TAKEN FROM WHERE.

  learner        benchmark_demo.QNet and benchmark_demo.Buffer, and the training loop of
                 benchmark_demo.train_eval copied line for line into train_cell() below. The
                 copy exists because train_eval builds its own GraphRouteEnv and takes no
                 environment argument. No learner value is changed: Adam at 1e-3, gamma 0.99,
                 buffer 20000, batch 64, target network every 200 gradient steps, epsilon from
                 1.0 to 0.05 over the first 60% of the episodes, masked greedy target.
  construction   anaheim_vi_perimeter.py at --band 0, exec'd up to its first statement, which is
                 what anaheim_shaped_vi_gform.py does. The link costs, the 13-node convex hull,
                 the 403 interior origins and the 38 published zones are therefore the ones the
                 optimum of Table IV is computed on, not a second reading of the same files.
  reward         Eq. (4) with the three terms of the optimum: R_d = 10 and R_x = 5 in units of
                 the mean link cost, beta = 1 on a potential of hop distance times the same
                 mean, the shaping increment of a leaving transition zero.

WHAT DIFFERS FROM THE GRID AND SIOUX FALLS CELLS, BY THE NETWORK.

  link cost      the published free-flow time of the link file, time-invariant, against the
                 congestion field of Eq. (9) the other two networks draw once per episode. The
                 Anaheim environment is therefore deterministic given the OD pair.
  boundary       a trip reaching any border node other than the destination ends there, against
                 a trip ending by taking a peripheral link on the other two networks.
  arrival        entering the destination node, against taking the exit action at it.
  actions        one per outgoing link, maxdeg = 6, with no exit action: the absorbing border
                 gives the policy no leaving action to take. The other two networks carry a
                 leaving action as the last slot, so their action count is maxdeg + 1.
  state          the coord form of benchmark_demo, 4 + maxdeg = 10.
  step cap       300, the author's setting of T-2081: ten times the longest shortest route of
                 the network (29 hops), which is the ratio the grid (120 / 9) and Sioux Falls
                 (60 / 6) carry.
  travel time    the cost of the move entering the destination is excluded, which is the
                 convention of the other two networks, where the arriving move is an exit action
                 carrying no link cost.

    python3 anaheim_cell_gform.py --episodes 8000 --seeds 10 --every 150
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = "/home/dhlee/review_paper/handoff/experiments/benchmark_network"
sys.path.insert(0, BENCH)
sys.path.insert(0, HERE)

import benchmark_demo as B          # noqa: E402  (imported, never written)
import anaheim_shaped_vi_gform as A  # noqa: E402  (the construction loader of the optimum)

SRC_BAND = 0.0                       # the convex hull, the narrow border of Table IV
MAX_STEPS = 300
EVAL_N, EVAL_SEED, MIN_SEP = 200, 12345, 3
R_GOAL, R_EXIT, BETA = 10.0, 5.0, 1.0


def construction():
    """Links, costs, border, origins and zones exactly as the optimum is solved on them."""
    ns = A.load(SRC_BAND)
    return ns


class AnaheimEnv(B.GraphRouteEnv):
    """benchmark_demo's environment on the Anaheim construction.

    Four overrides: the link cost is the published one, the border absorbs, arrival is entering
    the destination node, and the reward constants are in units of the mean link cost. The state,
    the observation, the step budget accounting and the greedy interface are inherited.
    """

    def __init__(self, ns, boundary="closed", reward="time_min", seed=0):
        self.COST, self.SCALE = ns["COST"], ns["SCALE"]
        self.BORDER = set(ns["BORDER"])
        self.ORIGINS, self.DESTS = list(ns["ORIGINS"]), list(ns["DESTS"])
        net = {"links": sorted(self.COST.keys()), "n_nodes": ns["NODES"],
               "coords": {k: (v[0], v[1]) for k, v in ns["COORDS"].items()},
               "state_mode": "coord", "od_mode": "anaheim_zones",
               "boundary": self.BORDER, "max_steps": MAX_STEPS, "min_sep": MIN_SEP}
        super().__init__(net, boundary=boundary, reward=reward, seed=seed,
                         beta=BETA, r_goal=R_GOAL, r_exit=R_EXIT)
        self.n_actions = self.maxdeg          # no leaving action: the border absorbs

    # -- the four overrides ------------------------------------------------
    def _cost(self, u, v):
        return self.COST[(u, v)]

    def _phi(self, node):
        d = self._phi_cache.get(self.dst)
        if d is None:
            d = B.hop_dist_to(self.dst, self.radj, self.n)
            self._phi_cache[self.dst] = d
        return self.SCALE * min(d[node], self.n)

    def available_actions(self):
        mask = np.zeros(self.n_actions, dtype=bool)
        for k in range(len(self.adj[self.pos])):
            mask[k] = True
        return mask

    def reset(self, od=None):
        self._cong = self.rng.rand(self.n)      # unused: the cost is time-invariant
        self._t = 0
        if od is None:
            while True:
                o = self.ORIGINS[self.rng.randint(len(self.ORIGINS))]
                d = self.DESTS[self.rng.randint(len(self.DESTS))]
                dd = self._phi_cache.get(d)
                if dd is None:
                    dd = B.hop_dist_to(d, self.radj, self.n)
                    self._phi_cache[d] = dd
                if o != d and MIN_SEP <= dd[o] < 1e8:
                    break
        else:
            o, d = od
        self.pos, self.dst = o, d
        self.done = False
        self.outcome = None
        return self._obs()

    def step(self, a):
        u = self.pos
        outs = self.adj[u]
        phi_b = self._phi(u)
        v = outs[a]
        cost = self._cost(u, v)
        self.pos = v
        self._t += 1
        reward = -cost
        if v == self.dst:
            if self.reward == "aligned":
                reward += self.beta * phi_b + self.r_goal * self.SCALE
            self.done = True
            self.outcome = "arrived"
            return self._obs(), reward, True, {"outcome": "arrived", "cost": cost}
        if self.boundary == "open" and v in self.BORDER:
            # the potential retains the value at the link of departure on a leaving transition,
            # so the shaping increment of this step is zero
            if self.reward == "aligned":
                reward -= self.r_exit * self.SCALE
            self.done = True
            self.outcome = "exited"
            return self._obs(), reward, True, {"outcome": "exited", "cost": cost}
        if self.reward == "aligned":
            reward += self.beta * (phi_b - self._phi(v))
        if self._t >= self.max_steps:
            self.done = True
            self.outcome = "timeout"
        elif not self.available_actions().any():
            self.done = True
            self.outcome = "timeout"
        return self._obs(), reward, self.done, {"outcome": self.outcome, "cost": cost}


def make_eval_od(ns, n=EVAL_N, seed=EVAL_SEED, min_sep=MIN_SEP):
    """The evaluation protocol of the other two networks on this construction: n draws with
    replacement, destinations from the published zones, origins from the interior, a minimum
    separation of three links and a reachable pair."""
    rng = np.random.RandomState(seed)
    radj, nodes = ns["RADJ"], ns["NODES"]
    origins, dests = list(ns["ORIGINS"]), list(ns["DESTS"])
    cache, ods = {}, []
    while len(ods) < n:
        o = origins[rng.randint(len(origins))]
        d = dests[rng.randint(len(dests))]
        dd = cache.get(d)
        if dd is None:
            dd = B.hop_dist_to(d, radj, nodes)
            cache[d] = dd
        if o != d and min_sep <= dd[o] < 1e8:
            ods.append((o, d))
    return ods


def evaluate(q, env, eval_od):
    """Greedy rollout. The travel time excludes the move entering the destination, which is the
    convention of the grid and Sioux Falls cells, where the arriving move carries no link cost."""
    arrived, tts = 0, []
    for od in eval_od:
        s = env.reset(od=od)
        mask = env.available_actions()
        tt, prev = 0.0, None
        info = {"outcome": None}
        while not env.done:
            with torch.no_grad():
                qq = q(torch.from_numpy(s).unsqueeze(0)).numpy()[0]
            a = int(np.argmax(np.where(mask, qq, -1e9)))
            s, _, d, info = env.step(a)
            mask = env.available_actions()
            if info["outcome"] == "arrived":
                break                      # the arriving move is not counted
            tt += info["cost"]
        if info["outcome"] == "arrived":
            arrived += 1
            tts.append(tt)
    return (100.0 * arrived / len(eval_od),
            float(np.mean(tts)) if tts else float("nan"), len(tts))


def train_cell(args):
    """benchmark_demo.train_eval, line for line, on AnaheimEnv, with checkpoints added.

    The checkpoint evaluation saves and restores the global generator states, as the grid runner
    does, so the trajectory is the one produced with no checkpoint recorded."""
    boundary, reward, seed, episodes, every = args
    ns = construction()
    eval_od = make_eval_od(ns)
    env = AnaheimEnv(ns, boundary=boundary, reward=reward, seed=seed)
    eval_env = AnaheimEnv(ns, boundary=boundary, reward=reward, seed=10000 + seed)
    torch.manual_seed(seed); np.random.seed(seed)
    q = B.QNet(env.state_dim, env.n_actions); qt = B.QNet(env.state_dim, env.n_actions)
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    buf = B.Buffer(20000, env.state_dim, env.n_actions)
    gamma, batch, target_every, upd = 0.99, 64, 200, 0
    curve = []

    def act(s, mask, eps):
        if np.random.rand() < eps:
            return int(np.random.choice(np.where(mask)[0]))
        with torch.no_grad():
            qq = q(torch.from_numpy(s).unsqueeze(0)).numpy()[0]
        return int(np.argmax(np.where(mask, qq, -1e9)))

    t0 = time.time()
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (episodes * 0.6))
        s = env.reset(); mask = env.available_actions()
        while not env.done:
            a = act(s, mask, eps); s2, r, d, _ = env.step(a); m2 = env.available_actions()
            buf.add(s, a, r, s2, float(d), m2); s, mask = s2, m2
            if len(buf) >= batch:
                bs, ba, br, bs2, bd, bm2 = buf.sample(batch)
                bs = torch.from_numpy(bs); ba = torch.from_numpy(ba); br = torch.from_numpy(br)
                bs2 = torch.from_numpy(bs2); bd = torch.from_numpy(bd); bm2 = torch.from_numpy(bm2)
                qv = q(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q2 = qt(bs2).masked_fill(bm2 < 0.5, -1e9).max(1)[0]
                    tgt = br + gamma * (1 - bd) * q2
                loss = F.smooth_l1_loss(qv, tgt)
                opt.zero_grad(); loss.backward(); opt.step(); upd += 1
                if upd % target_every == 0:
                    qt.load_state_dict(q.state_dict())
        if every and (ep + 1) % every == 0:
            np_state, torch_state = np.random.get_state(), torch.get_rng_state()
            rate, tt, _n = evaluate(q, eval_env, eval_od)
            np.random.set_state(np_state); torch.set_rng_state(torch_state)
            curve.append((ep + 1, rate, None if np.isnan(tt) else tt))
    rate, tt, n = evaluate(q, eval_env, eval_od)
    return (boundary, reward, seed, rate, (None if np.isnan(tt) else tt), n, curve,
            time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--every", type=int, default=150)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HERE, "anaheim_cells_gform.json"))
    a = ap.parse_args()
    cells = [("closed", "time_min"), ("open", "time_min"),
             ("closed", "aligned"), ("open", "aligned")]
    jobs = [(b, r, s, a.episodes, a.every) for (b, r) in cells for s in range(a.seeds)]
    out = {"%s_%s" % (b, r): {"steps": None, "curves": [None] * a.seeds,
                              "completion_per_seed": [None] * a.seeds,
                              "travel_time_per_seed": [None] * a.seeds,
                              "completing_trips": [None] * a.seeds,
                              "seconds": [None] * a.seeds} for (b, r) in cells}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for boundary, reward, seed, rate, tt, n, curve, secs in ex.map(train_cell, jobs):
            k = "%s_%s" % (boundary, reward)
            out[k]["steps"] = [int(e) for e, _, _ in curve]
            out[k]["curves"][seed] = [float(v) for _, v, _ in curve]
            out[k]["completion_per_seed"][seed] = float(rate)
            out[k]["travel_time_per_seed"][seed] = tt
            out[k]["completing_trips"][seed] = int(n)
            out[k]["seconds"][seed] = float(secs)
            print("== %-6s %-9s seed %d  completion %5.1f%%  travel time %s  (%.0f s)"
                  % (boundary, reward, seed, rate,
                     "n/a" if tt is None else "%.2f" % tt, secs), flush=True)
    for k, rec in out.items():
        rec["completion_mean"] = float(np.mean(rec["completion_per_seed"]))
        tts = [x for x in rec["travel_time_per_seed"] if x is not None]
        rec["travel_time_mean"] = float(np.mean(tts)) if tts else None
        print("%-18s completion %5.1f%%  travel time %s"
              % (k, rec["completion_mean"],
                 "n/a" if not tts else "%.2f" % rec["travel_time_mean"]))
    out["_protocol"] = {
        "network": "Anaheim, 416 nodes, 914 links",
        "border": "convex hull of the published coordinates, 13 nodes",
        "link_cost": "the published free-flow time, time-invariant",
        "mean_link_cost_min": construction()["SCALE"],
        "r_goal": R_GOAL, "r_exit": R_EXIT, "beta": BETA,
        "reward_units": "R_d, R_x and the potential are in units of the mean link cost",
        "potential": "hop distance to the destination times the mean link cost",
        "arrival": "entering the destination node",
        "exit": "entering any other border node, absorbing",
        "max_steps": MAX_STEPS, "episodes": a.episodes, "seeds": a.seeds,
        "checkpoint_every": a.every,
        "eval": "%d draws, seed %d, destinations the 38 published zones, origins the 403 "
                "interior nodes, minimum separation %d links" % (EVAL_N, EVAL_SEED, MIN_SEP),
        "travel_time": "excludes the move entering the destination",
        "learner": "benchmark_demo QNet 64-64, Adam 1e-3, gamma 0.99, buffer 20000, batch 64, "
                   "target every 200 updates, epsilon 1.0 to 0.05 over the first 60%",
        "threads": 1, "wall_seconds": time.time() - t0}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("wrote %s in %.0f s" % (os.path.basename(a.out), time.time() - t0))


if __name__ == "__main__":
    main()
