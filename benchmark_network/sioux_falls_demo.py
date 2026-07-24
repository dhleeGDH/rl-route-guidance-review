"""Boundary-open completion collapse on a real benchmark network (Sioux Falls).

The completion collapse demonstrated on the bespoke 5x5 grid in Section V invites the
objection that a regular, purpose-built topology is what produces it. This script reproduces the same four conditions on the canonical Sioux Falls network
(LeBlanc 1975; 24 nodes, 76 directed links), a standard, irregular transportation
benchmark, using the same time-varying congestion model, the same non-predictive local
state, and the same standard DQN as the grid demonstration. Only the topology changes.

Conditions (identical semantics to experiments/boundary_open_demo/env.py):
  boundary in {closed, open}; reward in {time_min, aligned}.
  time_min : r = -c(link)                      (early exit maximizes return on open)
  aligned  : r = -c(link) + beta*(phi_b-phi_a) with arrival bonus / exit penalty.

State (non-predictive, local): [cur/23, dst/23, present costs of up to MAXDEG out-links
padded with a sentinel]. Exposes only present, locally observable link costs.
Actions: index into the current node's out-links (0..deg-1); on an open boundary a
boundary node additionally offers an exit action (index MAXDEG) that leaves the network
into an absorbing OUT terminal, the vehicle-removal semantics of a microscopic simulator.
"""

import numpy as np
import torch
import heapq
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "boundary_open_demo"))
from dqn import QNet  # reuse the identical standard Q-network

# -- canonical Sioux Falls directed links (init, term) ----------------------
SF_LINKS = [
    (1, 2), (1, 3), (2, 1), (2, 6), (3, 1), (3, 4), (3, 12), (4, 3), (4, 5),
    (4, 11), (5, 4), (5, 6), (5, 9), (6, 2), (6, 5), (6, 8), (7, 8), (7, 18),
    (8, 6), (8, 7), (8, 9), (8, 16), (9, 5), (9, 8), (9, 10), (10, 9), (10, 11),
    (10, 15), (10, 16), (10, 17), (11, 4), (11, 10), (11, 12), (11, 14), (12, 3),
    (12, 11), (12, 13), (13, 12), (13, 24), (14, 11), (14, 15), (14, 23), (15, 10),
    (15, 14), (15, 19), (15, 22), (16, 8), (16, 10), (16, 17), (16, 18), (17, 10),
    (17, 16), (17, 19), (18, 7), (18, 16), (18, 20), (19, 15), (19, 17), (19, 20),
    (20, 18), (20, 19), (20, 21), (20, 22), (21, 20), (21, 22), (21, 24), (22, 15),
    (22, 20), (22, 21), (22, 23), (23, 14), (23, 22), (23, 24), (24, 13), (24, 21),
    (24, 23),
]
N_NODES = 24
# canonical Sioux Falls node coordinates (standard SiouxFalls_node.tntp), used so the
# non-predictive state carries real geometric position, exactly as the 5x5 grid state
# carries (row, col). Coordinates are only a position feature; they do not encode any
# future traffic.
SF_COORD = {
    1: (50000, 510000), 2: (320000, 510000), 3: (50000, 440000), 4: (130000, 440000),
    5: (220000, 440000), 6: (320000, 440000), 7: (420000, 380000), 8: (320000, 380000),
    9: (220000, 380000), 10: (220000, 320000), 11: (130000, 320000), 12: (50000, 320000),
    13: (50000, 50000), 14: (130000, 190000), 15: (220000, 190000), 16: (320000, 320000),
    17: (320000, 260000), 18: (420000, 320000), 19: (320000, 190000), 20: (320000, 50000),
    21: (220000, 50000), 22: (220000, 130000), 23: (130000, 130000), 24: (130000, 50000),
}
_XS = [c[0] for c in SF_COORD.values()]; _YS = [c[1] for c in SF_COORD.values()]
_XMIN, _XRNG = min(_XS), (max(_XS) - min(_XS))
_YMIN, _YRNG = min(_YS), (max(_YS) - min(_YS))


def _nc(node):
    x, y = SF_COORD[node]
    return (x - _XMIN) / _XRNG, (y - _YMIN) / _YRNG


# perimeter gateway nodes: the network's outer ring, designated as boundary nodes that
# offer an exit to unmodeled exterior roads on the open network.
BOUNDARY_NODES = {1, 2, 3, 7, 13, 18, 21, 24}

ADJ = {i: [] for i in range(1, N_NODES + 1)}
for u, v in SF_LINKS:
    ADJ[u].append(v)
MAXDEG = max(len(a) for a in ADJ.values())  # 5
UNAVAIL_COST = 9.0


def _hop_dist_to(dst):
    """Static shortest-path hop distance from every node to dst (for the potential)."""
    dist = {n: 1e9 for n in range(1, N_NODES + 1)}
    dist[dst] = 0
    pq = [(0, dst)]
    radj = {i: [] for i in range(1, N_NODES + 1)}
    for u, v in SF_LINKS:
        radj[v].append(u)  # reverse graph: distance TO dst
    while pq:
        d, n = heapq.heappop(pq)
        if d > dist[n]:
            continue
        for p in radj[n]:
            if d + 1 < dist[p]:
                dist[p] = d + 1
                heapq.heappush(pq, (d + 1, p))
    return dist


class SiouxFallsEnv:
    def __init__(self, boundary="closed", reward="time_min", congestion=0.6,
                 beta=1.0, r_goal=10.0, r_exit=5.0, max_steps=60, seed=0,
                 align_strength=1.0):
        assert boundary in ("closed", "open")
        assert reward in ("time_min", "aligned")
        self.boundary = boundary
        self.reward = reward
        self.congestion = congestion
        self.beta = beta
        self.r_goal = r_goal
        self.r_exit = r_exit
        self.max_steps = max_steps
        self.align_strength = align_strength
        self.rng = np.random.RandomState(seed)
        self.state_dim = 4 + MAXDEG  # cur(x,y), dst(x,y), up to MAXDEG present link costs
        self.n_actions = MAXDEG + 1  # last index is the exit action (open boundary only)
        self._cong = None
        self._phi_cache = {}
        self.reset()

    def _phi(self, node):
        d = self._phi_cache.get(self.dst)
        if d is None:
            d = _hop_dist_to(self.dst)
            self._phi_cache[self.dst] = d
        return d[node]

    def _link_cost(self, u, v):
        base = 1.0
        cong = self.congestion * self._cong[v - 1] * (0.5 + 0.5 * np.sin(self._t / 6.0))
        return base + max(0.0, cong)

    def reset(self, od=None):
        self._cong = self.rng.rand(N_NODES)
        self._t = 0
        if od is None:
            while True:
                o = self.rng.randint(1, N_NODES + 1)
                d = self.rng.randint(1, N_NODES + 1)
                if o != d and self._phi_at(o, d) >= 3:
                    break
        else:
            o, d = od
        self.pos = o
        self.dst = d
        self.done = False
        self.outcome = None
        return self._obs()

    def _phi_at(self, o, d):
        dd = self._phi_cache.get(d)
        if dd is None:
            dd = _hop_dist_to(d)
            self._phi_cache[d] = dd
        return dd[o]

    def _obs(self):
        u = self.pos
        outs = ADJ[u]
        costs = []
        for k in range(MAXDEG):
            if k < len(outs):
                costs.append(self._link_cost(u, outs[k]))
            else:
                costs.append(UNAVAIL_COST)
        cx, cy = _nc(u); dx, dy = _nc(self.dst)
        return np.array([cx, cy, dx, dy] + costs, dtype=np.float32)

    def available_actions(self):
        mask = np.zeros(self.n_actions, dtype=bool)
        outs = ADJ[self.pos]
        for k in range(len(outs)):
            mask[k] = True
        if self.boundary == "open" and self.pos in BOUNDARY_NODES:
            mask[MAXDEG] = True  # exit action
        return mask

    def step(self, a):
        assert not self.done
        u = self.pos
        outs = ADJ[u]
        phi_b = self._phi(u)

        # exit action on an open boundary node
        if a == MAXDEG:
            reward = -1.0
            if self.reward == "aligned":
                reward -= self.align_strength * self.r_exit
            self.done = True
            self.outcome = "exited"
            return self._obs(), reward, self.done, {"outcome": self.outcome}

        v = outs[a]
        cost = self._link_cost(u, v)
        self.pos = v
        self._t += 1
        reward = -cost
        if self.reward == "aligned":
            reward += self.align_strength * self.beta * (phi_b - self._phi(v))
        if v == self.dst:
            self.done = True
            self.outcome = "arrived"
            if self.reward == "aligned":
                reward += self.align_strength * self.r_goal
        elif self._t >= self.max_steps:
            self.done = True
            self.outcome = "timeout"
        return self._obs(), reward, self.done, {"outcome": self.outcome}


# -- self-contained training (own replay buffer sized to n_actions) ---------
class Buffer:
    def __init__(self, cap, sdim, nact):
        self.s = np.zeros((cap, sdim), np.float32); self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32); self.s2 = np.zeros((cap, sdim), np.float32)
        self.done = np.zeros(cap, np.float32); self.m2 = np.ones((cap, nact), np.float32)
        self.cap = cap; self.i = 0; self.full = False

    def add(self, s, a, r, s2, d, m2):
        i = self.i
        self.s[i], self.a[i], self.r[i] = s, a, r
        self.s2[i], self.done[i], self.m2[i] = s2, d, m2
        self.i = (i + 1) % self.cap; self.full = self.full or self.i == 0

    def __len__(self):
        return self.cap if self.full else self.i

    def sample(self, b):
        idx = np.random.randint(0, len(self), size=b)
        return (self.s[idx], self.a[idx], self.r[idx], self.s2[idx],
                self.done[idx], self.m2[idx])


def train_eval(boundary, reward, seed, episodes=3000, eval_od=None):
    env = SiouxFallsEnv(boundary=boundary, reward=reward, seed=seed)
    torch.manual_seed(seed); np.random.seed(seed)
    q = QNet(env.state_dim, env.n_actions); qt = QNet(env.state_dim, env.n_actions)
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    buf = Buffer(20000, env.state_dim, env.n_actions)
    gamma, batch, target_every, upd = 0.99, 64, 200, 0

    def act(s, mask, eps):
        avail = np.where(mask)[0]
        if np.random.rand() < eps:
            return int(np.random.choice(avail))
        with torch.no_grad():
            qq = q(torch.from_numpy(s).unsqueeze(0)).numpy()[0]
        return int(np.argmax(np.where(mask, qq, -1e9)))

    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (episodes * 0.6))
        s = env.reset(); mask = env.available_actions()
        while not env.done:
            a = act(s, mask, eps)
            s2, r, d, _ = env.step(a)
            m2 = env.available_actions()
            buf.add(s, a, r, s2, float(d), m2)
            s, mask = s2, m2
            if len(buf) >= batch:
                bs, ba, br, bs2, bd, bm2 = buf.sample(batch)
                bs = torch.from_numpy(bs); ba = torch.from_numpy(ba)
                br = torch.from_numpy(br); bs2 = torch.from_numpy(bs2)
                bd = torch.from_numpy(bd); bm2 = torch.from_numpy(bm2)
                qv = q(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q2 = q(bs2) if False else qt(bs2)
                    q2 = q2.masked_fill(bm2 < 0.5, -1e9).max(1)[0]
                    tgt = br + gamma * (1 - bd) * q2
                loss = torch.nn.functional.smooth_l1_loss(qv, tgt)
                opt.zero_grad(); loss.backward(); opt.step(); upd += 1
                if upd % target_every == 0:
                    qt.load_state_dict(q.state_dict())

    # greedy evaluation on a fixed OD set
    arrived = 0
    for od in eval_od:
        s = env.reset(od=od); mask = env.available_actions()
        while not env.done:
            with torch.no_grad():
                qq = q(torch.from_numpy(s).unsqueeze(0)).numpy()[0]
            a = int(np.argmax(np.where(mask, qq, -1e9)))
            s, _, d, info = env.step(a)
            mask = env.available_actions()
        arrived += info["outcome"] == "arrived"
    return arrived / len(eval_od)


def make_eval_od(n=200, seed=999):
    rng = np.random.RandomState(seed)
    ods = []
    while len(ods) < n:
        o = rng.randint(1, N_NODES + 1); d = rng.randint(1, N_NODES + 1)
        if o != d and _hop_dist_to(d)[o] >= 3:
            ods.append((o, d))
    return ods


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.episodes = 1, 800
    print(f"Sioux Falls: {N_NODES} nodes, {len(SF_LINKS)} links, MAXDEG={MAXDEG}, "
          f"boundary nodes={sorted(BOUNDARY_NODES)}")
    eval_od = make_eval_od()
    results = {}
    for boundary in ("closed", "open"):
        for reward in ("time_min", "aligned"):
            rates = [train_eval(boundary, reward, s, args.episodes, eval_od)
                     for s in range(args.seeds)]
            results[(boundary, reward)] = rates
            m = 100 * np.mean(rates); sd = 100 * np.std(rates)
            print(f"[{boundary:6s} | {reward:8s}] completion "
                  f"{m:5.1f}% (sd {sd:4.1f}) seeds={['%.0f'%(100*x) for x in rates]}")
    np.savez(str(Path(__file__).parent / "sioux_results.npz"),
             **{f"{b}_{r}": np.array(v) for (b, r), v in results.items()})
    print("saved sioux_results.npz")
