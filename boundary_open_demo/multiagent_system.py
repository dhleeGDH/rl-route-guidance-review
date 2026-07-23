"""System-level (multi-agent) boundary-open cell for Section V.

Section V's core demonstration uses an individual travel-time reward, but the corpus's largest
algorithm family is multi-agent with a system-level objective. This script tests, on the same 5x5 grid, whether the exit-dominance failure extends to a
system-level objective (total in-network travel time) with several congestion-coupled
vehicles, as Section V's scope argues verbally.

Setup. V vehicles share one parameter-shared Deep Q-Network and act sequentially each step,
each with its own OD pair. Congestion is endogenous: the cost of entering a cell rises with
the number of vehicles currently occupying it, so vehicles interact. The team reward at each
step is the negative of the total travel-time increment summed over all still-active vehicles
(the system objective), so a vehicle that leaves the network (terminal exit, open boundary)
or arrives stops contributing cost.

  reward = "sys_time" : r_t = -sum_v c(v, t)                    (minimize total in-network time)
  reward = "sys_aligned": the three terms of Eq. (4) applied per vehicle, a potential-shaping
      term on each in-grid move plus an arrival bonus and an exit penalty on leaving.

If a system-level travel-time reward on an open boundary also prefers expelling vehicles, the
team completion rate collapses under sys_time and recovers under sys_aligned, extending the
individual-reward result to the modal family. The bespoke individual-reward env is unchanged.
"""

import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dqn import QNet

N_SIDE = 5
ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N, E, S, W
UNAVAIL = 9.0


def perimeter_links(n):
    out = []
    for r in range(n):
        for c in range(n):
            for a, (dr, dc) in enumerate(ACTIONS):
                if not (0 <= r + dr < n and 0 <= c + dc < n):
                    out.append(((r, c), a))
    return out


class MultiVehicleGrid:
    def __init__(self, boundary="closed", reward="sys_time", n_veh=3, n_side=N_SIDE,
                 congestion=0.6, density_w=1.0, beta=1.0, r_goal=10.0, r_exit=5.0,
                 max_steps=50, seed=0):
        assert boundary in ("closed", "open")
        assert reward in ("sys_time", "sys_aligned")
        self.boundary = boundary; self.reward = reward
        self.n = n_side; self.V = n_veh
        self.congestion = congestion; self.density_w = density_w
        self.beta = beta                       # potential-shaping weight, as in env.py
        self.r_goal = r_goal; self.r_exit = r_exit; self.max_steps = max_steps
        self.rng = np.random.RandomState(seed)
        self.state_dim = 9  # cur(2), dst(2), 4 local costs, local vehicle density
        self.n_actions = 4
        self.reset()

    def _in(self, r, c):
        return 0 <= r < self.n and 0 <= c < self.n

    def _occ(self, cell):
        return sum(1 for v in range(self.V)
                   if self.active[v] and self.pos[v] == cell)

    def _edge_cost(self, r, c, a, extra_target=None):
        dr, dc = ACTIONS[a]; nr, nc = r + dr, c + dc
        if not self._in(nr, nc):
            return None
        base = 1.0
        cong = self.congestion * self._cong[nr, nc] * (0.5 + 0.5 * np.sin(self._t / 6.0))
        dens = self.density_w * self._occ((nr, nc))  # endogenous: cost rises with occupancy
        return base + max(0.0, cong) + dens

    def reset(self):
        self._cong = self.rng.rand(self.n, self.n); self._t = 0
        self.pos = []; self.dst = []; self.dst_link = []; self.active = [True] * self.V
        self.outcome = [None] * self.V
        links = perimeter_links(self.n)
        for _ in range(self.V):
            while True:
                o = (self.rng.randint(self.n), self.rng.randint(self.n))
                dl = links[self.rng.randint(len(links))]
                if abs(o[0] - dl[0][0]) + abs(o[1] - dl[0][1]) >= 3:
                    break
            self.pos.append(o); self.dst_link.append(dl); self.dst.append(dl[0])
        return

    def obs(self, v):
        r, c = self.pos[v]
        costs = []
        for a in range(4):
            ec = self._edge_cost(r, c, a)
            costs.append(UNAVAIL if ec is None else ec)
        dens = self._occ((r, c)) / self.V
        return np.array([r / (self.n - 1), c / (self.n - 1),
                         self.dst[v][0] / (self.n - 1), self.dst[v][1] / (self.n - 1),
                         costs[0], costs[1], costs[2], costs[3], dens], dtype=np.float32)

    def mask(self, v):
        r, c = self.pos[v]; m = np.zeros(4, bool)
        for a, (dr, dc) in enumerate(ACTIONS):
            if self._in(r + dr, c + dc):
                m[a] = True
            elif ((r, c), a) == self.dst_link[v]:
                m[a] = True                       # the vehicle's own destination link
            elif self.boundary == "open":
                m[a] = True                       # a wrong exit
        return m

    def step_all(self, actions):
        """Apply one action per active vehicle; return (team_reward, done)."""
        team = 0.0
        for v in range(self.V):
            if not self.active[v]:
                continue
            r, c = self.pos[v]; a = actions[v]; dr, dc = ACTIONS[a]; nr, nc = r + dr, c + dc
            if not self._in(nr, nc):  # off-grid
                if ((r, c), a) == self.dst_link[v]:
                    team += -1.0
                    if self.reward == "sys_aligned":
                        team += self.r_goal
                    self.active[v] = False; self.outcome[v] = "arrived"
                elif self.boundary == "closed":
                    team += -1.0  # unavailable: stay, unit time
                else:
                    team += -1.0
                    if self.reward == "sys_aligned":
                        team += -self.r_exit
                    self.active[v] = False; self.outcome[v] = "exited"
                continue
            # The aligned objective of Eq. (4) has three terms. Only the arrival bonus and
            # the exit penalty were applied here, which made this cell a different objective
            # from the one Section V-A defines and the one the single-vehicle cells use.
            # The potential-shaping term is applied per vehicle on its own in-grid move.
            phi_b = abs(r - self.dst[v][0]) + abs(c - self.dst[v][1])
            cost = self._edge_cost(r, c, a); self.pos[v] = (nr, nc); team += -cost
            if self.reward == "sys_aligned":
                phi_a = abs(nr - self.dst[v][0]) + abs(nc - self.dst[v][1])
                team += self.beta * (phi_b - phi_a)
        self._t += 1
        done = (self._t >= self.max_steps) or (not any(self.active))
        if done:
            for v in range(self.V):
                if self.active[v]:
                    self.outcome[v] = "timeout"; self.active[v] = False
        return team, done


class Buf:
    def __init__(self, cap, sd):
        self.s = np.zeros((cap, sd), np.float32); self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32); self.s2 = np.zeros((cap, sd), np.float32)
        self.d = np.zeros(cap, np.float32); self.m2 = np.ones((cap, 4), np.float32)
        self.cap = cap; self.i = 0; self.full = False

    def add(self, s, a, r, s2, d, m2):
        i = self.i; self.s[i], self.a[i], self.r[i] = s, a, r
        self.s2[i], self.d[i], self.m2[i] = s2, d, m2
        self.i = (i + 1) % self.cap; self.full = self.full or self.i == 0

    def __len__(self):
        return self.cap if self.full else self.i

    def sample(self, b):
        idx = np.random.randint(0, len(self), b)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx], self.m2[idx]


def train_eval(boundary, reward, seed, episodes=4000, n_veh=3, eval_eps=200):
    torch.manual_seed(seed); np.random.seed(seed)
    q = QNet(9, 4); qt = QNet(9, 4); qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=1e-3); buf = Buf(40000, 9)
    gamma, batch, tgt_every, upd = 0.99, 64, 200, 0

    def act(s, m, eps):
        if np.random.rand() < eps:
            return int(np.random.choice(np.where(m)[0]))
        with torch.no_grad():
            qq = q(torch.from_numpy(s).unsqueeze(0)).numpy()[0]
        return int(np.argmax(np.where(m, qq, -1e9)))

    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (episodes * 0.6))
        env = MultiVehicleGrid(boundary, reward, n_veh=n_veh, seed=seed * 100000 + ep)
        env.reset()
        done = False
        while not done:
            obs = {v: env.obs(v) for v in range(env.V) if env.active[v]}
            masks = {v: env.mask(v) for v in obs}
            acts = {v: act(obs[v], masks[v], eps) for v in obs}
            full_acts = [acts.get(v, 0) for v in range(env.V)]
            active_before = [v for v in obs]
            team, done = env.step_all(full_acts)
            for v in active_before:
                s2 = env.obs(v); m2 = env.mask(v)
                buf.add(obs[v], acts[v], team, s2, float(done), m2)
            if len(buf) >= batch:
                bs, ba, br, bs2, bd, bm2 = buf.sample(batch)
                bs = torch.from_numpy(bs); ba = torch.from_numpy(ba); br = torch.from_numpy(br)
                bs2 = torch.from_numpy(bs2); bd = torch.from_numpy(bd); bm2 = torch.from_numpy(bm2)
                qv = q(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q2 = qt(bs2).masked_fill(bm2 < 0.5, -1e9).max(1)[0]
                    tg = br + gamma * (1 - bd) * q2
                loss = torch.nn.functional.smooth_l1_loss(qv, tg)
                opt.zero_grad(); loss.backward(); opt.step(); upd += 1
                if upd % tgt_every == 0:
                    qt.load_state_dict(q.state_dict())

    # greedy evaluation: fraction of vehicles that arrive
    arrived = tot = 0
    for e in range(eval_eps):
        env = MultiVehicleGrid(boundary, reward, n_veh=n_veh, seed=10_000_000 + e)
        env.reset(); done = False
        while not done:
            acts = []
            for v in range(env.V):
                if not env.active[v]:
                    acts.append(0); continue
                s = env.obs(v); m = env.mask(v)
                with torch.no_grad():
                    qq = q(torch.from_numpy(s).unsqueeze(0)).numpy()[0]
                acts.append(int(np.argmax(np.where(m, qq, -1e9))))
            _, done = env.step_all(acts)
        arrived += sum(o == "arrived" for o in env.outcome); tot += env.V
    return arrived / tot


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--veh", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.seeds, a.episodes = 1, 1200
    print(f"Multi-vehicle system-level cell: V={a.veh}, seeds={a.seeds}, episodes={a.episodes}")
    res = {}
    for boundary in ("closed", "open"):
        for reward in ("sys_time", "sys_aligned"):
            rates = [train_eval(boundary, reward, s, a.episodes, a.veh) for s in range(a.seeds)]
            res[f"{boundary}_{reward}"] = np.array(rates)
            m, sd = 100 * np.mean(rates), 100 * np.std(rates)
            print(f"[{boundary:6s} | {reward:11s}] vehicle completion {m:5.1f}% (sd {sd:4.1f}) "
                  f"seeds={['%.0f' % (100 * x) for x in rates]}")
    np.savez(str(Path(__file__).parent / "multiagent_results.npz"), **res)
    print("saved multiagent_results.npz")
