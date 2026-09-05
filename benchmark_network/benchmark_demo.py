"""Boundary-open completion collapse on standard benchmark road networks.

The completion collapse demonstrated on the bespoke 5x5 grid in Section V invites the
objection that a regular, purpose-built grid is what produces it. This script reproduces the same four conditions on two widely used, independently
published transportation benchmark networks, with the same time-varying congestion
model, the same non-predictive local state, and the same standard DQN as the grid:

  Sioux Falls  (LeBlanc 1975; 24 nodes, 76 directed links). Link list and node
               coordinates are the canonical values from the Transportation Networks
               test-problem set. OD pairs are sampled across the network.
  Nguyen-Dupuis (Nguyen & Dupuis 1984; 13 nodes, 19 directed links). Canonical arc
               list. Its designated OD structure is used: origins {1, 4}, destinations
               {2, 3}.

Only the topology changes between these and the grid. Conditions match
experiments/boundary_open_demo/env.py exactly:
  boundary in {closed, open}; reward in {time_min, aligned}.
  time_min : r = -c(link)                         (early exit maximizes return on open)
  aligned  : r = -c(link) + beta*(phi_b - phi_a)  with arrival bonus / exit penalty.

A boundary node offers an exit action on the open network that leaves into an absorbing
OUT terminal, the vehicle-removal semantics of a microscopic simulator. Boundary nodes
are the network's peripheral through-nodes (documented per network below).
"""

import numpy as np
import torch
import heapq
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "boundary_open_demo"))
from dqn import QNet  # the identical standard Q-network used for the grid

# ------------------------------------------------------------------ networks
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
# canonical Sioux Falls node coordinates (longitude, latitude) from SiouxFalls_node.tntp
SF_COORD = {
    1: (-96.77041974, 43.61282792), 2: (-96.71125063, 43.60581298),
    3: (-96.77430341, 43.5729616), 4: (-96.74716843, 43.56365362),
    5: (-96.73156909, 43.56403357), 6: (-96.71164389, 43.58758553),
    7: (-96.69342281, 43.5638436), 8: (-96.71138171, 43.56232379),
    9: (-96.73124137, 43.54859634), 10: (-96.73143801, 43.54527088),
    11: (-96.74684071, 43.54413068), 12: (-96.78013678, 43.54394065),
    13: (-96.79337655, 43.49070718), 14: (-96.75103549, 43.52930613),
    15: (-96.73150355, 43.52940117), 16: (-96.71138171, 43.54674361),
    17: (-96.71138171, 43.54128009), 18: (-96.69407825, 43.54674361),
    19: (-96.71131617, 43.52959125), 20: (-96.71118508, 43.5153335),
    21: (-96.73097920, 43.51048509), 22: (-96.73124137, 43.51485818),
    23: (-96.75090441, 43.51485818), 24: (-96.74920028, 43.50316422),
}
# canonical Nguyen-Dupuis arcs (Nguyen & Dupuis 1984), verified against the published
# link table; origins {1,4}, destinations {2,3}.
ND_LINKS = [
    (1, 5), (1, 12), (4, 5), (4, 9), (5, 6), (5, 9), (6, 7), (6, 10), (7, 8),
    (7, 11), (8, 2), (9, 10), (9, 13), (10, 11), (11, 2), (11, 3), (12, 6),
    (12, 8), (13, 3),
]

NETWORKS = {
    # The boundary set of each benchmark is every node on the outside of the network as it is
    # published, which is the set drawn in red in Fig. 8. Interior nodes offer no exit.
    "sioux_falls": dict(
        links=SF_LINKS, n_nodes=24, coords=SF_COORD, state_mode="coord",
                # The outer face of the published drawing, derived by experiments/outer_face_sioux.py
        # from the rotation system of the coordinates under an Euler check. An earlier default
        # omitted node 8, which lies on the face: the graph carries no 6-7 link, so a border
        # without 8 does not close. The cells reported by Section V-D run on this set.
        od_mode="dest_boundary", boundary={1, 2, 3, 6, 7, 8, 12, 13, 18, 20, 21, 24},
        max_steps=60, min_sep=3, episodes=8000),
    "nguyen_dupuis": dict(
        links=ND_LINKS, n_nodes=13, coords=None, state_mode="onehot",
        od_mode="designated", origins=[1, 4], dests=[2, 3],
        boundary={1, 2, 3, 4, 8, 12, 13}, max_steps=30, min_sep=2, episodes=3000),
}
UNAVAIL_COST = 9.0

# A stranded node is charged the rest of its step budget at the minimum traversal cost, the
# handling described below. Setting this False restores the superseded handling, under which a
# stranded episode ended at no charge. nd_sink_control.py flips it to measure that difference
# and nothing else reads it.
STRANDED_CHARGE = True


def build(net):
    adj = {i: [] for i in range(1, net["n_nodes"] + 1)}
    radj = {i: [] for i in range(1, net["n_nodes"] + 1)}
    for u, v in net["links"]:
        adj[u].append(v); radj[v].append(u)
    return adj, radj


def hop_dist_to(dst, radj, n):
    dist = {i: 1e9 for i in range(1, n + 1)}
    dist[dst] = 0; pq = [(0, dst)]
    while pq:
        d, x = heapq.heappop(pq)
        if d > dist[x]:
            continue
        for p in radj[x]:
            if d + 1 < dist[p]:
                dist[p] = d + 1; heapq.heappush(pq, (d + 1, p))
    return dist


class GraphRouteEnv:
    def __init__(self, net, boundary="closed", reward="time_min", seed=0,
                 congestion=0.6, beta=1.0, r_goal=10.0, r_exit=5.0, align_strength=1.0):
        self.net = net
        self.adj, self.radj = build(net)
        self.n = net["n_nodes"]
        self.maxdeg = max(len(a) for a in self.adj.values())
        self.boundary = boundary
        self.reward = reward
        self.congestion = congestion; self.beta = beta
        self.r_goal = r_goal; self.r_exit = r_exit; self.align_strength = align_strength
        self.max_steps = net["max_steps"]
        self.state_mode = net["state_mode"]
        self.bnodes = net["boundary"]
        self.rng = np.random.RandomState(seed)
        if self.state_mode == "coord":
            xs = [c[0] for c in net["coords"].values()]
            ys = [c[1] for c in net["coords"].values()]
            self.xmin, self.xr = min(xs), (max(xs) - min(xs))
            self.ymin, self.yr = min(ys), (max(ys) - min(ys))
            self.state_dim = 4 + self.maxdeg
        else:
            self.state_dim = 2 * self.n + self.maxdeg
        self.n_actions = self.maxdeg + 1  # last index = exit (open boundary only)
        self._phi_cache = {}
        self.reset()

    def _phi(self, node):
        d = self._phi_cache.get(self.dst)
        if d is None:
            d = hop_dist_to(self.dst, self.radj, self.n); self._phi_cache[self.dst] = d
        # bound unreachable nodes (no directed path to dst) so the potential-shaping
        # term stays finite on directed benchmarks; strongly connected nets are unaffected.
        return min(d[node], self.n)

    def _cost(self, u, v):
        c = self.congestion * self._cong[v - 1] * (0.5 + 0.5 * np.sin(self._t / 6.0))
        return 1.0 + max(0.0, c)

    def reset(self, od=None):
        self._cong = self.rng.rand(self.n); self._t = 0
        if od is None:
            if self.net["od_mode"] == "designated":
                o = self.net["origins"][self.rng.randint(len(self.net["origins"]))]
                d = self.net["dests"][self.rng.randint(len(self.net["dests"]))]
            elif self.net["od_mode"] == "dest_interior":
                # Interior-destination control: the destination is a node that is NOT on the
                # perimeter, so arrival and a wrong exit are events at different nodes rather
                # than the same kind of node. This is the benchmark counterpart of the grid's
                # interior-destination control in Section V-B.
                inner = sorted(set(range(1, self.n + 1)) - set(self.bnodes))
                while True:
                    o = self.rng.randint(1, self.n + 1)
                    d = inner[self.rng.randint(len(inner))]
                    dd = self._phi_cache.get(d) or hop_dist_to(d, self.radj, self.n)
                    self._phi_cache[d] = dd
                    if o != d and dd[o] >= self.net["min_sep"] and dd[o] < 1e8:
                        break
            elif self.net["od_mode"] == "dest_boundary":
                bnodes = sorted(self.bnodes)
                while True:
                    o = self.rng.randint(1, self.n + 1)
                    d = bnodes[self.rng.randint(len(bnodes))]
                    dd = self._phi_cache.get(d) or hop_dist_to(d, self.radj, self.n)
                    self._phi_cache[d] = dd
                    if o != d and dd[o] >= self.net["min_sep"] and dd[o] < 1e8:
                        break
            else:
                while True:
                    o = self.rng.randint(1, self.n + 1); d = self.rng.randint(1, self.n + 1)
                    dd = self._phi_cache.get(d) or hop_dist_to(d, self.radj, self.n)
                    self._phi_cache[d] = dd
                    if o != d and dd[o] >= self.net["min_sep"] and dd[o] < 1e8:
                        break
        else:
            o, d = od
        self.pos, self.dst = o, d
        self.done = False; self.outcome = None
        return self._obs()

    def _obs(self):
        u = self.pos; outs = self.adj[u]
        costs = [self._cost(u, outs[k]) if k < len(outs) else UNAVAIL_COST
                 for k in range(self.maxdeg)]
        if self.state_mode == "coord":
            cx = (self.net["coords"][u][0] - self.xmin) / self.xr
            cy = (self.net["coords"][u][1] - self.ymin) / self.yr
            dx = (self.net["coords"][self.dst][0] - self.xmin) / self.xr
            dy = (self.net["coords"][self.dst][1] - self.ymin) / self.yr
            return np.array([cx, cy, dx, dy] + costs, dtype=np.float32)
        oh = np.zeros(2 * self.n, dtype=np.float32)
        oh[u - 1] = 1.0; oh[self.n + self.dst - 1] = 1.0
        return np.concatenate([oh, np.array(costs, dtype=np.float32)])

    def available_actions(self):
        mask = np.zeros(self.n_actions, dtype=bool)
        for k in range(len(self.adj[self.pos])):
            mask[k] = True
        # the exit action leaves the network. It is the way to arrive when at the
        # destination node, and a wrong exit at any other boundary node on the open
        # network. On the closed network only the destination node lets a route leave.
        if self.pos == self.dst:
            mask[self.maxdeg] = True
        elif self.boundary == "open" and self.pos in self.bnodes:
            mask[self.maxdeg] = True
        return mask

    def step(self, a):
        u = self.pos; outs = self.adj[u]; phi_b = self._phi(u)
        if a == self.maxdeg:  # leave the network here
            if u == self.dst:
                reward = -1.0
                if self.reward == "aligned":
                    reward += self.align_strength * (self.r_goal + self.beta * phi_b)
                self.done = True; self.outcome = "arrived"
                return self._obs(), reward, True, {"outcome": "arrived"}
            reward = -1.0
            if self.reward == "aligned":
                reward -= self.align_strength * self.r_exit
            self.done = True; self.outcome = "exited"
            return self._obs(), reward, True, {"outcome": "exited"}
        v = outs[a]; cost = self._cost(u, v); self.pos = v; self._t += 1
        reward = -cost
        if self.reward == "aligned":
            reward += self.align_strength * self.beta * (phi_b - self._phi(v))
        if self._t >= self.max_steps:
            self.done = True; self.outcome = "timeout"
        elif not self.available_actions().any():
            # A node with no outgoing arc and no exit is the wrong designated destination
            # of a directed benchmark. The closed boundary offers no way out, so the
            # vehicle is stranded inside the network and the trip fails. Ending the
            # episode here at no charge made stopping early the cheapest outcome under
            # the travel-time reward, and the policy routed to the nearer of the two
            # sinks whatever destination it was given. The rest of the step budget is
            # charged at the minimum traversal cost, which is what any vehicle that fails
            # to arrive pays on a closed network. The charge is a cost rather than an exit
            # penalty, so it enters both reward functions identically.
            if STRANDED_CHARGE:
                reward -= float(self.max_steps - self._t)
            self.done = True; self.outcome = "timeout"
        return self._obs(), reward, self.done, {"outcome": self.outcome}


# ------------------------------------------------------------------ training
class Buffer:
    def __init__(self, cap, sdim, nact):
        self.s = np.zeros((cap, sdim), np.float32); self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32); self.s2 = np.zeros((cap, sdim), np.float32)
        self.d = np.zeros(cap, np.float32); self.m2 = np.ones((cap, nact), np.float32)
        self.cap = cap; self.i = 0; self.full = False

    def add(self, s, a, r, s2, d, m2):
        i = self.i; self.s[i], self.a[i], self.r[i] = s, a, r
        self.s2[i], self.d[i], self.m2[i] = s2, d, m2
        self.i = (i + 1) % self.cap; self.full = self.full or self.i == 0

    def __len__(self):
        return self.cap if self.full else self.i

    def sample(self, b):
        idx = np.random.randint(0, len(self), size=b)
        return (self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx], self.m2[idx])


def train_eval(net, boundary, reward, seed, episodes, eval_od, return_travel_time=False):
    env = GraphRouteEnv(net, boundary=boundary, reward=reward, seed=seed)
    torch.manual_seed(seed); np.random.seed(seed)
    q = QNet(env.state_dim, env.n_actions); qt = QNet(env.state_dim, env.n_actions)
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    buf = Buffer(20000, env.state_dim, env.n_actions)
    gamma, batch, target_every, upd = 0.99, 64, 200, 0

    def act(s, mask, eps):
        if np.random.rand() < eps:
            return int(np.random.choice(np.where(mask)[0]))
        with torch.no_grad():
            qq = q(torch.from_numpy(s).unsqueeze(0)).numpy()[0]
        return int(np.argmax(np.where(mask, qq, -1e9)))

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
                loss = torch.nn.functional.smooth_l1_loss(qv, tgt)
                opt.zero_grad(); loss.backward(); opt.step(); upd += 1
                if upd % target_every == 0:
                    qt.load_state_dict(q.state_dict())

    # 2026-09-05. BOND item 1 asks a study to report the completion rate together with the
    # travel time of the completing trips, and the rate was reported alone on every replication
    # network. The evaluation below accumulates the in-network
    # traversal cost of each trip, on the convention of Appendix E: the arriving move is the
    # exit taken at the destination node and is excluded, as it is from the bespoke-grid
    # figures. The default return is unchanged, so every existing caller is unaffected.
    arrived = 0
    tts = []
    for od in eval_od:
        s = env.reset(od=od); mask = env.available_actions()
        tt = 0.0
        while not env.done:
            with torch.no_grad():
                qq = q(torch.from_numpy(s).unsqueeze(0)).numpy()[0]
            a = int(np.argmax(np.where(mask, qq, -1e9)))
            u = env.pos
            s, _, d, info = env.step(a); mask = env.available_actions()
            if a != env.maxdeg:
                tt += env._cost(u, env.pos)
        if info["outcome"] == "arrived":
            arrived += 1
            tts.append(tt)
    rate = arrived / len(eval_od)
    if return_travel_time:
        return rate, (float(np.mean(tts)) if tts else float("nan")), len(tts)
    return rate


def make_eval_od(net, n=200, seed=999):
    rng = np.random.RandomState(seed); adj, radj = build(net); ods = []
    if net["od_mode"] == "designated":
        pairs = [(o, d) for o in net["origins"] for d in net["dests"]]
        return [pairs[i % len(pairs)] for i in range(n)]
    # The evaluation set draws destinations the way training draws them. This branch once
    # drew from every node regardless of od_mode, which put 106 of the 200 Sioux Falls
    # evaluation destinations inside the network, where no boundary link leaves. The
    # evaluation then measured a different task than the one trained.
    bnodes = sorted(net["boundary"]) if net["od_mode"] == "dest_boundary" else None
    inner = (sorted(set(range(1, net["n_nodes"] + 1)) - set(net["boundary"]))
             if net["od_mode"] == "dest_interior" else None)
    while len(ods) < n:
        o = rng.randint(1, net["n_nodes"] + 1)
        if bnodes:
            d = bnodes[rng.randint(len(bnodes))]
        elif inner:
            d = inner[rng.randint(len(inner))]
        else:
            d = rng.randint(1, net["n_nodes"] + 1)
        dd = hop_dist_to(d, radj, net["n_nodes"])
        if o != d and net["min_sep"] <= dd[o] < 1e8:
            ods.append((o, d))
    return ods


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=0, help="override per-net episodes")
    ap.add_argument("--nets", nargs="+", default=list(NETWORKS.keys()))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--od-mode", default=None,
                    help="override each network's destination rule, e.g. dest_interior")
    ap.add_argument("--out", default=None, help="result file; defaults per destination rule")
    # 2026-08-25: a --smoke run at one seed and 300 episodes overwrote benchmark_results.npz, whose
    # keys are identical to a published run's. No gate reads that file, so nothing caught it. A
    # smoke run now writes its own file and can never land on a published name.
    # The published Sioux Falls border omits one node of the drawing's outer face, and the face
    # itself is derived by outer_face_sioux.py. This overrides the border set so the two can be
    # compared through one code path, under the same guard the destination rule carries.
    ap.add_argument("--boundary-nodes", nargs="+", type=int, default=None,
                    help="override the network's border set")
    args = ap.parse_args()
    # merge into any existing results so per-network runs accumulate into one file. A run under a
    # non-default destination rule produces different quantities under the same network name, so it
    # is given a file of its own: merging it into the published one would silently replace the
    # boundary-destination cells the manuscript reports, leaving no trace in the array shapes.
    # A smoke run is a wiring test, not a result. Sending it to a published file name is how
    # benchmark_results.npz acquired one-seed values on 2026-08-25.
    smoke_tag = "smoke_" if args.smoke else ""
    default_out = (smoke_tag + "benchmark_results.npz" if not args.od_mode
                   else "benchmark_results_%s.npz" % args.od_mode)
    npz_path = Path(__file__).parent / (args.out or default_out)
    assert not (args.od_mode and npz_path.name == "benchmark_results.npz"), \
        "a non-default destination rule must not be written into benchmark_results.npz"
    assert not (args.boundary_nodes and npz_path.name.startswith("benchmark_results.npz")), \
        "a non-default border set must not be written into benchmark_results.npz"
    allres = {}
    if npz_path.exists():
        z = np.load(str(npz_path)); allres = {k: z[k] for k in z.files}
    for name in args.nets:
        net = dict(NETWORKS[name])
        if args.od_mode:
            net["od_mode"] = args.od_mode
        if args.boundary_nodes:
            net["boundary"] = set(args.boundary_nodes)
        seeds = 1 if args.smoke else args.seeds
        episodes = args.episodes or (1500 if args.smoke else net["episodes"])
        print(f"\n=== {name}: {net['n_nodes']} nodes, {len(net['links'])} links, "
              f"boundary={sorted(net['boundary'])}, state={net['state_mode']}, "
              f"seeds={seeds}, episodes={episodes} ===")
        eval_od = make_eval_od(net)
        for boundary in ("closed", "open"):
            for reward in ("time_min", "aligned"):
                rates = [train_eval(net, boundary, reward, s, episodes, eval_od)
                         for s in range(seeds)]
                allres[f"{name}_{boundary}_{reward}"] = np.array(rates)
                m, sd = 100 * np.mean(rates), 100 * np.std(rates)
                print(f"[{boundary:6s} | {reward:8s}] completion {m:5.1f}% "
                      f"(sd {sd:4.1f}) seeds={['%.0f' % (100 * x) for x in rates]}")
        np.savez(str(npz_path), **allres)  # checkpoint after each network
    print("\nsaved %s" % npz_path.name)
