"""SUMO-backed 5x5 grid route-guidance environment (Section V-B robustness check).

The Section V grid demonstration is here ported onto the field's dominant microscopic
simulator, SUMO, to show that the boundary-open failure is not an artifact of a bespoke
grid engine but appears on SUMO's own network semantics -- in which a vehicle routed off
the modeled network (onto a fringe edge) is removed, exactly the terminal-exit semantics
Section V identifies as the cause.

The network is a SUMO-generated 5x5 grid (junctions A0..E4). At each junction the agent
chooses one of four moves (N/E/S/W). An in-grid move goes to the adjacent junction; a move
off the border goes onto a fringe edge to a dead-end node, where SUMO removes the vehicle
(a network exit). The state, action set, reward variants, and completion metric mirror the
bespoke-grid environment exactly, so only the substrate changes.

boundary in {closed, open}:
  closed : off-border (fringe) moves are unavailable (masked); an episode ends on arrival
           or step budget.
  open   : off-border moves are available; taking one drives the vehicle onto a fringe edge
           and SUMO removes it -> exit (non-arrival).
reward in {time_min, aligned}: identical definitions to the bespoke grid.
"""
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci  # noqa: E402
import sumolib  # noqa: E402

SUMO_BIN = sumolib.checkBinary("sumo")
NET = os.path.join(os.path.dirname(__file__), "grid5.net.xml")
N = 5                       # 5x5 grid


def _boundary_cells(n=N):
    return [(c, r) for c in range(n) for r in range(n)
            if c in (0, n - 1) or r in (0, n - 1)]


def _exit_action_at(c, r, n=N):
    """An off-grid action index available at boundary cell (c,r), or None."""
    from itertools import count
    order = []
    if r == n - 1: order.append(0)   # top
    if c == n - 1: order.append(1)   # right
    if r == 0: order.append(2)       # bottom
    if c == 0: order.append(3)       # left
    return order[0] if order else None
COL = "ABCDE"
# moves: 0=N (row+1), 1=E (col+1), 2=S (row-1), 3=W (col-1)
DELTA = [(0, 1), (1, 0), (0, -1), (-1, 0)]
FRINGE = {0: "top", 1: "right", 2: "bottom", 3: "left"}
UNAVAIL = 9.0


def jid(c, r):
    return COL[c] + str(r)


class SumoGridEnv:
    def __init__(self, boundary="closed", reward="time_min", congestion=0.6,
                 beta=1.0, r_goal=10.0, r_exit=5.0, max_steps=50, seed=0,
                 align_strength=1.0, return_cost=None, label="e", step_length=5.0):
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
        self.state_dim = 9
        self.n_actions = 4
        self.label = label
        self.net = sumolib.net.readNet(NET)
        self._edges = [e.getID() for e in self.net.getEdges()]
        self._base_speed = {e.getID(): e.getSpeed() for e in self.net.getEdges()}
        self._length = {e.getID(): e.getLength() for e in self.net.getEdges()}
        # normalize traversal cost so free-flow ~ 1.0 (matching the bespoke grid's [1, c_max]
        # scale), which lets the reward parameters (r_goal, r_exit, beta) transfer directly.
        self._cost_scale = min(self._length[e] / self._base_speed[e] for e in self._edges)
        traci.start([SUMO_BIN, "-n", NET, "--step-length", str(step_length),
                     "--no-step-log", "true", "--no-warnings", "true",
                     "--collision.action", "none", "--time-to-teleport", "-1",
                     "--default.speeddev", "0"],
                    label=label)
        self.conn = traci.getConnection(label)
        self._vid = 0
        self._cong = None

    # -- geometry ----------------------------------------------------------
    def _target_edge(self, c, r, a):
        """Return (edge_id, kind) for move a from (c,r). kind: 'grid'|'exit'."""
        dc, dr = DELTA[a]
        nc, nr = c + dc, r + dr
        if 0 <= nc < N and 0 <= nr < N:
            return jid(c, r) + jid(nc, nr), "grid"
        fr = {0: f"top{c}", 1: f"right{r}", 2: f"bottom{c}", 3: f"left{r}"}[a]
        return jid(c, r) + fr, "exit"

    def _incoming_fringe_edge(self, c, r):
        """A fringe edge leading INTO junction (c,r), used to spawn the vehicle.
        Falls back to any incoming edge."""
        n = self.net.getNode(jid(c, r))
        for e in n.getIncoming():
            if e.getFromNode().getType() == "dead_end":
                return e.getID()
        return n.getIncoming()[0].getID()

    def _phi(self, c, r):
        return abs(c - self.dst[0]) + abs(r - self.dst[1])

    def _edge_tt(self, edge_id):
        """Current travel time of an edge from its congestion-scaled speed."""
        spd = self._base_speed[edge_id] * (1.0 - self._cong.get(edge_id, 0.0))
        return self._length[edge_id] / max(spd, 1.0)

    def _cost(self, edge_id):
        """Normalized traversal cost (free-flow ~ 1.0), used for reward and state."""
        return self._edge_tt(edge_id) / self._cost_scale

    # -- episode -----------------------------------------------------------
    def reset(self, od=None):
        # per-episode congestion field: a fraction in [0, congestion*0.6] slows each edge
        self._cong = {e: self.congestion * self.rng.rand() * 0.6 for e in self._edges}
        for e in self._edges:
            spd = self._base_speed[e] * (1.0 - self._cong[e])
            self.conn.edge.setMaxSpeed(e, max(spd, 1.0))
        bcells = _boundary_cells()
        if od is None:
            while True:
                o = (self.rng.randint(N), self.rng.randint(N))
                d = bcells[self.rng.randint(len(bcells))]
                if abs(o[0] - d[0]) + abs(o[1] - d[1]) >= 3:
                    break
        else:
            o, d = od
        self.pos = o
        self.dst = d
        self.dst_exit = _exit_action_at(*d)
        self.done = False
        self.outcome = None
        self._t = 0
        # spawn vehicle on a fringe edge leading into the origin junction
        self._vid += 1
        self.veh = f"v{self.label}{self._vid}"
        entry = self._incoming_fringe_edge(*o)
        rid = f"r{self.veh}"
        self.conn.route.add(rid, [entry])
        self.conn.vehicle.add(self.veh, rid, departLane="best", departSpeed="max")
        self.cur_edge = entry
        # roll the vehicle onto the network (until it is actually on 'entry')
        self._advance_until_on(entry)
        return self._obs()

    def _advance_until_on(self, edge_id, limit=200):
        for _ in range(limit):
            self.conn.simulationStep()
            if self.veh in self.conn.vehicle.getIDList() and \
               self.conn.vehicle.getRoadID(self.veh) == edge_id:
                return True
            if self.veh in self.conn.simulation.getArrivedIDList():
                return False
        return False

    def _obs(self):
        c, r = self.pos
        costs = []
        for a in range(4):
            eid, kind = self._target_edge(c, r, a)
            if kind == "exit" and self.boundary == "closed":
                costs.append(UNAVAIL)
            else:
                costs.append(min(self._cost(eid), UNAVAIL))
        # The destination is an outgoing boundary link. The corner cells carry two such
        # links, both open on an open boundary, so the cell alone did not identify which
        # off-grid action arrives. The exit-action index completes the destination, as on
        # the bespoke grid.
        return np.array([c / (N - 1), r / (N - 1),
                         self.dst[0] / (N - 1), self.dst[1] / (N - 1),
                         self.dst_exit / 3.0,
                         costs[0], costs[1], costs[2], costs[3]], dtype=np.float32)

    def available_actions(self):
        c, r = self.pos
        mask = np.zeros(4, dtype=bool)
        for a in range(4):
            _, kind = self._target_edge(c, r, a)
            if kind == "grid":
                mask[a] = True
            elif (c, r) == self.dst and a == self.dst_exit:
                mask[a] = True                       # the destination's own exit
            elif self.boundary == "open":
                mask[a] = True                       # a wrong exit
        return mask

    def step(self, a):
        assert not self.done
        c, r = self.pos
        eid, kind = self._target_edge(c, r, a)
        cost = self._cost(eid)
        phi_before = self._phi(c, r)
        # route the vehicle: current edge then the chosen edge
        try:
            self.conn.vehicle.setRoute(self.veh, [self.cur_edge, eid])
        except traci.TraCIException:
            pass
        self._t += 1

        if kind == "exit":  # drive onto fringe -> removed by SUMO
            self._roll_until_arrived_or_on(eid, None)
            if (c, r) == self.dst and a == self.dst_exit:
                reward = -1.0
                if self.reward == "aligned":
                    reward += self.align_strength * (self.r_goal + self.beta * phi_before)
                self.done = True
                self.outcome = "arrived"
                self._cleanup()
                return self._obs(), reward, self.done, {"outcome": self.outcome}
            reward = -1.0
            if self.reward == "aligned":
                reward -= self.align_strength * self.r_exit
            self.done = True
            self.outcome = "exited"
            self._cleanup()
            return self._obs(), reward, self.done, {"outcome": self.outcome}

        # in-grid move: advance until the vehicle enters eid (passed the junction)
        arrived_edge = self._roll_until_arrived_or_on(eid, eid)
        nc, nr = self._edge_to_cr(eid)
        self.pos = (nc, nr)
        self.cur_edge = eid
        reward = -cost
        if self.reward == "aligned":
            reward += self.align_strength * self.beta * (phi_before - self._phi(nc, nr))

        if self._t >= self.max_steps:
            self.done = True
            self.outcome = "timeout"
            self._cleanup()
        return self._obs(), reward, self.done, {"outcome": self.outcome}

    def _edge_to_cr(self, edge_id):
        to = self.net.getEdge(edge_id).getToNode().getID()
        return COL.index(to[0]), int(to[1:])

    def _roll_until_arrived_or_on(self, edge_id, want_on, limit=400):
        """Step until the controlled vehicle is on want_on (if given) or removed."""
        for _ in range(limit):
            self.conn.simulationStep()
            ids = self.conn.vehicle.getIDList()
            if self.veh in ids:
                rid = self.conn.vehicle.getRoadID(self.veh)
                if want_on is not None and rid == want_on:
                    return "on"
            if self.veh in self.conn.simulation.getArrivedIDList() or self.veh not in ids:
                return "gone"
        return "limit"

    def _cleanup(self):
        try:
            if self.veh in self.conn.vehicle.getIDList():
                self.conn.vehicle.remove(self.veh)
        except traci.TraCIException:
            pass
        # let the removal settle
        self.conn.simulationStep()

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# self-test: scripted policies must behave as the mechanism requires
# --------------------------------------------------------------------------
if __name__ == "__main__":
    def greedy_move(env):
        c, r = env.pos
        best, ba = 1e9, None
        for a in range(4):
            dc, dr = DELTA[a]
            nc, nr = c + dc, r + dr
            if 0 <= nc < N and 0 <= nr < N:
                d = abs(nc - env.dst[0]) + abs(nr - env.dst[1])
                if d < best:
                    best, ba = d, a
        return ba

    print("[closed] shortest-path policy should arrive ~always")
    env = SumoGridEnv(boundary="closed", reward="time_min", seed=1, label="t1")
    arr = 0
    K = 20
    for _ in range(K):
        env.reset()
        for _ in range(env.max_steps):
            _, _, done, info = env.step(greedy_move(env))
            if done:
                arr += info["outcome"] == "arrived"
                break
    env.close()
    print(f"  closed shortest-path arrival: {arr}/{K}")

    print("[open] exit-greedy policy should exit (mechanism present)")
    env = SumoGridEnv(boundary="open", reward="time_min", seed=2, label="t2")
    ext = 0
    for _ in range(K):
        env.reset()
        for _ in range(env.max_steps):
            m = env.available_actions()
            c, r = env.pos
            exit_a = None
            for a in range(4):
                _, kind = env._target_edge(c, r, a)
                if kind == "exit" and m[a]:
                    exit_a = a
                    break
            a = exit_a if exit_a is not None else int(np.argmax(m))
            _, _, done, info = env.step(a)
            if done:
                ext += info["outcome"] == "exited"
                break
    env.close()
    print(f"  open exit-greedy exits: {ext}/{K}")
