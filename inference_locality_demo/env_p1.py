"""Grid environment for the inference-locality demonstration (Section V-B).

Design rationale (revised). Inference locality is the claim that a state built from
locally observable, current-time traffic cannot supply the FUTURE dynamic conditions a
time-consistent route needs. A demonstration must therefore make the deciding information
genuinely unavailable to any local policy, not merely to a naive one. Congestion here has
two parts:

  recurrent   : a fixed, deterministic per-cell background. It is learnable from history,
                so a well-trained local policy, or a historical-mean planner, already
                accounts for it. It is NOT where the predictive advantage comes from.
  incidents   : non-recurrent events, each a spike on one cell that switches on at a random
                FUTURE onset time and lasts a random duration. Onsets are drawn per episode
                and are not predictable from history. A predictive state carries a real-time
                forecast of them; a local, current-time state cannot see an incident until
                it has already switched on, i.e. until it is too late to have routed around.

Planners (Section V-B compares upper bounds for each state design, plus a climatology
control, so the reported gap is the value of a real-time forecast that a local state cannot
obtain, not an artifact of a weak baseline):

  predictive-optimal : time-expanded shortest path with full knowledge of the incidents
                       (a real-time forecast). Upper bound for a predictive state.
  reactive-local     : replans each step on the current congestion snapshot, assuming it
                       persists. Sees an incident only once it is active. A fair upper
                       bound for a local, non-predictive state, since no local policy can
                       observe an incident before its onset.
  historical-mean    : routes on the recurrent background only. Shows the incidents are not
                       recoverable from history, so a well-trained local policy cannot
                       anticipate them either.

The network is boundary-closed, so every policy completes; the reported quantity is
realized OD travel time and its variability.
"""

import numpy as np

N_SIDE = 5
ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N, E, S, W


class GridP1Env:
    def __init__(self, state_mode="predictive", n_side=N_SIDE,
                 recur_amp=0.6, n_incidents=3, inc_mag=6.0,
                 inc_dur=(4, 10), horizon_max=60, max_steps=60, seed=0):
        assert state_mode in ("local", "predictive")
        self.state_mode = state_mode
        self.n = n_side
        self.recur_amp = recur_amp
        self.n_incidents = n_incidents
        self.inc_mag = inc_mag
        self.inc_dur = inc_dur
        self.horizon_max = horizon_max
        self.max_steps = max_steps
        self.rng = np.random.RandomState(seed)
        self.n_actions = 4
        # local: pos, dst, current cost of 4 moves (8); predictive adds forecast of the
        # 4 moves at short lookahead (12)
        self.state_dim = 8 if state_mode == "local" else 12
        # fixed recurrent background over cells (same every episode -> learnable climatology)
        rb = np.random.RandomState(999).rand(self.n, self.n)
        self._recur = self.recur_amp * rb
        self.reset()

    def _in_grid(self, r, c):
        return 0 <= r < self.n and 0 <= c < self.n

    def _incident_load(self, r, c, t):
        load = 0.0
        for (ir, ic, onset, dur) in self._incidents:
            if ir == r and ic == c and onset <= t < onset + dur:
                load += self.inc_mag
        return load

    def _cell_congestion(self, r, c, t):
        return self._recur[r, c] + self._incident_load(r, c, t)

    def _cell_recurrent(self, r, c):
        return self._recur[r, c]

    def _edge_cost(self, r, c, a, t):
        dr, dc = ACTIONS[a]
        nr, nc = r + dr, c + dc
        if not self._in_grid(nr, nc):
            return None
        return 1.0 + self._cell_congestion(nr, nc, t)

    def _phi(self, r, c):
        return abs(r - self.dst[0]) + abs(c - self.dst[1])

    def reset(self, od=None):
        # sample non-recurrent incidents with random FUTURE onsets
        self._incidents = []
        for _ in range(self.n_incidents):
            ir, ic = self.rng.randint(self.n), self.rng.randint(self.n)
            onset = self.rng.randint(1, self.horizon_max // 2)
            dur = self.rng.randint(self.inc_dur[0], self.inc_dur[1])
            self._incidents.append((ir, ic, onset, dur))
        self._t = 0
        if od is None:
            while True:
                o = (self.rng.randint(self.n), self.rng.randint(self.n))
                d = (self.rng.randint(self.n), self.rng.randint(self.n))
                if abs(o[0] - d[0]) + abs(o[1] - d[1]) >= 3:
                    break
        else:
            o, d = od
        self.pos = o
        self.dst = d
        self.done = False
        self.outcome = None
        self.travel_time = 0.0
        return self._obs()

    def _obs(self):
        r, c = self.pos
        now = []
        for a in range(4):
            ec = self._edge_cost(r, c, a, self._t)
            now.append(ec if ec is not None else 9.0)
        base = [r / (self.n - 1), c / (self.n - 1),
                self.dst[0] / (self.n - 1), self.dst[1] / (self.n - 1)]
        if self.state_mode == "local":
            return np.array(base + now, dtype=np.float32)
        # predictive: forecast the four candidate moves a few steps ahead (real-time
        # incident forecast); a local state cannot see this
        fc = []
        for a in range(4):
            ec = self._edge_cost(r, c, a, self._t + 3)
            fc.append(ec if ec is not None else 9.0)
        return np.array(base + now + fc, dtype=np.float32)

    def available_actions(self):
        r, c = self.pos
        return np.array([self._in_grid(r + dr, c + dc) for dr, dc in ACTIONS], dtype=bool)

    def step(self, a):
        assert not self.done
        r, c = self.pos
        dr, dc = ACTIONS[a]
        nr, nc = r + dr, c + dc
        if not self._in_grid(nr, nc):
            cost = 1.0
            self.travel_time += cost
            self._t += 1
            if self._t >= self.max_steps:
                self.done = True; self.outcome = "timeout"
            return self._obs(), -cost, self.done, {"outcome": self.outcome,
                                                   "travel_time": self.travel_time}
        cost = self._edge_cost(r, c, a, self._t)
        self.pos = (nr, nc)
        self.travel_time += cost
        self._t += 1
        reward = -cost + 1.0 * (self._phi(r, c) - self._phi(nr, nc))
        if self.pos == self.dst:
            self.done = True; self.outcome = "arrived"; reward += 10.0
        elif self._t >= self.max_steps:
            self.done = True; self.outcome = "timeout"
        return self._obs(), reward, self.done, {"outcome": self.outcome,
                                                "travel_time": self.travel_time}
