"""Grid route-guidance environment for the boundary-open demonstration.

A controlled 5x5 grid MDP used in Section V of the survey to show, quantitatively,
that the configuration most route-guidance studies adopt (a travel-time-minimizing
reward evaluated on a boundary-closed network) hides a failure that appears on a
boundary-open network. The environment is a diagnostic instrument, not a proposed
method.

The destination is an outgoing boundary link, not an interior cell. A trip ends by
leaving the modeled area on a particular link, and arrival is taking that one link.
The origin is unconstrained and may be interior. This matches how a released system
sets a destination on a boundary stub, and it makes arrival and failure the same kind
of event, taking an outgoing boundary link, differing only in which link.

Two independent switches define the four conditions studied:

  boundary in {"closed", "open", "costly_return"}
      closed : the destination link is the only outgoing link on the perimeter. Every
               other boundary action is unavailable, so a vehicle that wanders can
               leave only where it was asked to. Every destination stays reachable.
      open   : other perimeter links are outgoing too, each leading into an absorbing
               OUT state; taking one that is not the destination ends the episode
               without arrival. This is the vehicle-removal semantics of the corpus's
               dominant microscopic simulator, so a wrong exit is a terminal event.
      costly_return : a boundary node has outward actions, but leaving is NOT
               terminal; the vehicle pays a fixed detour penalty (return_cost) and
               re-enters at the same node, the episode continuing. Models a vehicle
               that drives on unmodeled exterior roads and returns. Used only for the
               Section V robustness check answering whether the open-network collapse
               is an artifact of terminal (absorbing) exit rather than of open
               topology; with a non-terminal exit the network behaves closed.

  exit_density in (0, 1]: the share of non-destination perimeter links opened on the
      open condition. 1.0 opens all of them; a lower value opens a random subset.

  reward in {"time_min", "aligned"}
      time_min : r_t = -c(edge)  (minimize accumulated travel time). No arrival
                 bonus and no exit penalty, so on an open network the return is
                 maximized by leaving early rather than completing the trip.
      aligned  : r_t = -c(edge) + beta*(Phi(v_t) - Phi(v_{t+1})) with an arrival
                 bonus R_goal and an exit penalty R_exit, Phi the Manhattan
                 distance to the destination. Removes the early-exit incentive.

State (held fixed across all conditions, non-predictive / local by construction):
  [ cur_r/4, cur_c/4, dst_r/4, dst_c/4, dst_dir/3, c_N, c_E, c_S, c_W ]
  where dst_dir is the action index of the outgoing boundary link that is the destination,
  which identifies g at the four corner cells that have two such links.
  where c_* are the current-time traversal costs of the four candidate moves
  (a large sentinel for an unavailable move on a closed boundary). The state
  exposes only present, locally observable costs, matching the non-predictive
  dynamic state recorded for most corpus systems.

Actions: 0=N (r-1), 1=E (c+1), 2=S (r+1), 3=W (c-1).
"""

import numpy as np

N_SIDE = 5
ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N, E, S, W
UNAVAIL_COST = 9.0  # sentinel cost placed in the state for an unavailable move


def perimeter_links(n):
    """Every (cell, action) pair whose move leaves the n x n lattice."""
    out = []
    for r in range(n):
        for c in range(n):
            for a, (dr, dc) in enumerate(ACTIONS):
                if not (0 <= r + dr < n and 0 <= c + dc < n):
                    out.append(((r, c), a))
    return out


class GridRouteEnv:
    def __init__(self, boundary="closed", reward="time_min", n_side=N_SIDE,
                 congestion=0.6, beta=1.0, r_goal=10.0, r_exit=5.0,
                 max_steps=50, seed=0, align_strength=1.0, return_cost=1.0,
                 vanish_potential=False, exit_density=1.0):
        assert boundary in ("closed", "open", "costly_return")
        assert reward in ("time_min", "aligned")
        self.boundary = boundary
        self.reward = reward
        # share of non-destination perimeter links opened on the open condition
        self.exit_density = exit_density
        self._dst_link = None       # ((cell), action) whose off-grid move is the destination
        self._open_links = set()    # the other perimeter links open under the open condition
        # When True, the potential-based shaping term also fires on the exit transition
        # with the exit terminal's potential taken as zero (Phi vanishes at all terminals).
        # This makes the shaping policy-invariant (Ng et al.), so the accumulated shaping
        # of an exiting trajectory telescopes to the same beta*Phi(origin) as a completing
        # one and cannot tilt the optimum toward arrival. Used to separate the shaping
        # term's learning-aid role from its objective-alignment role (Section V, M6b).
        self.vanish_potential = vanish_potential
        # fixed detour penalty charged when the vehicle leaves and re-enters, under
        # the non-terminal costly_return boundary (Section V robustness check).
        self.return_cost = return_cost
        # align_strength scales the destination-aligned reward terms; 0 recovers the
        # plain travel-time objective, 1 is the full aligned objective. Used for the
        # reward dose-response sweep.
        self.align_strength = align_strength
        self.n = n_side
        self.congestion = congestion          # amplitude of per-episode congestion
        self.beta = beta                       # potential-shaping weight (aligned)
        self.r_goal = r_goal                   # arrival bonus (aligned)
        self.r_exit = r_exit                   # exit penalty (aligned)
        self.max_steps = max_steps
        self.rng = np.random.RandomState(seed)
        self.state_dim = 9
        self.n_actions = 4
        self._cong = None
        self.reset()

    # -- geometry helpers ---------------------------------------------------
    def _in_grid(self, r, c):
        return 0 <= r < self.n and 0 <= c < self.n

    def _edge_cost(self, r, c, a):
        """Current-time traversal cost of moving from (r,c) by action a.

        Base cost 1.0 plus a congestion term that depends on the destination
        cell and the current step, giving a time-varying dynamic network.
        Returns None if the move is not a within-grid edge.
        """
        dr, dc = ACTIONS[a]
        nr, nc = r + dr, c + dc
        if not self._in_grid(nr, nc):
            return None
        base = 1.0
        cong = self.congestion * self._cong[nr, nc] * (0.5 + 0.5 * np.sin(self._t / 6.0))
        return base + max(0.0, cong)

    def _phi(self, r, c):
        return abs(r - self.dst[0]) + abs(c - self.dst[1])

    def _is_dst_link(self, r, c, a):
        return self._dst_link is not None and self._dst_link == ((r, c), a)

    def _link_open(self, r, c, a):
        """Is this off-grid move an available exit under the current boundary condition?"""
        if self._is_dst_link(r, c, a):
            return True
        if self.boundary == "closed":
            return False
        return ((r, c), a) in self._open_links  # open or costly_return

    # -- episode API --------------------------------------------------------
    def reset(self, od=None):
        # per-episode congestion field over cells (dynamic traffic realization)
        self._cong = self.rng.rand(self.n, self.n)
        self._t = 0
        links = perimeter_links(self.n)
        if od is None:
            # origin anywhere, destination an outgoing boundary link, non-trivial separation
            while True:
                o = (self.rng.randint(self.n), self.rng.randint(self.n))
                dl = links[self.rng.randint(len(links))]
                if abs(o[0] - dl[0][0]) + abs(o[1] - dl[0][1]) >= 3:
                    break
        else:
            o, dl = od
        self.pos = o
        self._dst_link = dl
        self.dst = dl[0]                       # boundary cell the destination link leaves from
        rest = [l for l in links if l != dl]
        if self.boundary == "closed":
            self._open_links = set()
        elif self.exit_density >= 1.0:
            self._open_links = set(rest)
        else:
            k = int(round(self.exit_density * len(rest)))
            idx = self.rng.choice(len(rest), size=k, replace=False)
            self._open_links = {rest[i] for i in idx}
        self.done = False
        self.outcome = None  # "arrived" | "exited" | "timeout"
        return self._obs()

    def _obs(self):
        r, c = self.pos
        costs = []
        for a in range(4):
            ec = self._edge_cost(r, c, a)
            if ec is None:
                # off-grid move: on a closed boundary it is unavailable; on an
                # open boundary it is an exit. Either way no in-grid cost exists.
                costs.append(UNAVAIL_COST)
            else:
                costs.append(ec)
        # The destination of Eq. (2) is an outgoing boundary link, not a cell. Carrying the
        # cell alone left the four corner cells ambiguous: each has two outgoing links, both
        # available on an open boundary, and the two actions were observationally identical.
        # 105 of the 200 evaluation pairs have a corner destination, so more than half the
        # set was unidentifiable to the policy while the value iteration solved it knowing
        # the link. The direction of the destination link completes g.
        return np.array([r / (self.n - 1), c / (self.n - 1),
                         self.dst[0] / (self.n - 1), self.dst[1] / (self.n - 1),
                         self._dst_link[1] / 3.0,
                         costs[0], costs[1], costs[2], costs[3]], dtype=np.float32)

    def available_actions(self):
        """Action mask. In-grid moves are always available. An off-grid move is
        available if it is the destination link, or, on the open condition, if it is
        one of the opened perimeter links."""
        r, c = self.pos
        mask = np.zeros(4, dtype=bool)
        for a, (dr, dc) in enumerate(ACTIONS):
            if self._in_grid(r + dr, c + dc):
                mask[a] = True
            elif self.boundary == "costly_return":
                mask[a] = True
            elif self._link_open(r, c, a):
                mask[a] = True
        return mask

    def step(self, a):
        assert not self.done
        r, c = self.pos
        dr, dc = ACTIONS[a]
        nr, nc = r + dr, c + dc
        phi_before = self._phi(r, c)

        if not self._in_grid(nr, nc):
            # off-grid action
            if self._is_dst_link(r, c, a):
                # taking the destination link is arrival
                reward = -1.0
                if self.reward == "aligned":
                    reward += self.align_strength * self.r_goal
                    reward += self.align_strength * self.beta * (phi_before - 0.0)
                self.done = True
                self.outcome = "arrived"
                return self._obs(), reward, self.done, {"outcome": self.outcome}
            if self.boundary == "costly_return":
                # non-terminal exit: pay a fixed detour penalty and re-enter at the
                # same node; the episode continues.
                cost = self.return_cost
                self.pos = (r, c)
                reward = -cost
                self._t += 1
                if self._t >= self.max_steps:
                    self.done = True
                    self.outcome = "timeout"
                return self._obs(), reward, self.done, {"outcome": self.outcome}
            if not self._link_open(r, c, a):
                # a closed perimeter link: the move is unavailable, time passes in place
                reward = -1.0
                self.pos = (r, c)
                self._t += 1
                if self._t >= self.max_steps:
                    self.done = True
                    self.outcome = "timeout"
                return self._obs(), reward, self.done, {"outcome": self.outcome}
            # any other open perimeter link takes the vehicle out of the network
            reward = -1.0
            if self.reward == "aligned":
                reward -= self.align_strength * self.r_exit
                if self.vanish_potential:
                    reward += self.align_strength * self.beta * phi_before
            self.done = True
            self.outcome = "exited"
            return self._obs(), reward, self.done, {"outcome": self.outcome}

        # normal in-grid move
        cost = self._edge_cost(r, c, a)
        self.pos = (nr, nc)
        self._t += 1
        reward = -cost
        if self.reward == "aligned":
            reward += self.align_strength * self.beta * (phi_before - self._phi(nr, nc))

        if self._t >= self.max_steps:
            self.done = True
            self.outcome = "timeout"

        return self._obs(), reward, self.done, {"outcome": self.outcome}


# --------------------------------------------------------------------------
# self-test: verify the mechanism the experiment relies on, before any training
# --------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.RandomState(0)

    # 1. closed boundary: an optimal (shortest-path) policy always arrives
    env = GridRouteEnv(boundary="closed", reward="time_min", seed=1)
    arrived = 0
    for _ in range(500):
        env.reset()
        for _ in range(env.max_steps):
            r, c = env.pos
            # greedy Manhattan step toward destination among available moves
            best, ba = 1e9, None
            for a, (dr, dc) in enumerate(ACTIONS):
                nr, nc = r + dr, c + dc
                if env._in_grid(nr, nc):
                    d = abs(nr - env.dst[0]) + abs(nc - env.dst[1])
                    if d < best:
                        best, ba = d, a
            _, _, done, info = env.step(ba)
            if done:
                arrived += info["outcome"] == "arrived"
                break
    print(f"[closed] shortest-path policy arrival rate: {arrived/500:.2%} (expect ~100%)")

    # 2. open boundary + time_min: a myopic 'minimize steps to terminate' policy
    #    that is allowed to exit will leave early -> low arrival
    env = GridRouteEnv(boundary="open", reward="time_min", seed=2)
    exited = 0
    for _ in range(500):
        env.reset()
        for _ in range(env.max_steps):
            # policy: if an exit move is available, take it (it ends the episode
            # cheaply); this is the return-maximizing behavior under time_min
            mask = env.available_actions()
            r, c = env.pos
            exit_a = None
            for a, (dr, dc) in enumerate(ACTIONS):
                if not env._in_grid(r + dr, c + dc) and mask[a]:
                    exit_a = a
                    break
            a = exit_a if exit_a is not None else int(np.argmax(mask))
            _, _, done, info = env.step(a)
            if done:
                exited += info["outcome"] == "exited"
                break
    print(f"[open] early-exit policy exit rate: {exited/500:.2%} (mechanism available)")

    # 3. return comparison under time_min on open: exiting beats completing
    def rollout_return(policy, boundary, reward, seed, od):
        env = GridRouteEnv(boundary=boundary, reward=reward, seed=seed)
        env.reset(od=od)
        total = 0.0
        for _ in range(env.max_steps):
            r, c = env.pos
            if policy == "exit":
                mask = env.available_actions()
                a = None
                for aa, (dr, dc) in enumerate(ACTIONS):
                    if not env._in_grid(r + dr, c + dc) and mask[aa]:
                        a = aa; break
                if a is None:
                    a = int(np.argmax(mask))
            else:  # shortest path
                best, a = 1e9, 0
                for aa, (dr, dc) in enumerate(ACTIONS):
                    nr, nc = r + dr, c + dc
                    if env._in_grid(nr, nc):
                        d = abs(nr - env.dst[0]) + abs(nc - env.dst[1])
                        if d < best:
                            best, a = d, aa
            _, rew, done, _ = env.step(a)
            total += rew
            if done:
                break
        return total
    od = ((0, 0), (4, 4))
    g_exit = rollout_return("exit", "open", "time_min", 3, od)
    g_path = rollout_return("path", "open", "time_min", 3, od)
    print(f"[open/time_min] return: exit={g_exit:.2f}  complete={g_path:.2f} "
          f"-> exit_better={g_exit > g_path} (expect True: pathology present)")
