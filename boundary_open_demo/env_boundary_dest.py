# -*- coding: utf-8 -*-
"""The lattice with the destination as an outgoing boundary link.

The demonstration so far placed the destination on an interior cell and treated arrival as
entering that cell. A road network does not work that way. A trip ends by leaving the modelled
area on a particular link, and the released system reconstructed in Section V-G is set up
exactly so: its destination is a boundary stub, not an interior node.

That distinction is what separates the two boundary conditions cleanly:

    boundary-closed   the destination link is the only outgoing link on the perimeter, so a
                      vehicle that wanders can leave only where it was asked to leave
    boundary-open     other outgoing links exist on the perimeter, so a vehicle can leave the
                      network at a link that is not its destination

Arrival and failure then become the same kind of event, taking an outgoing boundary link, and
differ only in which link. That is the comparison the review is about.

The origin is unconstrained and may be interior. Only the destination has to sit on the
boundary.
"""
import numpy as np

from env import ACTIONS, UNAVAIL_COST, GridRouteEnv, N_SIDE


def perimeter_links(n):
    """Every (cell, action) pair whose move leaves the lattice."""
    out = []
    for r in range(n):
        for c in range(n):
            for a, (dr, dc) in enumerate(ACTIONS):
                if not (0 <= r + dr < n and 0 <= c + dc < n):
                    out.append(((r, c), a))
    return out


class BoundaryDestEnv(GridRouteEnv):
    """GridRouteEnv with the destination expressed as an outgoing boundary link.

    exit_density selects how many of the remaining perimeter links are open on the
    boundary-open condition. It is 1.0 by default, which opens all of them.
    """

    def __init__(self, exit_density=1.0, **kw):
        self.exit_density = exit_density
        self._dst_link = None
        self._open_links = set()
        super().__init__(**kw)

    # -- geometry ----------------------------------------------------------
    def _phi(self, r, c):
        """Distance to the boundary cell whose outgoing link is the destination."""
        if self._dst_link is None:
            return 0.0
        (dr, dc), _ = self._dst_link
        return abs(r - dr) + abs(c - dc)

    def _is_dst_link(self, r, c, a):
        return self._dst_link is not None and self._dst_link == ((r, c), a)

    def _link_open(self, r, c, a):
        """Is this off-grid move an available exit on the current boundary condition?"""
        if self._is_dst_link(r, c, a):
            return True
        if self.boundary == "closed":
            return False
        return ((r, c), a) in self._open_links

    # -- episode -----------------------------------------------------------
    def reset(self, od=None):
        self._cong = self.rng.rand(self.n, self.n)
        self._t = 0
        links = perimeter_links(self.n)
        if od is None:
            while True:
                o = (self.rng.randint(self.n), self.rng.randint(self.n))
                dl = links[self.rng.randint(len(links))]
                if abs(o[0] - dl[0][0]) + abs(o[1] - dl[0][1]) >= 3:
                    break
        else:
            o, dl = od
        self.pos = o
        self._dst_link = dl
        self.dst = dl[0]                       # the boundary cell the destination link leaves from
        # the remaining perimeter links that are open under the open condition
        rest = [l for l in links if l != dl]
        if self.boundary == "closed" or self.exit_density >= 1.0:
            self._open_links = set() if self.boundary == "closed" else set(rest)
        else:
            k = int(round(self.exit_density * len(rest)))
            idx = self.rng.choice(len(rest), size=k, replace=False)
            self._open_links = {rest[i] for i in idx}
        self.done = False
        self.outcome = None
        return self._obs()

    def available_actions(self):
        r, c = self.pos
        mask = np.zeros(4, dtype=bool)
        for a, (dr, dc) in enumerate(ACTIONS):
            if self._in_grid(r + dr, c + dc):
                mask[a] = True
            elif self._link_open(r, c, a):
                mask[a] = True
        return mask

    def _obs(self):
        r, c = self.pos
        costs = []
        for a in range(4):
            ec = self._edge_cost(r, c, a)
            costs.append(UNAVAIL_COST if ec is None else ec)
        # g of Eq. (2) is an outgoing boundary link. The cell alone does not name it at the
        # four corners, which have two outgoing links each and open both on an open boundary,
        # so the arriving action and a wrong exit shared one observation. 105 of the 200
        # evaluation pairs have a corner destination. The link direction completes g.
        d = self.dst if self._dst_link else (0, 0)
        dir_g = self._dst_link[1] / 3.0 if self._dst_link else 0.0
        return np.array([r / (self.n - 1), c / (self.n - 1),
                         d[0] / (self.n - 1), d[1] / (self.n - 1), dir_g,
                         costs[0], costs[1], costs[2], costs[3]], dtype=np.float32)

    def step(self, a):
        assert not self.done
        r, c = self.pos
        dr, dc = ACTIONS[a]
        nr, nc = r + dr, c + dc
        phi_before = self._phi(r, c)

        if not self._in_grid(nr, nc):
            if self._is_dst_link(r, c, a):
                # taking the destination link is arrival
                reward = -1.0
                if self.reward == "aligned":
                    reward += self.align_strength * self.r_goal
                    reward += self.beta * (phi_before - 0.0)
                self.done = True
                self.outcome = "arrived"
                return self._obs(), reward, self.done, {"outcome": self.outcome}
            if not self._link_open(r, c, a):
                # a closed perimeter link: the move is unavailable, time passes
                reward = -1.0
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

        # ordinary in-grid move
        cost = self._edge_cost(r, c, a)
        self.pos = (nr, nc)
        reward = -cost
        if self.reward == "aligned":
            reward += self.beta * (phi_before - self._phi(nr, nc))
        self._t += 1
        if self._t >= self.max_steps:
            self.done = True
            self.outcome = "timeout"
        return self._obs(), reward, self.done, {"outcome": self.outcome}


def make_eval_od(n=200, n_side=N_SIDE, seed=12345, min_sep=3):
    """Origin cells anywhere, destinations as outgoing boundary links."""
    rng = np.random.RandomState(seed)
    links = perimeter_links(n_side)
    ods = []
    while len(ods) < n:
        o = (rng.randint(n_side), rng.randint(n_side))
        dl = links[rng.randint(len(links))]
        if abs(o[0] - dl[0][0]) + abs(o[1] - dl[0][1]) >= min_sep:
            ods.append((o, dl))
    return ods
