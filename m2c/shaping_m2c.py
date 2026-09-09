# -*- coding: utf-8 -*-
"""M2-C: the discount-consistent shaping form with the published exit-terminal convention.

The published reward adds an UNDISCOUNTED difference of potentials on an in-grid move,

    r_t = -c(a_t, t) + beta * (Phi(n_t) - Phi(n_{t+1}))  [+ R_g on arrival, -R_x on an exit]

which telescopes to a constant only at gamma = 1. T-1901 replaces it with

    r_t = -c(a_t, t) + beta * (Phi(n_t) - gamma * Phi(n_{t+1}))

and takes Phi = 0 at arrival alone. T-1901 also took Phi = 0 at an exit, which made the shaping
policy-invariant, moved no optimum and left the learner without a signal; that variant is kept as a
control. Here the exit terminal keeps the published convention, Phi(exit terminal) = Phi(n_t), so the
exit transition carries beta * (1 - gamma) * Phi(n_t) and the potential of the node left behind is
still forfeited. A step-budget truncation keeps the ordinary increment, as in T-1901.

Nothing in repo_v11/ is modified. Each class here subclasses the published environment and adds
the difference between the new increment and the published one, so the published transition logic,
costs, geometry and terminal bookkeeping are used unchanged:

    in-grid move, stay in place, detour, truncation : + beta * (1 - gamma) * Phi(n_{t+1})
    arrival                                         : 0            (Phi already vanishes there)
    exit                                            : + beta * (1 - gamma) * Phi(n_t)

At gamma = 1 every correction vanishes, so the wrapper reproduces the published environment at its
default exactly. That identity is the regression baseline of this ticket.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.join(ROOT, "repo_v11")
for _p in ("boundary_open_demo", "benchmark_network", "sumo_corridor"):
    _d = os.path.join(REPO, _p)
    if _d not in sys.path:
        sys.path.insert(0, _d)


class DiscountConsistentShapingC(object):
    """Mixin: the discount-consistent increment, with the published exit-terminal convention."""

    gamma_shaping = 0.99

    def _phi_at_pos(self):
        pos = self.pos
        if isinstance(pos, tuple):
            return self._phi(*pos)
        return self._phi(pos)

    def step(self, a):
        aligned = getattr(self, "reward", None) == "aligned"
        scale = getattr(self, "align_strength", 1.0) * getattr(self, "beta", 0.0)
        phi_before = self._phi_at_pos() if aligned else 0.0
        obs, reward, done, info = super(DiscountConsistentShapingC, self).step(a)
        if aligned and scale:
            outcome = info.get("outcome")
            if outcome == "arrived":
                pass                                   # Phi(terminal) = 0 already
            elif outcome == "exited":
                # Phi(exit terminal) = Phi(n_t), so the increment is beta*(1-gamma)*Phi(n_t) and the
                # potential of the node left behind stays forfeited, as in the published reward.
                reward += scale * (1.0 - self.gamma_shaping) * phi_before
            else:
                reward += scale * (1.0 - self.gamma_shaping) * self._phi_at_pos()
        return obs, reward, done, info


def make(cls, gamma=0.99, **kw):
    """A discount-consistent subclass of a published environment class, variant C."""
    sub = type("M2C" + cls.__name__, (DiscountConsistentShapingC, cls), {"gamma_shaping": gamma})
    return sub(**kw)
