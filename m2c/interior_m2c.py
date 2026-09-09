# -*- coding: utf-8 -*-
"""M2-C step 3b: the interior-destination control and the 13x13 bracket, under both shaping forms.

Derived from repo_v11/boundary_open_demo/interior_deep_control.py, which is read and not modified.
That script already takes the potential to vanish on the exit transition, so the M2 change reaches
it only through the discount: published keeps beta*(Phi(n_t) - Phi(n_{t+1})) and m2 uses
beta*(Phi(n_t) - gamma*Phi(n_{t+1})), with the return discounted in both.
"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "repo_v11", "boundary_open_demo"))
from interior_destination import BETA, R_G, R_X          # noqa: E402
from interior_deep_control import eval_od, completion_is_cheaper   # noqa: E402  the published draw

from env import ACTIONS                                  # noqa: E402  the published order
E_COST = 1.0 + 0.6 * 0.5 * 0.5
NEG = -1e18


def solve(goal, n_side, boundary, reward, shaping="published", gamma=1.0, iters=8000, tol=1e-12):
    g = gamma if shaping in ("m2", "c") else 1.0

    def phi(r, c):
        return abs(r - goal[0]) + abs(c - goal[1])

    open_border = (boundary == "open")
    V = np.zeros((n_side, n_side))
    for _ in range(iters):
        Vn = np.full((n_side, n_side), NEG)
        for r in range(n_side):
            for c in range(n_side):
                if (r, c) == goal:
                    Vn[r, c] = 0.0
                    continue
                best = NEG
                for dr, dc in ACTIONS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n_side and 0 <= nc < n_side:
                        rew = -E_COST
                        if reward == "aligned":
                            rew += BETA * (phi(r, c) - g * phi(nr, nc))
                            if (nr, nc) == goal:
                                rew += R_G
                        best = max(best, rew + gamma * V[nr, nc])
                    elif open_border:
                        rew = -E_COST - (R_X if reward == "aligned" else 0.0)
                        if reward == "aligned":
                            # published and m2 collect the whole potential here; C keeps
                            # Phi(exit) = Phi(n_t), which leaves beta*(1-gamma)*Phi
                            rew += BETA * ((1.0 - gamma) * phi(r, c) if shaping == "c"
                                           else phi(r, c))
                        best = max(best, rew)
                Vn[r, c] = best
        if np.max(np.abs(Vn - V)) < tol:
            V = Vn
            break
        V = Vn
    return V


def arrives(o, goal, V, n_side, boundary, reward, shaping="published", gamma=1.0):
    g = gamma if shaping in ("m2", "c") else 1.0

    def phi(r, c):
        return abs(r - goal[0]) + abs(c - goal[1])

    open_border = (boundary == "open")
    pos = o
    for _ in range(8 * n_side):
        if pos == goal:
            return True
        r, c = pos
        best, arg = NEG, None
        for dr, dc in ACTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n_side and 0 <= nc < n_side:
                rew = -E_COST
                if reward == "aligned":
                    rew += BETA * (phi(r, c) - g * phi(nr, nc))
                    if (nr, nc) == goal:
                        rew += R_G
                q = rew + gamma * V[nr, nc]
                if q > best:
                    best, arg = q, (nr, nc)
            elif open_border:
                rew = -E_COST - (R_X if reward == "aligned" else 0.0)
                if reward == "aligned":
                    rew += BETA * ((1.0 - gamma) * phi(r, c) if shaping == "c" else phi(r, c))
                if rew > best:
                    best, arg = rew, None
        if arg is None:
            return False
        pos = arg
    return False


def make_od(n_side, margin=4, draws=200, seed=12345):
    """The published draw of interior_deep_control.eval_od, so the regression is comparable."""
    return [(o, g) for o, _l, g in eval_od(draws, n_side, margin, seed=seed)]


def main():
    out = {}
    for n_side in (13,):
        od = make_od(n_side)
        for boundary in ("closed", "open"):
            for reward in ("time_min", "aligned"):
                for shaping in ("published", "m2", "c"):
                    for gamma in (1.0, 0.99, 0.95):
                        cache = {}
                        ok = 0
                        for o, g in od:
                            key = (g, boundary, reward, shaping, gamma)
                            if key not in cache:
                                cache[key] = solve(g, n_side, boundary, reward, shaping, gamma)
                            ok += arrives(o, g, cache[key], n_side, boundary, reward, shaping, gamma)
                        v = round(100.0 * ok / len(od), 1)
                        out["%s|%s|%s|%s" % (boundary, reward, shaping, gamma)] = v
                        print("side %d  %-7s %-9s %-10s gamma=%-5s  %5.1f"
                              % (n_side, boundary, reward, shaping, gamma, v), flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "interior_m2c.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
