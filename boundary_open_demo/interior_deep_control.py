# -*- coding: utf-8 -*-
"""Does the separation survive where completing the trip is the cheaper route?

WHY THIS EXISTS. The reported cells put the destination
on a peripheral link and draw origins at least three links from it, on lattices whose deepest node
lies 2 or 3 links from the border. A destination at the border is reached by reaching the border,
so the nearest exit is never farther than the destination and the travel-time optimum is an exit
by construction. The 0.0% completion of the travel-time arm therefore carries little information,
and the interior-destination control of Supplementary S-I.E does not repair it: measured on its own
5x5 evaluation set, the destination is nearer than any exit on 0.0% of the drawn pairs.

WHAT THIS MEASURES. A 13x13 lattice, with both the destination and the origin drawn at least four
links inside the border. On that set the destination is nearer than the nearest exit on a large
share of the pairs, which is the geometry where Proposition 2 predicts that an exit does NOT
dominate. The prediction is quantitative: the travel-time optimum should complete about the share
of pairs on which completion is the cheaper route, rather than 0.0%.

The exact optimum accompanies every learned rate, as Section VI-A asks of every reported rate.

    python3 interior_deep_control.py [--n-side 13] [--margin 4] [--episodes 8000] [--seeds 10]

Writes interior_deep_control.json next to this file.
"""
import argparse, io, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch                                                    # noqa: E402
torch.set_num_threads(1)                                        # feedback_pin_torch_threads

from env import ACTIONS, perimeter_links                        # noqa: E402
from interior_destination import InteriorDestEnv, BETA, R_G, R_X, evaluate  # noqa: E402
from dqn import DQNAgent                                        # noqa: E402

E_COST = 1.0 + 0.6 * 0.5 * 0.5      # 1.15, the surrogate cost of optimal_vi_interior
NEG = -1e9


def eval_od(n_pairs, n_side, margin, seed):
    """Both endpoints at least `margin` links inside the border, separated by at least three."""
    rng = np.random.RandomState(seed)
    cells = [(r, c) for r in range(margin, n_side - margin) for c in range(margin, n_side - margin)]
    links = perimeter_links(n_side)
    out = []
    while len(out) < n_pairs:
        o = cells[rng.randint(len(cells))]
        g = cells[rng.randint(len(cells))]
        if abs(o[0] - g[0]) + abs(o[1] - g[1]) < 3:
            continue
        out.append((o, links[0], g))
    return out


def completion_is_cheaper(od, n_side):
    """Share of pairs where the destination is strictly nearer than the nearest exit move."""
    n = 0
    for o, _link, g in od:
        d_goal = abs(o[0] - g[0]) + abs(o[1] - g[1])
        d_exit = min(o[0], o[1], n_side - 1 - o[0], n_side - 1 - o[1]) + 1
        if d_goal < d_exit:
            n += 1
    return 100.0 * n / len(od)


def solve(goal, n_side, boundary, reward, iters=8000, tol=1e-12):
    """Exact value iteration on the surrogate cost, with the destination an interior cell."""
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
                            rew += BETA * (phi(r, c) - phi(nr, nc))
                            if (nr, nc) == goal:
                                rew += R_G
                        best = max(best, rew + V[nr, nc])
                    elif open_border:
                        rew = -E_COST - (R_X if reward == "aligned" else 0.0)
                        if reward == "aligned":
                            rew += BETA * (phi(r, c) - 0)   # the potential is not collected off-grid
                        best = max(best, rew)               # leaving ends the episode
                Vn[r, c] = best
        if np.max(np.abs(Vn - V)) < tol:
            V = Vn
            break
        V = Vn
    return V


def arrives(o, goal, V, n_side, boundary, reward):
    """Does the greedy policy of V reach the goal from o?"""
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
                    rew += BETA * (phi(r, c) - phi(nr, nc))
                    if (nr, nc) == goal:
                        rew += R_G
                q = rew + V[nr, nc]
                if q > best:
                    best, arg = q, (nr, nc)
            elif open_border:
                rew = -E_COST - (R_X if reward == "aligned" else 0.0)
                if reward == "aligned":
                    rew += BETA * phi(r, c)
                if rew > best:
                    best, arg = rew, None
        if arg is None:
            return False                     # the optimum leaves the network
        pos = arg
    return False


class DeepInteriorEnv(InteriorDestEnv):
    """Training draws its endpoints from the same deep interior as the evaluation set."""
    MARGIN = 4

    def reset(self, od=None):
        if od is not None:
            return super().reset(od=od)
        m, n = self.MARGIN, self.n
        cells = [(r, c) for r in range(m, n - m) for c in range(m, n - m)]
        for _ in range(500):
            o = cells[self.rng.randint(len(cells))]
            g = cells[self.rng.randint(len(cells))]
            if abs(o[0] - g[0]) + abs(o[1] - g[1]) >= 3:
                links = perimeter_links(n)
                return super().reset(od=(o, links[0], g))
        return super().reset(od=None)


def run_cell(boundary, reward, seed, episodes, od, n_side, max_steps=120):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = DeepInteriorEnv(boundary=boundary, reward=reward, seed=1000 + seed,
                          n_side=n_side, max_steps=max_steps)
    agent = DQNAgent(env.state_dim, env.n_actions, seed=seed)
    eps0, eps1 = 1.0, 0.05
    decay = int(0.6 * episodes)
    for ep in range(1, episodes + 1):
        env.reset()
        eps = max(eps1, eps0 - (eps0 - eps1) * ep / max(1, decay))
        s = env._obs()
        for _ in range(max_steps):
            mask = env.available_actions()
            a = agent.act(s, mask, eps)
            s2, r, done, _ = env.step(a)
            agent.buf.add(s, a, r, s2, float(done), env.available_actions().astype(np.float32))
            agent.learn()
            s = s2
            if done:
                break
    e = DeepInteriorEnv(boundary=boundary, reward=reward, seed=777,
                        n_side=n_side, max_steps=max_steps)
    arrived = 0
    for pair in od:
        e.reset(od=pair)
        for _ in range(max_steps):
            a = agent.act(e._obs(), e.available_actions(), eps=0.0)
            _, _, done, info = e.step(a)
            if done:
                # the environment reports the terminal state as info["outcome"]; reading a
                # key named "arrived" returned None for every trip and scored every cell 0.0%
                arrived += info["outcome"] == "arrived"
                break
    return 100.0 * arrived / len(od)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-side", type=int, default=13)
    ap.add_argument("--margin", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=8000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--vi-only", action="store_true")
    ap.add_argument("--out", default="interior_deep_control.json")
    a = ap.parse_args()
    DeepInteriorEnv.MARGIN = a.margin

    od = eval_od(200, a.n_side, a.margin, seed=12345)
    geom = completion_is_cheaper(od, a.n_side)
    print("%dx%d lattice, both endpoints at least %d links inside the border, %d draws"
          % (a.n_side, a.n_side, a.margin, len(od)))
    print("  destination nearer than the nearest exit on %.1f%% of the pairs\n" % geom)

    out = {"n_side": a.n_side, "margin": a.margin, "draws": len(od),
           "completion_cheaper_pct": geom, "episodes": a.episodes, "seeds": a.seeds,
           "optimum": {}, "learned": {}}

    for boundary in ("closed", "open"):
        for reward in ("time_min", "aligned"):
            hit = 0
            for o, _l, g in od:
                V = solve(g, a.n_side, boundary, reward)
                hit += int(arrives(o, g, V, a.n_side, boundary, reward))
            pct = 100.0 * hit / len(od)
            out["optimum"]["%s_%s" % (boundary, reward)] = pct
            print("  optimum  %-6s %-9s completes %5.1f%%" % (boundary, reward, pct), flush=True)

    if not a.vi_only:
        for boundary in ("closed", "open"):
            for reward in ("time_min", "aligned"):
                per = [run_cell(boundary, reward, s, a.episodes, od, a.n_side)
                       for s in range(a.seeds)]
                out["learned"]["%s_%s" % (boundary, reward)] = {
                    "mean": float(np.mean(per)), "sd": float(np.std(per)), "per_seed": per}
                print("  learned  %-6s %-9s completes %5.1f%% (sd %.1f)"
                      % (boundary, reward, np.mean(per), np.std(per)), flush=True)

    io.open(os.path.join(HERE, a.out), "w", encoding="utf-8").write(
        json.dumps(out, ensure_ascii=False, indent=1))
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
