# -*- coding: utf-8 -*-
"""T-1903 step 3 on the bespoke grid: regression, Table V, the discount sweep, and the pair audit."""
import io, json, os, sys
import numpy as np
import vi_m2c as V

sys.path.insert(0, V.ARC)
from env_boundary_dest import make_eval_od   # noqa: E402

ROWS = [("Travel time alone",       dict(reward="time_min")),
        ("Potential shaping alone", dict(reward="aligned", beta=1.0, r_goal=0.0,  r_exit=0.0)),
        ("Exit penalty alone",      dict(reward="aligned", beta=0.0, r_goal=0.0,  r_exit=5.0)),
        ("Arrival term alone",      dict(reward="aligned", beta=0.0, r_goal=10.0, r_exit=0.0)),
        ("All three terms",         dict(reward="aligned", beta=1.0, r_goal=10.0, r_exit=5.0))]
PUB = {"Travel time alone": 0.0, "Potential shaping alone": 4.0, "Exit penalty alone": 62.5,
       "Arrival term alone": 100.0, "All three terms": 100.0}
GAMMAS = [1.0, 0.999, 0.99, 0.95]


def per_pair(terms, gamma, shaping, eval_od):
    """(arrived?, exit step h, Phi at the node left) for every pair."""
    kw = {k: v for k, v in terms.items() if k != "reward"}
    cache, out = {}, []
    for o, dl in eval_od:
        key = (dl, gamma, shaping)
        if key not in cache:
            cache[key] = V.solve(dl, "open", terms["reward"], shaping=shaping, gamma=gamma, **kw)
        Vv, phi, links = cache[key]
        out.append(walk(o, dl, Vv, phi, links, terms["reward"], shaping, gamma, kw))
    return out


def walk(o, dst_link, Vv, phi, links, reward, shaping, gamma, kw):
    """Replay the greedy policy and report the outcome, the step count and the potential left."""
    from vi_m2c import ACTIONS, E_COST, NEG, N_SIDE, R_GOAL, R_EXIT, BETA
    R_GOAL_ = kw.get("r_goal", R_GOAL); R_EXIT_ = kw.get("r_exit", R_EXIT); BETA_ = kw.get("beta", BETA)
    N = kw.get("n") or N_SIDE
    g = gamma if shaping in ("m2", "c") else 1.0
    r, c = o
    for step in range(200):
        best, tgt = NEG, None
        for a, (ar, ac) in enumerate(ACTIONS):
            nr, nc = r + ar, c + ac
            if 0 <= nr < N and 0 <= nc < N:
                rw = -E_COST + (BETA_ * (phi(r, c) - g * phi(nr, nc)) if reward == "aligned" else 0.0)
                q, t = rw + gamma * Vv[nr, nc], (nr, nc)
            elif ((r, c), a) == dst_link:
                rw = -1.0 + (R_GOAL_ + BETA_ * phi(r, c) if reward == "aligned" else 0.0)
                q, t = rw, "arrived"
            elif ((r, c), a) in links:
                rw = -1.0 - (R_EXIT_ if reward == "aligned" else 0.0)
                if reward == "aligned" and shaping == "m2":
                    rw += BETA_ * phi(r, c)
                elif reward == "aligned" and shaping == "c":
                    rw += BETA_ * (1.0 - gamma) * phi(r, c)
                q, t = rw, "exited"
            else:
                continue
            if q > best:
                best, tgt = q, t
        if tgt == "arrived":
            return (True, step + 1, phi(r, c))
        if tgt == "exited" or tgt is None:
            return (False, step + 1, phi(r, c))
        r, c = tgt
    return (False, 200, phi(r, c))


def rate(pairs):
    return round(100.0 * sum(1 for a, _, _ in pairs if a) / len(pairs), 1)


def main():
    eval_od = make_eval_od(n=200, seed=12345)
    print("evaluation set: %d pairs, boundary-open 5x5 grid\n" % len(eval_od))

    print("=== regression: variant C at gamma = 1.0 against the printed Table V column ===")
    bad = []
    for name, terms in ROWS:
        v = rate(per_pair(terms, 1.0, "c", eval_od))
        ok = abs(v - PUB[name]) <= 0.05
        bad += [] if ok else [name]
        print("  %-24s C at gamma=1.0 %5.1f   printed %5.1f   %s" % (name, v, PUB[name], "OK" if ok else "FAIL"))
    print("  regression: %s\n" % ("passed" if not bad else "FAILED " + ", ".join(bad)))

    print("=== Table V, five rows: published / T-1901 (m2) / T-1903 (C) ===")
    print("%-24s %-10s %s" % ("Reward", "variant", "  ".join("g=%-7s" % g for g in GAMMAS)))
    table = {}
    for name, terms in ROWS:
        for shaping in ("published", "m2", "c"):
            vals = [rate(per_pair(terms, g, shaping, eval_od)) for g in GAMMAS]
            table["%s|%s" % (name, shaping)] = vals
            print("%-24s %-10s %s" % (name, shaping, "  ".join("%-9.1f" % v for v in vals)))
    json.dump(table, io.open("table_v_m2c.json", "w", encoding="utf-8"), indent=1)

    print("\n=== (a) aligned optimum on the boundary-open grid, all three terms ===")
    terms = dict(reward="aligned", beta=1.0, r_goal=10.0, r_exit=5.0)
    for g in GAMMAS:
        p_pub = per_pair(terms, g, "published", eval_od)
        p_c = per_pair(terms, g, "c", eval_od)
        diff = [(i, a, b) for i, (a, b) in enumerate(zip(p_pub, p_c)) if a[0] != b[0]]
        print("  gamma=%-7s published %5.1f   C %5.1f   differing pairs %d"
              % (g, rate(p_pub), rate(p_c), len(diff)))
        for i, a, b in diff[:8]:
            o, dl = eval_od[i]
            print("     pair %3d origin %s  published %s (h=%d, Phi=%d) -> C %s (h=%d, Phi=%d)"
                  % (i, o, "arrive" if a[0] else "exit", a[1], a[2],
                     "arrive" if b[0] else "exit", b[1], b[2]))

    print("\n=== Q1 and Q2 on the grid, variant C ===")
    tt = dict(reward="time_min")
    for g in GAMMAS:
        c_al = rate([w for w in closed_pairs(terms, g, "c", eval_od)])
        c_tt = rate([w for w in closed_pairs(tt, g, "c", eval_od)])
        o_al = rate(per_pair(terms, g, "c", eval_od))
        o_tt = rate(per_pair(tt, g, "c", eval_od))
        print("  gamma=%-7s closed: tt %5.1f aligned %5.1f | open: tt %5.1f aligned %5.1f"
              % (g, c_tt, c_al, o_tt, o_al))


def closed_pairs(terms, gamma, shaping, eval_od):
    kw = {k: v for k, v in terms.items() if k != "reward"}
    cache, out = {}, []
    for o, dl in eval_od:
        key = (dl, gamma, shaping)
        if key not in cache:
            cache[key] = V.solve(dl, "closed", terms["reward"], shaping=shaping, gamma=gamma, **kw)
        Vv, phi, links = cache[key]
        out.append(walk(o, dl, Vv, phi, links, terms["reward"], shaping, gamma, kw))
    return out


if __name__ == "__main__":
    main()
