# -*- coding: utf-8 -*-
# T-2068 (2026-09-16). The bespoke grid under the discount-carrying shaping of Eq. (4).
#
# Three things, all on the 5x5 grid of Fig. 1 and all at a discount:
#   A. the right-hand side of Eq. (7) as the body writes it, 1 - Delta/[(K - k_min) R_d], with
#      Delta the smallest positive gap in the DISCOUNTED travel-time return and K, k_min the
#      extreme transition counts of the two paths;
#   B. the 396 ordered OD pairs at graph distance 3 or more on the boundary-closed grid, greedy
#      route under each reward, under both shaping forms;
#   C. optimal completion by OD graph distance on the boundary-open grid, under both forms.
#
# The in-grid traversal cost is the deterministic surrogate the deposited value iteration uses,
# E_COST = 1 + 0.6*0.5*0.5 = 1.15, and the arrival link costs 1.0. Writes grid_gform.json here.
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNTREE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..",
                                       "repo_v11", "boundary_open_demo"))
sys.path.insert(0, RUNTREE)
sys.path.insert(0, HERE)

from env_boundary_dest import make_eval_od                                  # noqa: E402
import optimal_vi_boundary_dest_gform as G                                  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "boundary_open_demo"))
import importlib.util                                                        # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "published_vi", os.path.join(os.path.dirname(HERE), "boundary_open_demo",
                                 "optimal_vi_boundary_dest.py"))
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

N = 5
E_COST = 1.0 + 0.6 * 0.5 * 0.5
R_D_BODY, BETA = 10.0, 1.0
C_MIN = 1.0
TOL = 1e-9
GAMMAS_A = [0.90, 0.95, 0.99, 0.999, 0.9999]
GAMMAS_B = [1.0, 0.995, 0.99, 0.98, 0.95, 0.90]
GAMMAS_C = [1.0, 0.99, 0.95, 0.90]
NODES = [(r, c) for r in range(N) for c in range(N)]


def nbrs(u):
    r, c = u
    return [(r + a, c + b) for a, b in ((-1, 0), (0, 1), (1, 0), (0, -1))
            if 0 <= r + a < N and 0 <= c + b < N]


def man(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def vi(dest, gamma, shaped, g_form):
    """Value iteration on the boundary-closed grid. shaped=False is Eq. (3)."""
    V = {u: -1e18 for u in NODES}
    V[dest] = 0.0
    for _ in range(4 * len(NODES)):
        stable = True
        for u in NODES:
            if u == dest:
                continue
            best = -1e18
            for v in nbrs(u):
                r = -E_COST
                if shaped:
                    r += BETA * (man(u, dest) - (gamma if g_form else 1.0) * man(v, dest))
                    if v == dest:
                        r += R_D_BODY
                q = r + (0.0 if v == dest else gamma * V[v])
                if q > best:
                    best = q
            if best > V[u] + 1e-12:
                V[u] = best
                stable = False
        if stable:
            break
    return V


def greedy(u, dest, V, gamma, shaped, g_form):
    route, cur, seen = [u], u, set()
    while cur != dest and len(route) <= 4 * len(NODES):
        if cur in seen:
            return None
        seen.add(cur)
        best, arg = -1e18, None
        for v in sorted(nbrs(cur)):
            r = -E_COST
            if shaped:
                r += BETA * (man(cur, dest) - (gamma if g_form else 1.0) * man(v, dest))
                if v == dest:
                    r += R_D_BODY
            q = r + (0.0 if v == dest else gamma * V[v])
            if q > best + 1e-12:
                best, arg = q, v
        if arg is None:
            return None
        route.append(arg)
        cur = arg
    return route if cur == dest else None


def part_a(gamma):
    delta, K, k_min = None, 0, 10 ** 9
    for dest in NODES:
        V = vi(dest, gamma, False, False)
        kopt = {}
        for u in NODES:
            rt = greedy(u, dest, V, gamma, False, False)
            kopt[u] = None if rt is None else len(rt) - 1
        for u in NODES:
            if u == dest or kopt[u] is None:
                continue
            qs = [(v, -E_COST + (0.0 if v == dest else gamma * V[v])) for v in nbrs(u)]
            top = max(q for _, q in qs)
            for v, q in qs:
                gap = top - q
                if gap <= TOL:
                    continue
                kd = 1 if v == dest else (None if kopt[v] is None else kopt[v] + 1)
                if kd is None:
                    continue
                K = max(K, kopt[u], kd)
                k_min = min(k_min, kopt[u], kd)
                if delta is None or gap < delta:
                    delta = gap
    if delta is None:
        return {"gamma": gamma, "delta": None, "K": None, "k_min": None, "R_d": R_D_BODY,
                "eq7_rhs": None, "condition_gamma_gt_rhs": None}
    span = K - k_min
    rhs = 1.0 - delta / (span * R_D_BODY) if span > 0 else float("-inf")
    return {"gamma": gamma, "delta": delta, "K": K, "k_min": k_min, "R_d": R_D_BODY,
            "eq7_rhs": rhs, "condition_gamma_gt_rhs": bool(gamma > rhs)}


def part_b(gamma, g_form):
    pairs = same = 0
    for dest in NODES:
        Vt = vi(dest, gamma, False, False)
        Va = vi(dest, gamma, True, g_form)
        for o in NODES:
            if man(o, dest) < 3:
                continue
            pairs += 1
            a = greedy(o, dest, Vt, gamma, False, False)
            b = greedy(o, dest, Va, gamma, True, g_form)
            if a is not None and a == b:
                same += 1
    return pairs, round(100.0 * same / pairs, 1)


def part_c(gamma, mod, g_form):
    od = make_eval_od(n=200, seed=12345)
    out = {}
    for reward in ("time_min", "aligned"):
        cache, by = {}, {}
        for o, dl in od:
            if dl not in cache:
                cache[dl] = mod.solve(dl, "open", reward, gamma=gamma)
            V, phi, ol = cache[dl]
            k = man(o, dl[0]) + 1
            hit = mod.arrives(o, dl, V, phi, ol, reward, gamma=gamma)
            n, s = by.get(k, (0, 0))
            by[k] = (n + 1, s + (1 if hit else 0))
        out[reward] = {str(k): round(100.0 * s / n, 1) for k, (n, s) in sorted(by.items())}
        out[reward]["all"] = round(100.0 * sum(s for _, s in by.values())
                                   / sum(n for n, _ in by.values()), 1)
    return out


def main():
    res = {"A_eq7_rhs": [part_a(g) for g in GAMMAS_A],
           "A_body_printed": {"delta": 2.0 * C_MIN, "K": 10, "k_min": 3, "R_d": R_D_BODY,
                              "eq7_rhs": 1.0 - 2.0 * C_MIN / ((10 - 3) * R_D_BODY)},
           "B_route_identity": {}, "C_completion_by_distance": {}}
    print("A. Eq. (7) right-hand side on the bespoke grid, Delta measured at gamma")
    print("   %-8s %10s %4s %6s %12s  %s" % ("gamma", "Delta", "K", "k_min", "RHS", "gamma>RHS"))
    for r in res["A_eq7_rhs"]:
        print("   %-8s %10.6f %4d %6d %12.6f  %s"
              % (r["gamma"], r["delta"], r["K"], r["k_min"], r["eq7_rhs"],
                 r["condition_gamma_gt_rhs"]))
    print("   body prints Delta 2.0, K 10, k_min 3 -> %.4f" % res["A_body_printed"]["eq7_rhs"])

    print("\nB. same greedy route on the boundary-closed grid, %% of pairs")
    print("   %-8s %8s %8s" % ("gamma", "published", "gform"))
    for g in GAMMAS_B:
        n1, p1 = part_b(g, False)
        n2, p2 = part_b(g, True)
        res["B_route_identity"][str(g)] = {"pairs": n1, "published": p1, "gform": p2}
        print("   %-8s %7.1f%% %7.1f%%   (%d pairs)" % (g, p1, p2, n1))

    print("\nC. optimal completion by OD graph distance on the boundary-open grid")
    for g in GAMMAS_C:
        pub = part_c(g, P, False)
        gf = part_c(g, G, True)
        res["C_completion_by_distance"][str(g)] = {"published": pub, "gform": gf}
        for reward in ("time_min", "aligned"):
            print("   gamma %-7s %-9s published %s" % (g, reward, pub[reward]))
            print("   %-13s %-9s gform     %s" % ("", reward, gf[reward]))
    json.dump(res, io.open(os.path.join(HERE, "grid_gform.json"), "w", encoding="utf-8"), indent=1)
    print("\nwrote grid_gform.json")


if __name__ == "__main__":
    main()
