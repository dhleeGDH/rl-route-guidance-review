# -*- coding: utf-8 -*-
"""Eq. (6) at the route-set level, and Eq. (7) numerically, on the two real networks.

WHY THIS EXISTS. Cold review of v809, M7: "완성률 동일성은 Eq. (6)의 Πtt = Πda를 함의하지
않습니다." Correct. Section III-B verifies the route sets only on the 5x5 demonstration grid
("returning the same route on the 396 ordered origin-destination pairs"); on Sioux Falls and
Anaheim the manuscript checks identical optimal COMPLETION, which is a weaker statement. The same
comment asks for the Eq. (7) right-hand side on those networks, where Phi_max and K are far larger
than the grid's 8 and Delta can be far smaller.

WHAT IS COMPUTED, per network and boundary-closed:

1. Exact value iteration under each reward on the deterministic MDP of the link graph, at an
   undiscounted return and at the learner's discount of 0.99.
2. The optimal ACTION SET at every node under each reward, compared set to set with a tie
   tolerance. Identical action sets at every node imply identical induced optimal route sets,
   which is Eq. (6). Reporting a share of nodes rather than a boolean also shows where any
   divergence sits.
3. The Eq. (7) right-hand side from measured quantities: Delta the smallest positive action gap
   under the travel-time reward, K and k_min the longest and shortest optimal completing route in
   transitions over the evaluation set, Phi_max the largest potential.

The reward follows Eq. (4) as env_boundary_dest.py implements it: a per-step traversal cost, a
potential difference beta*(Phi(u) - Phi(v)) at every step with Phi zero at the destination, and an
arrival bonus R_g. Phi is the hop distance to the destination, which is what Appendix C states for
the published Sioux Falls drawing. The action-set identity is invariant to beta and R_g, so the
result does not rest on the magnitudes chosen; the Eq. (7) bound does, and both are reported.

    python3 eq7_route_sets.py                 # both networks, all zone destinations
    python3 eq7_route_sets.py --dests 5       # a smoke run over five destinations each

Writes eq7_route_sets.json next to this file.
"""
import argparse
import io
import json
import os
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
NETS = {"Sioux Falls": os.path.join(HERE, "networks", "SiouxFalls_net.tntp"),
        "Anaheim": os.path.join(HERE, "networks", "Anaheim_net.tntp")}
R_GOAL = 10.0            # in units of the mean traversal cost, as anaheim_vi.py sets it
BETA = 1.0               # one mean traversal cost per hop of potential
TOL = 1e-9


def read(path):
    zones = nodes = None
    links = []
    for ln in io.open(path, encoding="utf-8", errors="replace"):
        s = ln.strip()
        if s.startswith("<NUMBER OF ZONES>"):
            zones = int(s.split(">")[1])
        elif s.startswith("<NUMBER OF NODES>"):
            nodes = int(s.split(">")[1])
        elif s and not s.startswith(("~", "<")):
            f = s.rstrip(";").split()
            if len(f) >= 6:
                try:
                    links.append((int(f[0]), int(f[1]), float(f[4])))
                except ValueError:
                    pass
    return zones, nodes, links


def hop_potential(dest, radj, nodes):
    """Hop distance to the destination, the potential of Eq. (4) on a network without coordinates."""
    d = {dest: 0}
    q = deque([dest])
    while q:
        u = q.popleft()
        for p in radj.get(u, ()):
            if p not in d:
                d[p] = d[u] + 1
                q.append(p)
    far = max(d.values()) + 1 if d else 1
    return {n: d.get(n, far) for n in range(1, nodes + 1)}, far


def value_iteration(dest, adj, cost, nodes, gamma, phi, beta, r_goal):
    """Backward induction on the deterministic MDP, boundary closed so the destination is the
    only terminal. phi=None gives the travel-time reward of Eq. (3)."""
    V = {n: -1e18 for n in range(1, nodes + 1)}
    V[dest] = 0.0
    for _ in range(nodes + 2):
        stable = True
        for u in range(1, nodes + 1):
            if u == dest:
                continue
            best = -1e18
            for v in adj.get(u, ()):
                r = -cost[(u, v)]
                if phi is not None:
                    r += beta * (phi[u] - phi[v])
                    if v == dest:
                        r += r_goal
                nxt = r + (0.0 if v == dest else gamma * V[v])
                if nxt > best:
                    best = nxt
            if best > V[u] + 1e-12:
                V[u] = best
                stable = False
        if stable:
            break
    return V


def argmax_set(u, V, adj, cost, dest, gamma, phi, beta, r_goal):
    vals = []
    for v in adj.get(u, ()):
        r = -cost[(u, v)]
        if phi is not None:
            r += beta * (phi[u] - phi[v])
            if v == dest:
                r += r_goal
        vals.append((v, r + (0.0 if v == dest else gamma * V[v])))
    if not vals:
        return frozenset()
    top = max(q for _, q in vals)
    return frozenset(v for v, q in vals if q > top - 1e-7)


def rollout_cost(o, dest, V, adj, cost, gamma, phi, beta, r_goal, nodes):
    """Traversal cost actually paid along the greedy optimal route, or None where it does not
    complete. Ties are broken on the lowest node index, identically under both rewards, so a
    difference in cost is a difference the rewards produce rather than a difference in tie-breaking."""
    cur, paid, seen = o, 0.0, set()
    for _ in range(nodes + 2):
        if cur == dest:
            return paid
        if cur in seen or cur not in adj:
            return None
        seen.add(cur)
        best, arg = -1e18, None
        for v in sorted(adj[cur]):
            r = -cost[(cur, v)]
            if phi is not None:
                r += beta * (phi[cur] - phi[v])
                if v == dest:
                    r += r_goal
            q = r + (0.0 if v == dest else gamma * V[v])
            if q > best + 1e-12:
                best, arg = q, v
        paid += cost[(cur, arg)]
        cur = arg
    return None


def analyse(name, path, n_dests, gamma):
    zones, nodes, links = read(path)
    adj, radj, cost = {}, {}, {}
    for u, v, c in links:
        c = c if c > 0 else 1e-3
        adj.setdefault(u, []).append(v)
        radj.setdefault(v, []).append(u)
        cost[(u, v)] = c
    scale = sum(cost.values()) / len(cost)
    beta, r_goal = BETA * scale, R_GOAL * scale

    dests = list(range(1, zones + 1))
    if n_dests:
        dests = dests[:n_dests]

    same = diff = 0
    diverging = []
    gaps, ks, phimaxes = [], [], []
    tt_diff, rel, pairs, incomplete = [], [], [0], [0]
    for dest in dests:
        phi, far = hop_potential(dest, radj, nodes)
        reach = [n for n in range(1, nodes + 1) if phi[n] < far and n != dest]
        Vtt = value_iteration(dest, adj, cost, nodes, gamma, None, beta, r_goal)
        Vda = value_iteration(dest, adj, cost, nodes, gamma, phi, beta, r_goal)
        for u in reach:
            a = argmax_set(u, Vtt, adj, cost, dest, gamma, None, beta, r_goal)
            b = argmax_set(u, Vda, adj, cost, dest, gamma, phi, beta, r_goal)
            if a == b:
                same += 1
            else:
                diff += 1
                if len(diverging) < 12:
                    diverging.append({"dest": dest, "node": u,
                                      "travel_time": sorted(a), "aligned": sorted(b)})
        # Delta, K, k_min and Phi_max are properties of the UNDISCOUNTED problem: Eq. (7) asks
        # whether the undiscounted travel-time ordering survives a discount, so its return gap is
        # the undiscounted one. Computing Delta at the gamma under test gave Sioux Falls 1.0000 at
        # gamma 1 and 0.0099 at gamma 0.99, and so two different thresholds for one network.
        Vtt1 = Vtt if gamma == 1.0 else value_iteration(dest, adj, cost, nodes, 1.0, None,
                                                        beta, r_goal)
        for u in reach:
            best = max(-cost[(u, v)] + (0.0 if v == dest else Vtt1[v]) for v in adj.get(u, ()))
            for v in adj.get(u, ()):
                q = -cost[(u, v)] + (0.0 if v == dest else Vtt1[v])
                if best - q > TOL:
                    gaps.append(best - q)
        ks += [phi[n] for n in reach]
        phimaxes.append(max(phi[n] for n in reach))
        # The decisive quantity for the paper is not the action set but the travel time a reader
        # would see. A node where the two optimal action sets differ still yields the same travel
        # time wherever the alternatives cost the same.
        for u in reach:
            ctt = rollout_cost(u, dest, Vtt, adj, cost, gamma, None, beta, r_goal, nodes)
            cda = rollout_cost(u, dest, Vda, adj, cost, gamma, phi, beta, r_goal, nodes)
            if ctt is None or cda is None:
                incomplete[0] += 1
                continue
            pairs[0] += 1
            d = abs(ctt - cda)
            if d > 1e-9:
                tt_diff.append(d)
                rel.append(d / ctt)

    delta = min(gaps) if gaps else float("nan")
    K, k_min, phi_max = max(ks), min(ks), max(phimaxes)
    denom = (K - k_min) * r_goal + (K - 1) * beta * phi_max
    threshold = 1.0 - delta / denom if denom > 0 else 0.0
    return {"network": name, "zones": zones, "nodes": nodes, "links": len(links),
            "destinations_solved": len(dests), "gamma": gamma,
            "mean_traversal_cost": scale, "beta": beta, "r_goal": r_goal,
            "nodes_with_identical_optimal_action_set": same,
            "nodes_differing": diff,
            "share_identical": 100.0 * same / (same + diff) if same + diff else float("nan"),
            "examples_of_divergence": diverging,
            "delta_smallest_positive_action_gap": delta,
            "K_longest_hop": K, "k_min_shortest_hop": k_min, "phi_max": phi_max,
            "eq7_threshold": threshold,
            # the record holds what the manuscript prints, so a rounded figure is traceable
            "eq7_threshold_printed": float("%.6f" % threshold),
            "delta_printed": float("%.4f" % delta),
            "eq7_satisfied_at_0.99": bool(0.99 > threshold),
            "od_pairs_completing_under_both": pairs[0],
            "od_pairs_not_completing": incomplete[0],
            "pairs_with_a_different_travel_time": len(tt_diff),
            "share_pairs_with_a_different_travel_time":
                100.0 * len(tt_diff) / pairs[0] if pairs[0] else float("nan"),
            "max_absolute_travel_time_difference": max(tt_diff) if tt_diff else 0.0,
            "max_relative_travel_time_difference": max(rel) if rel else 0.0,
            "mean_relative_travel_time_difference_over_differing_pairs":
                (sum(rel) / len(rel)) if rel else 0.0}


def raw_penalty_vi(dest, adj, cost, nodes, phi, beta, r_goal):
    """Value iteration under a NON potential-based shaping, -beta*Phi(v) charged at every step."""
    V = {n: -1e18 for n in range(1, nodes + 1)}
    V[dest] = 0.0
    for _ in range(nodes + 2):
        stable = True
        for u in range(1, nodes + 1):
            if u == dest:
                continue
            best = -1e18
            for v in adj.get(u, ()):
                r = -cost[(u, v)] - beta * phi[v] + (r_goal if v == dest else 0.0)
                nxt = r + (0.0 if v == dest else V[v])
                if nxt > best:
                    best = nxt
            if best > V[u] + 1e-12:
                V[u] = best
                stable = False
        if stable:
            break
    return V


def raw_penalty_argmax(u, V, adj, cost, dest, phi, beta, r_goal):
    vals = [(v, -cost[(u, v)] - beta * phi[v] + (r_goal if v == dest else 0.0)
             + (0.0 if v == dest else V[v])) for v in adj.get(u, ())]
    if not vals:
        return frozenset()
    top = max(q for _, q in vals)
    return frozenset(v for v, q in vals if q > top - 1e-7)


def negative_control():
    """The identity at gamma = 1 must be a property of Eq. (4), not of the comparison code.

    The control must use shaping that is NOT potential-based. A first attempt added a constant to
    Phi and diverged at 0 nodes, which is correct rather than a bug: the telescoped sum is
    Phi(o) - Phi(dest), a per-origin constant, so ANY node-function potential leaves the optimal
    action set untouched at gamma = 1. That is the theorem being verified. Breaking it needs a term
    outside the difference form, here a raw distance penalty -beta*Phi(v) charged at each step.
    """
    zones, nodes, links = read(NETS["Sioux Falls"])
    adj, radj, cost = {}, {}, {}
    for u, v, c in links:
        c = c if c > 0 else 1e-3
        adj.setdefault(u, []).append(v)
        radj.setdefault(v, []).append(u)
        cost[(u, v)] = c
    scale = sum(cost.values()) / len(cost)
    beta, r_goal = BETA * scale, R_GOAL * scale
    bad = 0
    for dest in range(1, zones + 1):
        phi, far = hop_potential(dest, radj, nodes)
        Vtt = value_iteration(dest, adj, cost, nodes, 1.0, None, beta, r_goal)
        Vbk = raw_penalty_vi(dest, adj, cost, nodes, phi, beta, r_goal)
        for u in (n for n in range(1, nodes + 1) if phi[n] < far and n != dest):
            a = argmax_set(u, Vtt, adj, cost, dest, 1.0, None, beta, r_goal)
            b = raw_penalty_argmax(u, Vbk, adj, cost, dest, phi, beta, r_goal)
            if a != b:
                bad += 1
    print("  negative control: shaping outside the difference form diverges at %d node(s)"
          % bad, flush=True)
    if bad == 0:
        raise SystemExit("NEGATIVE CONTROL FAILED: the comparison reports identity for a reward "
                         "that cannot share the optima. The check detects nothing.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dests", type=int, default=0, help="limit destinations, 0 for every zone")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--out", default="eq7_route_sets.json")
    a = ap.parse_args()
    negative_control()
    out = []
    for name, path in NETS.items():
        r = analyse(name, path, a.dests, a.gamma)
        out.append(r)
        print("%-12s gamma %.2f  identical action set at %d of %d nodes (%.1f%%)"
              % (name, a.gamma, r["nodes_with_identical_optimal_action_set"],
                 r["nodes_with_identical_optimal_action_set"] + r["nodes_differing"],
                 r["share_identical"]), flush=True)
        print("             Delta %.4f  K %d  k_min %d  Phi_max %d  ->  Eq. (7) threshold %.6f"
              "   0.99 satisfies it: %s"
              % (r["delta_smallest_positive_action_gap"], r["K_longest_hop"],
                 r["k_min_shortest_hop"], r["phi_max"], r["eq7_threshold"],
                 r["eq7_satisfied_at_0.99"]), flush=True)
        print("             travel time differs on %d of %d completing pairs (%.2f%%), "
              "largest relative difference %.3f%%"
              % (r["pairs_with_a_different_travel_time"], r["od_pairs_completing_under_both"],
                 r["share_pairs_with_a_different_travel_time"],
                 100.0 * r["max_relative_travel_time_difference"]), flush=True)
    with io.open(os.path.join(HERE, a.out), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
