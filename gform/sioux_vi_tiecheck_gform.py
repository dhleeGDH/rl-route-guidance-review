# -*- coding: utf-8 -*-
"""Tie sensitivity of the Sioux Falls optimum of sioux_vi_gform.py.

A greedy rollout over an optimal value function has to break ties, and the completion rate is
what the tie rule decides wherever leaving and arriving carry the same return. sioux_vi_gform.py
prefers a move to the terminal action at an exact tie. This re-scores every cell with the
opposite rule, terminal action first, and counts the pairs where the two rules part.

    python3 sioux_vi_tiecheck_gform.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sioux_vi_gform as S   # noqa: E402

EPS = 1e-9


def rollout_terminal_first(o, dst, V, phi, exits, reward, cap=None):
    cap = cap or (4 * S.N)
    u = o
    for _ in range(cap):
        best, arg = S.NEG, None
        if u == dst:
            best = -1.0 + (S.R_GOAL + S.BETA * phi[u] if reward == "aligned" else 0.0)
            arg = "ARR"
        elif u in exits:
            best = -1.0 - (S.R_EXIT if reward == "aligned" else 0.0)
            arg = "EXIT"
        for v in S.ADJ[u]:
            rw = -S.E_COST + (S.BETA * (phi[u] - phi[v]) if reward == "aligned" else 0.0)
            if rw + V[v] > best + EPS:            # a move has to beat the terminal action
                best, arg = rw + V[v], v
        if arg == "ARR":
            return True
        if arg in ("EXIT", None):
            return False
        u = arg
    return False


def main():
    ods = S.B.make_eval_od(S.NET)
    out = {"tie_rule": "terminal action preferred at an exact tie", "cells": {}}
    print("Sioux Falls, optimum re-scored with the opposite tie rule")
    for boundary in ("closed", "open"):
        for reward in ("time_min", "aligned"):
            cache, a_move, a_term, differ = {}, 0, 0, 0
            for o, d in ods:
                key = (d, boundary, reward)
                if key not in cache:
                    cache[key] = S.solve(d, boundary, reward)
                V, phi, ex = cache[key]
                m = S.rollout(o, d, V, phi, ex, reward)[0]
                t = rollout_terminal_first(o, d, V, phi, ex, reward)
                a_move += m
                a_term += t
                differ += (m != t)
            lab = "%s %s" % (boundary, reward)
            out["cells"][lab] = {"move_first_pct": round(100.0 * a_move / len(ods), 1),
                                 "terminal_first_pct": round(100.0 * a_term / len(ods), 1),
                                 "pairs_that_differ": differ, "draws": len(ods)}
            print("  %-18s move-first %5.1f%%   terminal-first %5.1f%%   pairs differing %d"
                  % (lab, 100.0 * a_move / len(ods), 100.0 * a_term / len(ods), differ))
    json.dump(out, open(os.path.join(HERE, "sioux_vi_tiecheck_gform.json"), "w"), indent=1)
    print("wrote sioux_vi_tiecheck_gform.json")


if __name__ == "__main__":
    main()
