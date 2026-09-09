# -*- coding: utf-8 -*-
"""The arrival ceiling on boundary-open Anaheim, set by reachability rather than by the reward.

Opening the perimeter turns every border node other than the destination into an absorbing exit.
A pair can therefore arrive only where a route to the destination avoids every other border node.
This computes that share over the same 12,183 pairs the value iteration scores.
"""
import sys, io, heapq
sys.path.insert(0, "/home/dhlee/review_paper/handoff/experiments")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
src = open("/home/dhlee/review_paper/handoff/experiments/anaheim_vi_perimeter.py", encoding="utf-8").read()
ns = {"__file__": "/home/dhlee/review_paper/handoff/experiments/anaheim_vi_perimeter.py"}
exec(compile(src[:src.index("def solve(")], "h", "exec"), ns)
BORDER, ORIGINS, DESTS, RADJ, COST = ns["BORDER"], ns["ORIGINS"], ns["DESTS"], ns["RADJ"], ns["COST"]

def reach(dest, blocked):
    d = {dest}; q = [dest]
    while q:
        u = q.pop()
        if u in blocked and u != dest:
            continue
        for v in RADJ.get(u, []):
            if v not in d:
                d.add(v); q.append(v)
    return d

pairs = free = 0
for z in DESTS:
    rf = reach(z, set())
    rb = reach(z, BORDER)
    for o in ORIGINS:
        if o != z and o in rf:
            pairs += 1
            if o in rb:
                free += 1
print("scored pairs %d" % pairs)
print("pairs with a route avoiding every other border node: %d (%.1f%%)" % (free, 100.0 * free / pairs))
print("pairs forced through a border node: %d (%.1f%%)" % (pairs - free, 100.0 * (pairs - free) / pairs))
