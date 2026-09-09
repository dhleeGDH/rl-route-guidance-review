# -*- coding: utf-8 -*-
"""Arrival-reachable share of the retrained cell's own evaluation set.

Table XI's trip-completion item asks a study reporting a boundary-open completion rate to give
"the share of evaluation pairs able to arrive at all under that condition". Round 58 observed
that the retrained released implementation reported 0.0% arrival on the opened perimeter without
that share, which is the manuscript failing its own checklist on its own cell.

The quantity is computed here from the same border set, the same RNG and the same destination the
retraining uses, so it describes that cell and no other. A pair can arrive only where some route
from its origin to the destination enters no border node other than the destination itself.

    python3 arrival_reachable_eval_set.py

Result on 2026-08-19: 28 of the 30 pairs, 93.3%.
"""
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

import argparse as _ap
_A = _ap.ArgumentParser()
_A.add_argument("--band", type=float, default=0.10,
                help="border band as a fraction of the bounding box; 0 is the convex hull alone")
_ARGS, _ = _A.parse_known_args()
sys.argv = [sys.argv[0]]
import retrain_released as R

# Resolved at import: R.build() chdirs into the released repository, after which a relative
# __file__ resolves against that directory instead of this one.
OUT = Path(__file__).resolve().parent / ("arrival_reachable_eval_set.json" if _ARGS.band == 0.10
       else "arrival_reachable_eval_set_band%s.json" % ("hull" if _ARGS.band == 0 else str(_ARGS.band)))

DEST, N_ORIGINS, SEED, MAP = 15, 30, 12345, 2


def main():
    exits = R.border_nodes(Path(R.REPO) / "Maps" / "Anaheim" / "anaheim_nodes.geojson", band=_ARGS.band)
    m = R.build(MAP, DEST)
    rng = np.random.RandomState(SEED)
    pool = [x for x in range(1, m.n_node + 1) if x != DEST and x not in exits]
    origins = sorted(rng.choice(pool, size=min(N_ORIGINS, len(pool)), replace=False).tolist())

    adj = {}
    for u, v in m.G.edges():
        adj.setdefault(int(u), []).append(int(v))
    absorbing = {int(e) for e in exits} - {DEST}

    def reaches(o):
        if o in absorbing:
            return False
        seen, q = {o}, deque([o])
        while q:
            u = q.popleft()
            for v in adj.get(u, []):
                if v == DEST:
                    return True
                if v in absorbing or v in seen:
                    continue
                seen.add(v)
                q.append(v)
        return False

    ok = [o for o in origins if reaches(o)]
    print("network              : %d nodes, %d links" % (m.n_node, m.n_link))
    print("border nodes         : %d" % len(exits))
    print("destination          : %d" % DEST)
    print("origins              : %d" % len(origins))
    print("able to arrive at all: %d of %d (%.1f%%)" % (len(ok), len(origins),
                                                        100.0 * len(ok) / len(origins)))
    # A regression witness, in the manner of the other runners in this package.
    assert (len(origins), len(ok)) == (30, 28), (len(origins), len(ok))
    # Published so the manuscript's checks read the value rather than restating it.
    out = {"network": "Anaheim", "nodes": m.n_node, "links": m.n_link,
           "border_nodes": len(exits), "destination": DEST,
           "origins": len(origins), "arrival_reachable": len(ok),
           "ceiling_pct": round(100.0 * len(ok) / len(origins), 1),
           "unreachable_origins": sorted(set(origins) - set(ok))}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("\nOK: the published share is 28 of 30, written to arrival_reachable_eval_set.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
