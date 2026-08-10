# -*- coding: utf-8 -*-
"""How far the headline counts can move under any admissible coding of the contestable cells.

WHY THIS EXISTS. A prevalence count extracted by one reader invites the question of whether a
second reader would produce the same number. An agreement coefficient answers that question by
proxy, on a subsample, and it answers it weakly wherever one category holds most of the corpus.
This answers the question directly and over the whole corpus instead: it enumerates every coding
the recorded evidence admits and reports the range each count can take.

The contestable cells are the ones a second reader could plausibly move, and only those:

  state axis     unclear (6)          admits either value
                 local-prediction (12) counts as forecast-conditioned only if a short-range
                                      forecast is accepted as satisfying the requirement, which
                                      is the single judgment the criterion of Section III turns on
  boundary axis  unclear (3)          admits either value
  reward scope   unclear (3)          admits either value
  action gran.   unclear (7)          admits either value

Cells with an explicit value in the source text are transcription, not judgment, and are held
fixed. A repeat pass over 28 studies moved 3 of them on reward alignment and none on code
availability, so treating them as contestable would overstate the range.

    python3 coding_invariance.py            # bounds for each headline count
    python3 coding_invariance.py --json     # machine-readable, for the manuscript checks

Reads corpus_v9_coded.csv. Writes coding_invariance.json next to this file.
"""
import argparse
import csv
import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV = HERE.parent / "corpus" / "corpus_v9_coded.csv"

N = 94


def load():
    rows = [r for r in csv.DictReader(io.open(str(CSV), encoding="utf-8"))
            if str(r.get("in_reviewed_corpus", "")).strip().lower() in ("1", "true", "yes", "y")]
    if len(rows) != N:
        raise SystemExit("expected %d reviewed studies, found %d" % (N, len(rows)))
    return rows


def counts(rows, field):
    out = {}
    for r in rows:
        v = r[field].strip()
        out[v] = out.get(v, 0) + 1
    return out


def pct(k):
    return 100.0 * k / N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    rows = load()

    st = counts(rows, "predictive_representation")
    bd = counts(rows, "boundary_condition")

    roll = st.get("rollout-capable", 0)
    local = st.get("local-prediction", 0)
    st_unclear = st.get("unclear", 0)
    # Lowest admissible: only a rollout reaches every link of a full OD path, so a reader who
    # requires full-route reach counts these alone. Highest admissible: short-range forecasts and
    # every unclear cell are all credited.
    state_lo, state_hi = roll, roll + local + st_unclear

    bd_open = bd.get("boundary-open-addressed", 0)
    bd_unclear = bd.get("unclear", 0)
    bound_lo, bound_hi = bd_open, bd_open + bd_unclear

    # Both requirements at once. The upper bound credits every contestable cell on both axes at
    # once, which is the most favourable reading of the corpus available.
    both_lo = sum(1 for r in rows
                  if r["predictive_representation"].strip() == "rollout-capable"
                  and r["boundary_condition"].strip() == "boundary-open-addressed")
    both_hi = sum(1 for r in rows
                  if r["predictive_representation"].strip() in
                  ("rollout-capable", "local-prediction", "unclear")
                  and r["boundary_condition"].strip() in
                  ("boundary-open-addressed", "unclear"))

    # The exposed subset: an individual reward, a link-level action, and an unaddressed boundary.
    # Lowest admissible: only cells recorded with all three explicitly. Highest: every unclear
    # cell on reward scope and action granularity is read into the subset as well.
    def exposed(strict):
        n = 0
        for r in rows:
            rw = r["reward_alignment"].strip()
            ag = r["action_granularity"].strip()
            bc = r["boundary_condition"].strip()
            ok_rw = rw == "individual" if strict else rw in ("individual", "unclear")
            ok_ag = ag == "next-link" if strict else ag in ("next-link", "unclear")
            ok_bc = bc == "not-addressed" if strict else bc in ("not-addressed", "unclear")
            if ok_rw and ok_ag and ok_bc:
                n += 1
        return n

    exp_lo, exp_hi = exposed(True), exposed(False)

    # The claim axis: a route-optimality claim standing on a state that cannot deliver it. This is
    # the field with the lowest reproducibility against the published rule, so it is bounded the
    # same way as the others rather than reported as a point. Low credits only an explicit claim
    # on an explicitly instantaneous state; high adds every cell either side could move.
    import csv as _csv
    cc = list(_csv.DictReader(io.open(str(HERE.parent / "corpus" / "claim_charting.csv"),
                                      encoding="utf-8")))

    def _claim(x):
        c = (x["claimed_capability"] or "").strip()
        return ("claimed" if c.startswith("optimal-route-claim")
                else "modest" if c.startswith("modest") else "unresolved")

    def _state(x):
        v = (x["state_form"] or "").strip()
        return ("fc" if v in ("local-prediction", "rollout-capable")
                else "inst" if v == "none" else "unclear")

    cell = {}
    for x in cc:
        k = (_claim(x), _state(x))
        cell[k] = cell.get(k, 0) + 1
    gap_lo = cell.get(("claimed", "inst"), 0)
    gap_hi = (gap_lo + cell.get(("claimed", "unclear"), 0)
              + cell.get(("unresolved", "inst"), 0) + cell.get(("unresolved", "unclear"), 0))
    no_gap = sum(cell.get(("modest", s), 0) for s in ("fc", "inst", "unclear"))
    met = cell.get(("claimed", "fc"), 0)

    res = {
        "n": N,
        "state": {"low": state_lo, "high": state_hi,
                  "low_pct": round(pct(state_lo), 1), "high_pct": round(pct(state_hi), 1)},
        "boundary": {"low": bound_lo, "high": bound_hi,
                     "low_pct": round(pct(bound_lo), 1), "high_pct": round(pct(bound_hi), 1)},
        "both": {"low": both_lo, "high": both_hi,
                 "low_pct": round(pct(both_lo), 1), "high_pct": round(pct(both_hi), 1)},
        "exposed": {"low": exp_lo, "high": exp_hi,
                    "low_pct": round(pct(exp_lo), 1), "high_pct": round(pct(exp_hi), 1)},
        "claim_gap": {"low": gap_lo, "high": gap_hi, "no_gap": no_gap, "claim_met": met,
                      "cells": {"%s_%s" % k: v for k, v in sorted(cell.items())}},
    }
    res["minority_everywhere"] = {
        "state": res["state"]["high"] < N / 2,
        "boundary": res["boundary"]["high"] < N / 2,
        "both": res["both"]["high"] < N / 2,
    }

    (HERE / "coding_invariance.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")

    if a.json:
        print(json.dumps(res, indent=2))
        return
    print("Admissible range of each headline count over all codings of the contestable cells")
    print("(N = %d reviewed studies)\n" % N)
    for k, label in (("state", "forecast-conditioned state"),
                     ("boundary", "addressed open boundary"),
                     ("both", "both requirements met"),
                     ("exposed", "exposed subset")):
        d = res[k]
        print("  %-28s %2d to %2d   (%.1f%% to %.1f%%)"
              % (label, d["low"], d["high"], d["low_pct"], d["high_pct"]))
    print()
    c = res["claim_gap"]
    print("\n  claim axis: a route-optimality claim on a state that cannot deliver it")
    print("    the gap                    %2d to %2d" % (c["low"], c["high"]))
    print("    claim met                  %2d" % c["claim_met"])
    print("    no optimality claim        %2d" % c["no_gap"])
    print()
    for k, v in res["minority_everywhere"].items():
        print("  minority at every admissible coding, %-9s %s" % (k + ":", "yes" if v else "NO"))


if __name__ == "__main__":
    main()
