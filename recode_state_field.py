# -*- coding: utf-8 -*-
"""Recompute the state-form field from the released quotations.

The forecast-conditioned state is the property Section III-A defines, and the count of studies
that hold it is one of the review's headline figures. Reading a quotation and deciding whether
it names a forecast is a judgment, and a judgment made once by one reader is not checkable by
anyone else. This script removes that step for every study whose quotation the rule below
settles: the recorded value becomes the output of a stated rule over released text rather than
the residue of a reading, and anyone can rerun it.

The rule follows Section III-A. A study is forecast-conditioned when its state quotation names
an identifiable forecast of the conditions a candidate link will present at the time it would
be traversed, a decision-time rollout, or a fully known time-dependent cost schedule. It is not
forecast-conditioned when the quotation reads a present condition, when the mention of
prediction is negated, or when prediction is deferred to future work. Those three exclusions
are what a keyword list alone gets wrong, and they are why the rule carries a negative clause.

Cells the rule does not settle are not forced. They are printed, counted, and carried into the
headline figure as an interval whose ends assign them both ways, so a reader can see what the
unresolved judgments could do to the count.

    python recode_state_field.py                 # counts, interval, and disagreements
    python recode_state_field.py --list          # every study the rule and the record differ on
"""
import argparse
import csv
import io
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus", "corpus_v9_coded.csv")

# An identifiable statement about conditions at a future traversal time, a decision-time
# rollout, or a schedule from which such a condition is recoverable.
FORECAST = re.compile(
    r"predict\w*|forecast\w*|roll-?out|anticipat\w*|expected to travel"
    r"|future (?:road|traffic|state|cost|condition|usage)\w*"
    r"|time-dependent (?:cost|profile|schedule)"
    r"|Monte Carlo sampling",
    re.I)

# A mention that does not put a forecast in the state: prediction refused, deferred, or absent.
NOT_IN_STATE = re.compile(
    r"bypass\w*|could be further|future work|we do not|does not predict"
    r"|without\s+\w{0,12}\s*predict|no predict",
    re.I)

PREDICTIVE = {"local-prediction", "rollout-capable"}


def evidence(row):
    """The released text the rule reads for one study.

    The extraction fills pred_quote only where a study states something predictive, so an empty
    cell is itself the record that no such statement was found rather than a missing input. The
    state description is carried in staterep_quote, and reading both together is what lets the
    rule settle a study whose pred_quote is blank.
    """
    return " ".join(x for x in ((row.get("pred_quote") or ""),
                                (row.get("staterep_quote") or "")) if x.strip())


def classify(quote):
    """Return 'forecast', 'instantaneous', or None when the rule does not settle the cell."""
    if not quote or not quote.strip():
        return None
    if NOT_IN_STATE.search(quote):
        return "instantaneous"
    return "forecast" if FORECAST.search(quote) else "instantaneous"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print every disagreement in full")
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(io.open(CORPUS, encoding="utf-8"))
            if (r.get("in_reviewed_corpus") or "yes").strip() == "yes"]

    settled = agree = 0
    unsettled, differ = [], []
    for r in rows:
        recorded = r["predictive_representation"]
        if recorded not in PREDICTIVE and recorded != "none":
            continue                                       # unclear cells carry no state value
        rule = classify(evidence(r))
        if rule is None:
            unsettled.append(r)
            continue
        settled += 1
        rec = "forecast" if recorded in PREDICTIVE else "instantaneous"
        if rule == rec:
            agree += 1
        else:
            differ.append((r["idx"], rec, rule, evidence(r).strip()))

    n_forecast = sum(1 for r in rows if r["predictive_representation"] in PREDICTIVE)
    print("Studies reviewed                     %d" % len(rows))
    print("Recorded as forecast-conditioned     %d" % n_forecast)
    print()
    print("Cells the rule settles               %d" % settled)
    print("  rule reproduces the record         %d  (%.1f%%)" % (agree, 100.0 * agree / settled))
    print("  rule differs from the record       %d" % len(differ))
    print("Cells the rule leaves to judgment    %d" % len(unsettled))
    print()

    contested = len(differ) + len(unsettled)
    lo, hi = n_forecast - len(differ), n_forecast + contested
    print("Assigning every contested cell against the recorded reading, and then for it,")
    print("puts the forecast-conditioned count between %d and %d of %d, that is %.1f%% to %.1f%%."
          % (lo, hi, len(rows), 100.0 * lo / len(rows), 100.0 * hi / len(rows)))
    print("A forecast-conditioned state is the exception at both ends of that interval.")

    if differ:
        print("\nDisagreements:")
        for idx, rec, rule, q in differ:
            print("  idx %-4s recorded %-13s rule %-13s" % (idx, rec, rule))
            if a.list:
                print("        %s" % q[:300])


if __name__ == "__main__":
    main()
