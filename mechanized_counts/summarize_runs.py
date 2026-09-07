# -*- coding: utf-8 -*-
"""Put the two mechanized runs and the recorded values on one page.

The count the review reports is over all 94 studies, while the mechanized runs compare only the
studies that carry a recorded value and whose full text was matched and read. This script carries
the remainder at its recorded value and states how many studies that covers, so the machine figure
and the published one are read on the same denominator.
"""
import csv
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(HERE))
import recode_state_field as recorded_rule   # noqa: E402

# The corpus lives in experiments/corpus/. It sat in data/screened/ on the machine this
# was written on, and that path does not exist in the released package, so this script
# raised FileNotFoundError out of the box while Section II-B said any reader can rerun it.
CORPUS = os.path.join(ROOT, "corpus", "corpus_v9_coded.csv")
RUNS = [("positive clause only", "machine_state_counts.csv"),
        ("all three clauses", "machine_state_counts_v2.csv")]


def load(fn):
    return list(csv.DictReader(io.open(os.path.join(HERE, fn), encoding="utf-8")))


rows = [r for r in csv.DictReader(io.open(CORPUS, encoding="utf-8")) if r["idx"] != "93"]
n_all = len(rows)
n_rec = sum(1 for r in rows if r["predictive_representation"] in recorded_rule.PREDICTIVE)
n_unclear = sum(1 for r in rows
                if r["predictive_representation"] not in recorded_rule.PREDICTIVE
                and r["predictive_representation"] != "none")

print("Studies reviewed                                   %d" % n_all)
print("Recorded forecast-conditioned                      %d  (%.1f%%)"
      % (n_rec, 100.0 * n_rec / n_all))
print("Recorded unclear, carried as not forecast          %d" % n_unclear)
print()
print("%-24s %8s %8s %10s %9s %9s" % ("run", "compared", "machine", "over 94", "share", "agree"))
for label, fn in RUNS:
    out = load(fn)
    comp = [r for r in out if r["recorded"] and r["machine_w0"] in ("forecast", "instantaneous")]
    mach = sum(1 for r in comp if r["machine_w0"] == "forecast")
    agree = sum(1 for r in comp if r["machine_w0"] == r["recorded"])
    # studies outside the comparison keep the value the review recorded for them
    carried = sum(1 for r in out
                  if r["recorded"] == "forecast" and r not in comp
                  and r["machine_w0"] not in ("forecast", "instantaneous"))
    total = mach + carried
    if not comp:
        sys.exit("%s carries no study the machine rule could classify.\n"
                 "Rebuild it with machine_state_count.py and PDF_DIR set to the full texts."
                 % fn)
    print("%-24s %8d %8d %10d %8.1f%% %8.1f%%"
          % (label, len(comp), mach, total, 100.0 * total / n_all, 100.0 * agree / len(comp)))

print()
print("Published interval from the quotation-based rule    15 to 22 of 94  (16.0%% to 23.4%%)")
