# -*- coding: utf-8 -*-
"""Recompute the network-boundary count from the full texts rather than from picked quotations.

The boundary count is the review's second headline figure and was produced the same way the state
count was, by a reader picking one quotation out of each report. This script fixes that input. A
vocabulary derived from Section IV-A and released in boundary_lexicon.txt selects the sentences,
and a rule in the same file decides. The path from PDF to count is then rerunnable.

Text extraction, title matching, and sentence splitting are shared with machine_state_count.py, so
a difference between the two counts cannot come from how the PDFs were opened.

    python machine_boundary_count.py            # counts, agreement, and the disagreement table
    python machine_boundary_count.py --list     # print the sentence that settled each difference
"""
import argparse
import csv
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import machine_state_count as pipeline   # noqa: E402  shared extraction and matching

# The corpus lives in experiments/corpus/. It sat in data/screened/ on the machine this
# was written on, and that path does not exist in the released package, so this script
# raised FileNotFoundError out of the box while Section II-B said any reader can rerun it.
CORPUS = os.path.join(ROOT, "experiments", "corpus", "corpus_v9_coded.csv")
LEXICON = os.path.join(HERE, "boundary_lexicon.txt")
OUT = os.path.join(HERE, "machine_boundary_counts.csv")

AUTHOR_ROW = "93"
ADDRESSED = "boundary-open-addressed"


def settle(sents, lex):
    """Return the verdict and the sentence that produced it.

    Selection and decision are separate patterns over the same sentence. A sentence is eligible
    when the boundary vocabulary selects it, and it settles the study when it also matches one of
    the three forms Section IV-A names. Anything else leaves the study at silence, which is the
    value Section IV-A assigns to a report that fixes nothing about its edge.
    """
    for s in sents:
        if lex["DROP_SENTENCE"].search(s):
            continue
        if not lex["SELECT_BOUNDARY"].search(s):
            continue
        if lex["ADDRESSED"].search(s):
            return "addressed", s
    return "not-addressed", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print the settling sentence")
    a = ap.parse_args()

    lex = pipeline.load_lexicon(LEXICON)
    index = pipeline.build_pdf_index()
    keys = list(index.keys())
    rows = [r for r in csv.DictReader(io.open(CORPUS, encoding="utf-8"))
            if (r["idx"] or "").strip() != AUTHOR_ROW]

    out_rows, unmatched, evidence = [], [], {}
    for r in rows:
        idx = r["idx"]
        path, ratio = pipeline.match_pdf(r["title"], index, keys)
        field = r["boundary_condition"]
        recorded = ("addressed" if field == ADDRESSED else
                    "not-addressed" if field == "not-addressed" else "")
        row = {"idx": idx, "title": r["title"], "recorded_field": field, "recorded": recorded,
               "pdf": os.path.basename(path) if path else "", "match_ratio": "%.3f" % ratio,
               "machine": "", "settled_by": ""}
        if not path:
            unmatched.append((idx, r["title"], recorded))
            out_rows.append(row)
            continue
        sents = pipeline.sentences(pipeline.raw_text(path))
        verdict, why = settle(sents, lex)
        row["machine"] = verdict
        row["settled_by"] = why[:400]
        evidence[idx] = why
        out_rows.append(row)

    with io.open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    n_rec = sum(1 for r in rows if r["boundary_condition"] == ADDRESSED)
    comp = [r for r in out_rows if r["recorded"] and r["machine"]]
    agree = sum(1 for r in comp if r["recorded"] == r["machine"])
    mach = sum(1 for r in comp if r["machine"] == "addressed")
    carried = sum(1 for i, t, rec in unmatched if rec == "addressed")

    print("Studies reviewed                     %d" % len(rows))
    print("Full text matched and read           %d" % sum(1 for r in out_rows if r["machine"]))
    print("Recorded as addressing the boundary  %d  (over all %d)" % (n_rec, len(rows)))
    print()
    print("  studies compared                   %d" % len(comp))
    print("  recorded addressed, in that set    %d" % sum(1 for r in comp
                                                          if r["recorded"] == "addressed"))
    print("  machine addressed                  %d" % mach)
    print("  machine reproduces the record      %d  (%.1f%%)"
          % (agree, 100.0 * agree / len(comp) if comp else 0))
    print("  machine differs from the record    %d" % (len(comp) - agree))
    print("     recorded addressed, machine silent   %d"
          % sum(1 for r in comp if r["recorded"] == "addressed" and r["machine"] != "addressed"))
    print("     recorded silent, machine addressed   %d"
          % sum(1 for r in comp if r["recorded"] != "addressed" and r["machine"] == "addressed"))
    print()
    print("Carrying the %d studies with no full text at their recorded value" % len(unmatched))
    print("puts the machine count at %d of %d, that is %.1f%%."
          % (mach + carried, len(rows), 100.0 * (mach + carried) / len(rows)))

    if a.list:
        print("\nDifferences:")
        for r in comp:
            if r["recorded"] != r["machine"]:
                print("  idx %-4s recorded %-13s machine %-13s %s"
                      % (r["idx"], r["recorded"], r["machine"], r["title"][:46]))
                if r["settled_by"]:
                    print("        > %s" % r["settled_by"][:230])

    json.dump(evidence, io.open(os.path.join(HERE, "boundary_evidence.json"), "w",
                                encoding="utf-8"), indent=1, ensure_ascii=False)
    print("\nWrote %s" % os.path.basename(OUT))


if __name__ == "__main__":
    main()
