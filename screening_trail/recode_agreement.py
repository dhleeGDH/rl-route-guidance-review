# -*- coding: utf-8 -*-
"""Chance-corrected agreement for the 28-study repeat pass.

WHY THIS EXISTS. The supplementary material reports the repeat pass as raw agreement, at 68%, 64%
and 61% on the three fields turning on a definitional boundary. A chance-corrected figure belongs
alongside it: raw agreement flatters a field whose
values are concentrated in one category, and all three of these are.

The two passes are reconstructed from the released record rather than re-run. Pass one is the
recorded value in `corpus_v9_coded.csv`. Pass two is the same value except where
`recode_adjudication_r6.csv` records a disagreement, in which case it is the blinded recode value.
That reconstruction reproduces the published raw agreements exactly (19, 18 and 17 of 28), which
is the check that it is faithful.

Both coefficients are reported. Cohen's kappa is the conventional one and is deflated wherever one
category holds most of the sample. Gwet's AC1 is the marginal-robust alternative. Neither carries
a count in the manuscript: the headline numbers are bounded by enumeration instead, and this is
reported so a reader who wants the conventional statistic can see it.

    python3 recode_agreement.py

Writes recode_agreement.json next to this file.
"""
import csv
import io
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS = HERE.parent / "corpus" / "corpus_v9_coded.csv"
SAMPLE = HERE / "recode_sample_r6.csv"
DISAGREE = HERE / "recode_adjudication_r6.csv"

FIELDS = [("predictive_representation", "state form"),
          ("reward_temporality", "reward temporality"),
          ("generalization_mechanism", "generalization mechanism")]


def load():
    sample = [r["corpus_idx"] for r in csv.DictReader(io.open(str(SAMPLE), encoding="utf-8"))]
    corpus = {r["idx"]: r for r in csv.DictReader(io.open(str(CORPUS), encoding="utf-8"))}
    # The corpus file holds the ADJUDICATED value, which for a corrected cell equals the recode
    # value. Reading pass one from it would score those cells as agreements and report 82% where
    # the published figure is 68%. Pass one is the original_value the disagreement file records.
    dis = defaultdict(dict)
    for r in csv.DictReader(io.open(str(DISAGREE), encoding="utf-8")):
        dis[r["corpus_idx"]][r["field"]] = (r["original_value"], r["blind_recode_value"])
    ids = [i for i in sample if i in corpus
           and str(corpus[i].get("in_reviewed_corpus", "")).strip().lower()
           in ("1", "true", "yes", "y")]
    return ids, corpus, dis


def coefficients(pairs):
    """Cohen's kappa and Gwet's AC1 for one field, over (pass one, pass two) pairs."""
    n = len(pairs)
    cats = sorted({v for p in pairs for v in p})
    po = sum(1 for a, b in pairs if a == b) / n
    p1 = {c: sum(1 for a, _ in pairs if a == c) / n for c in cats}
    p2 = {c: sum(1 for _, b in pairs if b == c) / n for c in cats}
    pe_k = sum(p1[c] * p2[c] for c in cats)
    kappa = (po - pe_k) / (1 - pe_k) if pe_k < 1 else float("nan")
    # Gwet: chance agreement is built from the mean marginal of each category, which is what
    # stops a concentrated field from driving the coefficient toward zero.
    q = len(cats)
    pi = {c: (p1[c] + p2[c]) / 2.0 for c in cats}
    pe_g = sum(pi[c] * (1 - pi[c]) for c in cats) / (q - 1) if q > 1 else 0.0
    ac1 = (po - pe_g) / (1 - pe_g) if pe_g < 1 else float("nan")
    return po, kappa, ac1, n, q


def main():
    ids, corpus, dis = load()
    out = {"n": len(ids), "fields": {}}
    print("repeat pass over %d studies\n" % len(ids))
    print("  %-24s %8s %8s %8s %6s" % ("field", "raw", "kappa", "AC1", "cats"))
    for field, label in FIELDS:
        pairs = []
        for i in ids:
            rec = (corpus[i].get(field) or "").strip()
            if field in dis.get(i, {}):
                one, two = dis[i][field]
                pairs.append((one.strip(), two.strip()))
            else:
                pairs.append((rec, rec))
        po, k, ac1, n, q = coefficients(pairs)
        out["fields"][field] = {"label": label, "raw": round(100 * po, 1),
                                "kappa": round(k, 2), "ac1": round(ac1, 2), "categories": q}
        print("  %-24s %7.1f%% %8.2f %8.2f %6d" % (label, 100 * po, k, ac1, q))
    (HERE / "recode_agreement.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote recode_agreement.json")


if __name__ == "__main__":
    main()
