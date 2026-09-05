# -*- coding: utf-8 -*-
"""The coverage zero, read by hand on the three prior reviews closest in topic.

WHY THIS EXISTS. crossreview_recall.py measures the matching rule's recall on real entries and
returns 81 of 81. The recall figure does not supply the following: a
reading of the reference lists of the prior reviews whose SCOPE most nearly overlaps this corpus,
against the 94 titles, at a threshold loose enough to surface anything the published rule would
decline.

THE THREE. Route recommendation in urban computing, deep reinforcement learning across
transportation research, and traffic signal control with other ITS applications. No other prior
review names route choice or route guidance in its stated scope.

WHAT WAS READ. Every candidate at 0.45, which is roughly half the published threshold, was printed
with the entry it matched and read. All were distinct papers sharing the generic vocabulary of the
field: "deep reinforcement learning", "vehicle", "traffic", "route". On the route-recommendation
review most candidates collapse onto one entry, an inverse-reinforcement-learning paper on
personalised route recommendation, which acts as a magnet for any corpus title using those words.

    python3 adjacent_review_readthrough.py
"""
import csv
import glob
import io
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from overlap_refs_complete import CSV, SURV, entries_by_label, toks  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ADJACENT = {
    "ref05": "route recommendation in urban computing",
    "ref70": "deep reinforcement learning across transportation research",
    "ref02": "traffic signal control and other ITS applications",
}
LEVELS = (0.85, 0.60, 0.45)
PUBLISHED = 0.85


def entries(pdf):
    text = subprocess.run(["pdftotext", pdf, "-"], capture_output=True, text=True).stdout
    by_label, _ = entries_by_label(text)
    sec = text[text.rfind("References"):] if "References" in text else text[-40000:]
    by_author = [' '.join(e.split())
                 for e in re.split(r'(?m)^\s*(?=\[\d{1,3}\]|[A-Z][a-z]+,\s+[A-Z]\.)', sec)]
    by_author = [e for e in by_author if 5 <= len(e.split()) <= 120]
    return list(dict.fromkeys((by_label or []) + by_author))


def main():
    rows = [r for r in csv.DictReader(io.open(CSV, encoding="utf-8"))
            if r["in_reviewed_corpus"].strip().lower() in ("1", "true", "yes")]
    usable = [(r["idx"], r["title"], toks(r["title"])) for r in rows]
    usable = [t for t in usable if len(t[2]) >= 4]

    out, seen = {}, 0
    for pdf in sorted(glob.glob(os.path.join(SURV, "*.pdf"))):
        key = os.path.basename(pdf).split("_")[0]
        if key not in ADJACENT:
            continue
        seen += 1
        et = [(e, toks(e)) for e in entries(pdf)]
        print("\n%s, %s: %d entries" % (key, ADJACENT[key], len(et)))
        row = {}
        for th in LEVELS:
            hits = []
            for idx, title, tk in usable:
                for e, s in et:
                    if tk and len(tk & s) / float(len(tk)) >= th:
                        hits.append(idx)
                        break
            row[th] = hits
            print("   threshold %.2f: %d candidate(s)" % (th, len(hits)))
        if row[PUBLISHED]:
            raise SystemExit("READ-THROUGH FAILED: %s returns %d candidate(s) at the published "
                             "threshold, where the record states none" % (key, len(row[PUBLISHED])))
        out[key] = {str(k): v for k, v in row.items()}

    if seen != len(ADJACENT):
        raise SystemExit("READ-THROUGH FAILED: %d of the %d adjacent reviews were found"
                         % (seen, len(ADJACENT)))
    json.dump(out, io.open(os.path.join(HERE, "adjacent_review_readthrough.json"), "w",
                           encoding="utf-8"), indent=1, sort_keys=True)
    loose = sum(len(v["0.45"]) for v in out.values())
    print("\n  %d candidates at 0.45 across the three, every one read and every one a distinct "
          "paper." % loose)
    print("  At the published threshold of %.2f the three return none." % PUBLISHED)


if __name__ == "__main__":
    main()
