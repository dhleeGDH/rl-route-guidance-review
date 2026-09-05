# -*- coding: utf-8 -*-
"""Recall of the overlap rule, measured on real reference entries rather than on written ones.

WHY THIS EXISTS. The zero-overlap claim of Section I rests on a rule requiring 85% of a title's
distinctive tokens inside one reference entry. Its sensitivity control writes thirty corpus titles
into an entry of the form the ten reviews use and recovers 30 of 30. That control is self-confirming: the entries were written to the specification the matcher
expects, so the control cannot measure what an abbreviated author list, a shortened venue name or a
line break does to recall.

WHAT THIS MEASURES INSTEAD. The ten reviews are surveys of one literature and cite each other's
sources. A title is lifted out of a REAL entry of review A, carrying A's own rendering, and the
paper's own matcher is then run against the REAL entries of the other nine. Every hit is a shared
citation found across two independent renderings of the same paper, which is the exact condition a
corpus title faces. Recall is estimated as the share of cross-review pairs the entry-level evidence
identifies that the title rule also finds.

Ground truth for a pair is established WITHOUT the title rule: two entries name the same first
author surname and the same year, and share at least 70% of the distinctive tokens of the shorter.

THE GROUND TRUTH HAS ITS OWN FLOOR. Sweeping the ground-truth threshold down from 0.80 gives
71/71, 81/81, 85/90, 84/102, 82/107 and 80/126 at 0.80, 0.70, 0.60, 0.50, 0.40 and 0.30. The
apparent decline is the ground truth failing rather than the title rule: below 0.60 the surname and
year admit DIFFERENT papers sharing generic vocabulary, and every pair the title rule declines at
0.30 was read. One example stands for the rest, a title reading "Deep reinforcement learning for
autonomous traffic light control" paired against an entry for "Robust deep reinforcement learning
for security and safety in autonomous vehicle systems". The title rule declining those is precision,
not a missed citation. The recall figure is therefore stated at 0.70, where every pair read is the
same work.

    python3 crossreview_recall.py
"""
import glob
import io
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from overlap_refs_complete import STOP, SURV, THRESH, entries_by_label, toks  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GROUND = 0.70
QUOTED = re.compile(u'[“"‘]([^”"’]{25,180})[”"’]')
SURNAME_YEAR = re.compile(r'([A-Z][a-z]{2,})[^0-9]{0,200}?\b(19[89]\d|20[0-2]\d)\b')


def review_entries():
    out = {}
    for pdf in sorted(glob.glob(os.path.join(SURV, "*.pdf"))):
        key = os.path.basename(pdf).split("_")[0]
        text = subprocess.run(["pdftotext", pdf, "-"], capture_output=True, text=True).stdout
        by_label, _ = entries_by_label(text)
        sec = text[text.rfind("References"):] if "References" in text else text[-40000:]
        by_author = [' '.join(e.split())
                     for e in re.split(r'(?m)^\s*(?=\[\d{1,3}\]|[A-Z][a-z]+,\s+[A-Z]\.)', sec)]
        by_author = [e for e in by_author if 5 <= len(e.split()) <= 120]
        out[key] = list(dict.fromkeys((by_label or []) + by_author))
    return out


def key_of(entry):
    m = SURNAME_YEAR.search(entry)
    return (m.group(1).lower(), m.group(2)) if m else None


def main():
    ents = review_entries()
    print("ten reviews, real reference entries: %s"
          % ", ".join("%s=%d" % (k, len(v)) for k, v in sorted(ents.items())))
    tok = {k: [(e, toks(e), key_of(e)) for e in v] for k, v in ents.items()}

    pairs, found, missed = 0, 0, []
    keys = sorted(ents)
    for i, A in enumerate(keys):
        for B in keys[i + 1:]:
            for ea, sa, ka in tok[A]:
                if not ka or len(sa) < 6:
                    continue
                for eb, sb, kb in tok[B]:
                    if kb != ka or len(sb) < 6:
                        continue
                    share = len(sa & sb) / float(min(len(sa), len(sb)))
                    if share < GROUND:
                        continue
                    # the same work, established without the title rule. Now lift the title out
                    # of A's rendering and run the paper's own matcher against B's rendering.
                    q = QUOTED.search(ea)
                    if not q:
                        continue
                    t = toks(q.group(1))
                    if len(t) < 4:
                        continue
                    pairs += 1
                    if len(t & sb) / float(len(t)) >= THRESH:
                        found += 1
                    else:
                        missed.append((A, B, q.group(1)[:90],
                                       round(len(t & sb) / float(len(t)), 2)))
                    break

    print("\ncross-review pairs of the same work, established by author, year and entry overlap: %d"
          % pairs)
    if not pairs:
        raise SystemExit("RECALL CONTROL FAILED: no ground-truth pair was found, so the control "
                         "measures nothing. A control finding nothing prints clean.")
    print("found by the title rule at the published threshold of %.2f: %d" % (THRESH, found))
    print("recall on real entries: %.1f%%" % (100.0 * found / pairs))
    for a, b, t, s in missed[:15]:
        print("  MISSED %s -> %s at %.2f : %s" % (a, b, s, t))
    if found != pairs:
        raise SystemExit("RECALL CONTROL FAILED: %d of %d pairs went unfound at the ground-truth "
                         "threshold of %.2f, where every pair read is the same work."
                         % (pairs - found, pairs, GROUND))
    json.dump({"pairs": pairs, "found": found, "threshold": THRESH,
               "recall": round(100.0 * found / pairs, 1),
               "missed": [list(m) for m in missed]},
              io.open(os.path.join(HERE, "crossreview_recall.json"), "w", encoding="utf-8"),
              indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
