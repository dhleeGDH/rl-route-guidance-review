#!/usr/bin/env python3
"""Does the zero survive completing the three short reference lists?

WHY. Section I states that no study of the 94 appears among the entries recovered from the ten
prior reviews' reference lists, and the supplement discloses that three of the ten are recovered
short of the highest label they print. The zero could therefore be an artifact of that under-recovery. This answers it by re-segmenting the three from the numbered labels themselves and
re-running the manuscript's own matching rule.

TWO CONTROLS, both needed. Either alone passes a broken test.
  1. A fabricated title of ordinary domain vocabulary must match nothing. Without it, a bag of
     words over a whole reference section scores 88 to 100 percent on invented titles.
  2. Every entry is capped at 70 words. Without the cap the text after the last label lands in the
     final entry, which then accumulates enough vocabulary to match five corpus titles that are
     not there. Both controls were failed by a first attempt at this check.

    python3 overlap_completion_check.py
"""
import csv
import io
import json
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SURV = "/home/dhlee/review_paper/pdfs/surveys"
CSV = os.path.join(os.path.dirname(HERE), "corpus", "corpus_v9_coded.csv")
SHORT = ("ref43", "ref46", "ref47")
CAP = 70
FAKE = "Quantum Bicycle Semaphore Learning for Submarine Confectionery Navigation"

# Candidates adjudicated by hand, named so a NEW candidate cannot be absorbed into them.
# ref46 entry [39] is Wu, Cheng, Chu and Song, "An autonomous coverage path planning algorithm for
# maritime search and rescue of persons-in-water based on deep reinforcement learning", Ocean
# Engineering 2024. The corpus title it matches, "Optimization Path Planning Algorithm Based on
# Deep Reinforcement Learning", shares six of its seven distinctive tokens with it: path, planning,
# algorithm, based, deep, reinforcement, learning. They are different papers. The 85% rule cannot
# separate two short titles built from that vocabulary, which is a limit of the rule, not a match.
ADJUDICATED = {("ref46", 39): "maritime search-and-rescue coverage planning, a different paper"}
STOP = set("the a an of for and with in on to by from at as is are was were this that these those "
           "using use based via towards toward new novel study paper".split())


def toks(s):
    return [w for w in re.findall(r"[a-z]+", s.lower()) if len(w) >= 4 and w not in STOP]


def entries(tag):
    f = [x for x in os.listdir(SURV) if x.startswith(tag)][0]
    raw = subprocess.run(["pdftotext", "-q", os.path.join(SURV, f), "-"],
                         capture_output=True, text=True).stdout
    heads = [x.start() for x in
             re.finditer(r"(?im)^\s*(references|r\s*eferences|bibliography)\s*$", raw)]
    sec = " ".join((raw[heads[-1]:] if heads else raw).split())
    cuts = [(int(x.group(1)), x.start()) for x in re.finditer(r"\[(\d{1,3})\]", sec)]
    out = {}
    for i, (n, p) in enumerate(cuts):
        end = cuts[i + 1][1] if i + 1 < len(cuts) else len(sec)
        seg = " ".join(sec[p:end].split()[:CAP])
        if n not in out or len(seg) > len(out[n]):
            out[n] = seg
    return out


def matched(ent, title):
    tk = toks(title)
    if len(tk) < 4:
        return None
    for n, e in sorted(ent.items()):
        flat = " ".join(re.findall(r"[a-z]+", e.lower()))
        got = sum(1 for w in tk if re.search(r"\b%s\b" % re.escape(w), flat))
        if got / len(tk) >= 0.85:
            return n
    return None


def main():
    rows = [r for r in csv.DictReader(io.open(CSV, encoding="utf-8"))
            if r["in_reviewed_corpus"].strip().lower() in ("1", "yes", "true")]
    result = {"cap_words": CAP, "corpus": len(rows), "reviews": {}}
    total_hits = 0
    print("%-8s %8s %8s %9s  %s" % ("review", "entries", "labels", "hits", "fabricated control"))
    for tag in SHORT:
        ent = entries(tag)
        raw_hits = [(r["idx"], matched(ent, r["title"])) for r in rows]
        raw_hits = [(i, n) for i, n in raw_hits if n is not None]
        hits = [i for i, n in raw_hits if (tag, n) not in ADJUDICATED]
        for i, n in raw_hits:
            if (tag, n) in ADJUDICATED:
                print("     candidate idx %s at entry [%d] adjudicated: %s"
                      % (i, n, ADJUDICATED[(tag, n)]))
        ctl = matched(ent, FAKE)
        total_hits += len(hits)
        result["reviews"][tag] = {"entries": len(ent), "highest_label": max(ent) if ent else 0,
                                  "corpus_hits": hits, "fabricated_control_matched": ctl is not None}
        print("  %-6s %8d %8d %9d  %s" % (tag, len(ent), max(ent) if ent else 0, len(hits),
                                          "MATCHED (invalid)" if ctl else "no match"))
    result["total_corpus_hits"] = total_hits
    io.open(os.path.join(HERE, "overlap_completion_check.json"), "w", encoding="utf-8").write(
        json.dumps(result, indent=1))
    print("\ncorpus studies found in the completed lists: %d" % total_hits)
    print("the zero of Section I %s" % ("SURVIVES completion" if total_hits == 0 else "DOES NOT survive"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
