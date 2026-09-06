# -*- coding: utf-8 -*-
"""Which of the ten prior reviews names the boundary condition of a study area at all.

WHY THIS EXISTS. Table I carried three columns, and three reviewers over three rounds asked for a
column showing what separates this review from the ten. A corpus-size column is unavailable: nine
of the ten state no count anywhere in their text. The axis this review is built on is available,
and it is measured here from each review's own full text rather than asserted.

THE CONTROL COMES FIRST. A search that finds nothing prints the same clean zero as a search that
cannot run. Every review is therefore first tested for words it must contain. Only where the
control fires is a zero on the axis terms reported.

    python3 survey_axis_coverage.py
"""
import glob
import io
import json
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SURV = os.environ.get("SURVEY_PDF_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pdfs", "surveys"))
# "traffic" is absent from the two vehicle-routing reviews, so the control uses words no English
# survey of this literature can lack.
# 2026-08-31 (M3): the control was "reinforcement", "learning", "the", "algorithm", every one of
# which a FIRST PAGE alone satisfies, so a truncated extraction passed it and the zero it
# guarded evidenced nothing. Naming end-of-paper words failed too: IEEE small caps reach
# pdftotext as "R EFERENCES" and "C ONCLUSION". The control is therefore positional rather
# than lexical. The final tenth of each extraction must read as ordinary English prose, which
# no partial extraction can satisfy, and the word count of every review is reported so a
# reader sees the coverage the count rests on.
CONTROL = ["the", "of", "and", "in"]
TAIL_FRACTION = 0.10
TAIL_MIN_HITS = 20
AXIS = {
    # One term left the family for good on 2026-08-31, on the author's instruction: it is not his
    # word. It had been re-added by accident when the family was widened on 2026-08-29. The term
    # is removed from the INSTRUMENT and not merely from the printed table, so that the family
    # Appendix B prints is the family that was searched. Removing a term cannot raise a count,
    # and the boundary count was 0 of 10 with it present; the re-run below confirms 0 without it.
    # 2026-08-29: the cold review objected that four stems are too narrow and named network
    # truncation, sub-network extraction and the zone or centroid connector. The family is
    # widened here and the count re-measured. The manuscript prints the family, not the list,
    # since one member of it is a word the author bars.
    # 2026-08-31 (M3): the term was "boundary" and missed "boundaries", which is how the one
    # real occurrence in the ten reviews is written. A reviewer inferred an extraction failure
    # from the zero; the extraction is sound (5k-19k words each) and the SEARCH was narrow.
    # Stemmed here, so the count is honest before the sense of each hit is adjudicated.
    "boundary": [r"boundar", r"study area", r"peripheral", r"edge effect",
                 # the bare stems "truncat" and "clipped" are reinforcement-learning
                 # vocabulary (TRPO truncation, the PPO clipped objective, clipped double
                 # Q-learning) and returned three hits of that kind. The study-area sense is
                 # matched as a phrase instead.
                 r"network truncat", r"truncated network", r"sub-?network",
                 r"network extract", r"centroid connector", r"zone connector",
                 r"study region", r"spatial extent", r"network extent", r"delimit"],
    "completion": [r"completion rate", r"trip completion", r"arrival rate", r"success rate"],
}


# Every lexical hit of the boundary family, read in context and given a verdict. A hit counts for
# the claim only where it carries the STUDY-AREA sense, an area delimited from a larger network.
# 2026-08-31: stemming the family surfaced two hits in ref47, both "Lane boundaries", which is the
# lane-marking sense. The lexical count is 9 of 10 at zero; the adjudicated count is 10 of 10.
SENSE = {
    ("ref47", "boundar"): ("no", "both occurrences read \"Lane boundaries\", the lane-marking "
                                 "sense in a list of path-planning constraints beside traffic "
                                 "rules and obstacle avoidance, not an area delimited from a "
                                 "larger network"),
}


def adjudicate(key, pat, count):
    """Return the hits that carry the study-area sense."""
    verdict = SENSE.get((key, pat))
    if verdict is None:
        return count
    return count if verdict[0] == "yes" else 0


def main():
    out, bad = {}, []
    print("%-8s %7s %-22s %-10s %-12s" % ("review", "words", "tail control", "boundary", "completion"))
    for pdf in sorted(glob.glob(os.path.join(SURV, "*.pdf"))):
        key = os.path.basename(pdf).split("_")[0]
        t = re.sub(r"\s+", " ",
                   subprocess.run(["pdftotext", pdf, "-"], capture_output=True,
                                  text=True).stdout.lower())
        tail = t[int(len(t) * (1.0 - TAIL_FRACTION)):]
        ctl = {w: len(re.findall(r"\b%s\b" % w, tail)) for w in CONTROL}
        words = len(re.findall(r"[a-z][a-z'-]+", t))
        if sum(ctl.values()) < TAIL_MIN_HITS or len(t) < 20000:
            bad.append("%s: the final tenth does not read as prose (%s, %d chars)"
                       % (key, ctl, len(t)))
            continue
        row = {}
        for axis, pats in AXIS.items():
            hits = {p: len(re.findall(p, t)) for p in pats}
            row[axis + "_lexical"] = sum(hits.values())
            row[axis] = sum(adjudicate(key, p, n) for p, n in hits.items())
            row[axis + "_detail"] = {k: v for k, v in hits.items() if v}
        out[key] = row
        row["words"] = words
        print("%-8s %7d %-22s %-10d %-12d" % (key, words, str(ctl)[:20],
                                              row["boundary"], row["completion"]))

    if bad:
        raise SystemExit("AXIS COVERAGE FAILED: " + "; ".join(bad))
    nl = sum(1 for v in out.values() if v["boundary_lexical"] == 0)
    print("\n  %d of the %d return no LEXICAL hit of the boundary family." % (nl, len(out)))
    for k, v in sorted(out.items()):
        if v["boundary_lexical"]:
            sv = SENSE.get((k, "boundar"))
            print("    %s: %d hit(s), study-area sense %s -- %s"
                  % (k, v["boundary_lexical"], (sv[0] if sv else "UNADJUDICATED"),
                     (sv[1] if sv else "READ THEM AND RECORD A VERDICT")))
            assert sv is not None, "an unadjudicated lexical hit cannot enter the count"
    nb = sum(1 for v in out.values() if v["boundary"] == 0)
    nc = sum(1 for v in out.values() if v["completion"] == 0)
    print("\n  %d of the %d name no term of the boundary family anywhere in their full text."
          % (nb, len(out)))
    print("  %d of the %d name no completion-rate term." % (nc, len(out)))
    json.dump(out, io.open(os.path.join(HERE, "survey_axis_coverage.json"), "w",
                           encoding="utf-8"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
