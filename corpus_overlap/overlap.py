# -*- coding: utf-8 -*-
"""How many of the 94 corpus studies each prior review already cites.

A survey reader asks first how much of this corpus is coverage the
prior reviews do not have. Table I positions this review against ten of them on scope and purpose,
which paraphrase and do not discriminate. This measures the overlap directly.

Method. Each prior review's full text is searched for each corpus study's title. Matching is on a
normalized title with punctuation, case and whitespace removed, so line breaks and hyphenation in
the PDF do not defeat it. A title is counted as cited when its normalized form occurs anywhere in
the review's text, which is where a reference list lives.

Two controls, both asserted:
  1. a negative control title that occurs in no review must match nowhere;
  2. every corpus title must be long enough that a chance substring match is implausible, so titles
     under 25 normalized characters are reported separately rather than counted.
"""
import csv, io, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
SURVEYS = "/home/dhlee/review_paper/pdfs/surveys"
CSV = os.path.join(EXP, "corpus", "corpus_v9_coded.csv")

REFMAP = {"ref02": "[2]", "ref03": "[3]", "ref04": "[4]", "ref05": "[5]", "ref43": "[43]",
          "ref44": "[44]", "ref45": "[45]", "ref46": "[46]", "ref47": "[47]", "ref70": "[70]"}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def text_of(pdf):
    return subprocess.run(["pdftotext", pdf, "-"], capture_output=True, text=True).stdout


def main():
    rows = [r for r in csv.DictReader(io.open(CSV, encoding="utf-8"))
            if r["in_reviewed_corpus"].strip().lower() in ("1", "true", "yes")]
    titles = [(r["idx"], r["title"].strip()) for r in rows]
    short = [(i, t) for i, t in titles if len(norm(t)) < 25]
    usable = [(i, t) for i, t in titles if len(norm(t)) >= 25]
    print("corpus %d, usable titles %d, too short to match safely %d"
          % (len(titles), len(usable), len(short)), flush=True)

    NEG = "A completely fictitious title that occurs in no published review whatsoever"
    out = {}
    for f in sorted(os.listdir(SURVEYS)):
        if not f.endswith(".pdf"):
            continue
        key = f.split("_")[0]
        body = norm(text_of(os.path.join(SURVEYS, f)))
        if not body:
            print("  %-6s NO TEXT LAYER" % key, flush=True)
            out[key] = None
            continue
        assert norm(NEG) not in body, "negative control matched in %s" % key
        hits = [i for i, t in usable if norm(t) in body]
        out[key] = {"ref": REFMAP.get(key, key), "file": f, "hits": len(hits),
                    "of": len(usable), "idx": hits}
        print("  %-6s %-6s %3d of %d corpus studies cited"
              % (key, REFMAP.get(key, key), len(hits), len(usable)), flush=True)
    out["_meta"] = {"corpus": len(titles), "usable": len(usable),
                    "short_titles": [t for _, t in short]}
    json.dump(out, io.open(os.path.join(HERE, "overlap.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    print("wrote overlap.json")


if __name__ == "__main__":
    main()
