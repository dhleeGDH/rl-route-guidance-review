# -*- coding: utf-8 -*-
"""Corpus overlap with the prior reviews, matched reference entry against reference entry.

Three earlier methods failed at this, all by matching titles against a review's running
text: full-title substring found nothing, first-author-plus-year was 8 of 8 false on hand
inspection, and title-token windows were false too. README.md records that failure and names the
method a trustworthy answer needs, which is what this implements.

The difference is the unit. A reference ENTRY is about thirty words, so token overlap inside one
entry is meaningful, where the same test against a whole document is not. Each prior review's
reference section is isolated first, split into entries, and every corpus title is scored against
every entry. A match needs almost all of the title's distinctive tokens inside ONE entry.

Two controls:
  1. a fabricated title must match nowhere;
  2. every hit is printed with the entry it matched, for hand inspection. A count is not reported
     until those are read.
"""
import csv, io, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(os.path.dirname(HERE), "corpus", "corpus_v9_coded.csv")
SURV = "/home/dhlee/review_paper/pdfs/surveys"
THRESH = 0.85

STOP = set("a an the of in on for to and or with from by using based via toward towards".split())


def toks(t):
    return {w for w in re.findall(r"[a-z]+", t.lower()) if len(w) >= 4 and w not in STOP}


def ref_section(pdf):
    txt = subprocess.run(["pdftotext", pdf, "-"], capture_output=True, text=True).stdout
    flat = re.sub(r"-\n", "", txt)
    m = None
    for pat in (r"\nR\s?E\s?F\s?E\s?R\s?E\s?N\s?C\s?E\s?S", r"\nReferences\b", r"\nREFERENCES\b"):
        for m2 in re.finditer(pat, flat):
            m = m2
    return flat[m.end():] if m else ""


def entries(sec):
    parts = re.split(r"\n\s*\[\d{1,3}\]\s*", sec)
    if len(parts) < 10:
        parts = re.split(r"\n(?=\d{1,3}[.)]\s)", sec)
    if len(parts) < 10:
        # An author-year list with no numbering, as Elsevier sets it: a new entry begins at a
        # line whose start looks like a surname followed by an initial.
        parts = re.split(r"\n(?=[A-Z][a-zA-Z\-']+,\s*[A-Z]\.)", sec)
    return [re.sub(r"\s+", " ", p).strip() for p in parts if len(p.strip()) > 40]


def main():
    rows = [r for r in csv.DictReader(io.open(CSV, encoding="utf-8"))
            if r["in_reviewed_corpus"].strip().lower() in ("1", "true", "yes")]
    titles = [(r["idx"], r["title"].strip(), toks(r["title"])) for r in rows]
    usable = [t for t in titles if len(t[2]) >= 4]
    NEG = toks("A wholly fabricated title concerning imaginary quantum bicycles in Antarctica")
    print("corpus %d, titles with >=4 distinctive tokens %d" % (len(titles), len(usable)))

    out, allhits = {}, []
    for f in sorted(os.listdir(SURV)):
        if not f.endswith(".pdf"):
            continue
        key = f.split("_")[0]
        ents = entries(ref_section(os.path.join(SURV, f)))
        # A real reference entry runs about thirty words. Anything far longer means the split
        # failed and the "entry" is a blob that matches almost any title, which is exactly how the
        # first run produced fourteen distinct titles all matching one Huegle et al. entry.
        ents = [e for e in ents if 5 <= len(e.split()) <= 60]
        etok = [(e, toks(e)) for e in ents]
        assert not any(len(NEG & et) / len(NEG) >= THRESH for _, et in etok), \
            "negative control matched in %s" % key
        hits = []
        for idx, ti, tk in usable:
            for e, et in etok:
                if len(tk & et) / len(tk) >= THRESH:
                    hits.append((idx, ti, e[:150]))
                    break
        out[key] = {"entries": len(ents), "hits": [h[0] for h in hits]}
        allhits += [(key,) + h for h in hits]
        print("  %-6s %4d reference entries   %2d corpus studies cited" % (key, len(ents), len(hits)))
    json.dump(out, io.open(os.path.join(HERE, "overlap_refs.json"), "w", encoding="utf-8"), indent=1)
    print("\n--- every hit, for hand inspection ---")
    for k, idx, ti, e in allhits:
        print("[%s] idx %-4s %s\n        matched: %s" % (k, idx, ti[:62], e[:110]))
    print("\ntotal hits: %d" % len(allhits))


if __name__ == "__main__":
    main()
