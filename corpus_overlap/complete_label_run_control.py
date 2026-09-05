# -*- coding: utf-8 -*-
"""Does the under-recovery of three reference lists hide a corpus study?

overlap_refs_complete.py recovers 1245 entries across the ten related reviews and reports the
highest reference label printed in each list. Three lists recover fewer entries than that label:
42 against 69, 62 against 73 and 32 against 46. The recovered entries carry the zero count, so an
entry lost to the extraction is an entry never searched.

Whether the zero survives the entries the extraction drops is therefore open. Two of
the three short lists retain a complete run of labels, from 1 to the highest, which allows every
entry of those two lists to be delimited whether or not the entry text passes the length filter of
the main rule. This control searches every label-delimited entry of those two lists, 69 and 73 of
them, at the published threshold of 0.85.

The third list retains no complete run and is not reconstructable here; its 14 unrecovered entries
are adjudicated from the citation context of the review in Supplementary S-V.
"""
import csv, glob, io, json, os, re, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SURV = "/home/dhlee/review_paper/pdfs/surveys"
CSV = os.path.join(os.path.dirname(HERE), "corpus", "corpus_v9_coded.csv")
THRESH = 0.85
STOP = set("a an the of in on for to and or with from by using based via toward towards".split())


def toks(t):
    return {w for w in re.findall(r"[a-z]+", (t or "").lower()) if len(w) >= 4 and w not in STOP}


def label_run(text):
    """The longest run of consecutive reference labels, with every span kept."""
    marks = [(int(m.group(1)), m.start()) for m in re.finditer(r'\[(\d{1,3})\]', text)]
    best, run = [], []
    for n, pos in marks:
        if run and n != run[-1][0] + 1:
            if len(run) > len(best):
                best = run
            run = []
        run.append((n, pos))
    if len(run) > len(best):
        best = run
    if len(best) < 20:
        return None, None
    ents = [' '.join(text[p:(best[i + 1][1] if i + 1 < len(best) else min(len(text), p + 600))].split())
            for i, (n, p) in enumerate(best)]
    return ents, max(n for n, _ in best)


def main():
    rows = [r for r in csv.DictReader(io.open(CSV, encoding="utf-8"))
            if r["in_reviewed_corpus"].strip().lower() in ("1", "true", "yes")]
    usable = [(r["idx"], r["title"], toks(r["title"])) for r in rows if len(toks(r["title"])) >= 4]
    print("corpus %d\n" % len(usable))

    out = {}
    for pdf in sorted(glob.glob(os.path.join(SURV, "*.pdf"))):
        key = os.path.basename(pdf).split("_")[0]
        text = subprocess.run(["pdftotext", "-q", pdf, "-"], capture_output=True, text=True).stdout
        ents, high = label_run(text)
        if not ents or len(ents) != high:
            continue                      # the run is broken, so the list is not reconstructable
        hits = []
        for idx, title, tk in usable:
            for e in ents:
                if len(tk & toks(e)) / len(tk) >= THRESH:
                    hits.append({"idx": idx, "title": title, "entry": e[:200]})
                    break
        print("%-8s complete run of %d labels, every entry searched: %s"
              % (key, high, [h["idx"] for h in hits] or "no corpus study"))
        out[key] = {"labels": high, "entries_searched": len(ents),
                    "corpus_hits": [h["idx"] for h in hits], "hit_entries": hits}

    for k, v in out.items():
        for h in v["hit_entries"]:
            print("\n  HIT %s in %s\n    corpus title: %s\n    entry: %s" % (h["idx"], k, h["title"], h["entry"]))

    io.open(os.path.join(HERE, "complete_label_run_control.json"), "w", encoding="utf-8").write(
        json.dumps(out, ensure_ascii=False, indent=1))
    print("\nwrote complete_label_run_control.json")


if __name__ == "__main__":
    main()
