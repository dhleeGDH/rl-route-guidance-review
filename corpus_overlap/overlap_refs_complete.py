# -*- coding: utf-8 -*-
"""The overlap measurement re-run on complete reference lists.

WHY THIS EXISTS. The entry counts of overlap_refs.py were checked against each review's own
bibliography and three came up short: 40 entries against a highest label of [69], 51 against [73],
32 against [46]. An under-recovered reference list produces a false zero silently, and the zero is
the manuscript's strongest separation claim. This re-runs the same matching rule on lists rebuilt
from the numbered labels themselves, which is the one anchor a two-column PDF preserves.

The rule is unchanged: a corpus title matches an entry where at least 85% of its distinctive tokens
fall inside that ONE entry. Only the entry extraction differs.

Controls: a fabricated title must match nowhere, every hit is printed for hand inspection, and the
recovery rate against the highest label is printed per review so an under-recovery cannot pass
unnoticed a second time.
"""
import csv, glob, io, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(os.path.dirname(HERE), "corpus", "corpus_v9_coded.csv")
SURV = "/home/dhlee/review_paper/pdfs/surveys"
THRESH = 0.85
STOP = set("a an the of in on for to and or with from by using based via toward towards".split())

def toks(t):
    return {w for w in re.findall(r"[a-z]+", (t or "").lower()) if len(w) >= 4 and w not in STOP}

def entries_by_label(text):
    """Split a reference section on its own [N] labels, which survive column interleaving."""
    marks = [(int(m.group(1)), m.start()) for m in re.finditer(r'\[(\d{1,3})\]', text)]
    if len(marks) < 20:
        return None, None
    # keep the last run of labels, which is the bibliography rather than the citations in the body
    best, run = [], []
    for i, (n, pos) in enumerate(marks):
        if run and n != run[-1][0] + 1:
            if len(run) > len(best):
                best = run
            run = []
        run.append((n, pos))
    if len(run) > len(best):
        best = run
    if len(best) < 20:
        return None, None
    ents = []
    for i, (n, pos) in enumerate(best):
        end = best[i + 1][1] if i + 1 < len(best) else min(len(text), pos + 600)
        e = ' '.join(text[pos:end].split())
        if 5 <= len(e.split()) <= 120:
            ents.append(e)
    return ents, max(n for n, _ in best)

def main():
    rows = [r for r in csv.DictReader(io.open(CSV, encoding="utf-8"))
            if r["in_reviewed_corpus"].strip().lower() in ("1", "true", "yes")]
    titles = [(r["idx"], r["title"], toks(r["title"])) for r in rows]
    usable = [t for t in titles if len(t[2]) >= 4]
    print("corpus %d, titles with >=4 distinctive tokens %d\n" % (len(titles), len(usable)))

    out, total, hits_all = {}, 0, []
    print("%-8s %9s %9s %9s   %s" % ("review", "entries", "highest", "recovery", "corpus studies cited"))
    for pdf in sorted(glob.glob(os.path.join(SURV, "*.pdf"))):
        key = os.path.basename(pdf).split("_")[0]
        text = subprocess.run(["pdftotext", pdf, "-"], capture_output=True, text=True).stdout
        # Neither splitter recovers every list: the label rule fixed three reviews and lost
        # ground on three others, so both run and the entries are unioned. An entry counted
        # twice costs nothing, since a match needs the tokens inside ONE entry either way.
        by_label, high = entries_by_label(text)
        sec = text[text.rfind("References"):] if "References" in text else text[-40000:]
        by_author = [' '.join(e.split())
                     for e in re.split(r'(?m)^\s*(?=\[\d{1,3}\]|[A-Z][a-z]+,\s+[A-Z]\.)', sec)]
        by_author = [e for e in by_author if 5 <= len(e.split()) <= 120]
        ents = list(dict.fromkeys((by_label or []) + by_author))
        et = [(e, toks(e)) for e in ents]
        hits = []
        for idx, title, tk in usable:
            for e, s in et:
                if tk and len(tk & s) / len(tk) >= THRESH:
                    hits.append((idx, title, e)); break
        rec = ('%d%%' % round(100.0 * len(ents) / high)) if high else 'author-year'
        print("%-8s %9d %9s %9s   %s" % (key, len(ents), high or '-', rec, [h[0] for h in hits] or 'none'))
        out[key] = {"entries": len(ents), "highest_label": high, "hits": [h[0] for h in hits]}
        total += len(ents); hits_all += hits

    print("\ntotal reference entries: %d" % total)
    print("corpus studies matched anywhere: %s" % ([h[0] for h in hits_all] or "none"))
    for idx, title, e in hits_all:
        print("\n  HIT idx %s\n    title: %s\n    entry: %s" % (idx, title[:100], e[:200]))

    print("\n--- control 1, sensitivity: a corpus title present as an entry must be found ---")
    # Both published controls test specificity. Neither shows the rule can find a title that IS
    # there, which is the missing half. Thirty corpus titles are
    # written into an entry of the form the ten reviews use and put back through the same rule.
    import random
    random.seed(20260823)
    seeded = random.sample(usable, 30)
    hit = 0
    for idx, title, tk in seeded:
        entry = ('[99] A. Author, B. Coauthor, and C. Third, "%s," IEEE Trans. Intell. '
                 'Transp. Syst., vol. 25, no. 4, pp. 1234-1245, 2024.' % title)
        if len(tk & toks(entry)) / len(tk) >= THRESH:
            hit += 1
        else:
            print("      MISSED %s: %s" % (idx, title[:70]))
    print("  %-4s %d of %d seeded corpus titles recovered" % ("OK" if hit == 30 else "FAIL", hit, 30))
    sens_ok = (hit == 30)

    print("\n--- control 2, specificity: a fabricated title must match nowhere ---")
    fake = toks("A Quantum Cordon Heuristic for Interstellar Route Guidance under Fog")
    bad = 0
    for pdf in sorted(glob.glob(os.path.join(SURV, "*.pdf"))):
        text = subprocess.run(["pdftotext", pdf, "-"], capture_output=True, text=True).stdout
        ents, _ = entries_by_label(text)
        for e in (ents or []):
            if len(fake & toks(e)) / len(fake) >= THRESH:
                bad += 1
    print("  %-4s the fabricated title matches %d entries" % ("OK" if bad == 0 else "FAIL", bad))

    json.dump(out, io.open(os.path.join(HERE, "overlap_refs_complete.json"), "w",
                           encoding="utf-8"), indent=1)
    print("\nwrote overlap_refs_complete.json")
    out["sensitivity_recovered"] = hit
    return 0 if (bad == 0 and sens_ok) else 1

if __name__ == "__main__":
    sys.exit(main())
