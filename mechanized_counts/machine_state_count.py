# -*- coding: utf-8 -*-
"""Recompute the state-form count from the full texts rather than from picked quotations.

WHY. The forecast-conditioned count is a prevalence figure, and a prevalence figure invites the
question of who decided each case. experiments/recode_state_field.py already answers half of it:
the classification is a published rule, not a reading. What the rule reads, though, is a
quotation, and the quotation was picked by the same reader whose values are being checked. This
script removes that remaining step. The rule's input becomes every sentence of a study's full
text that a fixed vocabulary selects, so the whole path from PDF to count is rerunnable.

The classification rule is not restated here. It is imported from recode_state_field, unchanged,
so that a difference between this count and the recorded one can only come from the input.

The selection vocabulary is state_lexicon.txt, derived sentence by sentence from Section III-A
and fixed before any count was produced.

    python machine_state_count.py            # counts, agreement, and the disagreement table
    python machine_state_count.py --list     # print the selected sentences behind each difference

Outputs machine_state_counts.csv next to this file, one row per study.
"""
import argparse
import csv
import io
import json
import os
import re
import sys
import difflib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(HERE))

import recode_state_field as recorded_rule   # noqa: E402  the published classification rule

# The corpus lives in corpus/ of this repository. The full texts are copyrighted and are not
# redistributed; point PDF_DIR at a directory holding them, or let the script fall back to the
# extracted text shipped in _text_cache.
CORPUS = os.path.join(ROOT, "corpus", "corpus_v9_coded.csv")
PDF_BASE = os.environ.get("PDF_DIR", os.path.join(ROOT, "pdfs"))
PDF_DIRS = [PDF_BASE,
            os.path.join(PDF_BASE, "pdfs"),
            os.path.join(PDF_BASE, "pdfs_unlocked"),
            os.path.join(PDF_BASE, "pdfs_acquired")]
LEXICON = os.path.join(HERE, "state_lexicon.txt")
CACHE = os.path.join(HERE, "_text_cache")
OUT = os.path.join(HERE, "machine_state_counts.csv")
MISSING = ("the reviewed full texts are not present: PDF_DIR names no directory of PDFs and "
           "%s is absent.\n"
           "The full texts are copyrighted and are not redistributed here; set PDF_DIR to a "
           "directory holding them to rebuild this count." % CACHE)

AUTHOR_ROW = "93"                 # the author's own study, outside the reviewed corpus
MATCH_FLOOR = 0.80                # title similarity below which no PDF is claimed
PREDICTIVE = recorded_rule.PREDICTIVE


# --------------------------------------------------------------------------- lexicon

def load_lexicon(path):
    groups, current = {}, None
    for line in io.open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            groups[current] = []
            continue
        groups[current].append(line)
    return {k: re.compile("|".join("(?:%s)" % p for p in v), re.I) for k, v in groups.items()}


# --------------------------------------------------------------------------- pdf text

def norm_title(s):
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def build_pdf_index():
    index = {}
    for d in PDF_DIRS:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(".pdf"):
                index.setdefault(norm_title(os.path.splitext(fn)[0]), os.path.join(d, fn))
    if index:
        return index
    # The reviewed PDFs cannot be redistributed, so the released package ships the text already
    # extracted from them and this falls back to it. Without the fallback the script reported
    # "no PDF matched" for all 94 studies and every agreement figure came out at zero, which is
    # not the behaviour Section II-B describes.
    if os.path.isdir(CACHE):
        for fn in sorted(os.listdir(CACHE)):
            if fn.lower().endswith(".txt"):
                index.setdefault(norm_title(os.path.splitext(fn)[0]), os.path.join(CACHE, fn))
    return index


def match_pdf(title, index, keys):
    t = norm_title(title)
    if t in index:
        return index[t], 1.0
    m = difflib.get_close_matches(t, keys, n=1, cutoff=MATCH_FLOOR)
    if not m:
        return None, 0.0
    return index[m[0]], difflib.SequenceMatcher(None, t, m[0]).ratio()


def raw_text(path):
    """Full text of a PDF, cached, with the reference list removed.

    The reference list is dropped because its entries carry the vocabulary of every field a
    study cites, and none of them describe the study's own state.
    """
    if path.lower().endswith(".txt"):
        return io.open(path, encoding="utf-8").read()
    key = os.path.join(CACHE, re.sub(r"[^A-Za-z0-9]+", "_", os.path.basename(path))[:120] + ".txt")
    if os.path.exists(key):
        return io.open(key, encoding="utf-8").read()
    import fitz
    doc = fitz.open(path)
    text = "\n".join(pg.get_text("text") for pg in doc)
    doc.close()
    cuts = [m.start() for m in re.finditer(
        r"(?im)^\s*(?:\d+\.?\s*)?(?:references|bibliography|reference list)\s*$", text)]
    if cuts and cuts[-1] > 0.4 * len(text):
        text = text[:cuts[-1]]
    if not os.path.isdir(CACHE):
        os.makedirs(CACHE)
    io.open(key, "w", encoding="utf-8").write(text)
    return text


def sentences(text):
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", text)
    return [p.strip() for p in parts if 25 < len(p.strip()) < 700]


# --------------------------------------------------------------------------- selection

def select(sents, lex, window):
    """Sentences the fixed vocabulary hands to the rule.

    window=0 takes the matching sentence alone. window=1 also takes the sentence that follows it,
    the declared sensitivity variant, because a state definition often continues past its opening
    sentence. Both variants are reported; neither was chosen after seeing a count.
    """
    keep = set()
    for i, s in enumerate(sents):
        if lex["DROP_SENTENCE"].search(s):
            continue
        if lex["SELECT_STATE"].search(s) or lex["SELECT_ROLLOUT"].search(s):
            keep.add(i)
            for j in range(1, window + 1):
                if i + j < len(sents) and not lex["DROP_SENTENCE"].search(sents[i + j]):
                    keep.add(i + j)
    return [sents[i] for i in sorted(keep)]


# --------------------------------------------------------------------------- report

def summarize(label, pairs, n_rows):
    """pairs: list of (idx, recorded, machine) over studies that have both values."""
    agree = sum(1 for _, r, m in pairs if r == m)
    differ = [(i, r, m) for i, r, m in pairs if r != m]
    n_mach = sum(1 for _, _, m in pairs if m == "forecast")
    n_rec = sum(1 for _, r, _ in pairs if r == "forecast")
    print("%s" % label)
    print("  studies compared                   %d" % len(pairs))
    print("  recorded forecast-conditioned      %d" % n_rec)
    print("  machine forecast-conditioned       %d" % n_mach)
    print("  machine reproduces the record      %d  (%.1f%%)"
          % (agree, 100.0 * agree / len(pairs) if pairs else 0))
    print("  machine differs from the record    %d" % len(differ))
    rec_f_mach_i = sum(1 for _, r, m in differ if r == "forecast")
    print("     recorded forecast, machine instantaneous   %d" % rec_f_mach_i)
    print("     recorded instantaneous, machine forecast   %d" % (len(differ) - rec_f_mach_i))
    print()
    return differ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true",
                    help="print the selected sentences behind each difference")
    ap.add_argument("--lexicon", default=LEXICON,
                    help="selection vocabulary to run; state_lexicon_v2.txt is the second version")
    a = ap.parse_args()

    lex = load_lexicon(a.lexicon)
    print("Lexicon                              %s\n" % os.path.basename(a.lexicon))
    index = build_pdf_index()
    if not index:
        sys.exit(MISSING)
    keys = list(index.keys())
    rows = [r for r in csv.DictReader(io.open(CORPUS, encoding="utf-8"))
            if (r["idx"] or "").strip() != AUTHOR_ROW]

    out_rows, unmatched, unreadable = [], [], []
    evidence = {}
    for r in rows:
        idx = r["idx"]
        path, ratio = match_pdf(r["title"], index, keys)
        rec = r["predictive_representation"]
        recorded = ("forecast" if rec in PREDICTIVE else
                    "instantaneous" if rec == "none" else "")
        row = {"idx": idx, "title": r["title"], "recorded_field": rec, "recorded": recorded,
               "pdf": os.path.basename(path) if path else "", "match_ratio": "%.3f" % ratio,
               "n_selected_w0": "", "n_selected_w1": "", "machine_w0": "", "machine_w1": ""}
        if not path:
            unmatched.append((idx, r["title"]))
            out_rows.append(row)
            continue
        try:
            sents = sentences(raw_text(path))
        except Exception as exc:
            unreadable.append((idx, str(exc)[:80]))
            out_rows.append(row)
            continue
        for w in (0, 1):
            sel = select(sents, lex, w)
            verdict = recorded_rule.classify(" ".join(sel))
            row["n_selected_w%d" % w] = len(sel)
            row["machine_w%d" % w] = verdict or "unsettled"
            if w == 0:
                evidence[idx] = sel
        out_rows.append(row)

    tag = "_v2" if a.lexicon.endswith("v2.txt") else ""
    out_path = OUT.replace(".csv", tag + ".csv")
    with io.open(out_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    print("Studies reviewed                     %d" % len(rows))
    print("Full text matched and read           %d" % sum(1 for r in out_rows if r["machine_w0"]))
    print("  no PDF matched at ratio >= %.2f    %d" % (MATCH_FLOOR, len(unmatched)))
    print("  PDF matched but unreadable         %d" % len(unreadable))
    print("Recorded as forecast-conditioned     %d  (over all %d)"
          % (sum(1 for r in out_rows if r["recorded"] == "forecast"), len(rows)))
    print()

    diffs = {}
    for w, label in ((0, "Selected sentence alone (primary)"),
                     (1, "Selected sentence and the one after it (declared sensitivity)")):
        pairs = [(r["idx"], r["recorded"], r["machine_w%d" % w]) for r in out_rows
                 if r["recorded"] and r["machine_w%d" % w] in ("forecast", "instantaneous")]
        diffs[w] = summarize(label, pairs, len(rows))

    print("Studies with no PDF, carried at their recorded value:")
    for idx, t in unmatched:
        print("  idx %-4s %s" % (idx, t[:88]))
    if unreadable:
        print("Studies whose PDF would not extract:")
        for idx, e in unreadable:
            print("  idx %-4s %s" % (idx, e))

    print("\nDifferences under the primary variant:")
    for idx, rec, mach in diffs[0]:
        title = next(r["title"] for r in out_rows if r["idx"] == idx)
        print("  idx %-4s recorded %-13s machine %-13s  %s" % (idx, rec, mach, title[:58]))
        if a.list:
            for s in evidence.get(idx, []):
                if recorded_rule.FORECAST.search(s):
                    print("        > %s" % s[:260])

    json.dump({k: v for k, v in evidence.items()},
              io.open(os.path.join(HERE, "selected_sentences%s.json" % tag), "w",
                      encoding="utf-8"), indent=1, ensure_ascii=False)
    print("\nWrote %s" % os.path.basename(out_path))


if __name__ == "__main__":
    main()
