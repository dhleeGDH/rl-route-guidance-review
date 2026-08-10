# -*- coding: utf-8 -*-
"""The IEEE against non-IEEE split of the corpus, derived from the released record.

WHY THIS EXISTS. Section II-B once reported a split by search arm, at 45 studies from the
reproducible export against 48 from the citation arm. Those two total 93 rather than 94, and no
archived file attributes an arm to a study: the manuscript states that the first arm was not
logged per method, so that split cannot be reproduced and has been withdrawn.

The publisher split can be reproduced, and it answers the same question, which is whether a skew
in where the corpus was found moves either headline rate. It is derived here from two released
files and nothing else:

    manuscript/source_md/10_appendix_corpus_list.md   the 94 studies with their recorded values
    references/references_section.md                  the venue string of each reference

A study counts as IEEE when the word appears in its venue string. That rule is crude at the edges,
since a conference can be published by IEEE without carrying the word, and it is stated rather
than hidden so a reader can apply a different one to the same two files.

    python3 publisher_split.py

Writes publisher_split.json next to this file.
"""
import io
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
APPENDIX = ROOT / "manuscript" / "source_md" / "10_appendix_corpus_list.md"
REFS = ROOT / "references" / "references_section.md"

PRED_COL, BOUND_COL, COMPL_COL, CODE_COL = 3, 5, 6, 7


def corpus_rows():
    out = []
    for line in io.open(str(APPENDIX), encoding="utf-8"):
        s = line.strip()
        if not s.startswith("|"):
            continue
        m = re.match(r"\|\s*\[(\d+)\]†?\s*\|", s)
        if not m:
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        out.append((m.group(1), cells))
    return out


def venues():
    v = {}
    for line in io.open(str(REFS), encoding="utf-8"):
        m = re.match(r"\[(\d+)\]\s*(.*)", line.strip())
        if m:
            v[m.group(1)] = m.group(2)
    return v


def share(rows, sel, col, want):
    sub = [c for r, c in rows if sel(r)]
    det = [c for c in sub if c[col] not in ("?", "unclear", "")]
    hit = [c for c in det if any(c[col].startswith(w) for w in want)]
    return len(hit), len(det)


def main():
    rows, v = corpus_rows(), venues()
    if len(rows) != 94:
        raise SystemExit("expected 94 corpus rows, parsed %d" % len(rows))
    ieee = lambda r: "ieee" in v.get(r, "").lower()                       # noqa: E731
    n_i = sum(1 for r, _ in rows if ieee(r))
    res = {"ieee": n_i, "non_ieee": len(rows) - n_i, "fields": {}}
    for label, col, want in (("forecast_conditioned", PRED_COL, ("local", "roll")),
                             ("open_boundary", BOUND_COL, ("open",)),
                             ("trip_completion", COMPL_COL, ("yes",)),
                             ("code_release", CODE_COL, ("yes",))):
        a = share(rows, ieee, col, want)
        b = share(rows, lambda r: not ieee(r), col, want)
        res["fields"][label] = {"ieee": list(a), "non_ieee": list(b),
                                "ieee_pct": round(100.0 * a[0] / max(1, a[1]), 1),
                                "non_ieee_pct": round(100.0 * b[0] / max(1, b[1]), 1)}
    (HERE / "publisher_split.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("IEEE %d, non-IEEE %d\n" % (res["ieee"], res["non_ieee"]))
    for k, d in res["fields"].items():
        print("  %-22s IEEE %2d of %2d (%4.1f%%)   non-IEEE %2d of %2d (%4.1f%%)"
              % (k, d["ieee"][0], d["ieee"][1], d["ieee_pct"],
                 d["non_ieee"][0], d["non_ieee"][1], d["non_ieee_pct"]))
    print("\nwrote publisher_split.json")


if __name__ == "__main__":
    main()
