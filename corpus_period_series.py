# -*- coding: utf-8 -*-
"""How does the design space of the corpus move over time?

The coded corpus carries no publication year. The counts reported for the three periods
were therefore not reproducible from a repository artifact, which is the defect that
retired an earlier set of screening numbers. This recovers the year for every system by
joining the coded titles against the screening exports that do carry one, and then
recomputes every period count the review states.

The join is exact on a normalized title, with a close-match fallback at a 0.88 ratio for
the few titles whose punctuation differs between exports. Every one of the 94 systems
resolves. COREAN is excluded by title, since it is the authors' own system and is not part
of the reviewed corpus.

Period boundaries are inclusive of the cut years: up to 2018, 2019 through 2022, and 2023
onward.
"""
import collections
import csv
import difflib
import glob
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCREENED = os.path.join(ROOT, "data", "screened")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TITLE_COLS = ("title", "paper_title", "name")
YEAR_COLS = ("year", "pub_year", "publication_year", "date")


def normalize(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def load_corpus():
    path = os.path.join(SCREENED, "corpus_v9_coded.csv")
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if "COREAN" not in (r["title"] or "")]


def build_year_index():
    """Collect title -> year from every screening export that carries both fields."""
    index = {}
    for path in glob.glob(os.path.join(SCREENED, "*.csv")):
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            continue
        if not rows:
            continue
        cols = rows[0].keys()
        tcol = next((c for c in cols if c and c.lower() in TITLE_COLS), None)
        ycol = next((c for c in cols if c and c.lower() in YEAR_COLS), None)
        if not tcol or not ycol:
            continue
        for r in rows:
            t = normalize(r.get(tcol))
            m = re.search(r"(19|20)\d{2}", str(r.get(ycol) or ""))
            if t and m:
                index.setdefault(t, int(m.group(0)))
    return index


def resolve_years(corpus, index):
    keys = list(index)
    years, unresolved = {}, []
    for r in corpus:
        t = normalize(r["title"])
        y = index.get(t)
        if y is None:
            close = difflib.get_close_matches(t, keys, n=1, cutoff=0.88)
            y = index[close[0]] if close else None
        if y is None:
            unresolved.append(r["title"])
        else:
            years[r["idx"]] = y
    return years, unresolved


def period(year):
    if year <= 2018:
        return 0
    return 1 if year <= 2022 else 2


def main():
    corpus = load_corpus()
    years, unresolved = resolve_years(corpus, build_year_index())
    print("corpus size: %d" % len(corpus))
    print("years resolved: %d" % len(years))
    if unresolved:
        print("UNRESOLVED (counts below would be incomplete):")
        for t in unresolved:
            print("   ", t[:78])
        return 1

    ys = sorted(years.values())
    print("year range: %d to %d" % (ys[0], ys[-1]))
    print("published before 2018: %d, from 2018 onward: %d"
          % (sum(1 for y in ys if y < 2018), sum(1 for y in ys if y >= 2018)))

    per = {r["idx"]: period(years[r["idx"]]) for r in corpus}
    totals = [sum(1 for r in corpus if per[r["idx"]] == i) for i in (0, 1, 2)]
    print("\nperiods (to 2018 | 2019-2022 | 2023 on): %s" % totals)

    axes = [
        ("state_representation", ["graph-encoder", "raw-vector", "unclear"]),
        ("action_granularity", ["candidate-path", "next-link", "unclear"]),
        ("algorithm_family_coarse", ["Tabular-Q-Sarsa", "DQN-family", "Policy-gradient-AC"]),
    ]
    for field, values in axes:
        print("\n%s" % field)
        for v in values:
            c = collections.Counter(per[r["idx"]] for r in corpus if r[field] == v)
            counts = [c[i] for i in (0, 1, 2)]
            print("  %-22s %-14s total %d" % (v, counts, sum(counts)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
