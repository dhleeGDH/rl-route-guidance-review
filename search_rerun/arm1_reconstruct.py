# -*- coding: utf-8 -*-
"""Re-execute the first search arm and measure what it recovers (REVISION_PLAN 3.2).

Section II-B records that the first arm's per-database returns did not survive. The log is
unrecoverable; the queries are re-executable. rerun_arm1.py already reports the COUNT each open
interface returns under the protocol window. This script goes one step further and reports the
quantity a reader actually needs: of the 94 studies the corpus holds, how many the re-executed
queries return.

That is the decisive metric. A count alone says a query still runs. A recovery rate says how much
of the corpus a reader could rebuild from the queries the paper states.

  window        publication date filtered to the 5 July 2026 freeze
  deduplication DOI first, then a normalized title
  matching      a corpus title matches a returned record where at least 85% of its distinctive
                tokens appear in that record's title, the rule corpus_overlap/overlap_refs.py uses
  admission     NONE. A record the reconstruction returns and the original arm did not enters the
                sensitivity analysis and never the corpus.

Scopus is omitted deliberately: its interface requires an institutional key, so no reader could
rerun it from the paper alone. Crossref's public interface ranks and does not conjoin phrases, so
no comparable result set is defined there. Both are reported rather than worked around.

Output: arm1_reconstruct.csv (one row per corpus study) and arm1_reconstruct.json.

    python3 arm1_reconstruct.py
    python3 arm1_reconstruct.py --offline   # score from a previous fetch, no network
"""
import csv
import io
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(os.path.dirname(HERE), "corpus", "corpus_v9_coded.csv")
RAW = os.path.join(HERE, "arm1_records.json")
OUT_CSV = os.path.join(HERE, "arm1_reconstruct.csv")
OUT_JSON = os.path.join(HERE, "arm1_reconstruct.json")

FROM, TO = "2018-01-01", "2026-07-05"
UA = {"User-Agent": "route-guidance-review/1.0 (mailto:donghoun.lee88@gmail.com)"}

LEARNER = '"reinforcement learning"'
TASK = '("route guidance" OR "vehicle navigation" OR "route planning")'
DOMAIN = '(vehicle OR traffic OR road)'
PLAIN = 'reinforcement learning route guidance vehicle navigation route planning'

STOP = set("""the and for with from into over under this that these those are was were been being
have has had its their his her our your not but nor via per upon than then when where which who
whom whose what how why all any both each few more most other some such only own same too very
can will just should now""".split())


def get(url, tries=5, timeout=60):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:                                        # noqa: BLE001
            if k == tries - 1:
                return "__error__" + str(e)
            time.sleep(8 * (k + 1))


def toks(title):
    """Distinctive tokens: alphabetic, four characters or more, outside the stop list."""
    return {w for w in re.findall(r"[a-z0-9]+", (title or "").lower())
            if len(w) >= 4 and w not in STOP}


def norm(title):
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


# ------------------------------------------------------------------ interfaces
def openalex():
    q = "%s AND %s AND %s" % (LEARNER, TASK, DOMAIN)
    base = ("https://api.openalex.org/works?per-page=200&select=id,display_name,doi,"
            "publication_year&filter=" + urllib.parse.quote(
                "title_and_abstract.search:%s,from_publication_date:%s,to_publication_date:%s"
                % (q, FROM, TO), safe=":,"))
    out, cursor = [], "*"
    while cursor:
        raw = get(base + "&cursor=" + urllib.parse.quote(cursor))
        if raw.startswith("__error__"):
            return out, raw[9:]
        d = json.loads(raw)
        for w in d.get("results", []):
            out.append({"title": w.get("display_name"), "doi": (w.get("doi") or "").lower(),
                        "year": w.get("publication_year"), "source": "OpenAlex"})
        cursor = (d.get("meta") or {}).get("next_cursor")
        if not d.get("results"):
            break
        time.sleep(0.4)
    return out, None


def arxiv():
    q = ('all:"reinforcement learning" AND (all:"route guidance" OR all:"vehicle navigation" '
         'OR all:"route planning")')
    out = []
    for start in range(0, 400, 100):
        raw = get("http://export.arxiv.org/api/query?search_query=" + urllib.parse.quote(q)
                  + "&start=%d&max_results=100" % start)
        if raw.startswith("__error__"):
            return out, raw[9:]
        titles = re.findall(r"<entry>.*?<title>(.*?)</title>", raw, re.S)
        dates = re.findall(r"<entry>.*?<published>(\d{4})", raw, re.S)
        if not titles:
            break
        for i, t in enumerate(titles):
            y = int(dates[i]) if i < len(dates) else None
            if y and not (2018 <= y <= 2026):
                continue
            out.append({"title": " ".join(t.split()), "doi": "", "year": y, "source": "arXiv"})
        time.sleep(3)
    return out, None


def semantic_scholar():
    out = []
    for offset in range(0, 1000, 100):
        raw = get("https://api.semanticscholar.org/graph/v1/paper/search?query="
                  + urllib.parse.quote(PLAIN)
                  + "&year=2018-2026&limit=100&offset=%d&fields=title,externalIds,year" % offset)
        if raw.startswith("__error__"):
            return out, raw[9:]
        d = json.loads(raw)
        rows = d.get("data") or []
        for w in rows:
            doi = ((w.get("externalIds") or {}).get("DOI") or "").lower()
            out.append({"title": w.get("title"), "doi": doi, "year": w.get("year"),
                        "source": "Semantic Scholar"})
        if len(rows) < 100:
            break
        time.sleep(3)
    return out, None


NOT_RERUNNABLE = [
    ("Scopus", "requires an institutional API key; not rerunnable from the paper alone"),
    ("Crossref", "ranked retrieval only: the public interface does not conjoin phrases, so the "
                 "three clauses cannot be expressed and no comparable result set is defined"),
]


def fetch():
    records, errors, per_db = [], {}, {}
    for name, fn in (("OpenAlex", openalex), ("arXiv", arxiv),
                     ("Semantic Scholar", semantic_scholar)):
        rows, err = fn()
        per_db[name] = len(rows)
        if err:
            errors[name] = err
        records.extend(rows)
        print("  %-17s %5d records %s" % (name, len(rows), err or ""), flush=True)
    for name, why in NOT_RERUNNABLE:
        per_db[name] = None
        errors[name] = why
        print("  %-17s   n/a %s" % (name, why[:56]))
    return records, per_db, errors


def dedupe(records):
    seen_doi, seen_title, out = set(), set(), []
    for r in records:
        d, t = r["doi"], norm(r["title"])
        if d and d in seen_doi:
            continue
        if not d and t in seen_title:
            continue
        if d:
            seen_doi.add(d)
        seen_title.add(t)
        out.append(r)
    return out


def main():
    rows = list(csv.DictReader(io.open(CORPUS, encoding="utf-8-sig")))
    corpus = [r for r in rows
              if str(r.get("in_reviewed_corpus", "")).strip().lower() in ("1", "true", "yes", "y")]
    if len(corpus) != 94:
        sys.exit("FAILED: the record holds %d corpus studies, expected 94" % len(corpus))

    if "--offline" in sys.argv:
        if not os.path.exists(RAW):
            sys.exit("no arm1_records.json to score offline")
        blob = json.load(io.open(RAW, encoding="utf-8"))
        records, per_db, errors = blob["records"], blob["per_db"], blob["errors"]
        print("scoring %d cached records" % len(records))
    else:
        print("re-executing the stated queries, window %s to %s" % (FROM, TO))
        records, per_db, errors = fetch()
        json.dump({"records": records, "per_db": per_db, "errors": errors},
                  io.open(RAW, "w", encoding="utf-8"), ensure_ascii=False)

    uniq = dedupe(records)
    index = [(toks(r["title"]), r) for r in uniq]

    hits = []
    for c in corpus:
        ct = toks(c["title"])
        best, best_share = None, 0.0
        for rt, r in index:
            if not ct:
                continue
            share = len(ct & rt) / float(len(ct))
            if share > best_share:
                best, best_share = r, share
        hits.append({"idx": c["idx"], "title": c["title"][:120],
                     "recovered": "yes" if best_share >= 0.85 else "no",
                     "best_share": round(best_share, 3),
                     "matched_source": best["source"] if best and best_share >= 0.85 else "",
                     "matched_title": (best["title"][:120] if best and best_share >= 0.85 else "")})

    with io.open(OUT_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(hits[0].keys()))
        w.writeheader()
        w.writerows(hits)

    rec = sum(1 for h in hits if h["recovered"] == "yes")
    summary = {"window": [FROM, TO], "per_database": per_db, "errors": errors,
               "records_returned": len(records), "records_unique": len(uniq),
               "corpus": len(corpus), "recovered": rec,
               "recovery_pct": round(100.0 * rec / len(corpus), 1)}
    json.dump(summary, io.open(OUT_JSON, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    print()
    for k, v in per_db.items():
        print("  %-17s %s" % (k, v if v is not None else "not rerunnable"))
    print("\n  returned %d records, %d unique after deduplication" % (len(records), len(uniq)))
    print("  recovers %d of the %d corpus studies  (%.1f%%)"
          % (rec, len(corpus), 100.0 * rec / len(corpus)))
    print("\nwrote %s and %s" % (os.path.basename(OUT_CSV), os.path.basename(OUT_JSON)))


if __name__ == "__main__":
    main()
