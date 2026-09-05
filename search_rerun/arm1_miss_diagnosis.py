# -*- coding: utf-8 -*-
"""Why the re-executed queries miss the corpus studies they miss (REVISION_PLAN 3.2.3).

arm1_reconstruct.py reports the share of the corpus the stated queries return. A share alone does
not say what the shortfall means. Two causes are very different for a reader:

  indexed, not returned   the record is in the interface and the query did not select it. That is a
                          property of the query, and a wider query would recover it.
  not indexed / outside   the interface does not hold the record, or holds it outside the window.
                          No formulation of the query recovers it, and the record reaches the corpus
                          through the other arm or through citation searching.

Each missed corpus title is looked up in OpenAlex by title search alone, with no query clause and no
window, and the two causes are separated. Nothing here admits any record to the corpus.

    python3 arm1_miss_diagnosis.py
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
HITS = os.path.join(HERE, "arm1_reconstruct.csv")
OUT = os.path.join(HERE, "arm1_miss_diagnosis.csv")
UA = {"User-Agent": "route-guidance-review/1.0 (mailto:donghoun.lee88@gmail.com)"}

STOP = set("""the and for with from into over under this that these those are was were been being
have has had its their his her our your not but nor via per upon than then when where which who
whom whose what how why all any both each few more most other some such only own same too very
can will just should now""".split())


def toks(t):
    return {w for w in re.findall(r"[a-z0-9]+", (t or "").lower())
            if len(w) >= 4 and w not in STOP}


def lookup(title):
    url = ("https://api.openalex.org/works?per-page=25&select=display_name,publication_year"
           "&filter=" + urllib.parse.quote("title.search:" + title[:200], safe=":,"))
    for k in range(4):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except Exception:                                        # noqa: BLE001
            time.sleep(6 * (k + 1))
    return None


def main():
    rows = list(csv.DictReader(io.open(HITS, encoding="utf-8")))
    misses = [r for r in rows if r["recovered"] == "no"]
    print("%d of %d corpus studies were not returned by the stated queries" % (len(misses), len(rows)))
    out = []
    for i, r in enumerate(misses, 1):
        ct = toks(r["title"])
        d = lookup(r["title"])
        best, share, year = "", 0.0, None
        if d:
            for w in d.get("results", []):
                sh = len(ct & toks(w.get("display_name"))) / float(max(1, len(ct)))
                if sh > share:
                    best, share, year = w.get("display_name"), sh, w.get("publication_year")
        indexed = share >= 0.85
        in_window = bool(year and 2018 <= year <= 2026)
        out.append({"idx": r["idx"], "title": r["title"][:110],
                    "indexed_in_openalex": "yes" if indexed else "no",
                    "indexed_year": year if indexed else "",
                    "inside_window": ("yes" if in_window else "no") if indexed else "",
                    "best_share": round(share, 3),
                    "cause": ("query did not select it" if indexed and in_window else
                              "outside the window" if indexed else
                              "not found by title in the interface")})
        print("  %3d/%d idx %-4s %-28s %s" % (i, len(misses), r["idx"], out[-1]["cause"],
                                              r["title"][:44]), flush=True)
        time.sleep(0.4)

    with io.open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    import collections
    c = collections.Counter(o["cause"] for o in out)
    print()
    for k, v in c.most_common():
        print("  %-34s %d" % (k, v))
    print("\nwrote %s" % os.path.basename(OUT))


if __name__ == "__main__":
    main()
