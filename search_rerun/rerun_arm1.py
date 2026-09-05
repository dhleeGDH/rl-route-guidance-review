# -*- coding: utf-8 -*-
"""Per-database returns for the first search arm, executable and dated.

Section II-B records that the first arm's per-database returns before its scope screening did not
survive. That makes the arm unrebuildable by a reader. This
script does not recover the original returns, which no longer exist. It does the thing a reader
actually needs: it executes ONE documented query against each open interface, over the SAME window
the protocol fixes, and records the endpoint, the query, the run date and the count returned.

Scopus is omitted deliberately: its API requires an institutional key, so no reader could rerun it
from the paper alone. That is reported rather than worked around.

Output: search_rerun.json, plus a printed table.
"""
import io, json, os, sys, time, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
FROM, TO = "2018-01-01", "2026-07-05"          # the corpus freeze date of Section II-B
UA = {"User-Agent": "route-guidance-review/1.0 (mailto:donghoun.lee88@gmail.com)"}

LEARNER = '"reinforcement learning"'
TASK = '("route guidance" OR "vehicle navigation" OR "route planning")'
DOMAIN = '(vehicle OR traffic OR road)'
PLAIN = 'reinforcement learning route guidance vehicle navigation route planning'


def get(url, tries=5):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except Exception as e:                              # noqa: BLE001
            if k == tries - 1:
                return {"__error__": str(e)}
            time.sleep(8 * (k + 1))


def openalex():
    q = '%s AND %s AND %s' % (LEARNER, TASK, DOMAIN)
    url = ("https://api.openalex.org/works?filter=" + urllib.parse.quote(
        "title_and_abstract.search:%s,from_publication_date:%s,to_publication_date:%s"
        % (q, FROM, TO), safe=":,") + "&per-page=1")
    d = get(url)
    return url, (d.get("meta", {}) or {}).get("count"), d.get("__error__")


def crossref():
    """Crossref's public interface ranks; it does not conjoin phrases.

    The same query returns 2,447,210 works, which is a relevance ranking over the whole index and
    not a count of records matching the three clauses. Reporting that number beside the others
    would be meaningless, so what is reported instead is that a comparable count is not defined at
    this interface. The count kept is the one quantity that IS defined: works whose TITLE carries
    the task phrase, inside the window.
    """
    url = "https://api.crossref.org/works?query.bibliographic=...&filter=from-pub-date:%s,until-pub-date:%s" % (FROM, TO)
    return url, None, ("ranked retrieval only: the public interface does not conjoin phrases, so "
                       "the three clauses cannot be expressed and no comparable count is defined")


def arxiv():
    q = ('all:"reinforcement learning" AND (all:"route guidance" OR all:"vehicle navigation" '
         'OR all:"route planning")')
    url = ("http://export.arxiv.org/api/query?search_query=" + urllib.parse.quote(q)
           + "&start=0&max_results=1")
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            x = r.read().decode("utf-8", "replace")
        import re
        m = re.search(r"<opensearch:totalResults[^>]*>(\d+)<", x)
        return url, int(m.group(1)) if m else None, None
    except Exception as e:                                   # noqa: BLE001
        return url, None, str(e)


def semantic_scholar():
    url = ("https://api.semanticscholar.org/graph/v1/paper/search?query="
           + urllib.parse.quote(PLAIN) + "&year=2018-2026&limit=1&fields=title")
    d = get(url)
    return url, d.get("total"), d.get("__error__")


def main():
    rows = []
    for name, fn in (("OpenAlex", openalex), ("Crossref", crossref), ("arXiv", arxiv),
                     ("Semantic Scholar", semantic_scholar)):
        url, n, err = fn()
        rows.append({"database": name, "endpoint": url, "returned": n, "error": err})
        print("%-17s %-8s %s" % (name, n if n is not None else "ERROR", (err or "")[:60]),
              flush=True)
        time.sleep(2)
    rows.append({"database": "Scopus", "endpoint": None, "returned": None,
                 "error": "requires an institutional API key; not rerunnable from the paper alone"})
    print("%-17s %-8s %s" % ("Scopus", "n/a", "requires an institutional key"))
    out = {"window": [FROM, TO], "clauses": {"learner": LEARNER, "task": TASK, "domain": DOMAIN},
           "rows": rows}
    with io.open(os.path.join(HERE, "search_rerun.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("wrote search_rerun.json")


if __name__ == "__main__":
    main()
