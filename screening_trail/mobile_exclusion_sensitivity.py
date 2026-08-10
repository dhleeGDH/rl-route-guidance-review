# -*- coding: utf-8 -*-
"""What the "mobile" exclusion term removes from the search.

WHY THIS EXISTS. The IEEE Xplore arm applies an exclusion clause carrying the term "mobile", meant
to drop mobile-robot and handset work. "Mobile" also occurs in a vehicular sense, so the clause may
remove in-scope studies, and the loss would correlate with venue and community rather than fall at
random. The manuscript states the risk. This measures it.

The measurement does not run on IEEE Xplore. Automated access to the site is refused (HTTP 418),
its API needs a developer key the project does not hold, and an institutional subscription grants
reading rights rather than programmatic search. It runs on OpenAlex instead, which is open, covers
IEEE among every other publisher, and can be rerun by any reader without a subscription. The same
three clauses of the Xplore query are applied, once with the mobile exclusion and once without, and
the delta is the population the term removes.

    python3 mobile_exclusion_sensitivity.py             # counts and a screening list
    python3 mobile_exclusion_sensitivity.py --full      # write every delta record to CSV

Writes mobile_exclusion_delta.csv and mobile_exclusion_sensitivity.json next to this file.
"""
import argparse
import csv
import io
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAILTO = "donghoun.lee88@gmail.com"
BASE = "https://api.openalex.org/works"

# The three clauses of the Xplore query, as Section II-B and S-III state them.
LEARNER = ["deep reinforcement learning", "deep Q-network", "proximal policy optimization",
           "actor-critic"]
TASK = ["route guidance", "path planning", "dynamic routing", "traffic routing",
        "vehicle routing", "navigation"]
DOMAIN = ["vehicle", "traffic", "road", "driver", "urban", "transportation"]
EXCLUDE = ["pedestrian", "mobile", "robot", "maritime", "UAV", "obstacle"]

FROM, TO = "2018-01-01", "2026-12-31"


def get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:                                  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))
    return None


def q(learner, task, cursor=None, per_page=200):
    """One (learner, task) cell of the query grid, with no exclusion applied.

    OpenAlex takes one search value per filter entry and ANDs the entries, so the clause is
    expanded over its terms rather than passed as one boolean string. Negation is not available on
    a search filter (the API answers 400), so the exclusion clause and the domain clause are both
    applied to the returned title and abstract instead. Applying them here rather than at the
    server is what makes the delta exact: the same records are scored under both readings.
    """
    f = ["from_publication_date:" + FROM, "to_publication_date:" + TO,
         "title_and_abstract.search:" + '"%s"' % learner,
         "title_and_abstract.search:" + '"%s"' % task]
    # urlencode maps a space to "+", which OpenAlex rejects inside a quoted phrase (HTTP 400).
    filt = urllib.parse.quote(",".join(f), safe=':,"')
    return ("%s?filter=%s&per-page=%d&cursor=%s&mailto=%s&select=%s"
            % (BASE, filt, per_page, urllib.parse.quote(cursor or "*", safe=""), MAILTO,
               "id,title,publication_year,doi,primary_location,abstract_inverted_index"))


def sweep():
    seen = {}
    for learner in LEARNER:
        for task in TASK:
            cursor = "*"
            while cursor:
                d = get(q(learner, task, cursor))
                for w in d.get("results", []):
                    seen[w["id"]] = w
                cursor = d.get("meta", {}).get("next_cursor")
                if not d.get("results"):
                    break
            time.sleep(0.1)
    return seen


def text_of(w):
    """Title and abstract as one lowercase string. OpenAlex ships the abstract inverted."""
    parts = [(w.get("title") or "")]
    inv = w.get("abstract_inverted_index") or {}
    parts.extend(inv.keys())
    return " ".join(parts).lower()


def in_domain(t):
    return any(d in t for d in DOMAIN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()

    print("running the query once, with the exclusion applied afterwards ...")
    pool = sweep()
    print("  %d unique records over the learner and task clauses" % len(pool))

    others = [t.lower() for t in EXCLUDE if t != "mobile"]
    in_dom, with_excl, delta, mobile_only = {}, {}, {}, {}
    for k, w in pool.items():
        t = text_of(w)
        if not in_domain(t):
            continue
        in_dom[k] = w
        hits = [x for x in EXCLUDE if x.lower() in t]
        if not hits:
            with_excl[k] = w
            continue
        delta[k] = w
        if "mobile" in t and not any(o in t for o in others):
            mobile_only[k] = w

    res = {"pool": len(pool), "in_domain": len(in_dom),
           "with_exclusion": len(with_excl), "delta_all_terms": len(delta),
           "delta_mobile_only": len(mobile_only),
           "note": "OpenAlex stands in for IEEE Xplore, which refuses automated access (418)"}
    (HERE / "mobile_exclusion_sensitivity.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")

    rows = sorted(mobile_only.values(), key=lambda w: -(w.get("publication_year") or 0))
    with io.open(str(HERE / "mobile_exclusion_delta.csv"), "w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["year", "title", "doi", "venue"])
        for w in (rows if a.full else rows[:300]):
            loc = (w.get("primary_location") or {}).get("source") or {}
            wr.writerow([w.get("publication_year"), w.get("title"), w.get("doi"),
                         loc.get("display_name")])

    print("\npool over learner and task   %5d" % len(pool))
    print("in domain                   %5d" % len(in_dom))
    print("kept by the exclusion       %5d" % len(with_excl))
    print("removed, all six terms      %5d" % len(delta))
    print("removed by mobile alone     %5d   <- the title-and-abstract screening list"
          % len(mobile_only))
    print("\nwrote mobile_exclusion_delta.csv")


if __name__ == "__main__":
    main()
