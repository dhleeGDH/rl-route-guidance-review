# -*- coding: utf-8 -*-
"""How much the deep-only learner clause and the 2018 window cost the corpus.

WHY THIS EXISTS. The published query of the supplement names deep learners only, and the database
arm runs from 2018. Section IV-A states in consequence that the tabular and unspecified shares are
lower bounds. A lower bound stated without a value reads as an admission.
This measures it rather than arguing it.

The instrument is the one already used for the published OpenAlex bound: openalex_screen.screen()
and its corpus-membership key, unchanged. Only the query and the window differ, which is why the
pool and the results are written to their OWN files rather than over the published ones.

    NON-DEEP CLAUSE  "Q-learning" OR "SARSA" OR "temporal difference" OR "tabular"
                     with every deep term absent, so a record enters only on non-deep evidence
    WINDOW           1990-01-01 to 2026-07-05, the corpus freeze date, opened before 2018

Reports the pool size, how many of the 94 the pool recovers, and every record that survives the
same lexical screen without being in the corpus. Those are hand-adjudicated in STAGE_THREE of
openalex_screen.py, and any not listed there are printed for adjudication rather than counted.

    python3 nondeep_bound.py                    # fetch and screen
    python3 nondeep_bound.py --offline          # re-screen the stored pool
"""
import argparse, csv, io, json, os, sys, time, urllib.parse, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "screening_trail"))
import openalex_screen as base                                  # the published instrument

POOL = HERE / "nondeep_pool.csv"
OUT = HERE / "nondeep_bound.json"
MAILTO = base.MAILTO
FROM, TO = "1990-01-01", "2026-07-05"

# non-deep learner evidence only. The deep terms of the published clause are deliberately absent,
# so nothing enters this pool on the strength of a term the published query already carried.
LEARNER = ('("Q-learning" OR "SARSA" OR "temporal difference learning" OR '
           '"tabular reinforcement learning" OR "policy iteration")')
FILTER = ('title_and_abstract.search:%s AND '
          '("route guidance" OR "vehicle navigation" OR "route planning") AND '
          '(vehicle OR traffic OR road)' % LEARNER)
DATES = "from_publication_date:%s,to_publication_date:%s" % (FROM, TO)


# Every record this pool leaves outside the corpus, adjudicated at title and abstract. The
# published instrument's STAGE_THREE is not extended, since mutating it would change the reason
# recorded against a published figure; these are local to the non-deep pool.
STAGE_THREE_EXTRA = [
    ("Sarsa Learning Based Route Guidance System with Global and Local", "in-scope",
     "a centrally determined route guidance system guiding individual vehicles with Sarsa on two "
     "road networks, peer-reviewed journal, 2015 and therefore outside the database window"),
    ("On-line selection method of the traffic control and route guidance", "out-of-scope",
     "the decision is which control-and-guidance collaboration MODE to run, not a vehicle's route"),
    ("A Q-Learning-Based Approximate Solving Algorithm for Vehicular Route Game", "out-of-scope",
     "solves a route game to an equilibrium over route distributions, an aggregate allocation"),
    ("Intelligent Agents Approach To The On-Line Management And Control", "out-of-scope",
     "dynamic traffic routing to user equilibrium under a cell transmission model, aggregate flow"),
    ("Intelligent Path Planning for Off-Road Navigation", "out-of-scope",
     "off-road terrain traversal, no road network and no OD path over links; no abstract available"),
    ("DYNAMIC ROUTING FOR NAVIGATION IN CHANGING UNKNOWN MAPS", "out-of-scope",
     "an agent navigating an unknown map from visual input, with no modeled road network"),
    ("Multi-modal Mission Route Planning Method for Flying Vehicles", "out-of-scope",
     "flying vehicles over a three-dimensional mission map, an aerial class Section II-A excludes"),
]


def adjudicate(rec):
    t = rec["title"].lower()
    for prefix, outcome, why in STAGE_THREE_EXTRA:
        if t.startswith(prefix.lower()):
            return outcome, why
    outcome, why = base.adjudicate(rec)
    if why.startswith("read at title"):
        return "UNADJUDICATED", "no entry in either list; adjudicate before counting this record"
    return outcome, why


def fetch():
    rows, cursor, total = [], "*", None
    while cursor:
        url = ("https://api.openalex.org/works?per-page=200&select=id,display_name,"
               "publication_year,doi,abstract_inverted_index,primary_location"
               "&filter=%s,%s&cursor=%s&mailto=%s"
               % (urllib.parse.quote(FILTER, safe=':"()'), DATES, cursor, MAILTO))
        with urllib.request.urlopen(url, timeout=90) as r:
            d = json.load(r)
        total = d["meta"]["count"]
        for w in d["results"]:
            loc = (w.get("primary_location") or {}).get("source") or {}
            rows.append({"openalex_id": w["id"], "title": w["display_name"] or "",
                         "year": w.get("publication_year") or "",
                         "doi": w.get("doi") or "", "venue": loc.get("display_name") or "",
                         "abstract": base.inverted(w.get("abstract_inverted_index"))})
        cursor = d["meta"].get("next_cursor")
        print("   fetched %d of %d" % (len(rows), total), flush=True)
        if not d["results"]:
            break
        time.sleep(0.2)
    return rows, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()

    if a.offline:
        if not POOL.exists():
            sys.exit("no stored pool at %s" % POOL)
        rows = list(csv.DictReader(io.open(str(POOL), encoding="utf-8")))
        total = len(rows)
    else:
        rows, total = fetch()
        with io.open(str(POOL), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    corpus = base.corpus_titles()
    if not corpus:
        sys.exit("the corpus key read no rows; the bound would be meaningless")

    recovered, passed, outside, pre2018 = {}, [], [], 0
    for r in rows:
        key = base.norm(r["title"])
        if r.get("year") and str(r["year"]).isdigit() and int(r["year"]) < 2018:
            pre2018 += 1
        if key in corpus:
            recovered[key] = corpus[key]
            continue
        ok, why = base.screen(r)
        if ok:
            passed.append(r)
            outcome, note = adjudicate(r)
            outside.append({"title": r["title"], "year": r["year"], "venue": r["venue"],
                            "outcome": outcome, "why": note})

    print()
    print("--- the non-deep bound, %s to %s ---" % (FROM, TO))
    print("  query          %s" % LEARNER)
    print("  pool returned  %d records, of which %d are dated before 2018" % (total, pre2018))
    print("  corpus members recovered by this pool alone: %d of %d" % (len(recovered), len(corpus)))
    print("  survive the same lexical screen and are NOT in the corpus: %d" % len(passed))
    seen = {}
    for o in outside:
        seen[o["outcome"]] = seen.get(o["outcome"], 0) + 1
    for k in sorted(seen):
        print("      %-16s %d" % (k, seen[k]))
    unlisted = [o for o in outside if o["outcome"] == "out-of-scope"
                and o["why"].startswith("read at title")]
    print("  --- every record the pool leaves outside the corpus, for adjudication ---")
    for o in outside:
        print("      [%-14s] %-4s %s" % (o["outcome"], o["year"], o["title"][:96]))
    if not outside:
        print("      none")

    inscope = [o for o in outside if o["outcome"].startswith("in-scope")]
    print()
    print("  IN-SCOPE AND MISSED BY THE CORPUS: %d" % len(inscope))
    for o in inscope:
        print("      %s (%s)" % (o["title"][:88], o["year"]))

    json.dump({"filter": FILTER, "from": FROM, "to": TO, "returned": total,
               "pre_2018": pre2018, "recovered": len(recovered),
               "outside_after_screen": len(passed), "adjudicated": outside,
               "in_scope_missed": len(inscope)},
              io.open(str(OUT), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("\nwrote %s" % OUT)

    # control: the published pool, re-screened by the same call, must still recover what it did.
    print("--- control ---")
    pub = base.load_pool()
    rec = sum(1 for r in pub if base.norm(r["title"]) in corpus)
    print("  the published OpenAlex pool recovers %d of %d corpus members by the same key" %
          (rec, len(corpus)))


if __name__ == "__main__":
    main()
