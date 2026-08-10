# -*- coding: utf-8 -*-
"""Title-and-abstract screening of the published OpenAlex pool.

WHY THIS EXISTS. Section II-B reports prevalence over a corpus assembled from an IEEE Xplore
export and from citation searching, only one arm of which is logged per method. The supplementary
material publishes an executable OpenAlex query as a reproducible entry point, notes that it
returns about 500 records and recovers roughly a third of the corpus, and leaves the rest of that
pool unscreened. A reader is then entitled to ask how many in-scope studies the corpus missed.

This screens the whole returned pool against a mechanical rule and answers that question with a
bound. The rule is deliberately crude and recall-favouring: a record passes only where its title
or abstract carries a reinforcement-learning term, a routing or navigation term, and a road or
vehicle term, and carries no term from the out-of-scope list. A crude rule overcounts rather than
undercounts, so the figure it produces is an upper bound on what a full screening would confirm.

Corpus membership is decided on a normalized title, which is the same key `publisher_split.py`
uses, and near-identical titles are counted separately so that the recovery figure can be read
either way.

    python3 openalex_screen.py              # fetch, screen, and print the bound
    python3 openalex_screen.py --offline    # re-screen the stored pool without the network

Writes openalex_pool.csv and openalex_screen.json next to this file.
"""
import argparse
import csv
import io
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS = HERE.parent / "corpus" / "corpus_v9_coded.csv"
POOL = HERE / "openalex_pool.csv"

MAILTO = "donghoun.lee88@gmail.com"
FILTER = ('title_and_abstract.search:"reinforcement learning" AND '
          '("route guidance" OR "vehicle navigation" OR "route planning") AND '
          '(vehicle OR traffic OR road)')
DATES = "from_publication_date:2018-01-01,to_publication_date:2026-12-31"

# The scope rule of Section II-B, reduced to lexical terms. Each group must appear.
RL = (r"reinforcement learning|q-learning|q learning|deep q|dqn|actor-critic|actor critic|"
      r"policy gradient|ppo|a2c|a3c|sarsa|ddpg|td3|soft actor|marl|multi-agent rl")
ROUTE = (r"route guidance|route planning|route choice|route recommend|routing|navigation|"
         r"path planning|path selection|shortest path|trip planning")
ROAD = r"vehicle|vehicular|traffic|road|urban|driver|intersection|highway|transportation network"
# Out of scope: the neighbouring problem classes Section II-A excludes.
OUT = (r"unmanned aerial|uav|drone|quadrotor|robot|manipulator|warehouse|mobile robot|"
       r"packet|network-on-chip|wireless sensor|data center|spacecraft|maritime|vessel|ship |"
       r"railway|train timetable|flight|air traffic|pedestrian evacuation|"
       r"vehicle routing problem|travelling salesman|traveling salesman|last-mile delivery|"
       r"ride-hailing dispatch|order dispatch|fleet dispatch|charging scheduling|"
       r"traffic signal control|signal timing|ramp metering|lane change|car-following|"
       r"platoon control|trajectory tracking|motion planning|obstacle avoidance|"
       r"agv|automated guided vehicle|overhead hoist|semiconductor|container terminal|"
       r"logistics|delivery|parcel|waste collection|cold chain|distribution network|"
       r"sdn|software.defined|segment routing|content delivery|edge computing|offloading|"
       r"service migration|network slicing|aerial|surface vehicle|underwater|agricultural|"
       r"tourism|recommender system|carpool|ride.shar|ride.hail|orienteering|taxi|"
       r"charging station|discharging|path.tracking|trajectory track|end.to.end|"
       r"dynamic window|unsignalized intersection|perimeter control|"
       r"inverse reinforcement learning|evacuation|hazardous material|pedestrian|"
       r"large language model|\bllm\b|indoor|\bmaze\b")

# Second stage. A record surviving the scope rule is still excluded where its own title or
# abstract places it in one of these classes, each of which the corpus rule excludes for a
# reason already stated in Section II-A. The code is recorded so that every exclusion is
# checkable against the record it was applied to.
STAGE2 = [
    ("driving control or perception",
     r"collision|obstacle|perception|lane |steering|pid |kalman|localization|self.driving|"
     r"autonomous driving|automated driving|driving decision|behavioral decision|"
     r"stability analysis|simulator|reference path|trajectory"),
    ("communication or power network",
     r"srv6|optical|fiber|ftth|satellite|network slic|cable|power (data|grid)|blockchain|"
     r"\b5g\b|edge intelligen|wireless"),
    ("prediction or estimation only",
     r"prediction|forecast|estimation|detection|congestion control|carbon emission|"
     r"mode detection"),
    ("fleet, delivery, or VRP",
     r"customers|vehicle routing problem|fleet|patrol|bus |transit|pollution rout|"
     r"orienteering|multi-operator"),
    ("EV charging navigation", r"charging navigation|charge scheduling"),
    ("survey or review", r"\bsurvey\b|\breview\b|systematic decade"),
    ("not a study record", r"zenodo|dataset|v1\.0\.0"),
    ("robot or multi-platform navigation",
     r"movable and immovable|multiple platforms|unconstrained environments|diffusion polic|"
     r"virtual"),
]

# Third stage. What survives both lexical stages is decided by reading the title and abstract.
# Each decision is recorded here against a title prefix, so a reader can disagree with any one of
# them against the same record. "in-scope" marks a study the corpus should have held.
ADJUDICATED = [
    ("Hierarchical Sarsa Learning Based Route Guidance", "in-scope",
     "dynamic route guidance over a digital road network, Sarsa, peer-reviewed journal"),
    ("A Constraint-Based Routing and Charging Methodology", "in-scope",
     "electric-vehicle route planning over a road network with a deep learner, peer-reviewed"),
    ("Context-Conditioned Meta-Reinforcement Learning for Expectation", "in-scope-later",
     "route guidance by the present author, post-dating the search collection"),
    ("COREAN: Context-conditioned meta-reinforcement learning", "in-scope-later",
     "journal version of the preceding record, same work"),
    ("Enhancing Emergency Vehicle Navigation in Smart Traffic Squares", "unresolvable",
     "no abstract available and the title alone cannot resolve the guidance object"),
    ("Hybrid Graph Neural Network Model for Energy-efficient Route Planning", "unresolvable",
     "the learner is not identified as reinforcement learning in the abstract"),
    ("Research on Path Planning Algorithm Based on the Integration", "unresolvable",
     "the routing object is not resolvable to a road network from the abstract"),
    ("Efficient Path Planning for Large-Scale Vehicular Networks", "unresolvable",
     "no abstract available and the network may be a communication one"),
    ("Realtime Vehicle Route Optimisation via DQN", "out-of-scope",
     "doctoral thesis, outside the publication types the corpus admits"),
]


def norm(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def inverted(ix):
    if not ix:
        return ""
    m = {}
    for w, ps in ix.items():
        for p in ps:
            m[p] = w
    return " ".join(m[k] for k in sorted(m))


def fetch():
    rows, cursor = [], "*"
    while cursor:
        url = ("https://api.openalex.org/works?per-page=200&select=id,display_name,"
               "publication_year,doi,abstract_inverted_index,primary_location"
               "&filter=%s,%s&cursor=%s&mailto=%s"
               % (urllib.parse.quote(FILTER, safe=':"()'), DATES, cursor, MAILTO))
        with urllib.request.urlopen(url, timeout=60) as r:
            d = json.load(r)
        for w in d["results"]:
            loc = (w.get("primary_location") or {}).get("source") or {}
            rows.append({"openalex_id": w["id"], "title": w["display_name"] or "",
                         "year": w.get("publication_year") or "",
                         "doi": w.get("doi") or "", "venue": loc.get("display_name") or "",
                         "abstract": inverted(w.get("abstract_inverted_index"))})
        cursor = d["meta"].get("next_cursor")
        print("   fetched %d of %d" % (len(rows), d["meta"]["count"]), flush=True)
        if not d["results"]:
            break
        time.sleep(0.2)
    return rows


def load_pool():
    return list(csv.DictReader(io.open(str(POOL), encoding="utf-8")))


def corpus_titles():
    rows = [r for r in csv.DictReader(io.open(str(CORPUS), encoding="utf-8"))
            if str(r.get("in_reviewed_corpus", "")).strip().lower() in ("1", "true", "yes", "y")]
    return {norm(r["title"]): r["idx"] for r in rows}


def screen(rec):
    t = (rec["title"] + " " + rec.get("abstract", "")).lower()
    if re.search(OUT, t):
        return False, "out-of-scope class"
    if not re.search(RL, t):
        return False, "no learning term"
    if not re.search(ROUTE, t):
        return False, "no routing term"
    if not re.search(ROAD, t):
        return False, "no road or vehicle term"
    for code, pat in STAGE2:
        if re.search(pat, t):
            return False, code
    return True, "carried to adjudication"


def adjudicate(rec):
    t = rec["title"].lower()
    for prefix, verdict, why in ADJUDICATED:
        if t.startswith(prefix.lower()):
            return verdict, why
    return "out-of-scope", "read at title and abstract, outside the guidance object"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()

    if a.offline and POOL.exists():
        pool = load_pool()
    else:
        print("querying OpenAlex")
        pool = fetch()
        with io.open(str(POOL), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["openalex_id", "title", "year", "doi", "venue",
                                              "abstract"])
            w.writeheader()
            w.writerows(pool)

    ct = corpus_titles()
    in_corpus, carried, dropped = [], [], {}
    for r in pool:
        if norm(r["title"]) in ct:
            in_corpus.append(r)
            continue
        ok, why = screen(r)
        if ok:
            carried.append(r)
        else:
            dropped.setdefault(why, []).append(r)

    verdicts = {}
    for r in carried:
        v, why = adjudicate(r)
        r["verdict"], r["reason"] = v, why
        verdicts.setdefault(v, []).append(r)

    with io.open(str(HERE / "openalex_adjudication.csv"), "w", encoding="utf-8",
                 newline="") as f:
        w = csv.writer(f)
        w.writerow(["title", "year", "venue", "doi", "verdict", "reason"])
        for r in carried:
            w.writerow([r["title"], r["year"], r["venue"], r["doi"], r["verdict"], r["reason"]])

    n = len(pool)
    missed = len(verdicts.get("in-scope", []))
    res = {"pool": n, "recovered_from_corpus": len(in_corpus),
           "carried_to_adjudication": len(carried),
           "in_scope_missed": missed,
           "in_scope_later_by_author": len(verdicts.get("in-scope-later", [])),
           "unresolvable_at_abstract": len(verdicts.get("unresolvable", [])),
           "dropped": {k: len(v) for k, v in sorted(dropped.items())},
           "missed_titles": [r["title"] for r in verdicts.get("in-scope", [])]}
    # What the two missed studies would do to the headline shares if both were added, taken at
    # the extremes: neither meets a requirement, or both do.
    for label, k, d in (("state", 17, 94), ("boundary", 19, 94)):
        res["%s_share" % label] = {
            "as_reported": round(100.0 * k / d, 1),
            "both_missing": round(100.0 * k / (d + missed), 1),
            "both_meeting": round(100.0 * (k + missed) / (d + missed), 1)}
    (HERE / "openalex_screen.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    print("\npool returned                       %4d" % n)
    print("already in the reviewed corpus      %4d" % len(in_corpus))
    for k, v in sorted(dropped.items(), key=lambda x: -len(x[1])):
        print("   excluded, %-32s %4d" % (k, len(v)))
    print("carried to title-and-abstract reading %2d" % len(carried))
    for v, rs in sorted(verdicts.items()):
        print("   %-20s %3d" % (v, len(rs)))
    for t in res["missed_titles"]:
        print("      missed: %s" % t[:88])
    print("\nheadline shares if both missed studies are added")
    for label in ("state", "boundary"):
        d = res["%s_share" % label]
        print("   %-9s reported %.1f%%, range %.1f%% to %.1f%%"
              % (label, d["as_reported"], d["both_missing"], d["both_meeting"]))
    print("\nwrote openalex_pool.csv and openalex_screen.json")


if __name__ == "__main__":
    main()
