# -*- coding: utf-8 -*-
"""fulltext_scan.py : the three fields of the manuscript that require reading a full text.

Every rule is printed at the top of a run, so a reader can reproduce a count by hand.
The scan covers the 91 studies recorded as full-text in the archived study record. It writes one
row per study; the body counts are tallies of those columns.

    PDF_DIR=/path/to/full_texts python3 fulltext_scan.py [--out fulltext_scan]
"""
import argparse, csv, difflib, glob, json, os, re, subprocess, sys, statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "corpus", "corpus_v9_coded.csv")
# The full texts are copyrighted and are not redistributed with this repository.
# Point PDF_DIR at a directory holding them to reproduce the scan.
PDFDIR = os.environ.get("PDF_DIR", os.path.join(ROOT, "pdfs"))

# ---------------------------------------------------------------- rules (printed verbatim)
MEASURES = [("travel time", [r"travel\s+time"]),
            ("delay",       [r"\bdelay(?:s|ed)?\b"]),
            ("fuel/emission", [r"\bfuel\b", r"\bemission"]),
            ("reward/return", [r"\breward", r"\breturn\b"]),
            ("distance",    [r"\bdistance\b", r"\btravel\s+distance\b"]),
            ("throughput/queue", [r"\bthroughput\b", r"\bqueue"]),
            ("completion/arrival rate", [r"completion\s+rate", r"arrival\s+rate", r"success\s+rate"])]
BENCHMARKS = [r"sioux\s*falls", r"\bbraess\b", r"\banaheim\b", r"nguyen[\s-]*dupuis"]
BASELINES  = [r"shortest[\s-]*path", r"\bdijkstra\b", r"\bA\*", r"user\s+equilibrium",
              r"static\s+routing", r"fixed\s+routing", r"no\s+guidance",
              r"random\s+routing", r"greedy\s+routing", r"greedy\s+policy"]
SYNTHETIC  = [r"\bgrid\b", r"\blattice\b", r"synthetic\s+network", r"toy\s+network"]
CITIES = ["beijing","shanghai","chengdu","shenzhen","guangzhou","hangzhou","xi'an","xian","porto",
          "new york","manhattan","chicago","los angeles","san francisco","london","paris","berlin",
          "munich","cologne","bologna","turin","luxembourg","singapore","tokyo","seoul","daejeon",
          "austin","seattle","boston","dublin","monaco","ingolstadt","jinan","nanjing","wuhan",
          "changsha","kunming","zhengzhou","suzhou","tianjin","qingdao","harbin","dalian","xiamen",
          "hefei","sydney","melbourne","toronto","montreal","amsterdam","rome","milan","madrid",
          "barcelona","lisbon","athens","istanbul","cairo","delhi","mumbai","bangalore","jakarta",
          "bangkok","manila","hanoi","taipei","hong kong","osaka","nagoya","kyoto","sioux falls",
          "anaheim","shinjuku","brisbane","valencia","bilbao","zurich","vienna","prague","warsaw"]
REAL = [r"real\s+road\s+network", r"real[\s-]*world\s+network", r"real\s+traffic\s+network"] + \
       [r"\b" + re.escape(c) + r"\b" for c in CITIES]
NODECOUNT = r"([0-9][0-9,]{0,6})\s+(?:nodes|intersections)\b"

RULES = """
RULES AS APPLIED
  denominator        the 91 studies recorded as full-text in repo_v11/corpus/corpus_v9_coded.csv
  text               pdftotext -layout, first 40 pages
  measure region     from the first heading matching RESULT|EXPERIMENT|EVALUATION|CASE STUDY|
                     NUMERICAL|SIMULATION to the end of the text
  evaluation region  the same heading up to the next CONCLUSION|DISCUSSION|RELATED WORK|REFERENCES
  captions           every line beginning "TABLE <n>", "Table <n>", "Fig. <n>" or "FIGURE <n>"
  reported measure   the measure name appears after the first results-like heading (to the end of
                     the text) or inside a table or figure caption
                       travel time | delay | fuel/emission | reward/return | distance |
                       throughput/queue | completion/arrival rate
  benchmark network  Sioux Falls | Braess | Anaheim | Nguyen-Dupuis, anywhere in the text
  baseline           evaluation section only: shortest path | Dijkstra | A* | user equilibrium |
                     static routing | fixed routing | no guidance | random routing | greedy routing |
                     greedy policy
  substrate          evaluation section only.
                     real  = a named city or region, or "real road network" / "real-world network" /
                             "real traffic network"
                     synth = "grid" / "lattice" / "synthetic network" / "toy network"
                     a study may be both; "neither" where no cue appears
  node count         the first integer followed by "nodes" or "intersections"
"""

def norm(s): return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

def build_map():
    rows = [x for x in csv.DictReader(open(CORPUS, encoding="utf-8-sig", newline=""))
            if x["in_reviewed_corpus"].strip().lower() in ("yes", "true", "1")
            and x["source"].strip() == "full-text"]
    cand = {}
    for p in glob.glob(os.path.join(PDFDIR, "**", "*.pdf"), recursive=True):
        b = os.path.basename(p)[:-4]
        b = re.sub(r"^idx\d+[_ ]", "", b)          # idx-prefixed acquisitions
        b = re.sub(r"__[0-9a-f]{6,}$", "", b)      # hash-suffixed truncations
        cand.setdefault(norm(b), p)
    keys = list(cand)
    out = []
    for x in rows:
        n = norm(x["title"]); hit = cand.get(n)
        if not hit:                                 # truncated filename
            for k in keys:
                if len(k) >= 40 and (n.startswith(k) or k.startswith(n)): hit = cand[k]; break
        if not hit:
            m = difflib.get_close_matches(n, keys, n=1, cutoff=0.80)
            hit = cand[m[0]] if m else None
        if not hit:                                 # abbreviated filename: match on first page
            for k, p in cand.items():
                try: head = norm(text_of(p, pages=1))
                except Exception: continue
                if n[:60] and n[:60] in head: hit = p; break
        out.append((x["idx"], x["title"], hit))
    return out

def text_of(path, pages=40):
    return subprocess.run(["pdftotext", "-layout", "-q", "-l", str(pages), path, "-"],
                          capture_output=True, timeout=120).stdout.decode("utf-8", "ignore")

def regions(t):
    """Three regions.

    caps    every table and figure caption line.
    measure everything after the first results-like heading to the end of the text, plus captions.
    evalenv the evaluation section alone: the same heading up to the next CONCLUSION/DISCUSSION/
            RELATED WORK/REFERENCES heading, plus captions. Baseline and substrate read this region
            only, so that a phrase in an introduction or a motivation does not count.
    """
    caps = "\n".join(re.findall(r"^[ \t]*(?:TABLE|Fig\.?|FIGURE)\s*[IVXLCDM0-9]+.*$", t, re.M | re.I))
    m = re.search(r"^.{0,12}(?:RESULT|EXPERIMENT|EVALUATION|CASE STUDY|NUMERICAL|SIMULATION)",
                  t, re.M | re.I)
    measure = evalenv = ""
    if m:
        rest = t[m.start():]
        measure = rest                                   # widened: to the end of the references
        e = re.search(r"^.{0,12}(?:CONCLUSION|DISCUSSION|RELATED WORK|REFERENCES)", rest, re.M | re.I)
        evalenv = rest[:e.start()] if e else rest
    return ((measure + "\n" + caps).lower(),
            (evalenv + "\n" + caps).lower(),
            t.lower())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "fulltext_scan"))
    a = ap.parse_args()
    print(RULES)
    os.makedirs(a.out, exist_ok=True)
    mapped = build_map()
    unresolved = [(i, t) for i, t, p in mapped if not p]
    rows = []
    for idx, title, pdf in mapped:
        if not pdf:
            rows.append(dict(idx=idx, title=title, pdf="", **{k: "unc" for k, _ in MEASURES},
                             benchmark="unc", baseline="unc", substrate="unc", nodes=""))
            continue
        try: t = text_of(pdf)
        except Exception: t = ""
        reg, evalenv, full = regions(t)
        r = dict(idx=idx, title=title, pdf=os.path.relpath(pdf, ROOT))
        for name, pats in MEASURES:
            r[name] = "yes" if any(re.search(p, reg) for p in pats) else "no"
        r["benchmark"] = "yes" if any(re.search(p, full) for p in BENCHMARKS) else "no"
        r["baseline"]  = "yes" if any(re.search(p, evalenv) for p in BASELINES) else "no"
        isreal = any(re.search(p, evalenv) for p in REAL)
        issyn  = any(re.search(p, evalenv) for p in SYNTHETIC)
        r["substrate"] = "both" if (isreal and issyn) else "real" if isreal else "synthetic" if issyn else "neither"
        m = re.search(NODECOUNT, full)
        r["nodes"] = m.group(1).replace(",", "") if m else ""
        rows.append(r)
    cols = ["idx", "title", "pdf"] + [k for k, _ in MEASURES] + ["benchmark", "baseline", "substrate", "nodes"]
    csvp = os.path.join(a.out, "fulltext_scan_per_study.csv")
    with open(csvp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    n = len(rows)
    print("studies scanned : %d" % n)
    print("unresolved PDFs : %d %s" % (len(unresolved), [i for i, _ in unresolved] or ""))
    print("\nCOUNTS over %d full texts" % n)
    for name, _ in MEASURES:
        print("  reported measure %-26s %d" % (name, sum(1 for r in rows if r[name] == "yes")))
    print("  benchmark network                    %d" % sum(1 for r in rows if r["benchmark"] == "yes"))
    print("  shortest-path / baseline comparison  %d" % sum(1 for r in rows if r["baseline"] == "yes"))
    sub = [r["substrate"] for r in rows]
    both = sub.count("both")
    print("  substrate real (incl. both)          %d" % (sub.count("real") + both))
    print("  substrate synthetic (incl. both)     %d" % (sub.count("synthetic") + both))
    print("  substrate both                       %d" % both)
    print("  substrate neither                    %d" % sub.count("neither"))
    nodes = sorted(int(r["nodes"]) for r in rows if r["nodes"])
    if nodes:
        print("  node count stated                    %d  median %d  range %d to %d"
              % (len(nodes), int(statistics.median(nodes)), nodes[0], nodes[-1]))
    print("\nper-study output: %s" % os.path.relpath(csvp, ROOT))

if __name__ == "__main__":
    main()
