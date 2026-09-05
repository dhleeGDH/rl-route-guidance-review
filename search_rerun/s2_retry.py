# -*- coding: utf-8 -*-
"""Retry the Semantic Scholar leg of arm1_reconstruct.py, which returned HTTP 429.

The public graph interface is rate limited without a key. This retries the same query at a slower
pace and merges the result into arm1_records.json so arm1_reconstruct.py --offline can rescore.
"""
import io, json, os, time, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "arm1_records.json")
UA = {"User-Agent": "route-guidance-review/1.0 (mailto:donghoun.lee88@gmail.com)"}
PLAIN = 'reinforcement learning route guidance vehicle navigation route planning'

blob = json.load(io.open(RAW, encoding="utf-8"))
rows, err = [], None
for offset in range(0, 500, 100):
    ok = False
    for attempt in range(6):
        try:
            url = ("https://api.semanticscholar.org/graph/v1/paper/search?query="
                   + urllib.parse.quote(PLAIN)
                   + "&year=2018-2026&limit=100&offset=%d&fields=title,externalIds,year" % offset)
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=90) as r:
                d = json.loads(r.read().decode("utf-8", "replace"))
            ok = True
            break
        except Exception as e:                                   # noqa: BLE001
            err = str(e)
            time.sleep(30 * (attempt + 1))
    if not ok:
        break
    got = d.get("data") or []
    for w in got:
        rows.append({"title": w.get("title"),
                     "doi": ((w.get("externalIds") or {}).get("DOI") or "").lower(),
                     "year": w.get("year"), "source": "Semantic Scholar"})
    print("offset %d -> %d" % (offset, len(got)), flush=True)
    if len(got) < 100:
        break
    time.sleep(20)

if rows:
    blob["records"] = [r for r in blob["records"] if r["source"] != "Semantic Scholar"] + rows
    blob["per_db"]["Semantic Scholar"] = len(rows)
    blob["errors"].pop("Semantic Scholar", None)
    json.dump(blob, io.open(RAW, "w", encoding="utf-8"), ensure_ascii=False)
    print("merged %d Semantic Scholar records" % len(rows))
else:
    print("Semantic Scholar still unavailable: %s" % err)
