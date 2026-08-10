# -*- coding: utf-8 -*-
"""Recompute the capability-claim field from the full texts rather than from picked quotations.

WHY. Section III-C reports that a route-optimality or shortest-path objective is stated by 56 of
the 71 instantaneous-state studies. Until now that field carried a lower evidence grade than the
state and boundary axes for one reason: it had been recorded once, by one reader, and had never
been re-derived. It was not a coverage gap, since claim_charting.csv covers all 94 studies. This
script closes the remaining gap by applying a fixed rule to the cached full texts, so the whole
path from PDF to count is rerunnable, exactly as machine_state_count.py does for the state axis.

THE RULE, stated once. A study is recorded as making an optimal-route claim when at least one
sentence of its full text carries an optimality term governing a route object, together with a
marker that the sentence's subject is the study's own method or its stated objective, and is not
a background sentence. Every study with a cached full text and no such sentence is recorded as
modest. A study with no cached full text is recorded as unresolved and enters no agreement
figure.

The route-object requirement separates the optimality every Markov decision process has by
definition from a claim to return an optimal route, which is the distinction the review is built
on. Without it the rule settles a study on sentences introducing the Bellman equation or the
epsilon-greedy rule.

The selection vocabulary and the agency, route-object and hygiene patterns are in
claim_lexicon.txt.

    python3 machine_claim_count.py            # counts, agreement, and the disagreement table
    python3 machine_claim_count.py --list     # print the selected sentences behind each difference

Writes machine_claim_counts.csv next to this file, one row per study.
"""
import argparse
import csv
import difflib
import io
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # handoff/
CHART = os.path.join(ROOT, "experiments", "corpus", "claim_charting.csv")
CACHE = os.path.join(HERE, "_text_cache")
LEXICON = os.path.join(HERE, "claim_lexicon.txt")
OUT = os.path.join(HERE, "machine_claim_counts.csv")


def load_lexicon(path):
    groups, current = {}, None
    for line in open(path, encoding="utf-8"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("[") and s.endswith("]"):
            current = s[1:-1]
            groups[current] = []
            continue
        groups[current].append(s)
    return {k: re.compile("|".join(v), re.I) for k, v in groups.items()}


def sentences(text):
    return [s for s in re.split(r"(?<=[.])\s+", " ".join(text.split())) if len(s.split()) > 3]


def select(sents, lex):
    """Sentences the fixed vocabulary hands to the rule."""
    out = []
    for s in sents:
        if lex["DROP_SENTENCE"].search(s):
            continue
        if (lex["SELECT_CLAIM"].search(s) and lex["REQUIRE_AGENCY"].search(s)
                and lex["REQUIRE_ROUTE_OBJECT"].search(s)):
            out.append(s)
    return out


def norm(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print the sentences behind each difference")
    args = ap.parse_args()

    lex = load_lexicon(LEXICON)
    rows = list(csv.DictReader(open(CHART, encoding="utf-8")))
    idx_key = [c for c in rows[0] if c.endswith("idx")][0]
    files = os.listdir(CACHE)
    keys = [norm(f) for f in files]

    recorded_bucket = lambda v: ("optimal" if v.startswith("optimal")
                                 else "modest" if v.startswith("modest") else "unresolved")

    out_rows, pairs, diffs = [], [], []
    for r in rows:
        idx, title = str(r[idx_key]).strip(), r["title"]
        rec = recorded_bucket(r["claimed_capability"])
        m = difflib.get_close_matches(norm(title), keys, n=1, cutoff=0.4)
        if not m:
            machine, hits = "unresolved", []
        else:
            fn = files[keys.index(m[0])]
            hits = select(sentences(open(os.path.join(CACHE, fn), encoding="utf-8",
                                        errors="replace").read()), lex)
            machine = "optimal" if hits else "modest"
        out_rows.append({"idx": idx, "title": title, "state_form": r["state_form"],
                         "recorded": rec, "machine": machine,
                         "selected_sentences": len(hits),
                         "settled_by": hits[0][:300] if hits else ""})
        if rec != "unresolved" and machine != "unresolved":
            pairs.append((rec, machine))
            if rec != machine:
                diffs.append((idx, title, rec, machine, hits))

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    n = len(pairs)
    agree = sum(1 for a, b in pairs if a == b)
    rec_opt = sum(1 for r in out_rows if r["recorded"] == "optimal")
    mac_opt = sum(1 for r in out_rows if r["machine"] == "optimal")
    inst = [r for r in out_rows if r["state_form"] == "none"]
    print("studies                          : %d" % len(out_rows))
    print("comparable (both sides resolved) : %d" % n)
    print("recorded optimal-route claim     : %d" % rec_opt)
    print("machine  optimal-route claim     : %d" % mac_opt)
    print("agreement                        : %d / %d = %.1f%%" % (agree, n, 100.0 * agree / n))
    print("instantaneous state, recorded    : %d of %d" %
          (sum(1 for r in inst if r["recorded"] == "optimal"), len(inst)))
    print("instantaneous state, machine     : %d of %d" %
          (sum(1 for r in inst if r["machine"] == "optimal"), len(inst)))
    print("\ndisagreements: %d" % len(diffs))
    for idx, title, rec, mac, hits in diffs:
        print("  idx %-4s recorded=%-9s machine=%-9s  %s" % (idx, rec, mac, title[:62]))
        if args.list and hits:
            for s in hits[:2]:
                print("        %s" % s[:170])
    print("\nwrote %s" % os.path.basename(OUT))


main()
