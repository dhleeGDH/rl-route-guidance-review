# Corpus overlap with the ten prior reviews: attempted, NOT MEASURABLE from these PDFs

A survey reader wants one number first: of the 94 corpus studies, how many does
each prior review already cite. Three methods were tried on 2026-08-22 and ALL THREE produce false
counts. No number from this directory may be used.

1. **Full-title substring** (`overlap.py`). Zero hits in all ten reviews. A positive control shows
   the matcher works, so the zeros mean reference lists do not render titles as contiguous strings.
   Prefix matching then decays 14 -> 6 -> 4 -> 1 -> 0 as the prefix grows 25 -> 45 characters, which
   is the signature of generic title openings matching arbitrary text.
2. **First-author surname + year within 220 characters** (`overlap_surname.INVALID.json`). Produced
   a pattern that tracked topical distance and looked convincing: 8 hits for the route-recommendation
   review, 0 for motion planning. Hand-checking all 8 of the highest: **8 of 8 false**. Every one
   matched a different paper by a same-surname author in the same year. Surnames in this literature
   are dominated by a few families.
3. **Distinctive title tokens, 75% within a 500-character window** (`overlap_tokens.INVALID.json`).
   1 to 5 hits per review. The two checked by hand were both false; one matched a passage on
   "mobile sequential recommendations for taxi drivers" by another author.

**What a trustworthy answer needs.** Parse each review's reference list into structured entries
(authors, title, year, venue) and match on title token overlap between entries, not against running
text. That is a separate piece of work and was not completed here.

**What to say meanwhile.** The overlap is unmeasured. Do not assert it in either document.

---

## RESOLVED, 2026-08-22: measured by reference entry against reference entry

`overlap_refs.py` implements the method this file named as the trustworthy one. The unit is a
reference ENTRY of about thirty words, not a whole document, so token overlap inside one entry is
meaningful. Each prior review's reference section is isolated, split into entries, and every corpus
title is scored against every entry; a match needs at least 85% of the title's distinctive tokens
inside one entry.

Three guards, all of which fired during development and each of which changed the answer:
1. A fabricated title must match nothing. Passes.
2. An "entry" longer than 60 words means the split collapsed. Without this guard the first run
   reported 34 hits, fourteen of them distinct titles all matching one Huegle et al. blob.
3. Every hit is printed with the entry it matched and read by hand.

Entry counts after the guards are plausible reference-list sizes: 154, 130, 149, 157, 40, 128, 150,
51, 32, 194, for 1,185 entries across the ten reviews.

RESULT: one candidate match, `[ref46] idx 70` against "An autonomous coverage path planning
algorithm for maritime search and rescue", which is a different paper. **No corpus study appears in
any of the ten prior reviews' reference lists.**

Threshold sensitivity was checked rather than assumed: at 0.75, 0.65 and 0.55 the hit count rises to
1, 5 and 12 on one review, and every one of those was read and is false, generic RL-routing
vocabulary overlapping. Loosening therefore adds false positives and does not recover missed true
ones, which is what makes the strict threshold the right one rather than merely the safe one.

LIMIT: matching is on titles. A citation rendering a title very differently could be missed.
