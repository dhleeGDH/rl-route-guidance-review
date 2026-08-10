# Screening trail

The record of how the 1,155 title-and-abstract records were reduced to the 94 reviewed
studies, kept file by file. Section II-B describes the flow and Fig. 1 draws it.

## Two files here are superseded snapshots

`corpus_v9_coded.csv` and `claim_charting.csv` also exist in `experiments/corpus/`, and that
directory is the one every script reads. The copies here are earlier snapshots kept as part of
the trail, and they differ:

- `corpus_v9_coded.csv` here lacks the `in_reviewed_corpus` column. Every other cell is identical.
  That column is what excludes the author's own study from the 95 rows to give N = 94.
- `claim_charting.csv` here is missing the verbatim capability quotations for idx 0 and idx 75,
  added on 2026-08-09 so that all 56 recorded claims carry a quotation.

Use `experiments/corpus/` for any recount. Nothing in the manuscript is derived from the copies
in this directory.
