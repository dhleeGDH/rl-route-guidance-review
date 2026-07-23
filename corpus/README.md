# Corpus and extraction

## `corpus_v9_coded.csv`

The per-study extraction. One row per study, one column per recorded field, and beside each
field the verbatim quotation the value was read from. A reader who disagrees with a recorded
value can check it against the quotation without rerunning anything.

The file holds 95 rows and the reviewed corpus is 94 of them. The `in_reviewed_corpus` column
says which: row `idx` 93 is a study by the present author that post-dates the search window, is
cited in the roadmap as an exemplar rather than reviewed as corpus, and is excluded from every
count the paper reports. Keeping it here as a labelled row makes the extraction complete
without changing any denominator, and filtering on `in_reviewed_corpus == yes` reproduces the
94 the paper reports.

Fields whose values turn on a definitional boundary carry their criterion in the paper:

- `predictive_representation` — the state form criterion of Section III-A. The paper also
  states this field as a decision rule over the quotation column, released as
  `recode_state_field.py` at the repository root. Of the 88 studies carrying a value the rule
  settles 85 and reproduces the recorded reading in 83. It reads this column together with
  `staterep_quote`, since `pred_quote` is filled only where a study states something predictive
  and an empty cell is the record that no such statement was found.
- `boundary_condition` — whether the report addresses an open boundary, not whether the network
  is open by design. A study whose report is silent is recorded as not addressed.
- `reward_alignment` — individual, mixed, or system-level, recorded as a class. The magnitude of
  an arrival term is not recorded, since reward equations are quoted too inconsistently for it.

`unclear` marks a report that does not state the field. It is kept apart from any assigned
value and enters no count as if it were one.

## `master_pool_v1.csv`

The 1,155 candidates screened at title and abstract, with the `pool` column recording which
search arm returned each record: `gate1` for the IEEE Xplore export, `old` for the arm that
combined database and citation searching, and `both` for the 15 common to the two. The
prevalence figures split by arm in the paper are computed from this column.
