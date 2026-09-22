# fulltext_scan : partial verification of three full-text fields

`scripts/fulltext_scan.py` reads the 93 studies recorded as full-text in
`study_record_reviewed_studies.csv` and records, per study, the reported measures, the
benchmark network, the baseline comparison, the substrate and the stated node count.
The script prints its rules at the top of every run; `fulltext_scan_per_study.csv` holds one
row per study (93 rows, 0 unresolved PDFs). Re-executed at T-1994 over the 93-study record; the
91-study run it replaces reproduced exactly beforehand, row for row.

The scan is archived as a **partial verification**. It did not replace any count in the
manuscript, and no count was rewritten as a result of it.

## What the scan confirms

Four fields fall within three of the published coding and are taken as verified:

| field | published | scanned |
|---|---|---|
| benchmark road network | 12 | 13 |
| shortest-path comparison | 63 | 64 |
| substrate, both | 16 | 18 |
| node or intersection count stated | 40 | 39 |

The baseline field is the strongest of these. Under the first term list, which searched the whole
text for "random", "greedy" and "fixed", the scan returned 91 of 91. Restricting the search to the
evaluation section and dropping those three standalone terms brought it to 64 against a published
63, which reproduces the published coding almost exactly.

## Why the scan does not settle the other eleven fields

Two defects, both established by reading the studies concerned:

1. **The results region is detected by a heading regex.** The measure region begins at the first
   heading matching `RESULT|EXPERIMENT|EVALUATION|CASE STUDY|NUMERICAL|SIMULATION`. Eleven of the
   93 studies carry no such heading, so their region is empty and every measure scores `no`. This
   is a pure under-count and explains the shortfalls in travel time (78 against 87), delay (46
   against 59), fuel or emission (36 against 47) and completion (14 against 18). Example, idx 20,
   scored `no` for both delay and fuel: "to assess the impact of multiple traffic factors on
   vehicle delays" and "prolonged travel time, fuel wastage, and pollutant emissions".

2. **The widened region contains the bibliography.** Widening the measure region to the end of the
   text puts reference entries inside it for 30 of the 93 studies, so a title in a reference list
   scores as a reported measure. This is a pure over-count and explains reward or return (75
   against 45), distance (52 against 31) and throughput or queue (29 against 18). Example, idx 2,
   scored `yes` for reward: "Individual versus Difference Rewards on Reinforcement Learning for
   Route Choice, in: 2014 Brazilian Conference on Intelligent Systems, pp." The same region also
   counts method details, as in idx 1, "distance is calculated using Equation (20)".

The substrate rule fails for a third reason: a study that names its substrate through the simulator
rather than through a city is scored `neither`. idx 5, idx 14 and idx 15 each state
"Simulation of Urban MObility (SUMO)" in the evaluation section and are scored `neither`.

A scan fit to overturn the published coding would detect results sections structurally rather than
by heading, exclude the bibliography, and require a measure to sit beside a number in a results
table. It would also need a per-study published record to check against, which the archive does not
hold for these fields.
