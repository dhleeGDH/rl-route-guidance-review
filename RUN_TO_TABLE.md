# RUN_TO_TABLE: which run output backs which printed cell

Every measured value of the manuscript is listed here with the file that produces it, table by
table and figure by figure. Float numbers are the manuscript's own: Tables I to VIII in the body,
Figs. 1 to 5 in the body, and Tables S-1 to S-9 in the Supplementary.

Paths are relative to this package unless marked `repo_v11/`, which is the code and data archive
published at the GitHub tag of the same version. Exact computations, meaning value iteration and
reachability enumeration, take no seed and print no interval.

Two things have to be read before the tables below, because a reader who skips them will find
values that look wrong and are not.

## A. Two reward variants

Section V prints two shaping variants of the destination-aligned reward.

**Published.** The shaping term as first run. Its outputs are the `boundary_open_demo/`,
`sioux30/`, `anaheim/` and `benchmark_network/` files of this package.

**M2-C.** The discount-consistent form of the same term, which multiplies the shaping increment by
`(1 - gamma)` and is inert at `gamma = 1`. Its outputs are under `m2c/`, and the mixin that defines
it is `m2c/shaping_m2c.py`. Every exact row is identical under the two variants at `gamma = 1`, so
the regression is an identity rather than a coincidence.

The manuscript prints the published run throughout. Every one of the eight cells of Table V is
the published form: the 5x5-grid rows are `boundary_open_demo/travel_time_bdest_ingrid.json`
and `boundary_open_demo/travel_time_bdest_ingrid_8000.json`, and the Sioux Falls rows are
`gform/sioux_cells_3000_gform.json` and the thirty shards
`sioux30/dest_boundary_seed00.json` to `seed29.json`. The learner sweep of Supplementary
Table S-8 is `boundary_open_demo/dqn_sensitivity_30seed.json`, also the published form. No
printed value of the manuscript or of the supplement comes from an M2-C file; the M2-C
outputs under `m2c/` are the second arm of the same runs and are deposited as a record of it. A travel-time reward carries no shaping
term, so M2-C cannot reach it: those cells are identical under both variants by construction, and
`m2c/controls/sentinel_control_P.json` records the seed-for-seed reproduction that shows it.

Each row below names the variant where the distinction bears on which file to open.

## B. Two dispersion conventions

Both are a 95% percentile bootstrap over the per-seed values with 10,000 resamples, which is the
single convention Supplementary Section S-I states. They differ only in how the generator is
seeded, and they part by 0.1 at a bound.

| | Convention A | Convention B |
|---|---|---|
| Function | `table6_rows.py`, `ci(values, key)` | `repo_v11/bootstrap_travel_time_ci.py`, `boot(values, key)` |
| Seed | `(20260719 + crc32(key)) % 2**32` | `crc32(key) & 0x7FFFFFFF` |
| Key | `"<net>\|<reward>\|<episodes>\|<boundary>"`, from `cell_key()` | `"<repository-relative path>\|<cell>\|<field>"` |
| Draw | one vectorised `rng.choice` of shape (10000, n) | 10,000 separate `rng.choice` calls |
| Used by | every published-form cell of Tables IV and V and of Table S-3 | every M2-C cell, and every control under `m2c/` |

Convention B's key carries the path of the file **inside the development repository**, not inside
this package. The 5x5-grid M2-C cell is keyed
`m2c/train_m2c_g1.json|open|aligned|{}|3000|completion_per_seed`, and the same values keyed
on this package's `m2c/train_m2c_g1.json` return a different interval. The four M2-C cells of
Tables IV and V reproduce to the digit under the repository key and under no other.

**A third convention exists in the run files of the two sentinel controls.** The driver that wrote
them, `boundary_open_demo/exit_sentinel_control.py`, draws its own interval from
`numpy.random.default_rng(7)`, which is neither A nor B. The published-arm files
`boundary_open_demo/exit_sentinel_control.json`, `zero_shot_transfer.json` and
`m2c/controls/*_P.json` keep those bounds, since they are the record of the published run as it was
deposited. The M2-C arms that the manuscript prints, `m2c/controls/exit_sentinel_control_C.json` and
`sentinel_control_C.json`, were redrawn under convention B in the revision and now carry the printed
strings in `printed_open`, `printed_difference` and `printed_ci`. Their per-seed values and means
are the run and were not touched. `m2c/controls/summarize.py` prints every cell of both controls in
its printed form under both variants.

**A paired difference prints as the difference of the two printed cells**, which is the rule
`check_table_arithmetic.py` enforces, so a reader can take it from the cells beside it. Where the
two roundings part the printed difference is the smaller: the mean 24.45 rounds to 24.5 while the
printed cells 41.5 and 17.1 differ by 24.4, and 37.75 against 60.4 and 22.7 gives 37.7. The interval
beside it is the interval of the paired sample, drawn under convention B.

**The `printed` strings stored inside the run files are convention B.** `travel_time_bdest_ingrid.json`,
`travel_time_bdest_ingrid_8000.json`, `travel_time_sumo_3000.json` and `travel_time_sumo_8000.json`
each carry a `completion_ci95.printed` and a `travel_time_ci95.printed` written by
`bootstrap_travel_time_ci.py`. Table IV prints those same cells under convention A. Four completion
bounds therefore differ by 0.1 between the file and the table:

| Cell | Table IV | the file's own `printed` |
|---|---|---|
| SUMO, travel-time, closed, 3000 | 98.2 [96.6-99.5] | 98.2 (96.7 to 99.5) |
| SUMO, aligned, open, 3000 | 27.2 [18.8-36.6] | 27.2 (18.8 to 36.4) |
| SUMO, aligned, open, 8000 | 98.0 [94.9-99.8] | 98.0 (95.0 to 99.8) |
| grid, aligned, open, 3000, published form | 35.1 [26.1-43.0] | 35.1 (26.3 to 43.1) |

Neither is wrong. They are two draws of the same ten numbers under two seedings, and the table is
internally consistent because every one of its published-form cells uses convention A.

`table6_rows.py` is deposited here for its convention alone. Its `ci()`, `cell_key()` and `fmt()`
reproduce every published-form cell of Tables IV and V from the per-seed lists of the files named
below. Its `main()` does not run against this package: its loaders name the working-tree filenames
of the runs (`four_cells_boundary_dest.json`, `sumo_bdest_*.json`) rather than the deposited ones
(`travel_time_bdest_ingrid.json`, `travel_time_sumo_*.json`), which hold the same per-seed values
under different names, and it also reaches for benchmark outputs this package does not carry.

---

## Table I. The record of the 93 reviewed studies

| Column | Command | Output | Where the value prints |
|---|---|---|---|
| every cell | none; the record is the source | `study_record_reviewed_studies.csv`, filtered to `in_reviewed_studies == yes`, 93 rows | Table I, one row per study, and every count of Sections II-B, III-A, III-B and III-C |
| the simulator column of Supplementary Table S-7 | none | `study_record_simulator_audit_v6.csv` | Section III-C, the platform named in 53 of the 93 |

## Table II. The recorded values of the ten fields and the source of each

No measured value. The ten fields and their recorded values are the columns of the record above;
`gates/countcheck.py` is what holds the printed list and the record in step.

## Table III. Reward alignment and boundary condition by action type

| Row | Command | Output | Where the value prints |
|---|---|---|---|
| every cell | none; each cell is a tally of two columns of the record | `study_record_reviewed_studies.csv` | Table III, and the 83 of Section III-C |

## Table IV. OD trip completion rate of the optimal policy

| Row | Command | Output | Where the value prints |
|---|---|---|---|
| 5x5 grid | `python3 boundary_open_demo/optimal_vi_boundary_dest.py` | printed; the deposited copy of the same computation is `gform/grid_gform.json` | Table IV, the grid row, 100.0 / 100.0 / 0.0 / 100.0 |
| Sioux Falls | `python3 gform/sioux_vi_gform.py` | `gform/sioux_vi_gform.json` | Table IV, the Sioux Falls row |
| Anaheim, convex hull and wider border | `python3 gform/anaheim_shaped_vi_gform.py` | `gform/anaheim_shaped_vi_gform.json`, keys `hull` and `band` | Table IV, the two Anaheim rows, 21.1 / 97.8 and 5.2 / 46.7 |
| Attainable maximum, last column | `python3 anaheim_vi_perimeter.py` | `anaheim/anaheim_vi_perimeter.json`, `anaheim/anaheim_vi_perimeter_bandhull.json` | Table IV, the attainable maximum, and Supplementary Table S-3 |

## Table V. OD trip completion rate and travel time of the trained policy

| Row | Command | Output | Where the value prints |
|---|---|---|---|
| 5x5 grid, 3000 episodes | `python3 boundary_open_demo/travel_time_bdest.py` | `boundary_open_demo/travel_time_bdest_ingrid.json` | Table V, the four 3000-episode grid cells |
| 5x5 grid, 8000 episodes | the same runner at the longer budget | `boundary_open_demo/travel_time_bdest_ingrid_8000.json` | Table V, the four 8000-episode grid cells |
| 5x5 grid, destination-aligned, boundary-open | `python3 boundary_open_demo/travel_time_bdest.py` | `boundary_open_demo/travel_time_bdest_ingrid.json` and `boundary_open_demo/travel_time_bdest_ingrid_8000.json` | Table V, 34.2 at 3000 and 98.7 at 8000 across 30 seeds; the published run, as `check_table6_against_runs.py` reads it |
| Sioux Falls, 3000 episodes | `python3 gform/sioux_cells_3000_gform.py` | `gform/sioux_cells_3000_gform.json` | Table V, the Sioux Falls column at 3000 |
| Sioux Falls, 8000 episodes | `python3 sioux30/run_cell.py` once per seed | `sioux30/dest_boundary_seed00.json` to `seed29.json`, summarised in `sioux30/summary_30seed.json` | Table V, the Sioux Falls column at 8000, 95.7 / 99.8 / 0.0 / 85.9 |
| the cross condition | `python3 gform/cross_condition_8000_gform.py` | `gform/cross_condition_8000_gform.json` | Section IV-C, the 48.6% of a policy trained closed and evaluated open across 30 seeds |

## Table VI. The released implementation retrained on Anaheim

| Row | Command | Output | Where the value prints |
|---|---|---|---|
| the three conditions | `python3 gform/retrain_released_common_gform.py --seeds 30`, with `REPO_DIR` set as the README describes | `gform/retrain_released_common_gform.json` | Table VI, completion 98.0 / 58.7 / 0.0 and exit 0.0 / 41.2 / 100.0 against an attainable maximum of 100.0 / 100.0 / 93.3 across 30 seeds, and Supplementary S-I.D |
| the reachable set of the 30 origins | `python3 released_impl/arrival_reachable_eval_set.py` | `released_impl/arrival_reachable_eval_set.json`, `..._bandhull.json` | Supplementary S-I.D, the two origins that cannot reach the destination |

## Tables VII and VIII. The BOND checklist and the studies with released code

No measured value. Both are read from the record and from the full texts; `gates/countcheck.py`
recomputes every count of Table VII and every mark of Table VIII.

## Eq. (7), the exit-dominance threshold

| Item | Command | Output | Where the value prints |
|---|---|---|---|
| the threshold and the route sets | `python3 gform/eq7_threshold_gform.py`, `python3 gform/eq7_route_sets_gform.py` | `gform/eq7_threshold_gform.json`, `gform/eq7_route_sets_gform_gamma*.json` | Section V-C, the third problem, and Supplementary S-I |

## Figures

| Figure | Command | Output | Where it prints |
|---|---|---|---|
| Fig. 2, the selection flow | `python3 screening_trail/plot_assembly.py` | `fig_flow.png`; the counts are `screening_trail/assembly_stages.json`, asserted by `assembly_stages.py` | Fig. 2 and Supplementary Table S-7 |
| Fig. 3, the family-by-field grid | `python3 plot_family_merged.py` | `fig_family_grid.png` | Fig. 3; every cell is a tally of the record, held in step by `gates/countcheck.py` |
| Fig. 4, the three road networks | `python3 gform/plot_networks_gform.py` | `fig_networks.png`, over `networks/` | Fig. 4 |
| Fig. 4(b), the twelve nodes of the Sioux Falls border | `python3 benchmark_network/outer_face_sioux.py` | `benchmark_network/outer_face_sioux.json`; the trace of the outer face of the drawing, and the attainable maximum at that border and at the coordinate hull | Fig. 4(b) and Section IV-B |
| Fig. 5, the training curves | `python3 gform/plot_reward_condition_8000_gform.py` | `fig_reward_condition.png`, over `gform/four_cells_curves_8000_gform.json` | Fig. 5 |

`gates/check_figure_generators.py` renders each generator and compares the result with the placed
figure pixel by pixel, so a figure and its generator cannot part.

## Supplementary tables

Every file is listed with its anchor in `MANIFEST.md`; the commands are here.

| Table | Command | Output |
|---|---|---|
| S-1 Learner, seeds, budget and steps per episode | none; the protocol of each run file above | — |
| S-2 Pairs drawn, valid and distinct | `python3 check_eval_sets.py` | printed |
| S-3 Attainable maximum of each border set | `python3 anaheim_vi_perimeter.py`, `python3 benchmark_network/outerface12_intervals.py` | `anaheim/*.json`, `benchmark_network/outerface12_intervals.json` |
| S-4 The seven records removed by the exclusion clause | `python3 search_rerun/exclusion_clause_rerun.py` | `search_rerun/exclusion_clause_rerun.csv`, `exclusion_clause_records.csv` |
| S-5 Term families over the related reviews | `python3 survey_axis_coverage.py` | `survey_axis_coverage.json`; the reviews themselves are copyrighted and are not redistributed |
| S-6 The three reward categories | none; the `reward_category` column of the record | `study_record_reviewed_studies.csv` |
| S-7 Evaluation practice across the reviewed studies | `python3 eval_substrate/substrate.py`, `python3 eval_substrate/benchmark_network_adjudication.py`, `python3 fulltext_scan.py` | `eval_substrate/substrate.json`, `benchmark_network_adjudication.json`, `fulltext_scan/fulltext_scan_per_study.csv` |
| S-8 Learner sweep | `python3 boundary_open_demo/dqn_sensitivity.py` | `boundary_open_demo/dqn_sensitivity_30seed.json`, the published form the manuscript prints; `m2c/cells3/dqn_sensitivity_C.json` is the M2-C arm of the same sweep and is not printed |
| S-9 BOND answered for the controlled experiment | none; the experiment itself | — |

Three tables that stood here are now in the appendices of the manuscript itself: the search query
and record count of each source is Table IX, the selection of the reviewed studies is Table X, and
the evidence and recording rule of each field is Table XI. Their commands are
`python3 screening_trail/assembly_stages.py` for the selection, which writes
`screening_trail/assembly_stages.json`; the query counts are the queries of the appendix over
`reviewed_studies/ieee_xplore_export_records.csv`; and the recording rule is a rule, not a run.

| Table | Command | Output |
|---|---|---|
| IX Query, interface and record count | the queries of the appendix | `reviewed_studies/ieee_xplore_export_records.csv` |
| X Selection of the reviewed studies | `python3 screening_trail/assembly_stages.py` | `screening_trail/assembly_stages.json` |
| XI Evidence and recording rule of each field | none; the rule, not a run | — |

## Not included, and why

`repo_v11/screening_trail/_residual.json` and `repo_v11/related_reviews/overlap.json` back no printed
value; the manuscript sources contain no occurrence of the word overlap, and no count traces to the
screening residual.

`benchmark_network/benchmark_results_dest_interior.npz` holds the superseded Sioux Falls
interior-destination run at a border omitting one node of the outer face. It is the source of no
printed value of the manuscript or of the supplement and is kept as a record of the superseded
border alone, so it is not deposited; the Sioux Falls rows come from `sioux30/summary_30seed.json`.

`boundary_open_demo/matched_budget_grid.json` backs no cell of the present text; see Table IV above.

`released_impl/DRL-Router/`, the released implementation of [25] as its authors published it, is not
redistributed here. The retraining scripts of `released_impl/` expect that tree beside them and say
so when it is absent. A reader obtains it from the study's own release and reproduces the four
retrained cells against it.

## Values no run file stores literally

None. Every value printed in the manuscript and in the supplement is held by a run file named in the
tables above. Earlier versions of this package listed four figures here as computed rather than
stored, and each has since been either reproduced from a deposited file or withdrawn with the text
that printed it.

Every bootstrap bound in the manuscript is recomputable from the per-seed list of the file named for
its row above, under the convention section B assigns to it.

## Reading the study record

`study_record_reviewed_studies.csv` carries **95 rows**. Every count printed in the manuscript is
taken over the **93 rows whose `in_reviewed_studies` is `yes`**. The two remaining rows enter no
count, no denominator and no table: `idx` 93 is marked
`no (author own study, post-dates the search window; excluded from every count)`, and `idx` 26 is
marked `no (full text unobtainable at any institution available to this study; excluded from every
count, )`. **Apply the filter before any tally**; a tally over the raw file exceeds a printed
count wherever either study carries a value.

Worked example, the boundary condition:

    rows = [r for r in csv.DictReader(open("study_record_reviewed_studies.csv"))
            if r["in_reviewed_studies"].strip() == "yes"]        # 93 rows
    Counter(r["boundary_condition"] for r in rows)
    # boundary-open-addressed 10, not-addressed 83

which is the 10 / 83 of Section II-B, Table II and the abstract. Over the unfiltered 95 rows the
same tally returns 11 / 84, the eleventh addressed study being `idx` 93 and the 84th unaddressed
being `idx` 26, both excluded.

Two further columns record how the field reached those values:

    Counter(r["boundary_status"] for r in rows)
    # addressed 10, judged-not-addressed 9, not-addressed 74

    Counter(r["boundary_status_note"].split(" (")[0] for r in rows if r["boundary_status"] == "judged-not-addressed")
    # demand-inflow 6, agent-location 2, demand-pattern 1

    Counter(r["boundary_status_note"].split(":")[0].split(" (")[0] for r in rows if r["boundary_status"] == "addressed")
    # treatment stated 7, periphery named 3

`boundary_status` is a partition of the 93 and `boundary_condition` is the field the manuscript
prints; `addressed` and `boundary-open-addressed` agree row for row, and `judged-not-addressed` is
the nine studies re-read at full text for v1.5.0 and moved to not addressed. The 19 rows carrying a
non-empty `boundary_quote` are 10 + 9, which is the count v1.3.0 and v1.4.0 published. The third
tally is the evidence class of the ten, which Section IV-B prints as 7 stating the treatment of a
vehicle reaching a peripheral link and 3 naming a property of the periphery.

**Snapshot notice.** The study record of this package is the current coding. The v1.1 deposit holds
an earlier coding, superseded on the boundary field. From v1.9.0 the screening files travel with this
package, so every table of the Supplementary resolves inside it. See the release notes of this
version.

## Key names that differ from the printed term

| Printed term | File | Key | Note |
|---|---|---|---|
| attainable maximum, 97.8% and 46.7% | `anaheim/anaheim_vi_perimeter_bandhull.json`, `anaheim/anaheim_vi_perimeter.json` | `results.open_aligned_2.5x.arrive_rate` | the arrival rate of the aligned optimum at a 2.5x arrival term, which is the reachable share; no key is named attainable |
| instantaneous state | `study_record_reviewed_studies.csv` | `predictive_representation = none` | the forecast axis; `state_representation` is a different axis, the encoder (raw-vector, graph-encoder) |
| link-level action | the same file | `action_granularity = next-link` | |
| system-level reward | the same file | `reward_alignment = system` | |
| the boundary condition, 10 / 83 | the same file | `boundary_condition` | `boundary_status` and `boundary_status_note` decompose it; see above |
| a policy trained on a boundary-closed network and evaluated on a boundary-open network, 48.6% | `gform/cross_condition_8000_gform.json` | `cells.aligned.scored_open` | the 98.7% it is stated against is the boundary-open cell of Table V, `boundary_open_demo/travel_time_bdest_ingrid_8000.json`, key `open_aligned` |
| the destination is nearer than the nearest exit, 69.0% | `boundary_open_demo/interior_deep_control_vi.json` | `completion_cheaper_pct` | |
| travel-time optimum on the deep lattice, 81.0% | the same file | `optimum.open_time_min` | |
| the 396 ordered OD pairs | `boundary_open_demo/discount_condition.json` | `grid_check.pairs` | the Eq. (7) threshold is `gamma_star`, its inputs `phi_max`, `k_min`, `K`, `delta`, `denominator` |
| the M2-C 5x5-grid cells | `m2c/train_m2c_g1.json` | `open\|aligned\|{}\|3000` and `\|8000` | the cell name is the run's own parameter string; the empty braces are the default reward weights |
| the M2-C reward-term cells | `m2c/train_m2c_g3.json`, `m2c/train_m2c_g4.json` | `open\|aligned\|{"beta": ..., "r_exit": ..., "r_goal": ...}\|<budget>` | shaping alone is `beta` 1.0 with both bonuses zero |

