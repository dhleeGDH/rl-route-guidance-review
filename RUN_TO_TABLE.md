# RUN_TO_TABLE: which run output backs which printed cell

Every measured value of the manuscript is listed here with the file that produces it, table by
table and figure by figure. Float numbers are the manuscript's own: Tables I to VI in the body,
Figs. 1 to 5 in the body, and Tables S-1 to S-23 in the Supplementary.

Paths are relative to this package unless marked `repo_v11/`, which is the code and data archive
published at the GitHub tag of the same version. Exact computations, meaning value iteration and
reachability enumeration, take no seed and print no interval.

Two things have to be read before the tables below, because a reader who skips them will find
values that look wrong and are not.

## A. Two reward variants

Section V prints two shaping variants of the destination-aligned reward.

**Published.** The shaping term as first run. Its outputs are the `boundary_open_demo/`,
`sumo_corridor/`, `sioux30/`, `anaheim/` and `benchmark_network/` files of this package.

**M2-C.** The discount-consistent form of the same term, which multiplies the shaping increment by
`(1 - gamma)` and is inert at `gamma = 1`. Its outputs are under `m2c/`, and the mixin that defines
it is `m2c/shaping_m2c.py`. Every exact row is identical under the two variants at `gamma = 1`, so
the regression is an identity rather than a coincidence.

The manuscript prints M2-C for the destination-aligned cells of the bespoke grid, for Sioux Falls,
for the interior-destination grid cells, for the depth sweep, for the learner sweep, for the
non-terminal detour cell and for the two sentinel and truncation controls. It prints the published
run everywhere else, including every travel-time cell. A travel-time reward carries no shaping
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
this package. The bespoke-grid M2-C cell is keyed
`handoff/m2c/train_m2c_g1.json|open|aligned|{}|3000|completion_per_seed`, and the same values keyed
on this package's `m2c/train_m2c_g1.json` return a different interval. The four M2-C cells of
Tables IV and V reproduce to the digit under the repository key and under no other.

**A third convention exists in the run files of the two sentinel controls.** The driver that wrote
them, `boundary_open_demo/exit_sentinel_control.py`, draws its own interval from
`numpy.random.default_rng(7)`, which is neither A nor B. The published-arm files
`boundary_open_demo/exit_sentinel_control.json`, `zero_shot_transfer.json` and
`m2c/controls/*_P.json` keep those bounds, since they are the record of the published run as it was
deposited. The M2-C arms that the manuscript prints, `m2c/controls/exit_sentinel_control_C.json` and
`sentinel_control_C.json`, were redrawn under convention B by T-1969 and now carry the printed
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

## Table IV. Optimal and learned OD trip completion rate and travel time, by boundary condition

| Row | Printed | Variant | Source |
|---|---|---|---|
| Travel-time, optimal | 100.0 closed / 0.0 open | exact | value iteration on the bespoke grid, `boundary_open_demo/optimal_vi_boundary_dest.py` |
| Destination-aligned, optimal | 100.0 / 100.0 | exact | the same computation |
| Travel-time, 3000 | 99.9 [99.7-100.0], 5.37 [5.33-5.42], 0.0 | published | `boundary_open_demo/travel_time_bdest_ingrid.json`, cells `closed_time_min` and `open_time_min` |
| Destination-aligned, 3000, closed | 100.0, 5.28 [5.27-5.29] | published | the same file, cell `closed_aligned` |
| **Destination-aligned, 3000, opened** | **38.4 [29.6-47.6]**, 5.09 [4.76-5.33] | **M2-C** | **`m2c/train_m2c_g1.json`, cell `open\|aligned\|{}\|3000`**; the travel time is the published cell `open_aligned` of `travel_time_bdest_ingrid.json` |
| Travel-time, 8000 | 100.0, 5.25 [5.24-5.26], 0.0 | published | `boundary_open_demo/travel_time_bdest_ingrid_8000.json` |
| Destination-aligned, 8000, closed | 100.0, 5.23 [5.22-5.24] | published | the same file, cell `closed_aligned` |
| **Destination-aligned, 8000, opened** | **98.6 [96.5-100.0]**, 5.36 [5.30-5.44] | **M2-C** | **`m2c/train_m2c_g1.json`, cell `open\|aligned\|{}\|8000`**; the travel time is the published cell `open_aligned` of `travel_time_bdest_ingrid_8000.json` |
| Travel-time (SUMO), 3000 | 98.2 [96.6-99.5], 5.30 [5.28-5.32], 0.0 | published | `sumo_corridor/travel_time_sumo_3000.json` |
| Destination-aligned (SUMO), 3000 | 100.0, 5.16 [5.14-5.17], 27.2 [18.8-36.6], 4.85 [4.63-5.02] | published | the same file |
| Travel-time (SUMO), 8000 | 100.0, 5.15 [5.14-5.16], 0.0 | published | `sumo_corridor/travel_time_sumo_8000.json` |
| Destination-aligned (SUMO), 8000 | 100.0, 5.12 [5.12-5.13], 98.0 [94.9-99.8], 5.21 [5.17-5.24] | published | the same file |

The four SUMO rows are the published run because the M2-C rerun of those cells, which is
`m2c/cells/s5_*.json` and `m2c/cells3/sumo_interior_dest_*.json`, moved no cell outside its
published interval and was not adopted for the table.

**`travel_time_bdest_ingrid_8000.json` and `matched_budget_grid.json` are different cells.** The
first holds the sweep default, `open_aligned` 98.6; the second is a separate run at the matched
budget of 8000 episodes over ten seeds, `open_aligned` mean 97.65 over
[100, 100, 100, 100, 100, 99, 100, 92.5, 85, 100]. Earlier versions of the manuscript printed the
matched-budget 97.7 in this row. It now prints the M2-C 98.6, and `matched_budget_grid.json` backs
no printed cell of the present text. It is retained because the published-form Table IV of v1.4.0
and v1.5.0 quotes it.

`repo_v11/bootstrap_travel_time_ci.py` recomputes every travel-time column and interval of Table IV
from the four `travel_time_*.json` files above, under convention B. It also lists
`benchmark_network/travel_time_benchmark.json` as an input; that file is deliberately absent, as it
holds only the superseded ten-seed Sioux Falls run replaced by `sioux30/`.

## Table V. Optimum and learned completion on the boundary-open grid, by reward term and budget

| Row | Printed | Variant | Source |
|---|---|---|---|
| Optimum column | 0.0 / 4.0 / 100.0 / 62.5 / 100.0 | exact | `boundary_open_demo/term_optima_discounted.json`, key `published`; key `rows` gives the same five terms at discounts 0.99, 0.95 and 0.90. `m2c/table_v_m2c.json` records that variant C reproduces all five at every discount |
| Travel time alone, both budgets | 0.0 | published | `boundary_open_demo/ablation_10seed.json`, cell `travel time only`, and `table8_budget_intervals.json` |
| **Potential shaping alone** | **4.5 [4.1-5.2]** and **5.3 [5.1-5.5]** | **M2-C** | **`m2c/train_m2c_g3.json`**, cell `open\|aligned\|{"beta": 1.0, "r_exit": 0.0, "r_goal": 0.0}\|3000` and its 8000 counterpart |
| Arrival term alone, both budgets | 0.0 | M2-C and published agree | `m2c/train_m2c_g3.json`, cell with `r_goal` 10.0; the published run is `ablation_10seed.json`, cell `arrival bonus only` |
| Exit penalty alone, both budgets | 0.0 | M2-C and published agree | `m2c/train_m2c_g4.json`, cell with `r_exit` 5.0; the published run is `ablation_10seed.json`, cell `exit penalty only` |
| **All three terms** | **38.4 [29.6-47.6]** and **98.6 [96.5-100.0]** | **M2-C** | **`m2c/train_m2c_g1.json`**, the two cells of Table IV above |

The published-form counterparts of the last row are `ablation_10seed.json` cell `full aligned`
(35.15) and `table8_budget_intervals.json` (97.7 [94.5-100.0]). `ablation_bdest_matched.json` holds
the same four ablation cells at 3000 episodes as bare per-seed lists.
`repo_v11/boundary_open_demo/ablation_bdest.json` holds a five-seed version of the ablation
(shaping only 4.3, full aligned 30.2) and backs no printed value.

## Table I. The present review against three related reviews

| Column | Source |
|---|---|
| The differentiating column | `survey_axis_coverage.json`, generated by `survey_axis_coverage.py`, one entry per related review, keys `boundary`, `boundary_lexical`, `boundary_detail` and the trip-completion counterparts. The review full texts are copyrighted and are not redistributed; the script takes them from `SURVEY_PDF_DIR` |
| Scope and decision object | the reviews themselves; no run |

## Tables II, III and VI, and the counts of Section IV

| Float | Carries | Source |
|---|---|---|
| Table II, the corpus across the recorded fields | yes | `study_record_corpus_v9_coded.csv`, filtered to `in_reviewed_corpus == yes` |
| Table III, action type against reward alignment | yes | the same file |
| Table VI, the BOND checklist | no measured value | no run |

## Figures

| Figure | Carries | Source |
|---|---|---|
| Fig. 1, the assembly of the corpus | every count in the figure | `screening_trail/assembly_stages.json`, written and checked by `screening_trail/assembly_stages.py`, and drawn by `screening_trail/plot_assembly.py`. The record asserts every stage as a difference of the stage above it and Supplementary Table S-6 prints the same fifteen values |
| Fig. 2, experiment design | schematic | `repo_v11/boundary_open_demo/plot_experiment_design.py` |
| Fig. 3, the family-by-field grid | every cell | `plot_family_merged.py` at the top level. **The counts are literals inside that script, not a tally computed from the record**; it asserts the three totals the manuscript states and nothing else. Each cell is checkable against `study_record_corpus_v9_coded.csv`, and `scripts/countcheck.py` of the development repository is what holds the two in step |
| Fig. 4, the six evaluation networks | schematic | `repo_v11/benchmark_network/plot_networks.py`, over `networks/` |
| Fig. 5, the reward-condition curves | a measured trace | `repo_v11/boundary_open_demo/plot_reward_condition.py` |

## Supplementary tables

| Table | Variant | Source |
|---|---|---|
| S-1 Learner, seeds and budget of each cell | — | the protocol of each run file listed here; no measured value |
| S-2 Optimal action sets and travel times by discount | exact | `anaheim/eq7_route_sets_gamma0.99.json` and `anaheim/eq7_route_sets_gamma0.95.json`, key `share_identical`, for the 0.99 and 0.95 rows of both networks. `m2c/eq7_c.json` carries the same statistic at 1.0, 0.9999, 0.999, 0.99 and 0.95 under both variants, and `m2c/eq7_pairs_c.json` carries the pair-level statistic Section V-B prints |
| S-3 Interior-destination control, bespoke grid rows | published | `boundary_open_demo/interior_destination_3000_10seed.json` (88.1 [74.3-97.2] closed travel-time at 3000) and `interior_destination_10seed.json` (99.8 [99.4-100.0] at 8000), under convention A |
| S-3, bespoke grid, destination-aligned open | **M2-C** | `m2c/train_rest_interior.json`, cells `interior_grid\|open\|aligned\|3000` (95.0 [84.8-100.0]) and `\|8000` (100.0) |
| S-3, SUMO-executed grid rows | published | `sumo_corridor/sumo_interior_closed_time_min.json`, `sumo_interior_open_aligned.json` and the four `sumo_interior8k_*.json`, under convention A |
| S-3, Sioux Falls rows | published | `sioux30/summary_30seed.json`, keys `('dest_interior', ...)`, 30 seeds |
| S-4 Pairs drawn, valid and distinct | — | the `eval_pairs` field of each run file named here, and `sweep_extra.make_eval_od` for the lattice row; exact |
| S-5 Attainable maximum of each border set | exact | reachability over the published link lists of `networks/`; no seed |
| S-6 to S-10 Corpus assembly and screening | — | `repo_v11/screening_trail/`, `repo_v11/corpus/`. See the snapshot notice below |
| S-11 Learner sweep, all ten rows | **M2-C** | `m2c/cells3/dqn_sensitivity_C.json`, key `settings.<name>.cells."open aligned".completion_mean`. The published sweep is `boundary_open_demo/dqn_sensitivity_10seed.json`, whose ten values are 98.6, 24.5, 100.0, 99.95, 94.9, 94.65, 96.3, 98.0, 99.9 and 95.15 |
| S-12, S-13 BOND answered | — | the reports themselves; no run |
| S-14, S-15, S-18 to S-21 Corpus tables | — | `study_record_corpus_v9_coded.csv`, with `study_record_claim_charting.csv` and `gamma_todo.csv` for the title and reference joins |
| **S-22 Evaluation substrate**, 20 rows | — | `eval_substrate/substrate.json`, written by `eval_substrate/substrate.py` over `corpus_text/<idx>.txt`: the substrate partition, the benchmark total, the four node statistics, the seven measures and the five baselines |
| **S-22**, 5 rows | — | `eval_substrate/benchmark_network_adjudication.json`, key `per_network`: Sioux Falls 6, Braess 4, Anaheim 3, Nguyen-Dupuis 3, more than one 5 |
| **S-22**, 3 rows | — | `study_record_simulator_audit_v6.csv`, `category == named-platform` and `platform_names`: the evaluation environment named in 53 of 94, SUMO in 47 of 53, another engine in 6 of 53 |
| **S-22**, the random-or-greedy row | — | `eval_substrate/substrate.json`, adjudicated: the lexical count less one, the excluded occurrence being a false positive recorded in `substrate.py` |
| S-16, S-17 Queries and term families | — | `repo_v11/search_rerun/`, `fulltext_scan/`, `fulltext_scan.py` |
| S-23 Sioux Falls with the destination on the border | mixed | the two travel-time rows and all four travel-time columns from `sioux30/summary_30seed.json`, keys `('dest_boundary', ...)`, 30 seeds; the two destination-aligned completion cells, 99.9 [99.8-100.0] and 92.2 [88.4-95.6], from **`m2c/train_rest_sioux.json`** |
| S-I.B exit conventions, the travel-time detour cells | published | `boundary_open_demo/costly_return_intervals.json`, ten seeds at 3000 episodes on a 120-step budget: 46.25, 5.0 and 3.05 with the printed intervals; the per-seed values are in `costly_return/` |
| S-I.B, the destination-aligned detour cell, 89.2 | **M2-C** | `m2c/train_exitconv_c.json`, cell `costly_return, detour 1.0, aligned`. The published counterpart is the 82.85 of `costly_return_intervals.json` |
| S-I.B residual charge, 2.6 | published | `boundary_open_demo/residual_exit_control.py`; the M2-C rerun is `m2c/cells3/residual_exit_control_C.json` |
| S-I.C the 8x8 lattice, 19.9 (14.3 to 25.3) | published | `boundary_open_demo/grid8_10seed.json`, cell `open_aligned` |
| S-I.E interior destination on the deep lattice | exact | `boundary_open_demo/interior_deep_control_vi.json`: `completion_cheaper_pct` 69.0 and `optimum.open_time_min` 81.0, at n_side 13, margin 4, 200 draws. `m2c/interior_m2c.json` records the same bracket under both variants at three discounts |
| S-I.E the depth sweep, learned | **M2-C** | `m2c/train_rest_depth.json`, ten cells at 8000 episodes, every one 100.0. The published sweep is `boundary_open_demo/interior_depth_sweep.json` with `interior_depth_intervals.json` |
| S-I.E the 7x7 SUMO cell, 97.2 (92.8 to 99.7) | published | re-run from `repo_v11/sumo_corridor/sumo_recon.py`; no output file is deposited, and the driver reproduces both bounds |
| S-II Study-area geometry | exact | reachability over `networks/`; no seed |
| S-V Learner sensitivity | **M2-C** | `m2c/cells3/dqn_sensitivity_C.json`, as Table S-11 |

## Section V-D and Section V-C, the values not in a table

| Printed | Variant | Source |
|---|---|---|
| Anaheim, 21.1% and 94.1% at the 13-node hull, attainable maximum 97.8% | exact | `anaheim/anaheim_vi_perimeter_bandhull.json`, `results.open_time_min.arrive_rate`, `results.open_aligned.arrive_rate` and `results.open_aligned_2.5x.arrive_rate` |
| Anaheim, 5.2% and 45.6% at the 95-node band, attainable maximum 46.7% | exact | `anaheim/anaheim_vi_perimeter.json`, the same three keys |
| Anaheim per-zone geometry and cost model | exact | `anaheim/anaheim_vi_all_zones.json` |
| The 15,289 pairs and the 12,183 pairs | exact | the `pairs` field of the two files above |
| Cross-condition evaluation, 17.1% (12.9 to 21.1) and 41.5% (35.6 to 48.1), paired 24.4 | **M2-C** | `m2c/controls/exit_sentinel_control_C.json`, cells `high_time_min` and `high_aligned`, field `scored_open`; the paired row is `paired.high`. The published counterpart is `boundary_open_demo/exit_sentinel_control.json` at 17.1 / 40.4 / 23.3, identical to `m2c/controls/exit_sentinel_control_P.json` |
| Sentinel-encoding control, 22.7% (18.2 to 28.9) and 60.4% (57.0 to 64.2), paired 37.7 | **M2-C** | the same file, cells `matched_time_min` and `matched_aligned`, and `paired.matched`. The travel-time cell is identical under both variants |
| Arrival-sentinel replacement, 38.4% to 44.1%, paired -5.7 (-21.4 to 9.9) | **M2-C** | `m2c/controls/sentinel_control_C.json`, cells `high_open_aligned` and `matched_open_aligned`, and `paired_open_aligned` |
| Bootstrap-at-cap check, no cell moving by more than 0.8 points | **M2-C** | `m2c/controls/truncation_control_C.json`, the `delta` of each of the four cells. The published counterpart is `boundary_open_demo/truncation_control.json`, identical to `m2c/controls/truncation_control_P.json` |
| The premise of the sentinel control: 16 of 25 nodes, 55.2% of steps | published | `boundary_open_demo/exit_sentinel_control.json`, key `premise` |
| Eq. (7) threshold 0.9859 and the 396 ordered OD pairs | exact | `boundary_open_demo/discount_condition.json`, keys `gamma_star` and `grid_check.pairs`, with the inputs `phi_max`, `k_min`, `K`, `beta`, `R_g`, `delta` and `denominator`. Under M2-C the threshold is 0.9714; `m2c/eq7_c.py` derives it |
| Zero-shot transfer, closed to open | published | `boundary_open_demo/zero_shot_transfer.json`, which duplicates the `high_*` cells of `exit_sentinel_control.json` and adds the paired rows |
| The 12-node outer face of Sioux Falls | published | `benchmark_network/outerface12_intervals.json`, with the border set in key `border` |
| The constrained formulation at a fixed multiplier | published | `boundary_open_demo/constrained_fixed25_10seed.json` |
| The published router retrained on Anaheim, 94.0% and 0.0% at the 95-node band | published | `released_impl/retrain_released_10seed.json`, `results.closed.mean` and `results.open.mean`, ten seeds at 3000 iterations over 30 origins and the one destination of the published configuration |
| The same retraining at the 13-node hull, 96.0% and 40.0% | published | `released_impl/retrain_released_hull_10seed.json`, the same two keys |
| The published router against its own baseline, 96.0% and 100.0% closed, 42.9% and 100.0% at the hull | published | `released_impl/baseline_arm_hull.json`, key `relative_to_attainable`, which also carries the 4.0-point and 57.1-point gaps |
| The attainable maximum of the retraining draw, 93.3% | exact | `released_impl/arrival_reachable_eval_set_bandhull.json`, and `arrival_reachable_eval_set.json` for the 95-node band |
| Arrival episodes, 14.4 against 397.0 in 3000 episodes | published | `boundary_open_demo/arrival_visits.json`, keys `arms."arrival term alone".arrived_mean` and `arms."all three terms".arrived_mean`, written by `arrival_visits.py` |

## Not included, and why

`repo_v11/screening_trail/_residual.json` and `repo_v11/corpus_overlap/overlap.json` back no printed
value; the manuscript sources contain no occurrence of the word overlap, and no count traces to the
screening residual.

`benchmark_network/benchmark_results_dest_interior.npz` holds the superseded Sioux Falls
interior-destination run at the border omitting one node of the outer face, which gives 56.1% where
Table S-3 prints 29.1%. Supplementary Section S-I.E states the supersession. The printed row comes
from `sioux30/summary_30seed.json` and the superseded file is not deposited.

`boundary_open_demo/matched_budget_grid.json` backs no cell of the present text; see Table IV above.

`released_impl/DRL-Router/`, the released implementation of [25] as its authors published it, is not
redistributed here. The retraining scripts of `released_impl/` expect that tree beside them and say
so when it is absent. A reader obtains it from the study's own release and reproduces the four
retrained cells against it.

## Values no run file stores literally

Two printed values are computed rather than recorded. A reader recomputes each from the sample named
beside it rather than searching the package for the number.

| Printed | Where | Why it is not stored | Recompute from |
|---|---|---|---|
| 60.8% | Table S-5, the band of a fifth | arithmetic in the table: 253 border nodes of 416 | the border column of the same row |
| 25.3 | S-I.C, upper bound of `19.9% (14.3 to 25.3)` | a percentile bootstrap bound | `boundary_open_demo/grid8_10seed.json`, cell `open_aligned`, per-seed values, convention A at key `grid8\|aligned\|3000\|open` |

Two further bounds were listed here in v1.4.0 and v1.5.0 as unstored, namely the 76.2 of Table S-3's
`85.5 [76.2-93.3]` and the 8.6 of its `6.2 [4.0-8.6]`. Both are reproduced exactly by convention A
over `sumo_corridor/sumo_interior_closed_time_min.json` and `sumo_interior_open_aligned.json`, and
the entries are withdrawn.

Every other bootstrap bound in the manuscript is recomputable from the per-seed list of the file
named for that row above, under the convention section B assigns to it.

## Reading the study record

`study_record_corpus_v9_coded.csv` carries **95 rows**. Every count printed in the manuscript is
taken over the **94 rows whose `in_reviewed_corpus` is `yes`**. The remaining row is marked
`no (author own study, post-dates the search window; excluded from every count)` and enters no
count, no denominator and no table. **Apply the filter before any tally**; a tally over the raw
file exceeds every printed count by one wherever that study carries a value.

Worked example, the boundary condition:

    rows = [r for r in csv.DictReader(open("study_record_corpus_v9_coded.csv"))
            if r["in_reviewed_corpus"].strip() == "yes"]        # 94 rows
    Counter(r["boundary_condition"] for r in rows)
    # boundary-open-addressed 10, not-addressed 81, unclear 3

which is the 10 / 81 / 3 of Section II-B, Table II and the abstract. Over the unfiltered 95 rows the
same tally returns 11 / 81 / 3, the eleventh being `idx` 93, which is excluded.

Two further columns record how the field reached those values:

    Counter(r["boundary_status"] for r in rows)
    # addressed 10, judged-not-addressed 9, not-addressed 72, unclear 3

    Counter(r["boundary_status_note"] for r in rows if r["boundary_status"] == "judged-not-addressed")
    # demand-inflow 6, agent-location 2, demand-pattern 1

`boundary_status` is a partition of the 94 and `boundary_condition` is the field the manuscript
prints; `addressed` and `boundary-open-addressed` agree row for row, and `judged-not-addressed` is
the nine studies re-read at full text for v1.5.0 and moved to not addressed. The 19 rows carrying a
non-empty `boundary_quote` are 10 + 9, which is the count v1.3.0 and v1.4.0 published.

**Snapshot notice.** The corpus record of this package is the current coding. The copy under
`repo_v11/corpus/corpus_v9_coded.csv`, which `repo_v11/screening_trail/` and the S-6 to S-10 tables
resolve against, is the coding as of v1.4.0 and is superseded on the boundary field. See the release
notes of v1.5.0 and of this version.

## Key names that differ from the printed term

| Printed term | File | Key | Note |
|---|---|---|---|
| attainable maximum, 97.8% and 46.7% | `anaheim/anaheim_vi_perimeter_bandhull.json`, `anaheim/anaheim_vi_perimeter.json` | `results.open_aligned_2.5x.arrive_rate` | the arrival rate of the aligned optimum at a 2.5x arrival term, which is the reachable share; no key is named attainable |
| instantaneous state | `study_record_corpus_v9_coded.csv` | `predictive_representation = none` | the forecast axis; `state_representation` is a different axis, the encoder (raw-vector, graph-encoder) |
| link-level action | the same file | `action_granularity = next-link` | |
| system-level reward | the same file | `reward_alignment = system` | |
| the boundary condition, 10 / 81 / 3 | the same file | `boundary_condition` | `boundary_status` and `boundary_status_note` decompose it; see above |
| policies trained on a boundary-closed network, 17.1% and 41.5% | `m2c/controls/exit_sentinel_control_C.json` | `cells.high_time_min`, `cells.high_aligned`, field `scored_open` | `high` marks the sentinel at its published value; `cells.matched_*` carries the base-cost control, 22.7% and 60.4% |
| paired difference, 24.4 and 37.7 points | the same file | `paired.high`, `paired.matched` | each with `printed_cells`, `printed_difference` and `printed_ci` |
| the destination is nearer than the nearest exit, 69.0% | `boundary_open_demo/interior_deep_control_vi.json` | `completion_cheaper_pct` | |
| travel-time optimum on the deep lattice, 81.0% | the same file | `optimum.open_time_min` | |
| the 396 ordered OD pairs | `boundary_open_demo/discount_condition.json` | `grid_check.pairs` | the Eq. (7) threshold is `gamma_star`, its inputs `phi_max`, `k_min`, `K`, `delta`, `denominator` |
| the M2-C bespoke-grid cells | `m2c/train_m2c_g1.json` | `open\|aligned\|{}\|3000` and `\|8000` | the cell name is the run's own parameter string; the empty braces are the default reward weights |
| the M2-C reward-term cells | `m2c/train_m2c_g3.json`, `m2c/train_m2c_g4.json` | `open\|aligned\|{"beta": ..., "r_exit": ..., "r_goal": ...}\|<budget>` | shaping alone is `beta` 1.0 with both bonuses zero |
| the non-terminal detour cells | `boundary_open_demo/costly_return_intervals.json` | `cells."costly_return_time_min@1.0"` and siblings | the detour cost is the part after the at sign |
