# v1.7.0: the figure generators

Supersedes v1.6.0 of the same concept record, DOI 10.5281/zenodo.21523970, which always resolves to
the newest version. Cited as tag v1.7.0 in Section V and in Supplementary Section S-VI of the
manuscript.

v1.6.0 was tagged on GitHub and its Zenodo upload was held by an access failure, so no version DOI
was minted for it. This version supersedes it on the record and carries everything it held.

## What is new in v1.7.0

**The generator of Fig. 1 (`screening_trail/`).** Three files: `assembly_stages.py` holds the
fifteen stage counts as constants and asserts every stage as a difference of the stage above it,
`assembly_stages.json` is the record it writes, and `plot_assembly.py` draws the figure from that
record and writes nothing of its own. Supplementary Table S-6 prints the same fifteen values, and
the development repository's `check_flow_stages.py` holds the two in step. Up to v1.6.0 none of the
three was published and the figure's counts could be read only off the page.

**The generator of Fig. 3 (`plot_family_merged.py`).** The script that draws the family-by-field
grid. Its counts are literals inside it rather than a tally computed from the study record, which
`RUN_TO_TABLE.md` now states in the Fig. 3 row rather than leaving a reader to infer; the script
asserts the three totals the manuscript states, and every cell is checkable against
`study_record_corpus_v9_coded.csv`.

**Fig. 1 is redrawn.** The figure v1.6.0 described was the PRISMA 2020 flow diagram in everything
but its three stage bands, box for box and phrase for phrase. This review is neither registered nor
PRISMA-compliant, so the resemblance claimed a protocol it does not run. The figure is now two
bands and four levels: the two routes side by side and unequal, the pool decomposed as the record
decomposes it, one reading band split left and right with the records leaving stated inside it, and
the corpus split by how each study was read. The counts are unchanged; `assembly_stages.json` is
the same record under a new name.

**Both scripts resolve their output inside this package.** `plot_assembly.py` and
`plot_family_merged.py` wrote into a working-tree path that does not exist here. Each now writes
beside itself when no manuscript tree is found, which is the treatment `fulltext_scan.py` and the
`eval_substrate/` scripts already had.

**The two `sioux30` runners name tag v1.7.0** as the source of `benchmark_demo.py`.

## What v1.6.0 added

Supplementary Section S-VI states that every result reported in the study is recomputed from the
archived tag. Of the versions up to v1.5.0 that was true of most values and not of all: the scripts
and outputs behind Table S-22, behind the discounted column of Table V, behind the learner sweep,
behind the retrained published implementation and behind several controls existed only in the
author's working tree. v1.6.0 deposited them, and the discount-consistent shaping
variant with them.

**The discount-consistent shaping variant (`m2c/`).** The shaping term of the destination-aligned
reward is multiplied by `(1 - gamma)`, which makes the increment consistent with the learner's
discount and leaves the term inert at `gamma = 1`. Every exact row of the study is identical under
the two forms at `gamma = 1`, so the regression is an identity. `m2c/shaping_m2c.py` defines the
mixin, `m2c/selftest_shaping_c.py` checks it at the transition level, and the run outputs are
`m2c/train_m2c_g*.json`, `m2c/train_rest*.json`, `m2c/train_exitconv_c.json`, `m2c/cells/`,
`m2c/cells3/` and `m2c/controls/`. `RUN_TO_TABLE.md` names, cell by cell, which printed value
reports which form. Travel-time cells carry no shaping term and are identical under both;
`m2c/controls/sentinel_control_P.json` records the seed-for-seed reproduction that shows it.

**The evaluation-substrate record (`eval_substrate/`, `corpus_text/`).** `substrate.py` and its
`substrate.json` produce 20 of the 29 rows of Supplementary Table S-22 and the five substrate
figures of Section IV-A, reading the 91 extracted full texts of `corpus_text/`.
`benchmark_network_adjudication.py` and its JSON produce 5 further rows. Until this version these
existed only in the working tree, and `RUN_TO_TABLE.md` attributed the whole table to four record
CSVs that produce three of its rows.

**The scripts behind the exact rows.** Seventeen scripts that produce deposited artefacts, at the
paths their outputs already occupied: `term_optima_discounted.py`, `eq7_route_sets.py`, the four
Anaheim solvers, `travel_time_bdest.py`, `matched_budget_grid.py`, `table8_budget_intervals.py`,
`ablation_bdest_matched.py`, `interior_depth_sweep.py`, `interior_depth_intervals.py`,
`outerface12_intervals.py`, `costly_return_intervals.py`, `exit_sentinel_control.py`,
`truncation_control.py`, `sumo_interior_dest.py` and `zero_shot_transfer.py`. The eighteenth is
`boundary_open_demo/optimal_vi_boundary_dest.py`, which is replaced by the 152-line form that
threads the discount through `solve()` and `arrives()`; the 118-line form published up to v1.5.0
takes no discount parameter, so no value printed at the learner's discount was reproducible from it.

**The network data (`networks/`).** `Anaheim_net.tntp`, `Anaheim_trips.tntp`, `anaheim_nodes.geojson`,
`SiouxFalls_net.tntp`, `SiouxFalls_node.tntp` and `ChicagoSketch_net.tntp`, which the Anaheim and
Sioux Falls solvers read, together with `exposure_ratio_real_networks.py`, the module those solvers
import for the border rule.

**The retrained published implementation (`released_impl/`).** The four retrained cells of
Section V-D and the baseline comparison of Section VI: `retrain_released_10seed.json` at the 95-node
band, `retrain_released_hull_10seed.json` at the 13-node hull, `baseline_arm_hull.json` for the
comparison against the study's own shortest-path reference, the two evaluation-set files and the four
scripts. The released implementation of [25] itself is not redistributed; the scripts expect it beside
them and say so when it is absent.

**Outputs that backed a printed value and were not deposited.**
`boundary_open_demo/arrival_visits.json` (14.4 arrivals against 397.0 in 3000 episodes),
`interior_destination_3000_10seed.json` and `interior_destination_10seed.json` (the bespoke-grid
rows of Table S-3), `dqn_sensitivity_10seed.json` (the learner sweep), `discount_condition.json`
(the Eq. (7) threshold and its inputs), `interior_deep_control_vi.json`, `ablation_10seed.json`,
`exit_sentinel_control.json`, `truncation_control.json`, `zero_shot_transfer.json` and
`term_optima_discounted.json`.

**`table6_rows.py`.** The dispersion convention behind every published-form cell of Tables IV and V.
The study states one convention, a 95% percentile bootstrap over the per-seed values at 10,000
resamples, and applies it under two seedings that part by 0.1 at a bound. `RUN_TO_TABLE.md` section B
states both, says which cells use which, and lists the four bounds where a run file's own stored
string differs from the table for that reason. Only the convention functions of this script run
against the package; its loaders name working-tree filenames.

**Two columns in the study record.** `study_record_corpus_v9_coded.csv` gains `boundary_status` and
`boundary_status_note` at the end; every other cell of all 95 rows and 31 original columns is
unchanged. `boundary_status` partitions the reviewed 94 into `addressed` 10, `judged-not-addressed`
9, `not-addressed` 72 and `unclear` 3. `boundary_status_note` is filled on the nine alone and records
the class of evidence that failed the Table S-9 rule: `demand-inflow` 6, `agent-location` 2,
`demand-pattern` 1. The 19 rows carrying a non-empty `boundary_quote`, which is the count v1.3.0 and
v1.4.0 published, are now decomposed inside the file as 10 + 9.

**A correction to the v1.5.0 notes.** Those notes classify the nine recoded studies as "five rest on
sentences about demand entering the network, three on the location of agents at corners or on the
demand pattern, and one on the fog-node architecture". The fog-node reading does not hold. The study
in question is `idx` 30, whose recorded boundary quote is a demand-inflow sentence, and whose full
text contains no occurrence of boundary, border, peripheral or exit in the study-area sense; the fog
node is the computing architecture that partitions the network for the action space, and no sentence
connects it to the treatment of a vehicle reaching a peripheral link. The recode itself stands. The
classification is 6 demand-inflow, 2 agent-location and 1 demand-pattern, which is what
`boundary_status_note` records.

**`RUN_TO_TABLE.md`, rewritten.** All 24 rows of the v1.4.0 map were audited against the artefacts
they name. Six were misattributed, one named a path that does not exist, eight floats had no row at
all, no figure appeared anywhere, the table numbers were off by one against the manuscript, and the
worked example printed a pre-recode tally beside post-recode prose. The map is rewritten from the
artefacts: every body table, every figure and every supplementary table now has a row, the float
numbers are the manuscript's, and each row states which shaping variant its cell reports.

**`README.md`.** A top-level guide to the package, with the superseded-records notice for a reader
who downloads an older version.

## What v1.5.0 added

**survey_axis_coverage.py and survey_axis_coverage.json.** The scan behind the differentiating
column of Table I, run over the full text of each of the ten related reviews. The record carries,
per review, the study-area boundary family and the trip-completion family of Supplementary
Table S-17, each as a hit count with the matched terms, together with a positional control that
distinguishes a search finding nothing from a search that could not run. The published column
reports the boundary family; the trip-completion family is recorded here and reported nowhere in
the manuscript, by author decision.

**Corrected boundary record.** Every study recorded as addressing the boundary condition was re-read
at full text against the rule Supplementary Table S-9 states for the field, namely a statement of the
treatment of a vehicle reaching a peripheral link. Nine did not meet it. The nine move to not
addressed, so the field reads 10 addressed, 81 not addressed and 3 unclear across the 94 studies, in
the manuscript, in Supplementary Table S-21 and in study_record_corpus_v9_coded.csv here. Three of
the nine carry a link-level action and an individual reward, so the group of studies at issue moves
from 41 to 44, and Tables S-19 and S-20 carry 44 rows.

**Scripts resolve their inputs against this package.** `fulltext_scan.py` reads
`study_record_corpus_v9_coded.csv` at the top level of the package and takes the full texts from
the directory named by `PDF_DIR`, which are copyrighted and are not redistributed here.
`survey_axis_coverage.py` takes the review full texts from `SURVEY_PDF_DIR`. The two `sioux30/`
scripts import the environment module from `benchmark_network/` of the repository at the GitHub
tag of this version, or from the directory named by `BENCHMARK_DIR`, and say so when the module is
absent. The v1.4.0 copies pointed at a working-tree layout instead.

## What v1.4.0 added

**RUN_TO_TABLE.md.** The first mapping from a printed cell to the run output that produces it. See
the rewrite above.

**The run outputs behind the two experiment tables** (`boundary_open_demo/`, `sumo_corridor/`,
`anaheim/`, `benchmark_network/`). v1.3.0 carried the Sioux Falls rerun and the full-text scan alone,
so a reader could not recompute the grid and SUMO travel-time columns, the reward-term table, the
Anaheim optima or the attainable maxima. Those files were added there.
`repo_v11/bootstrap_travel_time_ci.py` recomputes every travel-time column and interval from them.

**The interior-deep control** (`boundary_open_demo/interior_deep_control_vi.json`). The 13x13 lattice
computation behind Section V-B: the destination is nearer than the nearest exit on 69.0% of the 200
pairs, and the travel-time optimum completes 81.0%. Both are exact and carry no seed.

**The exit-convention cells** (`costly_return/`). The non-terminal detour cells of Supplementary
Section S-I.B, rerun at ten seeds, 3000 episodes and a 120-step budget, with the per-seed values
retained. The rerun reproduces the published means: 46.2, 5.0, 3.0 and 82.8.

## Reproducing

    python3 eval_substrate/substrate.py
    python3 eval_substrate/benchmark_network_adjudication.py
    python3 repo_v11/bootstrap_travel_time_ci.py
    python3 repo_v11/boundary_open_demo/costly_return.py --seeds 10 --max_steps 120 --out <path>.csv

The two `eval_substrate/` scripts read the study record at the top level of this package and the
full texts from `corpus_text/` beside them; `CORPUS_CSV` overrides the record. The v1.5.0 copies
resolved the record against a working-tree layout instead.

The `m2c/` scripts import the published environment, learner and evaluation draw from
`boundary_open_demo/` of the repository at this tag, so that the variant is the only thing that
moves. Point Python at that directory to run them:

    PYTHONPATH=<repo>/repo_v11/boundary_open_demo python3 m2c/selftest_shaping_c.py
    PYTHONPATH=<repo>/repo_v11 python3 m2c/controls/summarize.py

`summarize.py` needs only `bootstrap_travel_time_ci.py` from that tree and prints every control cell
of Section V-C in its printed form under both variants.

`repo_v11/` is read-only throughout; nothing in this release writes to it.
