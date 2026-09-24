# RL route guidance review: data, code and run outputs

This package holds the study record, the run outputs and the scripts behind every measured value of
the manuscript *Reinforcement learning for navigation-level route guidance: the study-area boundary
and the identifiability of the routing objective* (single author, Donghoun Lee).

Cited in the manuscript as tag v1.10.0 of
<https://github.com/dhleeGDH/rl-route-guidance-review>, under the concept DOI
10.5281/zenodo.21523970, which always resolves to the newest version.

## Where to start

**`RUN_TO_TABLE.md` is the index.** Every printed table, figure and body value has a row there
naming the file that produces it, the key inside that file, and which of the two reward variants the
cell reports. Read its sections A and B before comparing any number with any file: the study prints
two shaping variants and applies one dispersion convention under two seedings, and a reader who
skips those two pages will find values that look wrong and are not.

**`RELEASE_NOTES.md`** says what each version added and what changed since the last one.

## One external dependency

Every runner of this package imports only what the package carries, with one exception. The
retraining of Section IV-C calls the released implementation of the reviewed study [25], published
at <https://github.com/YoZo-X/DRL-Router>. That repository is not redistributed here, since it
carries no licence of its own. The runs of this study used its commit
`e204293ffaa401f4eb86bd4527a61d01b85171d1` of 11 October 2021, the head of its default branch,
which carries no tag. `released_impl/` holds this study's wrappers around it; to run them,
clone that repository and give its path, either as `--repo <path>` or by setting `REPO_DIR` to the
directory holding it. Nothing else in the package needs it.

## Layout

| Path | Holds |
|---|---|
| `study_record_reviewed_studies.csv` | the field record of the review: 95 rows, of which the 93 with `in_reviewed_studies == yes` carry every printed count; the other two are the author's own post-window study and a study whose full text no institution available to the author holds |
| `study_record_claim_charting.csv`, `study_record_simulator_audit_v6.csv`, `gamma_todo.csv` | the claim charting, the simulator audit and the reference join |
| `reviewed_studies_text/` | the 93 extracted full texts the lexical scans read, by `idx`, with a `<idx>.README.txt` beside each text acquired, recovered or withdrawn after the first extraction (13, 22, 26, 40, 48) |
| `eval_substrate/` | the evaluation-substrate scan behind Supplementary Table S-22 and the substrate figures of Section IV-A |
| `fulltext_scan/`, `fulltext_scan.py` | the term-family scan behind Supplementary Tables S-16 and S-17 |
| `survey_axis_coverage.json`, `survey_axis_coverage.py` | the scan behind the differentiating column of Table I |
| `boundary_open_demo/` | the bespoke-grid and lattice runs, the exact solvers and the controls |
| `sumo_corridor/` | the SUMO-executed grid runs |
| `sioux30/` | the Sioux Falls runs at 30 seeds |
| `anaheim/` | the Anaheim value iteration and the Eq. (7) route sets |
| `benchmark_network/`, `costly_return/` | the outer-face intervals and the non-terminal detour cells |
| `networks/` | the published link and node data of Anaheim, Sioux Falls and Chicago Sketch |
| `released_impl/` | the retraining of the published router of [25] under each boundary condition |
| `m2c/` | the discount-consistent shaping variant: the mixin, its self-test, its run outputs and its controls |
| `table6_rows.py` | the dispersion convention behind the published-form cells of Tables IV and V |
| `screening_trail/` | the record behind Fig. 1, the script that writes and checks it, and the script that draws the figure |
| `plot_family_merged.py`, `plot_family_reporting.py` | the script that draws Fig. 3, and the superseded single-group form of the same grid that the appendix gate reads as its record |
| `boundary_open_demo/plot_experiment_design.py`, `boundary_open_demo/plot_reward_condition.py`, `benchmark_network/plot_networks.py` | the scripts that draw Figs. 2, 5 and 4, at the revisions that produced the placed figures |
| `gates/` | two gates of the development repository, deposited as they run there: `check_figure_generators.py`, which renders every generator and compares the result pixel by pixel with the placed figure, and `countcheck.py`, which recomputes every printed count from Table A-1 and the record. Both resolve manuscript paths and do not run inside this package |

## Superseded records

A reader who downloads an older version, or who opens the repository tree rather than this package,
should know two things.

**The boundary field was corrected in v1.5.0.** Versions v1.3.0 and v1.4.0 record 19 studies as
addressing the boundary condition of the study area. Every one of those 19 was re-read at full text
against the rule Supplementary Table S-9 states for the field, which is a statement of the treatment
of a vehicle reaching a peripheral link. Nine did not meet it and moved to not addressed. In v1.8.0 the field
reads 10 addressed and 83 not addressed over the 93 reviewed studies, with no unclear value left. The direction
of the correction is downward: no study moved into the addressed group. The `boundary_status` and
`boundary_status_note` columns of the study record decompose the old 19 as 10 + 9 and record why
each of the nine moved.

**The record was recomposed at 93 reviewed studies in v1.8.0.** Up to v1.7.0 the record held 94 reviewed
studies, three of them read from a title and an abstract alone. Two of the three, `idx` 13 and
`idx` 40, were read at full text and every field re-recorded from the text, with the evidence
sentence of each value in `reviewed_studies_text/13.README.txt` and `40.README.txt`; the third, `idx` 26,
could not be obtained from any institution available to the author and left the record. Every
reviewed study is therefore a full text, one denominator of 93 applies to every field but the
generalization mechanism (91), and every count the manuscript prints moved with the record. The
lexical scans of `eval_substrate/` and `fulltext_scan/` were re-executed over the 93 texts.

**The screening files travel with this package from v1.9.0.** The export records, the
exclusion-clause re-execution and its two outputs are under `reviewed_studies/` and `search_rerun/`
here, so the screening tables of the Supplementary resolve inside the package. The v1.1 deposit
holds an earlier coding of the record, superseded on the boundary field; the current coding is
`study_record_reviewed_studies.csv` at the top level of this package, and every count printed in the
manuscript is taken over it.

## Reading the study record

Filter to the reviewed studies before any tally:

    rows = [r for r in csv.DictReader(open("study_record_reviewed_studies.csv"))
            if r["in_reviewed_studies"].strip() == "yes"]        # 93 rows

The two remaining rows enter no count, no denominator and no table: `idx` 93 is the author's own
study, which post-dates the search window, and `idx` 26 is the study withdrawn in v1.8.0, whose full
text no institution available to the author holds. `RUN_TO_TABLE.md`, under "Reading the study record", gives the worked
tallies of the boundary field and the two status columns.

## What is not here

The full texts of the reviewed studies and of the related reviews are copyrighted and are not
redistributed. `reviewed_studies_text/` holds the extractions the lexical scans read; the scans that need the
PDFs themselves take their directory from `PDF_DIR` or `SURVEY_PDF_DIR`. The released implementation
of [25] is likewise not redistributed; `released_impl/` holds this study's retraining of it and
expects that tree beside it.
