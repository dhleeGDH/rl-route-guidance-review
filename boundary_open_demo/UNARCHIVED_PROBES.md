# Probes with no archived output, and what replaced them

Three scripts in this folder print to standard output and write no result file. Nothing in the
manuscript rests on them, and this note records that so a reader comparing the folder against the
text does not take a printed figure for a reported one.

## state_condition_seeds.py, state_condition_noisy.py

Both repeat the planner comparison of `state_condition_demo.py` over seeds or under a noisy
forecaster.

`state_condition_seeds.py` now writes `state_condition_seeds.json`, so the seed spread and the
per-trip distribution behind Section V-B are archived. The figures it produces reproduce the ones
the manuscript has always printed: 5.5% snapshot excess, 12.2% mean ETA error, 49.2% strictly
slower, over ten seeds. `state_condition_summary.csv` holds seed 12345 alone, at 5.28 / 12.47 /
46.00, and is one draw from that spread rather than a competing value. An earlier note in this
file called the two figures a mismatch. They are not.

`state_condition_noisy.py` still writes no output file, so the six-item checklist of Section VI-D,
which asks for the per-seed values behind each reported figure, cannot be met for it. Nothing in
the manuscript rests on it.

Earlier revisions of Section V-B reported the travel-time margin between the two planners as
evidence for the state requirement. That use has been withdrawn, for one reason and not for the
archiving: the comparison is between two planners that both fix a path at departure, and no study
in the corpus does that. The matched contrast is `state_condition_learned.py`, which runs the same state difference
through the learner, reward, network, cost model, and budget of the four cells. It finds the
instantaneous state 0.7% faster, which is the null Section V-B now reports.

The state requirement carries no performance claim in the current manuscript. Section III states
it as a definitional condition on the object a navigation claim names, and Section V-B uses the
null above to explain why travel time cannot test it.

## n2_log.txt

A three-seed probe of an 8x8 boundary-open aligned cell at 6000 episodes, recorded as a single
log line with no accompanying script. Its configuration cannot be reconstructed, so it supports
nothing. The training-budget result the manuscript reports comes from `budget_control.py`, which
is paired within seed over ten seeds under the current configuration.
