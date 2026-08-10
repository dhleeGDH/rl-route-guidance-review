# This retraining was attempted and left incomplete. It supports no conclusion.

`train_open.py` retrains the released XRouting policy of [50] on a boundary-open variant of its
own network, with the released reward and the released architecture. It is the only attempt in
this program to put the failure to a published system rather than to a reconstruction, and it did
not finish.

## What stopped it

The 2022 environment starts a fresh SUMO every episode and leaks until a native access violation
after roughly a dozen resets, so training runs in chunks under an external relaunch loop.
`run_open/state.json` records `done_iters: 16` against `XR_ITERS = 50`. The reward is still moving
at the point it stopped, at -1162.7 mean on iteration 15 and -1142.3 on iteration 16.

## Why the 100.0% in run_open/eval_completion2.json must not be read as a result

Three reasons, each sufficient.

1. The policy is roughly one third trained. Section V-B's own grid cell needed 8000 episodes to
   converge and reads 30.8% at 3000, so an early checkpoint completing trips says nothing about
   the objective it was trained under.
2. The evaluation covers **6 episodes**. No dispersion can be computed from it.
3. The exit geometry of the modified network was never measured against the bound of Section V-B.
   Exit dominance requires `k > rho * c_max + 1`. Where the destination is nearer than the
   nearest exit the travel-time reward is expected to complete trips, so 100.0% completion with
   zero boundary departures is consistent with the paper's own analysis rather than a
   counterexample to it. Which of the two holds here is undetermined.

## Two evaluation files, one superseded

`run_open/eval_completion.json` reports 0.0% over 8 episodes. It is **wrong** and superseded by
`eval_completion2.json`. It counted arrivals through a `traci.vehicle.remove` call, while the
released environment resolves arrival through SUMO's own arrival event, so no arrival was ever
counted. `eval_completion2.py` reads the per-episode tripinfo and compares `arrivalLane` against
the destination edge, which is the correct test. The 0.0% is retained only so that the correction
is visible.

## Reproducing it

The stack is Windows-only as archived: SUMO with traci, ray 2022 with the RLlib PPOTrainer of that
era, the released XRouting tree at `C:\xro`, and the checkpoint paths under `D:\`. It cannot be
resumed on the machine this package was assembled on.

## What the manuscript says instead

Section V-C and S-I.H report what was established from the released code without retraining: its
reward is the pure travel-time form, and its shipped network places the destination on a boundary
link with no other exit at the perimeter, which resolves that study boundary-closed. That
finding does not depend on this run.
