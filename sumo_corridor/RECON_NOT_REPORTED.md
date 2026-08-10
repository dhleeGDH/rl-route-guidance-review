# sumo_recon_summary.csv is invalid. Use sumo_recon_budget.csv.

`sumo_recon.py` reconstructs the formulation of [50] on a 7x7 SUMO grid, varying only the boundary
and the reward. Its docstring still says "Section V-G". That section no longer exists in this
form: the manuscript replaced the reconstruction with direct examination of the study's released
code, which is stronger evidence about that study than any re-implementation of its reward.
Section V-C and S-I.H carry what came out of that examination.

## Why the earlier summary must not be read

`sumo_recon_summary.csv` reports closed/time_min at 5.17%, open/time_min at 0.00%, open/aligned at
28.00% and closed/aligned at 100.00%. Those figures are the product of two defects found on
2026-08-09, either of them fatal on its own.

**The 7x7 run used 5x5 boundary geometry.** `sumo_env._boundary_cells(n=N)` and
`_exit_action_at(c, r, n=N)` bound `N` as a *default argument*, evaluated once at import with
N = 5. `sumo_recon.py` sets `sumo_env.N = 7` after importing, which moves the module global and
leaves the defaults at 5. Column and row index 4 therefore counted as the perimeter of a 7x7 grid
and the real perimeter at index 6 counted as interior with no exit action. The boundary condition,
which is the variable the whole experiment turns on, was applied to the wrong cells.

**Half the evaluation set could not be completed by any policy.** `make_eval_od` drew destinations
uniformly over all 49 cells, and 103 of the 200 pairs named an interior cell where the environment
offers no arrival action. Section S-I.E records the same correction for the 5x5 SUMO replication,
where it produced the superseded 72.5%.

Both are fixed. `_boundary_cells` and `_exit_action_at` now resolve `N` at call time, which is
byte-identical to the old behaviour for every 5x5 network and therefore changes nothing the
manuscript reports from `sumo_env`. `make_eval_od` now draws destinations from the boundary cells,
as training does.

## The replacement

`sumo_recon_budget.py` reruns all four cells on the corrected setup, five seeds, trained once to
8000 episodes and evaluated at 1500, 4000 and 8000 paired within seed. Results in
`sumo_recon_budget.{json,csv}` and the run log in `sumo_recon_budget_log.txt`.

The 1500-episode column is there because the original run used that budget on a 49-node network,
where the 25-node grid of Section V-B needs 8000 episodes to converge and reads 30.8% at 3000. The
boundary-open travel-time cell is the control: its optimum is leaving, so a longer budget must not
lift it.
