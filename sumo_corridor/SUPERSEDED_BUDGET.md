# sumo_budget_summary.csv and sumo_budget_curves.csv are superseded

These two files were produced before the SUMO evaluation set was corrected, and no figure in
them is comparable to any figure in the manuscript.

The earlier evaluation set drew destinations from cells including interior ones, where the
environment offers no arrival action, so a trip to such a cell could not be completed and the
completion rate was computed over a set the policy could not satisfy. Section S-I.E records the
correction. The evaluation set now draws destinations as training does, from the boundary cells.

The visible consequence is the `b50_e3000` row. It averages 72.5% over its five seeds, against
the 20.7% the manuscript reports for the same nominal cell under the corrected set. The
manuscript states that the superseded 72.5% appears in neither the body nor the supplement, and
this file is where it survives.

Every other row inherits the same defect. In particular `b50_e6000`, at 99.5%, is NOT a
matched-budget replication of the manuscript's SUMO cell and must not be read as one. The only
training-budget control the manuscript relies on is `boundary_open_demo/budget_control.py`, which
is paired within seed and run under the current configuration.

Re-running the budget probe under the corrected evaluation set would make these rows usable. That
run has not been performed.
