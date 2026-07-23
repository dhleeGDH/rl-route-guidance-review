# Reinforcement Learning for Vehicle Route Guidance: Code and Data

Environment, planner, training, and analysis code for the controlled experiments reported in
*Reinforcement Learning for Vehicle Route Guidance: A Critical Review of Reward Design and
Evaluation Boundary Conditions*, together with the screened corpus and the per-study extraction
the review is built on.

The experiments answer one question. A route-guidance policy trained on a network whose
boundary is closed, and scored by travel time, converges to a return that says nothing about
whether trips arrive. Opening the boundary makes leaving the network the optimum under that
reward, which exact value iteration confirms, and OD trip completion falls from 99.5% to 0.0%.
A destination-aligned reward restores arrival as the optimum. The code reproduces both results
and the robustness checks around them.

## Layout

| Directory | Contents |
|---|---|
| `boundary_open_demo/` | The 5x5 lattice environment, the DQN and PPO learners, exact value iteration, the planner comparison of Section V-B with its noisy-forecaster control, and every cell of Section V |
| `benchmark_network/` | Sioux Falls and Nguyen-Dupuis replications and the figure that reports them |
| `sumo_corridor/` | The SUMO port of the four cells |
| `xrouting_mc4/` | Inspection notes on a released system's reward formulation |

## Reproducing the main result

```
cd boundary_open_demo
python four_cells_boundary_dest.py      # the four cells of Table IV
python optimal_vi_boundary_dest.py      # exact value iteration, the optimum each cell admits
python plot_headline.py                 # Fig. 6
```

Each script writes its results to a JSON file beside itself and prints a summary. Seeds are
fixed in the scripts, so a rerun reproduces the reported numbers rather than a sample near them.

Other cells follow the same pattern:

```
python exit_density.py                  # completion against exit geometry
python ablation.py                      # which reward terms recover completion
python constrained_arrival.py           # arrival as a constraint, with the multiplier adapted
python replay_composition.py            # what the learner replays, and the lattice gap
python interior_destination.py          # the four cells with an interior destination
python costly_return.py --max_steps 120 # a non-terminal wrong exit
```

The benchmark networks and the SUMO port have their own entry points:

```
cd ../benchmark_network && python benchmark_demo.py && python plot_benchmark.py
cd ../sumo_corridor && python sumo_train.py
```

## Corpus and extraction

`corpus/` carries the screened corpus and the per-study extraction. Every non-trivial recorded
value is anchored to a verbatim quotation from the study it describes, so any cell can be
checked against its source without rerunning anything. The state field is also recorded as a
decision rule over those quotations, which reproduces the recorded value in 85 of the 88
studies the rule resolves; the three it does not resolve are marked.

## Requirements

Python 3.9, with `numpy`, `torch`, `matplotlib`, and `pandas`. The SUMO port additionally needs
SUMO 1.25 with `traci` on the path. `pip install -r requirements.txt` covers the rest.

## Citation

If you use this code, please cite the paper. A BibTeX entry will be added on publication.

## License

MIT. See `LICENSE`.
