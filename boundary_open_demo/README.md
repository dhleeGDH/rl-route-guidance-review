# Boundary-Open Demonstration (Section V)

A controlled 5x5 grid experiment for the survey's Section V. It shows,
quantitatively, that the configuration most DRL route-guidance studies adopt
(a travel-time-minimizing reward evaluated on a network whose boundary is not
treated as open) hides a failure that appears once the boundary is open. The
experiment is a **diagnostic of that configuration, not a proposed method**.

## What it demonstrates

Four conditions from two independent switches, `boundary in {closed, open}` and
`reward in {time_min, aligned}`:

| Condition | Meaning | Expected |
|-----------|---------|----------|
| closed, time_min | typical setup on a closed network | completes |
| open, time_min | typical setup on an open network | early-exit failure |
| open, aligned | reward reformulated to remove the early-exit incentive | recovers |
| closed, aligned | reformulated reward on a closed network | completes |

The contrast between the two `time_min` conditions is the point: a boundary-closed
test makes the typical reward look adequate, while the same reward on a
boundary-open network collapses the OD trip completion rate. The `open, aligned`
condition holds the state fixed and changes only the reward, which attributes the
failure to reward design rather than to task difficulty.

## Files

- `env.py` — `GridRouteEnv`, the grid MDP with the two switches (run `python env.py`
  to execute the environment self-tests).
- `dqn.py` — a standard MLP Deep Q-Network with a replay buffer, target network,
  epsilon-greedy exploration, and action masking.
- `train.py` — trains all four conditions across seeds and records the OD trip
  completion rate during training.
- `plot.py` — produces the two Section V figures from `results.npz`.

## Reproduce

```
python env.py                                              # environment self-tests
python train.py --episodes 3000 --seeds 5 --max_steps 120  # writes results.npz, summary.csv
python plot.py                                              # writes the two figures
```

`python train.py --smoke` runs a fast single-seed check.

## Result (5 seeds, 3000 episodes, horizon 120)

| Condition | Final OD trip completion rate |
|-----------|------------------------------|
| closed, time_min (typical) | 82.3% +/- 10.2% |
| open, time_min (typical) | 0.0% +/- 0.0% |
| open, aligned (control) | 93.7% +/- 7.0% |
| closed, aligned | 99.4% +/- 1.0% |

The typical setup completes the majority of trips on a closed network and none on an open
network; changing only the reward restores completion on the open network.

## Dependencies

Python 3, `numpy`, `torch`, `matplotlib`. No simulator or external graph library
is required; the environment is self-contained for exact reproducibility.

## Determinism

Each run seeds NumPy and PyTorch. Evaluation uses one fixed held-out set of OD
pairs (`make_eval_od`, seed 12345) shared across every condition and seed, scored
greedily, so completion rates are comparable across conditions.
