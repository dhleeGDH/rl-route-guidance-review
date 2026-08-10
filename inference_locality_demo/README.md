# Inference-Locality Demonstration (Section V-B)

A controlled grid study for the survey's Section V, complementary to the
boundary-open demonstration. It shows that a state built only from locally
observed, current-time traffic carries a measurable travel-time and reliability
cost under time-varying conditions, even where it still reaches the destination.
It is a diagnostic of a state-design property, **not** a proposed method.

## What it demonstrates

Inference locality is a claim about the information a state makes available, not
about a particular learner, so it is demonstrated at the level of the planners
that upper-bound each state design, on a boundary-closed network (so completion is
not at issue) with a congestion band that sweeps across the grid over time:

- **predictive-optimal** — minimum-travel-time OD path on the time-expanded
  network, computed with knowledge of how congestion evolves. Upper-bounds any
  policy whose state carries a predictive component.
- **reactive-local** — an optimal closed-loop policy using only current-time
  congestion: each step it shortest-paths assuming the present persists, moves
  once, observes, and replans. Upper-bounds any local, non-predictive policy, and
  is exactly the "replan reactively" strategy.

The gap isolates the value of predictive information over reactive replanning.

## Result (200 OD pairs x 20 congestion realizations)

| Planner | Mean OD travel time | Std |
|---------|--------------------|-----|
| predictive-optimal | 11.04 | 4.39 |
| reactive-local | 12.83 | 7.26 |

Predictive information lowers mean travel time by 14% and cuts its variability
(standard deviation) by about 40%, avoiding a heavy tail of long trips that
reactive replanning incurs by routing on stale congestion. Locality does not
prevent arrival; it degrades travel-time reliability, the quantity a
navigation-level service must control.

## Files

- `env_p1.py` — grid with a moving congestion band and a local/predictive state
  switch (`python env_p1.py` runs an oracle sanity check).
- `p1_planner.py` — the predictive-optimal vs reactive-local comparison; writes
  `fig_v3_inference_locality.png` and `summary_p1.csv`.
- `dqn.py` — shared DQN (unused by the planner study; retained for parity with the
  boundary demo).

## Reproduce

```
python env_p1.py            # oracle sanity check
python p1_planner.py        # writes the figure and summary
```

## Dependencies

Python 3, `numpy`, `matplotlib`.
