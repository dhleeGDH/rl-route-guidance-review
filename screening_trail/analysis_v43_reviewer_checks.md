# Supplementary analysis for the v43 revision (reviewer checks)

Reproducible from `corpus_v9_coded.csv` (idx 93 = author exemplar, excluded; N = 94).

## M7. Arrival-term presence within the 41-system exposed population

Exposed population = next-link action AND individual reward AND boundary not-addressed (n = 41).
Classification read from each system's charted reward quotation (`align_quote`, `rt_quote`):

- Guaranteed inversion (n=10): quoted reward is a complete travel-time or link-cost term,
  no arrival/terminal reward and no destination-progress term:
  [117], [24], [78], [63], [47], [89], [65], [28], [111], [116]
- Explicit terminal arrival reward, magnitude unstated (n=10): [76], [20], [118], [39], [92], [98], [99], [56], [128], [129]
- Destination-progress term, magnitude unstated (n=9): [77], [61], [80], [81], [120], [59], [97], [113], [114]
- Undetermined from a goal-only or partial quotation (n=12): [75], [79], [44], [119], [37], [58], [38], [27], [121], [51], [42], [131]

Total 10 + 10 + 9 + 12 = 41.

## M4. Full-census re-derivation of the three decisive fields (all 94)

Re-derived each recorded value against its released source quotation, forecast-conditioning
+ boundary + trip-completion, across all 94 systems. No value required correction.
- forecast-conditioning: 3 systems flagged by a prediction keyword resolved to the recorded
  non-predictive value (prediction deferred to future work, explicitly bypassed, or a
  non-forecast quantity); counts 17 predictive / 71 non-predictive / 6 unclear stand.
- boundary: 0 not-addressed systems carried a boundary-handling quotation; the 3 unclear are
  the 3 abstract-only systems. Counts 19 open / 72 not-addressed / 3 unclear stand.
- trip-completion: every cell matched its quotation. Count 16 reported / 91 stands.
- M2b known-schedule boundary case (deterministic time-dependent cost schedule): 0 among the
  non-predictive systems, so it does not move the 71.

## M3. Open-boundary natural experiment (the 19 boundary-open systems)

By reward: individual 8, mixed 7, system 4. Systems that address the open boundary predominantly
reintroduce destination alignment, several naming the boundary-open failure as the reason:
- [19]: large negative reward on boundary-link terminal states + explicit terminal reward term.
- [124]: large penalty as the state approaches a defined boundary (return drops sharply near it).
- [40]: reports vehicles otherwise circle and cannot find the exit; rewards proximity to destination.
- [91]: rule constraints to keep vehicles from being directed to edge roads and left unable to arrive.

## M6b. Shaping decomposition (boundary-open 5x5, 5 seeds)

- shaping-alone, non-vanishing Phi(exit): 28.9% (reproduces the main ablation).
- shaping-alone, vanishing Phi(exit) [policy-invariant]: 0.0% across all 5 seeds.
The isolated recovery of shaping is therefore entirely its objective-alignment (non-invariance)
effect, not a learning aid. Code: `experiments/boundary_open_demo/shaping_decomp.py`
(env flag `vanish_potential`).

## M5/M6a budget diagnostic (SUMO open-aligned, 5 seeds)

- decision budget 50 -> 100 steps: mean 72.5% -> 74.0%, sample SD 18.4 -> 16.3 pp (unresolved).
- training budget 3000 -> 6000 episodes: mean 99.5%, sample SD 1.1 pp (resolved).
Code: `experiments/sumo_corridor/sumo_m5_budget.py`.
