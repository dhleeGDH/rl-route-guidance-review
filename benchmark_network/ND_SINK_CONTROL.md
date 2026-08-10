# What the Nguyen-Dupuis stranded-node handling changes

Nguyen-Dupuis is directed and its two designated destinations have no outgoing arc, so a vehicle
sent to the destination it was not given reaches a node with no move and, on the boundary-closed
network, no exit. `benchmark_demo.py` charges that vehicle the rest of its step budget at the
minimum traversal cost, and the manuscript reports figures from that handling. An earlier revision
ended the episode there at no charge.

`nd_sink_control.py` runs both handlings over the same five seeds, 3000 episodes, and 30-step
budget, with `STRANDED_CHARGE` the only difference. Results are archived in
`nd_sink_control.json`.

| Cell | charged (reported) | free (superseded) | difference |
|---|---|---|---|
| closed, travel-time | 100.0 (0.0) | 47.4 (2.6) | 52.6 |
| closed, aligned | 100.0 (0.0) | 100.0 (0.0) | 0.0 |
| open, travel-time | 0.0 (0.0) | 0.0 (0.0) | 0.0 |
| open, aligned | 100.0 (0.0) | 100.0 (0.0) | 0.0 |

The correction moves one cell. The boundary-open collapse under the travel-time reward and the
full recovery under the aligned reward are identical under both handlings, so no conclusion of
Section V turns on the choice. What the free ending does change is the boundary-closed comparison:
at 47.4% against 100.0% the two rewards would separate there, and that separation is a property of
the free stop rather than of either reward. The charged handling removes it, which is why the
manuscript reports the two rewards as indistinguishable on every boundary-closed network.

Written into the supplementary material at S-I.F.
