# MOGA + TDMA WSN Simulation

The lifetime comparison now runs packet traffic and radio energy accounting in
ns-3. MOGA selects cluster heads; LSA routing has been removed. The optional
GraphSAGE variant changes the intra-cluster objective space but uses the same
MOGA selection and TDMA traffic model.

## Network and links

- Each run samples 100 sensors from the WSN CSV using its seed. The sink is
  placed at the mean sensor coordinates.
- ns-3 uses a fixed Euclidean `RangePropagationLossModel`. Its range is the
  maximum pairwise distance among all sensors and the sink (+1 m), so the
  deployment is dense: every node reaches the BS and every possible CH.
  RSSI is not used.
- The GraphSAGE graph and the node degree use the separate neighbour range
  `Network.communication_range` (13.95 m), not the radio range.
- After every re-cluster the BS assigns each live sensor to its nearest CH and
  fixes this assignment until the next re-cluster (see TDMA below). CHs may
  have no members.

## MOGA

Each chromosome is a unique set of live node indices, with the CH count set to
10% of the live sensors (at least one).

Objectives, min-max normalized using the initial population's bounds:

1. Minimize the sum of member-to-CH distances.
2. Minimize the sum of CH-to-sink distances.
3. Minimize the negative sum of CH residual energy.

The GA returns its final first non-dominated front. The deployed CH set is the
front member with the lowest sum of the three normalized objectives (Eq. 1).
Feasible solutions dominate infeasible ones; among infeasible solutions,
lower normalized constraint violation takes precedence. The constraints are:

- CHs must be live and have positive residual energy.
- Each CH's residual energy must not exceed its own initial energy.
- For a non-empty cluster, CH-to-sink distance must not exceed the farthest
  member's distance to the sink.
- Each member-to-CH distance must not exceed the maximum pairwise distance
  within that cluster. An empty cluster has no distance constraint.

Survivor selection takes the best half as the parent pool, creates `P`
children, then selects `P` survivors from the combined `2P` population by
non-dominated fronts. Crowding distance breaks ties when truncating a front.
Defaults: population 100, 30 generations, crossover probability 0.8, initial
mutation rate 0.2. If the best fitness fails to improve for 10 generations,
the mutation rate increases by 1.5x, capped at 0.9. Objective normalization
stays fixed within a MOGA run so fitness comparisons across generations are
meaningful.

## TDMA and transmitted data

The BS assigns a two-level TDMA schedule after each re-cluster and keeps it
for every round until the next re-cluster. Slots are sized from payload
length at 11 Mbps with a 2 ms guard interval. Each round:

0. Only after a re-cluster: the BS broadcasts the schedule (CH slot order and
   each CH's member slot order). Every live sensor receives it.
1. Level 1, one frame per CH: the CH broadcasts a short beacon to its members,
   then each member sends a JSON UDP payload to its CH in its own slot:
   CSV temperature and humidity, current residual energy and degree. The
   residual energy is the piggybacked network-state data.
2. Level 2: each CH sends its aggregate (its own record and those of the
   members it received) directly to the BS in its slot; there are no inter-CH
   routes. The frame ends with emergency slots for orphaned members (below).

Only the scheduled sender and receivers of a slot pay radio energy; every
other node sleeps (idle current only). Energy is debited slot by slot, so a
node whose battery cannot cover a slot dies in that slot: it does not
transmit, and packets addressed to it are lost.

Degree is the count of other sensors within the 13.95 m neighbour range.

## CH failure and re-cluster triggers

A member whose CH sent no beacon (CH dead), or whose packet got no MAC ACK
(CH died mid-frame), is orphaned. It sends its record directly to the BS in
an emergency slot at the end of the round. The BS identifies senders by IPv4
address; data from a node that is not a scheduled CH means a CH failed.

After each round the BS re-clusters (MOGA, then a new schedule broadcast) when:

1. `ch_failure`: it received data directly from a non-CH node;
2. `no_live_ch`: no scheduled CH is alive;
3. `predicted` (`--method predictive` only): XGBoost predicts `recluster`.
   `watch` calls `handle_watch()` (placeholder) and `healthy` keeps the schedule.

The `moga` and `graphsage` baselines re-cluster every round (`periodic`).
Each round's reason is stored in `trajectory.recluster_reason` and the totals
in `recluster_reasons`. Data lost while a re-cluster is computed is not modelled.

## Running

Set `WSN_DATASET_CSV` to the input CSV and `NS3_BINDINGS_DIR` to the ns-3
`bindings/python` directory. Run:

```text
python run_compare.py
```

The script compares MOGA and MOGA+GraphSAGE over three seeds and 3,000 rounds.
`ns3_wsn_sim.py` can also run one method directly with `--seed`, `--rounds`,
`--method`, `--dataset-csv`, and `--output`.

## Scope and deviations

This is a project-specific adaptation, not a literal reproduction of the
paper. The paper's initial random clustering, 100-chromosome population and
LSA route search are not used in this TDMA setup; members reach the BS
directly only after a CH failure. The MOGA population and adaptation settings, Euclidean
communication range, TDMA slot timing, and ns-3 radio model are explicit
implementation choices.
