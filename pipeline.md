# MOGA + TDMA WSN Simulation

The lifetime comparison now runs packet traffic and radio energy accounting in
ns-3. MOGA selects cluster heads; LSA routing has been removed. The optional
GraphSAGE variant changes the intra-cluster objective space but uses the same
MOGA selection and TDMA traffic model.

## Network and links

- Each run samples 100 sensors from the WSN CSV using its seed. The sink is
  placed at the mean sensor coordinates.
- ns-3 uses a fixed Euclidean `RangePropagationLossModel`. Its range is the
  maximum pairwise distance among all sensors and the sink, so every node and
  every sensor pair is in range. RSSI is not used.
- All live sensors are assigned to their nearest live CH. Members never send
  directly to the sink. If a CH dies during a round, MOGA chooses a new CH set
  after that round's CH-to-sink phase; surviving sensors are assigned to the
  nearest selected CH in the next round. CHs may have no members.

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

Each round has two serialized TDMA phases. Slots are sized from payload length
at 11 Mbps with a 2 ms guard interval:

1. Every cluster head schedules one slot for each of its members. Each member
   sends a JSON UDP payload containing its CSV temperature and humidity,
   current residual energy, and degree.
2. Each CH has one slot to send an aggregate payload with those values for
   itself and its assigned members directly to the sink.

CH-to-sink transmissions are direct; there are no inter-CH routes. After all
CH slots, MOGA runs once to produce the next round's clustering. Nodes use
zero idle and CCA radio current to model sleeping outside their TDMA slots;
ns-3 accounts for transmission and reception energy.

Degree is the count of other sensors within the configured radio range. Since
the range is derived from the maximum pairwise distance, this deployment is
fully connected and every sensor has degree `N - 1`.

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
paper. The paper's initial random clustering, 100-chromosome population,
LSA route search, and sink-adjacent direct-member traffic are not used in this
TDMA setup. The MOGA population and adaptation settings, Euclidean
communication range, TDMA slot timing, and ns-3 radio model are explicit
implementation choices.
