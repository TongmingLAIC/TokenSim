# Explicit global Ring communication

`CollectiveQuery.schedule` selects `hierarchical` by default or `global_ring` for an explicit global Ring.
The hierarchical estimator retains its existing behavior and results.
Its `algorithm="ring"` selects a Ring within each topology level and does not specify a single Ring across every rank.
Use `global_ring` when the reference explicitly forces NCCL Ring across the complete communicator.

The global schedule preserves logical rank order and maps each successor edge to its lowest common topology level.
GPU endpoints own level-zero bandwidth, while higher-level endpoints own their node or rack injection budgets.
Concurrent outgoing and incoming flows share each full-duplex endpoint budget independently.
Contiguous placement can have one Ring edge crossing each node boundary; interleaved placement can have several crossing edges sharing the same NICs.
The model uses the maximum simultaneous route service per round and serializes the algorithm's rounds.
It adds software launch once and retains the existing link-efficiency and complete-buffer byte conventions.
This is an analytical synchronous-round model and does not simulate packet arbitration or NCCL channel/protocol selection.

```python
query = CollectiveQuery(
    "all_reduce",
    message_bytes,
    topology.group_layout(tuple(range(16))),
    dtype="fp32",
    algorithm="ring",
    schedule="global_ring",
)
estimate = model.estimate(query)
```

The supported operations are AllReduce, AllGather and ReduceScatter.
AllGather uses the complete gathered-buffer convention of `CollectiveQuery`; its local input size is the complete size divided by participant count.
This differs from the per-rank-input convention of `flat_collective`, which remains unchanged.
Measured packages must match the requested algorithm, schedule and platform; these conditions remain part of their source provenance.
Analytical extrapolation from a measured anchor preserves the requested schedule.
