"""Price an explicit global ring with shared hierarchical endpoint budgets."""

from collections import Counter
from functools import lru_cache

from TokenSim.comm.collective_types import (
    CollectiveEstimate,
    LevelEstimate,
    endpoint_bandwidth,
    _rounds,
)


@lru_cache(maxsize=256)
def route_loads(topology, ranks):
    """Count simultaneous ring edges sharing each full-duplex endpoint."""
    outgoing = [Counter() for _ in topology.levels]
    incoming = [Counter() for _ in topology.levels]
    coordinates = {rank: topology.coordinates(rank) for rank in ranks}
    for source, target in zip(ranks, ranks[1:] + ranks[:1]):
        if source == target:
            continue
        left, right = coordinates[source], coordinates[target]
        level = next(i for i in range(len(left)) if left[i] == right[i])
        source_endpoint = source if level == 0 else left[level - 1]
        target_endpoint = target if level == 0 else right[level - 1]
        outgoing[level][source_endpoint] += 1
        incoming[level][target_endpoint] += 1
    return tuple(
        max((*send.values(), *receive.values()), default=0)
        for send, receive in zip(outgoing, incoming)
    )


def estimate_global_ring(query, topology, links, launch_us):
    """Use the slowest simultaneous route per round, then serialize rounds.

    Level zero budgets belong to GPUs; higher-level budgets belong to the
    corresponding node/rack endpoint and are shared by all crossing edges.
    Link latency includes the represented path's hop costs.
    Message-size efficiency uses the collective buffer, as in the existing
    hierarchical estimator, rather than applying a launch penalty per shard.
    """
    ranks = query.layout.ranks
    rounds, injected = _rounds(query.operation, "ring", len(ranks), query.message_bytes)
    if not rounds:
        return CollectiveEstimate(
            0, query.operation, len(ranks), query.message_bytes, "analytical", "trivial"
        )
    per_round = injected / rounds
    loads = route_loads(topology, ranks)
    estimates = []
    for index, simultaneous in enumerate(loads):
        if not simultaneous:
            continue
        level = topology.levels[index]
        link = links.get(level.link)
        budget = endpoint_bandwidth(level, link, query.message_bytes, collective=True)
        time_us = rounds * (link.latency_us + per_round * simultaneous / budget * 1e6)
        estimates.append(
            LevelEstimate(
                level.name,
                query.layout.fan[index],
                "ring",
                rounds,
                per_round * simultaneous,
                level.link,
                time_us,
            )
        )
    latency = launch_us + max((level.latency_us for level in estimates), default=0)
    return CollectiveEstimate(
        latency,
        query.operation,
        len(ranks),
        query.message_bytes,
        "analytical",
        f"analytical:global_ring:{topology.topology_id}",
        tuple(estimates),
        launch_us,
        dict(
            schedule="global_ring",
            simultaneous_edges_per_endpoint=loads,
            rounds=rounds,
            level_timing="Concurrent per round; maximum route service, not summed levels",
        ),
    )
