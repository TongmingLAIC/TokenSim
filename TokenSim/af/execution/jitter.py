"""Mean-one lognormal disturbances keyed by request work, without global RNG."""

from dataclasses import replace
import hashlib
import math
import random


def request_multiplier(seed, identity, request_id, cv):
    if not cv:
        return 1.0
    digest = hashlib.sha256(repr((seed, identity, request_id)).encode()).digest()
    normal = random.Random(int.from_bytes(digest[:16], "big")).gauss(0, 1)
    sigma = math.sqrt(math.log1p(cv * cv))
    return math.exp(sigma * normal - sigma * sigma / 2)


def perturb_service(
    estimate, config, block, layer, stage, *, network=False, query=None
):
    direction = getattr(query, "direction", None)
    collective = network and direction not in {"a_to_f", "f_to_a", None}
    cv = (
        (config.tp_cv if collective else config.network_cv)
        if network
        else config.compute_cv
    )
    rank_cv = config.rank_compute_cv if not network else 0
    participants = getattr(query, "physical_endpoints", ())
    tp_rank_cv = (
        config.tp_rank_cv
        if network
        and len(participants) > 1
        and getattr(query, "operation", None)
        in {"all_reduce", "all_gather", "reduce_scatter", "tp_collective"}
        else 0
    )
    if (
        not config.enabled
        or not (cv or rank_cv or tp_rank_cv)
        or not estimate.service_duration
    ):
        return estimate
    edges = getattr(query, "routing_edges", ()) if query is not None else ()
    requests = sorted(
        {edge.request_id for edge in edges} or set(block.request_ids), key=repr
    )
    # Work identities exclude submission order, owner placement and selected scheme.
    # Averaging models shared service variation; it imposes no batch-max penalty.
    identity = (
        "network" if network else "compute",
        layer,
        stage,
        block.phase,
        tuple(sorted(set(block.context_lengths))),
    )
    multiplier = sum(
        request_multiplier(config.seed, identity, r, cv) for r in requests
    ) / len(requests)
    if rank_cv:
        multiplier *= request_multiplier(
            config.seed,
            ("rank", layer, stage, tuple(sorted(set(block.context_lengths)))),
            getattr(query, "physical_worker", None) or block.source_a_worker_id,
            rank_cv,
        )
    if tp_rank_cv:
        multiplier *= max(
            request_multiplier(
                config.seed,
                ("tp-rank", layer, stage, tuple(sorted(set(block.context_lengths)))),
                endpoint,
                tp_rank_cv,
            )
            for endpoint in participants
        )
    fixed = (
        min(
            estimate.service_duration,
            (estimate.details.get("fixed_latency_us") or 0) / 1e6,
        )
        if network
        else 0
    )
    return replace(
        estimate,
        service_duration=fixed + (estimate.service_duration - fixed) * multiplier,
        details={
            **estimate.details,
            "synthetic_jitter": {
                "seed": config.seed,
                **(
                    {"tp_rank_cv": tp_rank_cv, "tp_participants": len(participants)}
                    if tp_rank_cv
                    else {}
                ),
                "request_cv": cv,
                "rank_compute_cv": rank_cv,
                "collective": collective,
                "multiplier": multiplier,
                "base_service_s": estimate.service_duration,
                "model": "mean_one_request_lognormal_average",
            },
        },
    )
