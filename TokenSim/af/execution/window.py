"""Limit within-iteration overlap while preserving block geometry and A owners."""

from dataclasses import replace

from TokenSim.af.trace.metrics import AFTraceEvent


def run_admitted_block(executor, block, slots):
    if slots is None:
        yield from executor._run_block(block)
        return
    queued_at = executor.env.now
    with slots.request() as reservation:
        yield reservation
        admitted_at = executor.env.now
        trace = executor.trace
        indices = trace.iteration_event_indices
        offset = len(indices.get(block.iteration_id, ())) if trace.enabled else 0
        node_id = f"i{block.iteration_id}:b{block.block_id}:admission"
        if trace.enabled:
            trace.record(
                AFTraceEvent(
                    iteration_id=block.iteration_id,
                    block_id=block.block_id,
                    layer_id=-1,
                    phase="decode",
                    event_type="synchronize",
                    operation="microbatch_admission",
                    resource_id=(
                        f"pipeline_window:{block.placement.a_domain}:{block.placement.a_instance}"
                        if executor.cluster_config.af.max_inflight_scope == "owner"
                        else f"pipeline_window:i{block.iteration_id}"
                    ),
                    node_id=node_id,
                    ready_at=queued_at,
                    start_at=admitted_at,
                    end_at=admitted_at,
                    queue_duration=admitted_at - queued_at,
                    service_duration=0.0,
                    request_count=block.request_count,
                    token_count=block.token_count,
                    source_id="runtime:microbatch-admission-window",
                    dataset_version="runtime",
                    match_type="analytical",
                    source_domain=block.placement.a_domain,
                    dp_rank=block.placement.a_dp_rank,
                    details={
                        "max_inflight_blocks": slots.capacity,
                        "scope": (
                            "per_attention_owner"
                            if executor.cluster_config.af.max_inflight_scope == "owner"
                            else "per_attention_owner_iteration"
                        ),
                        **(
                            {"admitted_owner_cohorts": slots.count}
                            if executor.cluster_config.af.max_inflight_scope == "owner"
                            else {}
                        ),
                        "wait_reason": "microbatch_window",
                    },
                )
            )
        yield from executor._run_block(block)
        if trace.enabled:
            for index in indices.get(block.iteration_id, ())[offset:]:
                event = trace.events[index]
                if (
                    event.block_id == block.block_id
                    and event.node_id != node_id
                    and event.event_type != "host_prepare"
                    and (
                        not event.predecessor_ids
                        or all(
                            p.startswith(f"i{block.iteration_id}:prepare:")
                            for p in event.predecessor_ids
                        )
                    )
                ):
                    trace.events[index] = replace(
                        event,
                        predecessor_ids=tuple(
                            dict.fromkeys((*event.predecessor_ids, node_id))
                        ),
                    )
