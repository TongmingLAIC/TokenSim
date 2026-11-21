"""Blocking P/D timing for deployments without a shared endpoint fabric."""

from dataclasses import replace
from math import ceil

from TokenSim.errors import ConfigurationError
from TokenSim.kv_transfer.connectors.base import flatten_connectors
from TokenSim.kv_transfer.connectors.metadata import KVConnectorMetadata
from TokenSim.kv_transfer.connectors.mooncake import MooncakeConnector
from TokenSim.kv_transfer.connectors.multi import MultiConnector
from TokenSim.kv_transfer.connectors.noop import NoopConnector
from TokenSim.kv_transfer.connectors.p2p import P2PConnector
from TokenSim.llm.llm_request import RequestStatus

from .lifecycle import assign_decode_coordinate, select_decode_target


def validate_legacy_handoff(engine):
    """Require explicit endpoint transport for native grouped KV copies."""
    for source in engine.prefill_workers.workers:
        if source.role != "prefill":
            continue
        for target in engine.decode_workers.workers:
            managers = (
                getattr(worker.scheduler, "block_manager", None)
                for worker in (source, target)
            )
            if any(getattr(manager, "grouped_cache", False) for manager in managers):
                raise ConfigurationError("Grouped P/D requires pd_endpoint_limits")


def _requires_shard_reshaping(source, target):
    return any(
        worker.parallel_config.tensor_parallel_size != 1
        or worker.parallel_config.pipeline_parallel_size != 1
        for worker in (source, target)
    ) and any(worker.execution_domain is not None for worker in (source, target))


def _direct_sender(connector):
    for leaf in flatten_connectors(connector):
        if isinstance(leaf, (MooncakeConnector, P2PConnector)):
            return leaf
    return None


def _receiver_path(connector, transport):
    if isinstance(connector, MultiConnector):
        for index, child in enumerate(connector.children):
            path = _receiver_path(child, transport)
            if path is not None:
                return (index, *path)
        return None
    if isinstance(connector, (type(transport), NoopConnector)):
        return ()
    return None


def _arrive_after_copy(engine, worker, requests, plan):
    from TokenSim.llm.engine.messages import Task

    trace = getattr(engine, "trace_session", None)
    if trace is not None and plan.latency:
        trace._record_load(worker, (plan,), plan.latency)
    yield engine.env.timeout(plan.latency)
    worker.connector.stats.record_transfer(
        blocks=plan.blocks, bytes_=plan.bytes, latency=plan.latency, kind="load"
    )
    ready = [request for request in requests if not request.cancel_requested]
    if ready:
        for req in ready:
            assign_decode_coordinate(engine, req, worker)
        engine.send_task(worker, Task.ADD, ready)


def dispatch_legacy_handoff(engine, sender, requests):
    from TokenSim.llm.engine.messages import Task

    num_per_worker = ceil(len(requests) / len(engine.decode_workers))
    for i in range(0, len(requests), num_per_worker):
        batch = requests[i : i + num_per_worker]
        target = select_decode_target(engine, batch)
        if _requires_shard_reshaping(sender, target):
            raise ConfigurationError(
                "Independent legacy P/D requires TP1/PP1 on both sides"
            )
        manager = getattr(sender.scheduler, "block_manager", None)
        block_counts = (
            manager.get_request_block_counts(batch)
            if manager is not None
            else {
                req.id: ceil(req.num_computed_tokens / sender.block_size)
                for req in batch
            }
        )
        source = _direct_sender(sender.connector)
        transport = source or P2PConnector(
            sender.kv_transfer_config, sender.cache_config
        )
        latency = (
            sender.estimate_transfer_latency(target.id, sum(block_counts.values()))
            if isinstance(transport, P2PConnector)
            else None
        )
        plan = transport.build_transfer_plan(
            requests=batch,
            source_worker_id=sender.id,
            target_worker_id=target.id,
            kind="local" if sender is target else "load",
            latency=latency,
            request_block_counts=block_counts,
        )
        if source is None:
            sender.connector.stats.record_transfer(
                blocks=plan.blocks, bytes_=plan.bytes, latency=plan.latency, kind="save"
            )
        else:
            source.bind_connector_metadata(
                KVConnectorMetadata(connector_name=source.name, saves=[plan])
            )
            source.wait_for_save()
        for req in batch:
            if manager is not None:
                manager.commit_finished_cache(req)
                manager.release_request_blocks(req)
            sender.connector.on_request_released(req)
            req.status = RequestStatus.WAITING_FOR_KV

        path = _receiver_path(target.connector, transport)
        if target is sender or path is not None:
            if path:
                plan = replace(plan, extra={**plan.extra, "connector_path": path})
            metadata = KVConnectorMetadata(
                connector_name=transport.name,
                loads=[] if sender is target else [plan],
            )
            for req in batch:
                assign_decode_coordinate(engine, req, target)
            engine.send_task(target, Task.ADD, batch, metadata)
        else:
            # The receiver's store only handles prefix objects. Deliver the
            # request on the P/D timeline without feeding it a storage lookup.
            for req in batch:
                req.owner_worker_ids.discard(sender.id)
            engine.env.process(_arrive_after_copy(engine, target, batch, plan))
