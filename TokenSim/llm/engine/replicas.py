"""Build ordinary serving replicas and validate their request limits."""

from collections import defaultdict

from TokenSim.block.block_manager import BlockManager
from TokenSim.block.hybrid.manager import HybridKVBlockManager
from TokenSim.config.serving_domains import resolve_worker_config
from TokenSim.errors import ConfigurationError
from TokenSim.placement.policies import replica_key


def validate_layouts(cluster, workers, model, parallel, config, kv_transfer):
    if parallel.enable_expert_parallel and not model.is_moe:
        raise ConfigurationError("enable_expert_parallel requires a MoE model config")
    for worker in workers:
        effective = worker.parallel_config or parallel
        settings = resolve_worker_config(config, worker.simulation_overrides)
        if effective.enable_expert_parallel and not model.is_moe:
            raise ConfigurationError(
                f"worker group {worker.name} enables expert parallelism on a dense model"
            )
        if settings.prefill_schedule_interval > 1 and (
            not settings.dp_lockstep
            or effective.data_parallel_size <= 1
            or settings.batching != "paged-attn"
        ):
            raise ConfigurationError(
                f"worker group {worker.name}: prefill_schedule_interval requires "
                "paged-attn and multiple synchronized DP ranks"
            )
        if effective.pipeline_parallel_size > 1:
            if cluster.af.enabled:
                raise ConfigurationError(
                    "The ordinary worker layout in an AF cluster requires PP1; "
                    "configure AF pipeline stages on their AF execution domains"
                )
            connector = worker.kv_transfer or kv_transfer
            if (
                settings.dp_lockstep
                or model.components is not None
                or worker.role not in {None, "hybrid"}
                or set(worker.capabilities) & {"prefill", "decode_full"}
                != {"prefill", "decode_full"}
                or connector.kv_connector != "NoopConnector"
            ):
                raise ConfigurationError(
                    "Ordinary PP requires hybrid workers, uniform KV without offload, "
                    "and independent DP steps"
                )


def bind_ordinary_replicas(workers):
    """Keep a scheduler's logical capacity within the narrowest rank's pool."""
    replicas = defaultdict(list)
    for worker in workers:
        if not worker.af_runtime_bindings and worker.parallel_config.world_size > 1:
            replicas[replica_key(worker)].append(worker)
    for ranks in replicas.values():
        first = next(
            worker for worker in ranks if worker.tp_rank == worker.pp_rank == 0
        )
        first.forward_stages = tuple(
            sorted(
                (worker for worker in ranks if worker.tp_rank == 0),
                key=lambda w: w.pp_rank,
            )
        )
        manager = getattr(first.scheduler, "block_manager", None)
        if manager is None:
            continue
        limits = [worker.scheduler.block_manager for worker in ranks]
        gpu = min(limit.num_total_gpu_blocks for limit in limits)
        cpu = min(limit.num_total_cpu_blocks for limit in limits)
        if gpu == manager.num_total_gpu_blocks and cpu == manager.num_total_cpu_blocks:
            continue
        if manager.grouped_cache:
            first.scheduler.block_manager = HybridKVBlockManager(
                manager.layout, gpu, first.model
            )
        else:
            first.scheduler.block_manager = BlockManager(
                first.block_size, gpu, cpu, first.model
            )


def validate_request_capacity(engine, requests):
    validate_pool = (
        ("prefill", engine.prefill_workers.workers),
        ("decode", engine.decode_workers.workers),
    )
    for request in requests:
        request.validate_prompt_length(engine.model_spec.max_position_embeddings)
    for role, workers in validate_pool:
        for worker in workers:
            manager = getattr(worker.scheduler, "block_manager", None)
            if manager is None:
                continue
            available = available_gpu_slots(manager) - manager.watermark_blocks
            layout = getattr(worker.cache_config, "hybrid_layout", None)
            for request in requests:
                # A dedicated prefill worker never computes the sampled token.
                context = request.prefill_len
                if role == "decode":
                    context = min(
                        engine.model_spec.max_position_embeddings,
                        context + request.decode_len - 1,
                    )
                if layout is not None:
                    context = min(
                        layout.max_model_len, context + 2 * layout.speculative_tokens
                    )
                    needed = layout.max_request_slots(context)
                else:
                    needed = -(-context // worker.block_size)
                if needed > available:
                    raise ConfigurationError(
                        f"request {request.id} cannot fit on eligible {role} worker "
                        f"{worker.id}: required_blocks={needed}, available_blocks={available}, "
                        f"prefill_len={request.prefill_len}, decode_len={request.decode_len}, "
                        f"block_size={worker.block_size}"
                    )


def available_gpu_slots(manager):
    """Count data slots, excluding the null page used by grouped layouts."""
    return manager.num_total_gpu_blocks - int(manager.grouped_cache)
