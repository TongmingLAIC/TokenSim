"""Independent complete-model parallel groups for dedicated prefill/decode pools."""

from TokenSim.config.parallel_config import ParallelRankInfo
from TokenSim.errors import ConfigurationError
from dataclasses import replace

OVERRIDE_FIELDS = {
    "gpu_memory_utilization",
    "runtime_memory_profile",
    "max_num_seqs",
    "max_num_batched_tokens",
    "long_prefill_token_threshold",
    "prefill_schedule_interval",
    "adaptive_long_prefill_threshold",
    "mtp_num_speculative_tokens",
    "mtp_acceptance_length",
    "mtp_acceptance",
    "mtp_seed",
}


def resolve_worker_config(config, overrides):
    unknown = set(overrides) - OVERRIDE_FIELDS
    if unknown:
        raise ConfigurationError(
            f"Unsupported worker simulation overrides: {sorted(unknown)}"
        )
    return replace(config, **overrides) if overrides else config


def assign_domain_ranks(cluster, workers, global_parallel):
    if cluster.af.enabled:
        raise ConfigurationError(
            "Complete-model group parallelism cannot be mixed with AF domains"
        )
    if global_parallel.world_size != 1:
        raise ConfigurationError(
            "Explicit serving groups require the global parallel layout to be omitted"
        )
    if cluster.kv_cache_capacity_tokens_total is not None:
        raise ConfigurationError(
            "Independent serving groups require a per-replica KV quota or native capacity"
        )
    for group in cluster.worker_groups:
        if group.parallel_config is None or group.role not in {"prefill", "decode"}:
            raise ConfigurationError(
                "Every independent serving group needs dedicated prefill/decode role and parallel_config"
            )
        if group.num_workers != group.parallel_config.world_size:
            raise ConfigurationError(
                f"Serving group {group.name} needs one worker per parallel rank: "
                f"got {group.num_workers}, expected {group.parallel_config.world_size}"
            )
        for rank, worker in enumerate(w for w in workers if w.name == group.name):
            worker.rank_info = ParallelRankInfo.from_global_rank(
                rank, group.parallel_config
            )
    return workers
