"""Grouped AF KV with identical replicated latent shards on every A TP rank."""

from TokenSim.block.hybrid.manager import HybridKVBlockManager
from TokenSim.af.runtime.topology import AFCacheConfig


class CompressedReplicaKV(HybridKVBlockManager):
    def __init__(self, plan, replica):
        super().__init__(
            plan.compressed_kv_layout, replica.logical_gpu_blocks, plan.model
        )
        self.replica = replica
        self.owner_instances = {replica.domain: replica.instance}

    def rank_status(self, rank):
        free, used, _ = self.get_gpu_status()
        return dict(
            gpu_capacity=dict(self.replica.local_gpu_blocks)[rank],
            gpu_used=used,
            cpu_capacity=0,
            cpu_used=0,
        )


def cache_configuration(plan, replica):
    cache = AFCacheConfig.from_plan(plan, replica)
    cache.hybrid_layout = plan.compressed_kv_layout
    cache.physical_slot_bytes = plan.compressed_kv_layout.physical_slot_bytes
    cache.memory_budget = None
    return cache
