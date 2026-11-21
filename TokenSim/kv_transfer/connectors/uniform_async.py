"""Queued uniform storage with native keys, GPU ownership and pluggable media."""

from dataclasses import replace, asdict

from TokenSim.errors import ConfigurationError
from TokenSim.kv_transfer.storage.online_config import OnlineStoragePolicy
from TokenSim.kv_transfer.storage.uniform_state import (
    UniformStoreState,
    UniformStoreView,
)
from .block_store.client import UniformStoreClient
from TokenSim.kv_transfer.io_observation import IOObservation
from .uniform_async_load import UniformLoadMixin
from .uniform_async_save import UniformSaveMixin


class UniformAsyncStore(UniformLoadMixin, UniformSaveMixin, UniformStoreClient):
    name = "UniformAsyncStore"
    async_offload = True
    supports_grouped_cache = False

    def __init__(self, config, cache_config=None):
        extra = dict(config.kv_connector_extra_config)
        self.policy = OnlineStoragePolicy.from_mapping(extra.pop("online_policy", {}))
        extra.setdefault("load_async", True)
        extra.setdefault("transfer_overlap", True)
        if not extra["load_async"] or not extra["transfer_overlap"]:
            raise ConfigurationError(
                "UniformAsyncStore requires asynchronous transfer overlap"
            )
        if cache_config is None or cache_config.hybrid_layout is not None:
            raise ConfigurationError(
                "UniformAsyncStore requires a uniform cache geometry"
            )
        if cache_config.parallel_config.pipeline_parallel_size != 1:
            raise ConfigurationError("UniformAsyncStore currently requires PP1")
        super().__init__(replace(config, kv_connector_extra_config=extra), cache_config)
        if self.block_bytes > self.policy.save_batch_bytes:
            raise ConfigurationError(
                "A physical KV object must fit the configured save batch"
            )
        if not self.policy.backend_includes_transport and (
            self.options.fixed_latency_us is None
            or self.options.bandwidth_bytes_per_s is None
        ):
            raise ConfigurationError(
                "Explicit transport requires a fixed latency and bandwidth"
            )
        self.loads = {}
        self.write_jobs = 0
        self.pending_writes = []
        self._saved_request_ids = set()
        self.policy_stats = dict(reuse_rejected_bytes=0)

    def bind_runtime(self, worker):
        super().bind_runtime(worker)
        if worker.mtp_num_speculative_tokens:
            raise ConfigurationError("Uniform queued storage requires ordinary decode")
        self.worker = worker
        self.env = worker.env
        self.scheduler = worker.scheduler
        self.manager = worker.scheduler.block_manager
        self.manager.transfer_references = {}
        if not hasattr(self.service, "uniform_queue_state"):
            self.service.uniform_queue_state = UniformStoreState(
                self.env, self.service, self.policy
            )
        self.state = self.service.uniform_queue_state
        if self.state.policy != self.policy:
            raise ConfigurationError(
                "Shared queued clients require matching online policies"
            )
        self.store = UniformStoreView(self.state)
        self.observation = IOObservation(worker)
        self.changed = self.env.event()

    def pin_gpu(self, blocks):
        for block in blocks:
            self.manager.gpu_allocator.acquire(block)
            key = block.block_number
            self.manager.transfer_references[key] = (
                self.manager.transfer_references.get(key, 0) + 1
            )

    def unpin_gpu(self, blocks):
        for block in blocks:
            self.manager.gpu_allocator.free(block)
            key = block.block_number
            remaining = self.manager.transfer_references[key] - 1
            if remaining:
                self.manager.transfer_references[key] = remaining
            else:
                self.manager.transfer_references.pop(key)

    def wire_latency(self, size):
        if self.policy.backend_includes_transport:
            return 0.0
        return (
            self.options.fixed_latency_us / 1e6
            + size / self.options.bandwidth_bytes_per_s
        )

    def record_transfer(self, kind, blocks, size, seconds, *, blocking):
        wait = seconds if blocking else 0.0
        self.stats.record_transfer(
            blocks=blocks,
            bytes_=size,
            latency=seconds,
            kind=kind,
            blocking_latency=wait,
        )
        self.mooncake_stats.record_transfer(
            bytes_=size, latency=seconds, kind=kind, blocking_latency=wait
        )

    def notify(self):
        event, self.changed = self.changed, self.env.event()
        if not event.triggered:
            event.succeed()
        from TokenSim.llm.llm_engine import Task, WorkerStatus

        if (
            self.scheduler.waiting
            or self.scheduler.running
            or self.scheduler.loaded_waiting
        ):
            if self.worker.status == WorkerStatus.WAITING:
                self.worker.status = WorkerStatus.RUNNING
                self.worker.send_task(self.worker, Task.STEP)

    def on_request_released(self, req):
        super().on_request_released(req)
        self._saved_request_ids.discard(req.id)

    def request_finished(self, req, block_ids):
        self.on_request_released(req)
        return False, None

    def describe(self):
        return dict(
            backend=self.name,
            storage=self.store.describe(),
            online_policy=asdict(self.policy),
            active_loads=len(self.loads),
            active_writes=self.write_jobs,
            policy_stats=self.policy_stats,
        )
