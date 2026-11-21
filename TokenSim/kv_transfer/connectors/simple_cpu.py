"""Native SimpleCPU: confirmed eager stores and request-local asynchronous loads.

Source contract: vllm/v1/simple_kv_offload at 30f5c01d4ebb. Each rank has
its own packed CPU pool and full-duplex PCIe queues. DMA references outlive
request references; cancelled loads never publish their destination pages.
"""

from TokenSim.errors import ConfigurationError
from TokenSim.kv_transfer.storage.native import NativePageStore, QueuedKVLink
from TokenSim.config.validation import integer, number, boolean

from .grouped_offload import AsyncOffloadConnector


class SimpleCPUOffloadConnector(AsyncOffloadConnector):
    name = "SimpleCPUOffloadConnector"

    def __init__(self, config, cache_config=None):
        super().__init__(config, cache_config)
        extra = dict(config.kv_connector_extra_config)
        allowed = {
            "cpu_bytes_to_use_per_rank",
            "cpu_bytes_to_use",
            "host_available_bytes",
            "lazy_offload",
            "transfer_utilization",
        }
        unknown = set(extra) - allowed
        if unknown:
            raise ConfigurationError(f"unknown SimpleCPU options: {sorted(unknown)}")
        if boolean(extra.get("lazy_offload", False), "lazy_offload"):
            raise ConfigurationError("SimpleCPU supports eager offload")
        if "cpu_bytes_to_use_per_rank" in extra and "cpu_bytes_to_use" in extra:
            raise ConfigurationError(
                "choose either total or per-rank CPU byte capacity"
            )
        for name in (
            "cpu_bytes_to_use_per_rank",
            "cpu_bytes_to_use",
            "host_available_bytes",
        ):
            if name in extra:
                integer(extra[name], name, minimum=1)
        number(
            extra.get("transfer_utilization", 1.0),
            "transfer_utilization",
            positive=True,
            maximum=1,
        )
        self.extra = extra

    def bind_runtime(self, worker):
        self.worker, self.env = worker, worker.env
        self.manager = worker.scheduler.block_manager
        self.scheduler = worker.scheduler
        world_size = len(worker.engine.workers)
        capacity = self.extra.get("cpu_bytes_to_use_per_rank")
        if capacity is None:
            capacity = self.extra.get("cpu_bytes_to_use", 0) // world_size
        host_bytes = self.extra.get("host_available_bytes")
        if host_bytes is not None and capacity * world_size > host_bytes:
            raise ConfigurationError("CPU runtime budget exceeds available host DRAM")
        stride = self.manager.layout.physical_slot_bytes
        # Both source pools reserve block zero as the null page.
        slots = int(capacity) // stride
        if slots < 2:
            raise ConfigurationError("SimpleCPU needs a null slot and usable CPU slots")
        self.budget_bytes = int(capacity)
        self.store = NativePageStore(
            (slots - 1) * stride, allocation_unit=stride, scope=f"worker:{worker.id}"
        )
        link = worker.hardware_context.link(worker.device.host_link)
        utilization = self.extra.get("transfer_utilization", 1.0)
        args = dict(utilization=utilization, latency_us=link.latency_us)
        self.read_link = QueuedKVLink(
            self.env, link.bandwidth_per_direction_bytes_per_s, **args
        )
        self.write_link = QueuedKVLink(
            self.env, link.bandwidth_per_direction_bytes_per_s, **args
        )
        self.manager.publish_callbacks.append(self._queue_write)
        self.changed = self.env.event()
