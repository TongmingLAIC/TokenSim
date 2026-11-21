"""Environment-shared L3 capacity, aggregate network and backend arbitration."""

from dataclasses import dataclass

from TokenSim.errors import ConfigurationError
from TokenSim.kv_transfer.native_storage import NativePageStore, QueuedKVLink
from TokenSim.kv_transfer.online_storage_config import OnlineStoragePolicy
from TokenSim.kv_transfer.storage_queue import StorageQueue


@dataclass(frozen=True)
class TierPage:
    key: tuple
    page: object
    backing: object
    tier: int

    @property
    def bytes(self):
        return self.page.bytes


class TieredService:
    def __init__(self, env, options, node_bandwidth):
        self.env, self.options = env, options
        self.store = NativePageStore(options.l3_bytes, scope="tiered:shared-l3")
        self.read = QueuedKVLink(
            env,
            min(node_bandwidth, options.network_bytes_per_second),
            utilization=options.transfer_utilization,
            latency_us=options.network_latency_us,
        )
        self.write = QueuedKVLink(
            env,
            min(node_bandwidth, options.network_bytes_per_second),
            utilization=options.transfer_utilization,
            latency_us=options.network_latency_us,
        )
        self.queue = StorageQueue(
            env, OnlineStoragePolicy(read_priority=options.read_priority)
        )
        self.namespaces = {}
        self.completed = {"read": 0, "write": 0}
        self.pending = {"read": 0, "write": 0}
        self.clients = []

    def namespace(self, descriptor):
        return self.namespaces.setdefault(descriptor, object())

    def transfer(self, kind, size):
        self.pending[kind] += size
        try:
            if kind == "write":
                yield self.env.process(self.write.transfer(size))
            duration = self.options.backend_latency_us / 1e6 + size / (
                self.options.backend_bytes_per_second
                * self.options.transfer_utilization
            )
            # The backend is a shared half-duplex service; network directions
            # have separate queues. No measured SSD/DRAM accuracy is implied.
            if kind == "write":
                self.queue.backlog_bytes += size
                self.queue.peak_backlog_bytes = max(
                    self.queue.peak_backlog_bytes, self.queue.backlog_bytes
                )
            yield self.env.process(self.queue.execute(kind, size, lambda: duration))
            if kind == "read":
                yield self.env.process(self.read.transfer(size))
            self.completed[kind] += size
        finally:
            self.pending[kind] -= size

    def coverage(self):
        """Count distinct ready content objects; this does not imply full-prefix validity."""
        l1, l2 = {}, {}
        replicated_l2 = 0
        for client in self.clients:
            sizes = {g.group_id: g.group_bytes for g in client.manager.layout.groups}
            for block in getattr(client.manager, "blocks", ()):
                key = block.block_hash
                if key is not None:
                    l1[client.store.remote_key(key)] = sizes[key[0]]
            for key, page in client.store.local.pages.items():
                if page.ready:
                    size = client.store.payload_bytes.get(key, page.bytes)
                    l2[client.store.remote_key(key)] = size
                    replicated_l2 += size
        l3 = {key: page.bytes for key, page in self.store.pages.items() if page.ready}
        union = dict(l1)
        union.update(l2)
        union.update(l3)
        return dict(
            l1_unique_payload_bytes=sum(l1.values()),
            l2_unique_payload_bytes=sum(l2.values()),
            l2_replicated_payload_bytes=replicated_l2,
            l3_unique_payload_bytes=sum(l3.values()),
            l2_l3_overlap_payload_bytes=sum(l2[k] for k in l2.keys() & l3.keys()),
            all_tier_unique_payload_bytes=sum(union.values()),
            all_tier_unique_objects=len(union),
            scope="complete content objects across physical shard namespaces; not reusable context coverage",
        )

    def describe(self):
        return dict(
            storage=self.store.describe(),
            network_read=self.read.describe(),
            network_write=self.write.describe(),
            backend_queue=self.queue.describe(),
            completed_bytes=dict(self.completed),
            pending_bytes=dict(self.pending),
            backend=self.options.backend,
            backend_bytes_per_second=self.options.backend_bytes_per_second,
            backend_latency_us=self.options.backend_latency_us,
            service_scope="one environment-shared remote service; full-duplex node network, serial backend",
        )


def shared_service(worker, options):
    # Engines in one environment share finite capacity and aggregate bandwidth.
    name = "_tokensim_tiered_l3"
    service = getattr(worker.env, name, None)
    node = worker.hardware_context.link(worker.nettype)
    descriptor = (
        options.l3_bytes,
        options.network_bytes_per_second,
        options.network_latency_us,
        options.backend,
        options.backend_bytes_per_second,
        options.backend_latency_us,
        options.transfer_utilization,
        options.read_priority,
    )
    if service is None:
        service = TieredService(
            worker.env, options, node.bandwidth_per_direction_bytes_per_s
        )
        service.descriptor = descriptor
        setattr(worker.env, name, service)
    elif service.descriptor != descriptor:
        raise ConfigurationError(
            "Shared tiered L3 clients must agree on service resources"
        )
    return service
