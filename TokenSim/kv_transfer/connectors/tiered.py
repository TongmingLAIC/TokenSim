"""Experimental grouped L2/L3 policies with completion-visible immutable copies."""

import simpy

from TokenSim.errors import ConfigurationError
from TokenSim.kv_transfer.storage.native import QueuedKVLink
from TokenSim.kv_transfer.storage.tiered_config import TieredConfig
from TokenSim.kv_transfer.storage.tiered_service import TierPage, shared_service
from TokenSim.kv_transfer.storage.tiered_store import TieredStore
from .grouped_offload import AsyncOffloadConnector


class TieredKVConnector(AsyncOffloadConnector):
    name = "TieredKVConnector"

    def __init__(self, config, cache_config=None):
        super().__init__(config, cache_config)
        self.options = TieredConfig.from_mapping(config.kv_connector_extra_config)
        self.flush_batch = []
        self.flush_scheduled = False
        self.tail = None
        self.refused_l3_writes = 0
        self.pending_eviction_bytes = 0

    def bind_runtime(self, worker):
        self.worker, self.env = worker, worker.env
        self.manager, self.scheduler = worker.scheduler.block_manager, worker.scheduler
        self.service = shared_service(worker, self.options)
        self.service.clients.append(self)
        groups = tuple(
            (g.group_id, g.group_bytes, g.block_tokens)
            for g in self.manager.layout.groups
        )
        namespace = self.service.namespace(
            (
                worker.model_spec.cache_signature(),
                worker.parallel_config.tensor_parallel_size,
                worker.parallel_config.pipeline_parallel_size,
                worker.tp_rank,
                worker.pp_rank,
                groups,
            )
        )
        self.store = TieredStore(self, namespace)
        stride = self.manager.layout.physical_slot_bytes
        if self.options.staging_bytes_per_rank < stride:
            raise ConfigurationError("Staging must hold a physical GPU slot")
        host = worker.hardware_context.link(worker.device.host_link)
        link_args = dict(
            utilization=self.options.transfer_utilization, latency_us=host.latency_us
        )
        self.read_link = QueuedKVLink(
            self.env, host.bandwidth_per_direction_bytes_per_s, **link_args
        )
        self.write_link = QueuedKVLink(
            self.env, host.bandwidth_per_direction_bytes_per_s, **link_args
        )
        self.staging_lock = simpy.Resource(self.env, capacity=1)
        self.manager.publish_callbacks.append(self._queue_write)
        self.changed = self.env.event()
        self.budget_bytes = self.options.l2_bytes_per_rank

    def chunks(self, rows):
        count = max(
            1,
            self.options.staging_bytes_per_rank
            // self.manager.layout.physical_slot_bytes,
        )
        for start in range(0, len(rows), count):
            yield rows[start : start + count]

    def _queue_write(self, group, index, block, req):
        if self.config.kv_role == "kv_consumer":
            return
        controller = getattr(self.scheduler, "cache_controller", None)
        if controller is not None and controller.skip_write(self.manager, block, req):
            return
        if self.options.write_policy == "write_around":
            key = self.store.remote_key(block.block_hash)
            raw = self.service.store.reserve_write(key, group.group_bytes)
            if raw is None:
                self.refused_l3_writes += int(key not in self.service.store.pages)
                return
            page = TierPage(block.block_hash, raw, self.service.store, 3)
        else:
            page = self.store.reserve_write(block.block_hash, group.group_bytes)
            if page is None:
                if not getattr(req, "cache_one_shot", False):
                    self.store.one_shot.discard(block.block_hash)
                return
            if getattr(req, "cache_one_shot", False):
                self.store.one_shot.add(block.block_hash)
        self.manager.pin_transfer(block)
        self.pending_writes.append((page, block))

    def _save(self, batch):
        if self.options.write_policy != "write_around":
            yield from super()._save(batch)
            if self.options.write_policy == "write_through":
                for page, _ in batch:
                    self.enqueue_flush(page.page)
            else:
                self.store.pressure_flush()
            return
        issued = self.env.now
        unpinned = published = 0
        try:
            for chunk in self.chunks(batch):
                with self.staging_lock.request() as lock:
                    yield lock
                    yield self.env.process(
                        self.write_link.transfer(
                            self._transfer_bytes(p for p, _ in chunk)
                        )
                    )
                    for _, block in chunk:
                        self.manager.unpin_transfer(block)
                        unpinned += 1
                    self.notify()
                    yield self.env.process(
                        self.service.transfer("write", sum(p.bytes for p, _ in chunk))
                    )
                    for page, _ in chunk:
                        self.store.finish_write(page)
                        published += 1
            self._record_transfer(
                kind="save",
                blocks=len(batch),
                bytes_=self._transfer_bytes(p for p, _ in batch),
                latency=self.env.now - issued,
                blocking_latency=0,
            )
        finally:
            for page, _ in batch[published:]:
                self.store.abort_write(page)
            for _, block in batch[unpinned:]:
                self.manager.unpin_transfer(block)
            self.write_jobs -= 1
            self.notify()

    def enqueue_flush(self, page, *, evict=False):
        if not page.ready:
            return False
        key = self.store.remote_key(page.key)
        if key in self.service.store.pages:
            return False
        remote = self.service.store.reserve_write(
            key, self.store.payload_bytes[page.key]
        )
        if remote is None:
            self.refused_l3_writes += 1
            return False
        # A writeback lease is not a new read/reuse of its LRU victim.
        self.store.local.acquire(page, touch=False)
        if evict:
            self.pending_eviction_bytes += page.bytes
        self.flush_batch.append((page, remote, evict, page.last_touch))
        if not self.flush_scheduled:
            self.flush_scheduled = True
            self.write_jobs += 1
            self.env.process(self._flush())
        return True

    def _flush(self):
        yield self.env.timeout(0)
        batch, self.flush_batch = self.flush_batch, []
        self.flush_scheduled = False
        published = 0
        try:
            for chunk in self.chunks(batch):
                yield self.env.process(
                    self.service.transfer("write", sum(row[1].bytes for row in chunk))
                )
                for local, remote, evict, touched in chunk:
                    self.service.store.finish_write(remote)
                    self.store.local.release(local)
                    published += 1
                    if evict:
                        self.pending_eviction_bytes -= local.bytes
                        if local.last_touch == touched and self.store.has_remote(
                            local.key
                        ):
                            self.store.drop_local(local)
        finally:
            for local, remote, evict, _ in batch[published:]:
                self.service.store.abort_write(remote)
                self.store.local.release(local)
                if evict:
                    self.pending_eviction_bytes -= local.bytes
            self.write_jobs -= 1
            self.notify()

    def transfer_load(self, targets):
        local = [row for row in targets if row[2].tier == 2]
        remote = [row for row in targets if row[2].tier == 3]
        if local:
            yield self.env.process(
                self.read_link.transfer(self._transfer_bytes(row[2] for row in local))
            )
        for chunk in self.chunks(remote):
            with self.staging_lock.request() as lock:
                yield lock
                yield self.env.process(
                    self.service.transfer("read", sum(row[2].bytes for row in chunk))
                )
                yield self.env.process(
                    self.read_link.transfer(
                        self._transfer_bytes(row[2] for row in chunk)
                    )
                )
                for row in chunk:
                    self.store.promote(row[2])

    def capture_profile_baseline(self):
        super().capture_profile_baseline()
        self.profile_baseline.update(
            shared_l3=self.service.describe(), storage=self.store.describe()
        )

    def capture_end_of_workload(self):
        before = self.describe()
        self.tail = dict(
            started_at=self.env.now, before=before, coverage=self.service.coverage()
        )

    def post_workload_drain(self):
        if self.tail is None:
            self.capture_end_of_workload()
        if self.options.drain_tail:
            for key, page in list(self.store.local.pages.items()):
                if (
                    self.options.write_policy == "selective_write_back"
                    and key in self.store.one_shot
                ):
                    continue
                self.enqueue_flush(page)
            while self.write_jobs:
                yield self.changed
        self.tail.update(
            completed_at=self.env.now,
            remaining_l3_missing_bytes=self.store.missing_bytes(),
            completed_l3_write_bytes=self.service.completed["write"],
        )

    def describe(self):
        return dict(
            backend=self.name,
            policy=self.options.write_policy,
            storage=self.store.describe(),
            shared_l3=self.service.describe(),
            staging_bytes_per_rank=self.options.staging_bytes_per_rank,
            staging_scope="reserved within L2; one serialized remote staging stream per rank",
            read=self.read_link.describe(),
            write=self.write_link.describe(),
            loading_requests=len(self.loads),
            write_jobs=self.write_jobs,
            refused_l3_writes=self.refused_l3_writes,
            pending_eviction_bytes=self.pending_eviction_bytes,
            watermark_eviction=self.options.watermark_eviction,
            tail=self.tail,
            profile_baseline=self.profile_baseline,
            acknowledgment="L2 visible after GPU DMA; L3 visible after network and backend completion",
            packing="L2 and PCIe use physical GPU stride; L3 stores group payload bytes",
        )
