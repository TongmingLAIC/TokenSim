"""DMA ownership and asynchronous import/export for grouped KV slots."""

from TokenSim.errors import ConfigurationError
from TokenSim.llm.llm_request import RequestStatus

from .base import BaseKVConnector


class AsyncOffloadConnector(BaseKVConnector):
    supports_grouped_cache = True
    async_offload = True

    def __init__(self, config, cache_config=None):
        super().__init__(config, cache_config)
        if cache_config is None or cache_config.hybrid_layout is None:
            raise ConfigurationError("Grouped offload requires native grouped KV")
        self.loads = {}
        self.pending_writes = []
        self.write_jobs = 0
        self.profile_baseline = None

    def notify(self):
        if not self.changed.triggered:
            self.changed.succeed()
        self.changed = self.env.event()
        coordinator = getattr(self.worker.engine, "dp_coordinator", None)
        if coordinator is not None and self.worker.id in coordinator.worker_ids:
            coordinator.notify()
        elif (
            self.scheduler.running
            or self.scheduler.waiting
            or self.scheduler.loaded_waiting
        ):
            from TokenSim.llm.llm_engine import Task, WorkerStatus

            if self.worker.status == WorkerStatus.WAITING:
                self.worker.status = WorkerStatus.RUNNING
                self.worker.send_task(self.worker, Task.STEP)

    def probe(self, req, local_plan):
        if self.config.kv_role == "kv_producer":
            return local_plan.hit_tokens, ()
        return self.manager.plan_external_reuse(
            req, self.store, local_tokens=local_plan.hit_tokens
        )

    def begin_load(self, req, boundary, hits):
        pages = [p for _, pairs in hits for _, p in pairs]
        acquired = []
        try:
            for page in pages:
                self.store.acquire(page)
                acquired.append(page)
            targets = self.manager.allocate_external_destinations(req, hits)
        except Exception:
            for page in reversed(acquired):
                self.store.release(page)
            raise
        for _, _, _, block in targets:
            self.manager.pin_transfer(block)
        req.external_cached_context_tokens = boundary - req.gpu_cached_context_tokens
        req.gpu_cached_prefill_tokens = min(
            req.prefill_len, req.gpu_cached_context_tokens
        )
        req.external_cached_prefill_tokens = min(
            req.external_cached_context_tokens,
            max(0, req.prefill_len - req.gpu_cached_prefill_tokens),
        )
        req.cached_prefill_tokens = min(req.prefill_len, boundary)
        req.cached_prefill_blocks = req.cached_prefill_tokens // req.block_size
        req.effective_prefill_tokens = max(0, req.prefill_len - boundary)
        req.status = RequestStatus.WAITING_FOR_KV
        self.scheduler.loading[req.id] = req
        self.loads[req.id] = self.env.process(self._load(req, targets, boundary))

    def _load(self, req, targets, boundary):
        issued = self.env.now
        control = getattr(req, "control_prefetch", False)
        ready = False
        transferred = False
        try:
            bytes_ = self._transfer_bytes(page for _, _, page, _ in targets)
            yield self.env.process(self.transfer_load(targets))
            transferred = True
            if not req.cancel_requested and req.id in self.manager.tables:
                self.manager.finish_external_import(req, targets, boundary)
                if req.needs_recompute:
                    req.recompute_tokens = max(0, req.context_len - boundary)
                req.start_context_build(boundary)
                req.status = RequestStatus.WAITING
                if control:
                    req.control_prefetch_callback(self.scheduler, req)
                else:
                    self.scheduler.loaded_waiting.append(req)
                    ready = True
        finally:
            if not ready:
                self.manager.release_request_blocks(req)
                self.scheduler.loading.pop(req.id, None)
                if control:
                    req.release_logical_blocks()
            for _, _, page, block in targets:
                self.store.release(page)
                self.manager.unpin_transfer(block)
            self.loads.pop(req.id, None)
            if transferred:
                self._record_transfer(
                    blocks=len(targets),
                    bytes_=bytes_,
                    latency=self.env.now - issued,
                    kind="load",
                )
            self.notify()

    def transfer_load(self, targets):
        size = self._transfer_bytes(page for _, _, page, _ in targets)
        yield self.env.process(self.read_link.transfer(size))

    def _queue_write(self, group, index, block, req):
        if self.config.kv_role == "kv_consumer":
            return
        controller = getattr(self.scheduler, "cache_controller", None)
        if controller is not None and controller.skip_write(self.manager, block, req):
            return
        page = self.store.reserve_write(block.block_hash, group.group_bytes)
        if page is None:
            return
        self.manager.pin_transfer(block)
        self.pending_writes.append((page, block))

    def wait_for_save(self):
        if self.pending_writes:
            batch, self.pending_writes = self.pending_writes, []
            self.write_jobs += 1
            self.env.process(self._save(batch))
        return 0.0

    def _save(self, batch):
        issued = self.env.now
        published = 0
        try:
            bytes_ = self._transfer_bytes(page for page, _ in batch)
            yield self.env.process(self.write_link.transfer(bytes_))
            for page, _ in batch:
                self.store.finish_write(page)
                published += 1
        finally:
            for index, (page, block) in enumerate(batch):
                if index >= published:
                    self.store.abort_write(page)
                self.manager.unpin_transfer(block)
            self.write_jobs -= 1
            if published == len(batch):
                self._record_transfer(
                    blocks=len(batch),
                    bytes_=bytes_,
                    latency=self.env.now - issued,
                    kind="save",
                    blocking_latency=0.0,
                )
            self.notify()

    def _transfer_bytes(self, pages):
        return sum(self.manager.layout.physical_slot_bytes for _ in pages)

    def _record_transfer(self, **values):
        self.stats.record_transfer(**values)

    def request_finished(self, req, block_ids):
        self.scheduler.loading.pop(req.id, None)
        return False, None

    def capture_profile_baseline(self):
        self.profile_baseline = dict(
            at=self.env.now,
            read=self.read_link.describe(),
            write=self.write_link.describe(),
        )

    def describe(self):
        return {
            "backend": self.name,
            "profile_baseline": self.profile_baseline,
            "runtime_budget_per_rank_bytes": self.budget_bytes,
            "storage": self.store.describe(),
            "read": self.read_link.describe(),
            "write": self.write_link.describe(),
            "loading_requests": len(self.loads),
            "write_jobs": self.write_jobs,
        }
