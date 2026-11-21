"""Pinned compressed KV handoff between independent full-model DP/EP domains."""

from TokenSim.errors import ConfigurationError
from .lifecycle import EndpointHandoff
from TokenSim.llm.llm_request import RequestStatus


class GroupedPDHandoff(EndpointHandoff):
    @staticmethod
    def reuse_destinations(destination, req, source_pages):
        plan = destination.plan_reuse(req)
        available = {gid: dict(pairs) for gid, pairs in plan.group_hits}
        local, missing = [], []
        for gid, pairs in source_pages:
            local_pairs, missing_pairs = [], []
            for index, source in pairs:
                resident = available.get(gid, {}).get(index)
                if (
                    resident is not None
                    and source.cached
                    and source.block_hash == resident.block_hash
                ):
                    local_pairs.append((index, resident))
                else:
                    missing_pairs.append((index, source))
            local.append((gid, tuple(local_pairs)))
            missing.append((gid, tuple(missing_pairs)))
        needed = sum(b.ref_count == 0 for _, pairs in local for _, b in pairs)
        needed += sum(len(pairs) for _, pairs in missing)
        return tuple(local), tuple(missing), needed

    def __init__(self, engine, specification):
        workers = engine.request_workers
        if engine.config.batching != "paged-attn" or any(
            w.role not in {"prefill", "decode"}
            or w.execution_domain is None
            or w.parallel_config.tensor_parallel_size != 1
            or w.parallel_config.pipeline_parallel_size != 1
            or not w.scheduler.block_manager.grouped_cache
            for w in workers
        ):
            raise ConfigurationError(
                "Grouped P/D transport requires independent TP1/PP1 dedicated serving domains with native grouped KV"
            )
        layouts = [
            (
                layout.groups,
                layout.logical_block_tokens,
                layout.physical_slot_bytes,
                layout.retention_interval,
            )
            for layout in [w.scheduler.block_manager.layout for w in workers]
        ]
        if any(layout != layouts[0] for layout in layouts):
            raise ConfigurationError(
                "Grouped P/D transport requires identical physical KV geometry"
            )
        super().__init__(engine, specification)
        for worker in workers:
            worker.scheduler.block_manager.pd_handoff = self

    def notify(self):
        from TokenSim.llm.engine.messages import Task, WorkerStatus

        self._signal_change()
        coordinator = self.engine.dp_coordinator
        if coordinator is not None:
            coordinator.notify()
        for worker in self.engine.request_workers:
            if coordinator is not None and worker.id in coordinator.worker_ids:
                continue
            if worker.status == WorkerStatus.WAITING and (
                worker.scheduler.waiting or worker.scheduler.loaded_waiting
            ):
                worker.status = WorkerStatus.RUNNING
                self.engine.send_task(worker, Task.STEP)

    def dispatch(self, sender, requests):
        manager = sender.scheduler.block_manager
        for req in requests:
            target = self._select_decode_target(req)
            tables = manager.tables[req.id]
            pages = tuple(
                (
                    spec.group_id,
                    tuple(
                        (index, block)
                        for index, block in tables[spec.group_id].items()
                        if index * spec.block_tokens < req.num_computed_tokens
                    ),
                )
                for spec in manager.groups
            )
            blocks = [block for _, pairs in pages for _, block in pairs]
            destination = target.scheduler.block_manager
            if (
                len(blocks) + destination.watermark_blocks
                >= destination.num_total_gpu_blocks
            ):
                raise ConfigurationError(
                    "Grouped handoff cannot fit a request at its receiver"
                )
            pinned = []
            try:
                for block in blocks:
                    manager.pin_transfer(block)
                    pinned.append(block)
                manager.release_request_blocks(req)
                sender.connector.on_request_released(req)
                self._start(
                    req, sender, target, pages=pages, blocks=pinned, destinations=[]
                )
            except Exception:
                for block in pinned:
                    manager.unpin_transfer(block)
                raise

    def _transfer(self, job):
        req, target = job["request"], job["target"]
        destination = target.scheduler.block_manager
        self._record(job, "receiver_wait", pages=len(job["blocks"]))
        while True:
            if req.cancel_requested:
                self._finish_cancel(job)
                return
            local, missing, needed = self.reuse_destinations(
                destination, req, job["pages"]
            )
            if (
                destination.get_num_free_blocks()
                >= needed + destination.watermark_blocks
            ):
                break
            yield self.changed
        if req.cancel_requested:
            self._finish_cancel(job)
            return
        targets = destination.allocate_handoff_destinations(req, missing, local)
        for row in targets:
            destination.pin_transfer(row[-1])
            job["destinations"].append(row)
        count = len(targets)
        bytes_ = count * destination.layout.physical_slot_bytes
        self._record(
            job,
            "transfer_start",
            bytes=bytes_,
            receiver_allocated_pages=count,
            receiver_reused_pages=sum(len(pairs) for _, pairs in local),
        )
        yield from self._copy(job["sender"].id, [(target.id, bytes_)])
        if req.cancel_requested:
            self._finish_cancel(job)
            return
        destination.finish_handoff(req, targets)
        self._unpin(job)
        self._deliver(job, bytes_)

    def _enqueue_decode(self, job):
        req = job["request"]
        req.status = RequestStatus.WAITING
        job["target"].scheduler.loaded_waiting.append(req)

    def _unpin(self, job):
        source = job["sender"].scheduler.block_manager
        for block in job["blocks"]:
            source.unpin_transfer(block)
        job["blocks"] = []
        destination = job["target"].scheduler.block_manager
        for _, _, _, block in job["destinations"]:
            destination.unpin_transfer(block)
        job["destinations"] = []

    def _release_destination(self, job):
        req = job["request"]
        scheduler = job["target"].scheduler
        destination = scheduler.block_manager
        if req.id in destination.tables:
            destination.release_request_blocks(req)
        scheduler.loaded_waiting = [
            r for r in scheduler.loaded_waiting if r.id != req.id
        ]
        return {}

    def report(self):
        return dict(
            network=self.fabric.report(),
            lifecycle=self.events,
            pending_requests=len(self.jobs),
            pinned_source_pages=sum(len(j["blocks"]) for j in self.jobs.values()),
            pinned_destination_pages=sum(
                len(j["destinations"]) for j in self.jobs.values()
            ),
            receiver_reserved_pages=0,
            payload_definition="Resident computed native grouped slots, including physical stride padding; no TP reshaping",
        )
