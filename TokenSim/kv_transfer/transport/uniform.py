"""Pinned P/D handoff with receiver reservations and shared endpoint transfers."""

import math

from TokenSim.errors import ConfigurationError
from .lifecycle import EndpointHandoff


class PDHandoff(EndpointHandoff):
    def __init__(self, engine, specification):
        if (
            engine.parallel_config.tensor_parallel_size != 1
            or engine.parallel_config.pipeline_parallel_size != 1
        ):
            raise ConfigurationError(
                "Endpoint P/D transport requires TP1 independent ordinary workers"
            )
        if engine.config.batching != "paged-attn":
            raise ConfigurationError("Endpoint P/D transport requires paged-attn")
        workers = engine.request_workers
        if any(
            not w.af_runtime_bindings
            and (
                w.parallel_config.tensor_parallel_size != 1
                or w.parallel_config.pipeline_parallel_size != 1
            )
            for w in workers
        ):
            raise ConfigurationError(
                "Endpoint P/D transport requires TP1/PP1 ordinary workers"
            )
        if any(w.role not in {"prefill", "decode"} for w in workers):
            raise ConfigurationError(
                "Endpoint P/D transport requires dedicated prefill/decode roles"
            )
        if any(
            w.scheduler.block_manager.grouped_cache or w.connector.async_offload
            for w in workers
        ):
            raise ConfigurationError("Endpoint P/D transport requires uniform GPU KV")
        super().__init__(engine, specification)
        for worker in workers:
            worker.scheduler.block_manager.transfer_reservations.clear()
            worker.scheduler.block_manager.pd_handoff = self

    def notify(self):
        from TokenSim.llm.engine.messages import Task, WorkerStatus

        self._signal_change()
        for worker_id in list(self.blocked_workers):
            worker = self.engine.workers[worker_id]
            self.blocked_workers.discard(worker_id)
            if worker.status == WorkerStatus.WAITING and worker.scheduler.waiting:
                worker.status = WorkerStatus.RUNNING
                self.engine.send_task(worker, Task.STEP)

    def dispatch(self, sender, requests):
        manager = sender.scheduler.block_manager
        for req in requests:
            target = self._select_decode_target(req)
            destination = target.scheduler.block_manager
            count = math.ceil(req.context_len / target.block_size)
            if count + destination.watermark_blocks > destination.num_total_gpu_blocks:
                raise ConfigurationError(
                    "P/D handoff cannot fit one request at its receiver"
                )
            pinned = []
            try:
                for block in manager.block_table.get_blocks(req.id):
                    manager.gpu_allocator.acquire(block)
                    pinned.append(block)
                manager.commit_finished_cache(req)
                manager.release_request_blocks(req)
                sender.connector.on_request_released(req)
                self._start(req, sender, target, blocks=pinned, reserved=0, count=count)
            except Exception:
                for block in pinned:
                    manager.gpu_allocator.free(block)
                raise

    def _payloads(self, job):
        target = job["target"]
        manager = target.scheduler.block_manager
        if not hasattr(manager, "rank_keys"):
            size = int(
                len(job["blocks"])
                * target.block_size
                * target.cache_config.size_per_token
            )
            return [(target.id, size)]
        workers = {
            w.physical_worker_ref.identity: w.id
            for w in self.engine.workers
            if w.physical_worker_ref is not None
        }
        payloads = []
        for key in manager.rank_keys:
            binding = self.engine.af_runtime_plan.binding(*key)
            size = len(job["blocks"]) * binding.kv_partition.bytes_per_block
            payloads.append((workers[binding.physical_identity], int(size)))
        return payloads

    def _transfer(self, job):
        req = job["request"]
        destination = job["target"].scheduler.block_manager
        self._record(job, "receiver_wait")
        while (
            destination.get_gpu_status()[0]
            < job["count"] + destination.watermark_blocks
        ):
            if req.cancel_requested:
                self._finish_cancel(job)
                return
            yield self.changed
        if req.cancel_requested:
            self._finish_cancel(job)
            return
        destination.reserve_gpu_blocks(job["count"])
        destination.transfer_reservations[req.id] = job["count"]
        job["reserved"] = job["count"]
        source = job["sender"]
        payloads = self._payloads(job)
        bytes_ = sum(size for _, size in payloads)
        self._record(
            job, "transfer_start", bytes=bytes_, receiver_reserved_blocks=job["count"]
        )
        yield from self._copy(source.id, payloads)
        self._unpin(job)
        if req.cancel_requested:
            self._finish_cancel(job)
            return
        self._deliver(job, bytes_)

    def _enqueue_decode(self, job):
        from TokenSim.llm.engine.messages import Task

        self.engine.send_task(job["target"], Task.ADD, [job["request"]])

    def _unpin(self, job):
        source = job["sender"].scheduler.block_manager
        for block in job["blocks"]:
            source.gpu_allocator.free(block)
        job["blocks"] = []

    def _release_destination(self, job):
        req = job["request"]
        scheduler = job["target"].scheduler
        destination = scheduler.block_manager
        reserved = destination.transfer_reservations.pop(req.id, 0)
        destination.release_reserved_blocks(reserved)
        scheduler.waiting = [r for r in scheduler.waiting if r.id != req.id]
        return {"receiver_reserved_blocks": reserved}

    def report(self):
        return dict(
            network=self.fabric.report(),
            lifecycle=self.events,
            pending_requests=len(self.jobs),
            pinned_source_pages=sum(len(j["blocks"]) for j in self.jobs.values()),
            receiver_reserved_pages=sum(
                sum(w.scheduler.block_manager.transfer_reservations.values())
                for w in self.engine.request_workers
            ),
        )


def build_pd_handoff(engine, kv_config):
    specification = kv_config.kv_connector_extra_config.get("pd_endpoint_limits")
    if specification is not None and kv_config.kv_connector not in {
        "NoopConnector",
        "P2PConnector",
    }:
        raise ConfigurationError(
            "Endpoint P/D transport requires a GPU-only P2P connector"
        )
    if specification is None:
        from .legacy import validate_legacy_handoff

        validate_legacy_handoff(engine)
        return None
    if any(w.scheduler.block_manager.grouped_cache for w in engine.request_workers):
        from .grouped import GroupedPDHandoff

        return GroupedPDHandoff(engine, specification)
    return PDHandoff(engine, specification)
