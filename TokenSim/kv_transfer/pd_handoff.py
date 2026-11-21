"""Pinned P/D handoff with receiver reservations and shared endpoint transfers."""

import math

from TokenSim.errors import ConfigurationError
from TokenSim.llm.llm_request import RequestStatus
from TokenSim.kv_transfer.endpoint_fabric import EndpointFabric


class PDHandoff:
    def __init__(self, engine, specification):
        self.engine = engine
        self.env = engine.env
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
            not w.af_runtime_bindings and w.parallel_config.tensor_parallel_size != 1
            for w in workers
        ):
            raise ConfigurationError(
                "Endpoint P/D transport requires TP1 ordinary workers"
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
        self.fabric = EndpointFabric(self.env, **specification)
        self.jobs = {}
        self.events = []
        self.blocked_workers = set()
        self.changed = self.env.event()
        for worker in workers:
            worker.scheduler.block_manager.transfer_reservations = {}
            worker.scheduler.block_manager.pd_handoff = self
            for manager in getattr(
                worker.scheduler.block_manager, "rank_managers", {}
            ).values():
                manager.transfer_reservations = (
                    worker.scheduler.block_manager.transfer_reservations
                )

    def notify(self):
        from TokenSim.llm.engine.messages import Task, WorkerStatus

        event, self.changed = self.changed, self.env.event()
        if not event.triggered:
            event.succeed()
        for worker_id in list(self.blocked_workers):
            worker = self.engine.workers[worker_id]
            self.blocked_workers.discard(worker_id)
            if worker.status == WorkerStatus.WAITING and worker.scheduler.waiting:
                worker.status = WorkerStatus.RUNNING
                self.engine.send_task(worker, Task.STEP)

    def dispatch(self, sender, requests):
        from TokenSim.llm.engine.messages import Task

        manager = sender.scheduler.block_manager
        for req in requests:
            target = self._select_decode_target(req)
            destination = target.scheduler.block_manager
            count = math.ceil(req.context_len / target.block_size)
            if count + destination.watermark_blocks > destination.num_total_gpu_blocks:
                raise ConfigurationError(
                    "P/D handoff cannot fit one request at its receiver"
                )
            blocks = list(manager.block_table.get_blocks(req.id))
            for block in blocks:
                manager.gpu_allocator.acquire(block)
            manager.commit_finished_cache(req)
            manager.release_request_blocks(req)
            sender.connector.on_request_released(req)
            req.status = RequestStatus.WAITING_FOR_KV
            # The transport retains source ownership until physical DMA completion.
            req.owner_worker_ids = {sender.id}
            self.jobs[req.id] = dict(
                request=req,
                sender=sender,
                source_dp_rank=getattr(
                    sender, "dp_rank", getattr(req, "dp_rank", None)
                ),
                target=target,
                blocks=blocks,
                reserved=0,
                delivered=False,
                count=count,
                task=Task.ADD,
            )
            self.env.process(self._checked_transfer(req.id))

    def _uses_decode_coordinates(self):
        return getattr(self.engine, "af_runtime_plan", None) is not None or any(
            getattr(worker, "execution_domain", None) is not None
            for worker in getattr(self.engine, "request_workers", ())
        )

    def _select_decode_target(self, req):
        if not self._uses_decode_coordinates():
            return self.engine.decode_workers.select_transfer_target([req])
        source_rank = req.dp_rank
        req.dp_rank = None
        try:
            return self.engine.decode_workers.select_transfer_target([req])
        finally:
            req.dp_rank = source_rank

    def _assign_decode_coordinate(self, job):
        if self._uses_decode_coordinates() and hasattr(job["target"], "dp_rank"):
            job["request"].dp_rank = job["target"].dp_rank

    def _checked_transfer(self, request_id):
        job = self.jobs[request_id]
        try:
            yield from self._transfer(job)
        except Exception as error:
            self.engine.record_worker_failure(job["sender"].id, error)

    def _record(self, job, kind, **fields):
        self.events.append(
            dict(
                time=self.env.now,
                kind=kind,
                request_id=job["request"].id,
                source=job["sender"].id,
                target=job["target"].id,
                **(
                    dict(
                        source_dp_rank=job.get("source_dp_rank"),
                        target_dp_rank=getattr(job["target"], "dp_rank", None),
                    )
                    if self._uses_decode_coordinates()
                    else {}
                ),
                **fields,
            )
        )

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
        yield self.env.all_of(
            [
                self.env.process(self.fabric.transfer(source.id, target_id, size))
                for target_id, size in payloads
            ]
        )
        self._unpin(job)
        if req.cancel_requested:
            self._finish_cancel(job)
            return
        job["delivered"] = True
        self._assign_decode_coordinate(job)
        req.owner_worker_ids = {job["target"].id}
        self._record(job, "transfer_complete", bytes=bytes_)
        self.engine.send_task(job["target"], job["task"], [req])
        self.notify()

    def _unpin(self, job):
        source = job["sender"].scheduler.block_manager
        for block in job["blocks"]:
            source.gpu_allocator.free(block)
        job["blocks"] = []

    def _finish_cancel(self, job):
        req = job["request"]
        self._unpin(job)
        destination = job["target"].scheduler.block_manager
        reserved = destination.transfer_reservations.pop(req.id, 0)
        destination.release_reserved_blocks(reserved)
        job["target"].scheduler.waiting = [
            r for r in job["target"].scheduler.waiting if r.id != req.id
        ]
        self.jobs.pop(req.id, None)
        self._record(job, "cancel_released", receiver_reserved_blocks=reserved)
        for owner in list(req.cancel_pending_worker_ids):
            req.acknowledge_cancel(self.env, owner)
        req.acknowledge_cancel(self.env)
        self.notify()

    def cancel(self, req):
        job = self.jobs.get(req.id)
        if job is None:
            return False
        if job["delivered"]:
            self._finish_cancel(job)
        self.notify()
        return True

    def admitted(self, req):
        job = self.jobs.pop(req.id, None)
        if job:
            self._record(job, "decode_admitted")
            self.notify()

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
        return None
    if any(w.scheduler.block_manager.grouped_cache for w in engine.request_workers):
        from TokenSim.kv_transfer.grouped_pd_handoff import GroupedPDHandoff

        return GroupedPDHandoff(engine, specification)
    return PDHandoff(engine, specification)
