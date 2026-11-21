"""Shared request lifecycle for endpoint-mode P/D handoffs."""

from TokenSim.llm.llm_request import RequestStatus

from .fabric import EndpointFabric


def uses_decode_coordinates(engine):
    return getattr(engine, "af_runtime_plan", None) is not None or any(
        getattr(worker, "execution_domain", None) is not None
        for worker in getattr(engine, "request_workers", ())
    )


def select_decode_target(engine, requests):
    if not uses_decode_coordinates(engine):
        return engine.decode_workers.select_transfer_target(requests)
    req = requests[0]
    source_rank = req.dp_rank
    req.dp_rank = None
    try:
        return engine.decode_workers.select_transfer_target(requests)
    finally:
        req.dp_rank = source_rank


def assign_decode_coordinate(engine, req, target):
    if uses_decode_coordinates(engine) and hasattr(target, "dp_rank"):
        req.dp_rank = target.dp_rank
        req.dp_domain = getattr(target, "execution_domain", None)


class EndpointHandoff:
    def __init__(self, engine, specification):
        self.engine, self.env = engine, engine.env
        self.fabric = EndpointFabric(self.env, **specification)
        self.jobs = {}
        self.events = []
        self.blocked_workers = set()
        self.changed = self.env.event()

    def _signal_change(self):
        event, self.changed = self.changed, self.env.event()
        if not event.triggered:
            event.succeed()

    def _start(self, req, sender, target, **state):
        req.status = RequestStatus.WAITING_FOR_KV
        req.owner_worker_ids = {sender.id}
        job = dict(
            request=req,
            sender=sender,
            source_dp_rank=getattr(sender, "dp_rank", req.dp_rank),
            target=target,
            delivered=False,
            **state,
        )
        self.jobs[req.id] = job
        self.env.process(self._checked_transfer(job))

    def _deliver(self, job, byte_count):
        job["delivered"] = True
        self._assign_decode_coordinate(job)
        job["request"].owner_worker_ids = {job["target"].id}
        self._record(job, "transfer_complete", bytes=byte_count)
        self._enqueue_decode(job)
        self.notify()

    def _checked_transfer(self, job):
        try:
            yield from self._transfer(job)
        except Exception as error:
            try:
                self._finish(job, "transfer_failed")
            except Exception as cleanup_error:
                error.add_note(f"P/D cleanup failed: {cleanup_error}")
            self.engine.record_worker_failure(job["sender"].id, error)

    def _copy(self, source_id, payloads):
        copies = []
        try:
            for target_id, size in payloads:
                copies.append(
                    self.env.process(self.fabric.transfer(source_id, target_id, size))
                )
            yield self.env.all_of(copies)
        except Exception:
            for copy in copies:
                if copy.is_alive:
                    copy.defused = True
                    copy.interrupt("P/D transfer failed")
            for copy in copies:
                if not copy.processed:
                    try:
                        yield copy
                    except Exception:
                        pass
            raise

    def _finish(self, job, kind):
        req = job["request"]
        try:
            fields = self._release_destination(job)
        finally:
            try:
                self._unpin(job)
            finally:
                self.jobs.pop(req.id, None)
                req.owner_worker_ids.clear()
        self._record(job, kind, **fields)
        for owner in tuple(req.cancel_pending_worker_ids):
            req.acknowledge_cancel(self.env, owner)
        req.acknowledge_cancel(self.env)
        self.notify()

    def _finish_cancel(self, job):
        self._finish(job, "cancel_released")

    def _select_decode_target(self, req):
        return select_decode_target(self.engine, [req])

    def _assign_decode_coordinate(self, job):
        assign_decode_coordinate(self.engine, job["request"], job["target"])

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
                    if uses_decode_coordinates(self.engine)
                    else {}
                ),
                **fields,
            )
        )

    def cancel(self, req):
        job = self.jobs.get(req.id)
        if job is None:
            return False
        if job["delivered"]:
            self._finish_cancel(job)
        else:
            self.notify()
        return True

    def admitted(self, req):
        job = self.jobs.pop(req.id, None)
        if job:
            self._record(job, "decode_admitted")
            self.notify()
