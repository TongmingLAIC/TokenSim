"""Run one worker step and coordinate the shared scheduler boundaries."""

from TokenSim.errors import SimulationStateError
from .af_worker import run_worker_compute
from .messages import WorkerStatus, _merge_connector_metadata


def accept_requests(worker, message):
    requests = message.requests or []
    worker.pending_admissions -= len(requests)
    if message.metadata is not None:
        worker.pending_connector_metadata = _merge_connector_metadata(
            message.metadata, worker.pending_connector_metadata
        )
    worker.scheduler.add_requests(requests)
    update_worker_status(worker)


def has_ready_work(worker):
    scheduler = worker.scheduler
    return bool(
        scheduler.running
        or scheduler.waiting
        or getattr(scheduler, "loaded_waiting", ())
    )


def update_worker_status(worker):
    worker.status = (
        WorkerStatus.RUNNING if has_ready_work(worker) else WorkerStatus.WAITING
    )


def finish_worker_step(worker):
    worker.finalize_connector_step()
    if worker.role == "prefill":
        running = worker.scheduler.running
        completed = [
            request
            for request in running
            if not request.is_context_build and not request.cancel_requested
        ]
        if completed:
            worker.engine.dispatch_prefill_to_decode(worker, completed)
            worker.scheduler.running = [
                request
                for request in running
                if request.is_context_build or request.cancel_requested
            ]
    update_worker_status(worker)


def wait_for_external_work(worker):
    handoff = getattr(worker.engine, "pd_handoff", None)
    if handoff is not None and handoff.jobs:
        handoff.blocked_workers.add(worker.id)
        return True
    connector = worker.connector
    if connector.async_offload and (
        connector.loads or connector.write_jobs or connector.pending_writes
    ):
        return True
    controller = getattr(worker.engine, "cache_controller", None)
    return controller is not None and controller.pending


def execute_worker_step(worker):
    prior_preemptions = worker.preempted_cnt
    running, load_latency = worker.prepare_step()
    if load_latency:
        trace = getattr(worker.engine, "trace_session", None)
        if trace is not None:
            trace.record_connector_load(worker, load_latency)
        yield worker.env.timeout(load_latency)
    active = [request for request in running if not request.cancel_requested]
    if active:
        latency = yield from run_worker_compute(worker, active)
        save_latency = worker.complete_step(active, latency)
        if save_latency:
            yield worker.env.timeout(save_latency)
    finish_worker_step(worker)
    if active or load_latency or not has_ready_work(worker):
        worker.stalled_at = None
        return bool(active or load_latency)
    if wait_for_external_work(worker):
        worker.status = WorkerStatus.WAITING
        worker.stalled_at = None
        return False
    if worker.preempted_cnt > prior_preemptions and worker.stalled_at != worker.env.now:
        # A preemption can make room for the next admission at the same time.
        worker.stalled_at = worker.env.now
        return True
    raise SimulationStateError(f"worker {worker.id} cannot schedule queued requests")
