"""Advance workload events and settle storage without extending a run cutoff."""

from math import inf

from TokenSim.errors import SimulationStateError
from TokenSim.kv_transfer.storage.native import validate_external_stores


def _advance(engine, waiting, deadline):
    """Advance pending work in the half-open interval ending at deadline."""
    env = engine.env
    while waiting() and not engine.failure_event.triggered:
        next_event = env.peek()
        if next_event == inf:
            break
        if deadline is not None and next_event >= deadline:
            if env.now < deadline:
                # SimPy gives a numeric deadline priority over normal events
                # at the same timestamp. Never leave a timer after early exit.
                env.run(until=deadline)
            break
        env.step()
    engine.raise_if_failed()
    return not waiting()


def run_simulation(engine, source, *, replay=None, sim_time=None):
    """Return the termination reason after workload execution and KV cleanup."""
    env = engine.env
    if sim_time is None:
        env.run(until=source if replay is not None else None)
        engine.raise_if_failed()
    else:
        waiting = (
            (lambda: not source.processed)
            if replay is not None
            else (lambda: env.peek() != inf)
        )
        if not _advance(engine, waiting, sim_time):
            if env.now >= sim_time:
                return "sim_time_limit"
            raise SimulationStateError(
                "event queue exhausted before the workload ended"
            )

    if not _finish_kv(engine, replay, sim_time):
        return "sim_time_limit"
    return "completed"


def _finish_kv(engine, replay, sim_time):
    env = engine.env
    controller = engine.cache_controller
    if controller is not None:
        controller.close()
    connectors = [
        worker.connector
        for worker in engine.request_workers
        if worker.connector.async_offload
    ]
    timeout = replay.cancellation_timeout if replay is not None else 10.0
    deadline = env.now + timeout
    if replay is not None and replay.issue_window_end_at is not None:
        deadline = replay.issue_window_end_at + replay.grace_period + timeout
    cutoff = deadline if sim_time is None else min(deadline, sim_time)

    def pending():
        return any(
            connector.loads or connector.write_jobs or connector.pending_writes
            for connector in connectors
        ) or (controller is not None and controller.pending)

    if not _advance(engine, pending, cutoff):
        if sim_time is not None and env.now >= sim_time:
            return False
        raise SimulationStateError(
            "KV transfers did not finish within the cancellation drain deadline"
        )

    for connector in connectors:
        capture = getattr(connector, "capture_end_of_workload", None)
        if capture is not None:
            capture()
    for connector in connectors:
        drain = getattr(connector, "post_workload_drain", None)
        if drain is not None:
            process = env.process(drain())
            if not _advance(engine, lambda: not process.processed, sim_time):
                if sim_time is not None and env.now >= sim_time:
                    return False
                raise SimulationStateError("event queue exhausted during KV audit")
    for worker in engine.request_workers:
        manager = getattr(worker.scheduler, "block_manager", None)
        if manager is not None and hasattr(manager, "validate_conservation"):
            manager.validate_conservation()
    validate_external_stores(connectors)
    return True


def request_observation_duration(requests, now):
    """Time from simulation start through the last request or a partial cutoff."""
    if any(
        not request.is_done and request.cancelled_at is None for request in requests
    ):
        return now
    return max(
        (
            timestamp
            for request in requests
            for timestamp in (request.completed_at, request.cancelled_at)
            if timestamp is not None
        ),
        default=0.0,
    )
