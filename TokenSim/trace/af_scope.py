"""Attach A/F iteration ownership and completion to the global trace."""

from dataclasses import replace

from TokenSim.trace.events import TraceEvent


class AFTraceObserver:
    def __init__(self, session):
        self.session = session
        self.owners = {}

    def submit(self, worker_id, iteration):
        self.session.endpoint_transfers.sync()
        dependencies = dict(
            self.session.request_tails[request_id]
            for request_id in iteration.request_ids
            if request_id in self.session.request_tails
        )
        self.owners[iteration.iteration_id] = (
            worker_id,
            iteration.request_ids,
            dependencies,
            iteration.run_phase,
        )

    def record(self, event):
        owner = self.owners.get(event.iteration_id)
        if owner is None:
            return event
        details = dict(event.details, run_phase=owner[3])
        if not event.predecessor_ids:
            details["external_predecessors"] = owner[2]
        return replace(event, details=details, run_phase=owner[3])

    def finish(self, iteration):
        owner = self.owners.pop(iteration.iteration_id, None)
        if owner is None:
            return
        session = self.session
        worker_id, ids, _, run_phase = owner
        events = session.engine.af_executor.trace.events_for(iteration.iteration_id)
        ends = {
            event.node_id: event.end_at
            for event in events
            if abs(event.end_at - iteration.end_at) < 1e-12
        }
        node = f"af:completion:{iteration.iteration_id}"
        session.recorder.record(
            TraceEvent(
                iteration_id=session.iteration,
                block_id=worker_id,
                layer_id=-1,
                phase="decode",
                event_type="synchronize",
                resource_id=f"af:completion:{worker_id}",
                ready_at=iteration.end_at,
                start_at=iteration.end_at,
                end_at=iteration.end_at,
                service_duration=0,
                queue_duration=0,
                request_count=len(ids),
                token_count=0,
                source_id="af_execution",
                operation="af_completion",
                worker_id=worker_id,
                node_id=node,
                predecessor_ids=tuple(ends),
                details={"request_ids": ids, "run_phase": run_phase},
                run_phase=run_phase,
            )
        )
        session._complete(session.iteration, ends)
        session.iteration += 1
        for request_id in ids:
            session.request_tails[request_id] = node, iteration.end_at
