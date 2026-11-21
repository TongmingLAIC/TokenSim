"""Record the endpoint P/D transport interval and request dependency."""

from TokenSim.trace.events import TraceEvent


class EndpointTransferObserver:
    def __init__(self, session):
        self.session = session
        self.pending = {}
        self.cursor = 0

    def sync(self):
        handoff = getattr(self.session.engine, "pd_handoff", None)
        if handoff is None:
            return
        while self.cursor < len(handoff.events):
            self._record(handoff.events[self.cursor])
            self.cursor += 1

    def _record(self, event):
        request_id = event["request_id"]
        kind = event["kind"]
        session = self.session
        if kind == "transfer_start":
            tail = session.request_tails.get(request_id)
            self.pending[request_id] = (
                event,
                session.iteration,
                {} if tail is None else {tail[0]: tail[1]},
                session.run_phase,
            )
            session.iteration += 1
        if kind not in {"transfer_complete", "cancel_released", "transfer_failed"}:
            return
        entry = self.pending.pop(request_id, None)
        if entry is None:
            return
        complete = kind == "transfer_complete"
        row = self._event(entry, event["time"], complete=complete)
        session.recorder.record(row)
        session._complete(row.iteration_id, entry[2])
        if complete:
            session.request_tails[request_id] = row.node_id, row.end_at

    def partial_events(self, cutoff):
        return (
            self._event(entry, cutoff, complete=False)
            for entry in self.pending.values()
            if entry[0]["time"] < cutoff
        )

    @staticmethod
    def _event(entry, end, *, complete):
        event, iteration, dependencies, phase = entry
        source, target = event["source"], event["target"]
        start = event["time"]
        return TraceEvent(
            iteration_id=iteration,
            block_id=0,
            layer_id=-1,
            phase="prefill",
            event_type="transfer",
            resource_id=f"pd:{source}:{target}",
            ready_at=start,
            start_at=start,
            end_at=end,
            queue_duration=0,
            service_duration=end - start,
            request_count=1,
            token_count=0,
            source_id="endpoint_fabric",
            node_id=f"pd:transfer:{iteration}",
            predecessor_ids=tuple(dependencies),
            worker_id=target,
            operation="kv_transfer",
            run_phase=phase,
            payload_bytes=event["bytes"] if complete else None,
            incomplete=not complete,
            details=dict(
                request_ids=(event["request_id"],),
                source_workers=(source,),
                target_worker=target,
                planned_payload_bytes=event["bytes"],
                cost_scope="endpoint_transfer_including_launch",
                run_phase=phase,
            ),
        )
