"""Online lifecycle observations shared by uniform storage policy clients."""

import json
from pathlib import Path

from TokenSim.errors import ConfigurationError
from TokenSim.kv_transfer.cache_control import NativeCacheController
from TokenSim.kv_transfer.uniform_prefetch import UniformPrefetch
from TokenSim.kv_transfer.io_observation import request_record, request_metrics


class UniformCacheController:
    configure_log = NativeCacheController.configure_log
    record = NativeCacheController.record

    def __init__(self, engine, connectors):
        self.engine = engine
        self.env = engine.env
        self.connectors = connectors
        self.mode = "uniform_online_storage"
        self.event_path = None
        self.changed = self.env.event()
        self.closed = False
        self.prefetch = UniformPrefetch(self)
        self.owners = {}
        self.stats = dict(completions=0, cancellations=0)
        self.request_rows = []
        for connector in connectors:
            connector.scheduler.cache_controller = self

    @staticmethod
    def owner(req):
        return str(req.chat_id or req.cache_salt or req.id)

    def notify(self):
        event, self.changed = self.changed, self.env.event()
        if not event.triggered:
            event.succeed()

    def completed(self, scheduler, req):
        self.stats["completions"] += 1
        self.request_rows.append(request_record(req, self.env.now, "successful"))
        self.owners.pop(req.id, None)
        if req.measurement_phase == "profile" and not req.cancel_requested:
            self.prefetch.capture(scheduler, req)

    def defer_admission(self, scheduler, req):
        self.owners[req.id] = scheduler
        return self.prefetch.blocks(req)

    def resumed(self, req):
        scheduler = self.owners.get(req.id)
        if scheduler is not None:
            self.prefetch.resumed(scheduler, req)

    def client_resuming(self, req):
        self.prefetch.client_resuming(req)

    def tool_wait(self, req):
        self.prefetch.tool_wait(req)

    def play_finished(self, identity):
        self.prefetch.finish_play(identity)

    def cancelled(self, req):
        self.stats["cancellations"] += 1
        self.request_rows.append(request_record(req, self.env.now, "cancelled"))
        self.owners.pop(req.id, None)
        self.prefetch.release(self.owner(req))

    def close(self):
        self.closed = True
        self.prefetch.close()

    @property
    def pending(self):
        return self.prefetch.pending

    def write_summary(self, directory):
        unique = {id(c.state): c.state for c in self.connectors}
        self.prefetch.stats["wasted_bytes"] = (
            self.prefetch.stats["issued_bytes"] - self.prefetch.stats["useful_bytes"]
        )
        summary = dict(
            **self.stats,
            prefetch=self.prefetch.stats,
            request_metrics=request_metrics(self.request_rows),
            stores=[state.describe() for state in unique.values()],
            per_worker_io={
                c.worker.id: c.observation.summary() for c in self.connectors
            },
        )
        path = Path(directory)
        (path / "cache_control_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        with (path / "storage_io.jsonl").open("w") as stream:
            for state in unique.values():
                for event in state.queue.events:
                    stream.write(json.dumps(event) + "\n")
        with (path / "storage_request_metrics.jsonl").open("w") as stream:
            for row in self.request_rows:
                stream.write(json.dumps(row) + "\n")
        with (path / "storage_request_io.jsonl").open("w") as stream:
            for connector in self.connectors:
                for row in connector.observation.loads:
                    stream.write(
                        json.dumps(dict(worker_id=connector.worker.id, **row)) + "\n"
                    )


def attach_uniform_controller(engine):
    from TokenSim.kv_transfer.connectors.uniform_async import UniformAsyncStore

    connectors = [
        w.connector
        for w in engine.request_workers
        if isinstance(w.connector, UniformAsyncStore)
    ]
    if not connectors:
        return
    if engine.cache_controller is not None:
        raise ConfigurationError(
            "Uniform online storage uses its own lifecycle controller"
        )
    engine.cache_controller = UniformCacheController(engine, connectors)
