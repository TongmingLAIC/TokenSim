"""Write cache-policy events and queued storage summaries."""

import json
from pathlib import Path
from TokenSim.kv_transfer.io_observation import request_metrics


class CacheControlLog:
    def __init__(self, env, mode):
        self.env, self.mode = env, mode
        self.event_path = None

    def configure_log(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.event_path = directory / "cache_control.jsonl"
        self.event_path.write_text("")
        self.record("controller_enabled", mode=self.mode)

    def record(self, kind, **values):
        if self.event_path is not None:
            with self.event_path.open("a") as stream:
                stream.write(
                    json.dumps(dict(time_s=self.env.now, kind=kind, **values)) + "\n"
                )


def write_summary(directory, summary):
    Path(directory, "cache_control_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


def write_uniform_summary(controller, directory):
    unique = {id(c.state): c.state for c in controller.connectors}
    controller.prefetch.stats["wasted_bytes"] = (
        controller.prefetch.stats["issued_bytes"]
        - controller.prefetch.stats["useful_bytes"]
    )
    summary = dict(
        **controller.stats,
        prefetch=controller.prefetch.stats,
        request_metrics=request_metrics(controller.request_rows),
        stores=[state.describe() for state in unique.values()],
        per_worker_io={
            c.worker.id: c.observation.summary() for c in controller.connectors
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
        for row in controller.request_rows:
            stream.write(json.dumps(row) + "\n")
    with (path / "storage_request_io.jsonl").open("w") as stream:
        for connector in controller.connectors:
            for row in connector.observation.loads:
                stream.write(
                    json.dumps(dict(worker_id=connector.worker.id, **row)) + "\n"
                )
