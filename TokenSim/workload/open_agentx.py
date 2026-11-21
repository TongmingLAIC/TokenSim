"""Open arrival of complete agent trees using native dependency and tool replay."""

import json
import math
from pathlib import Path

from TokenSim.errors import WorkloadValidationError
from TokenSim.workload.agentx_replay.core import AgentXReplay


def read_arrivals(path):
    try:
        rows = [
            json.loads(line)
            for line in Path(path).read_text().splitlines()
            if line.strip()
        ]
    except (OSError, ValueError) as error:
        raise WorkloadValidationError(
            f"Invalid root arrival schedule: {error}"
        ) from error
    if not rows:
        raise WorkloadValidationError("Root arrival schedule is empty")
    previous = -1.0
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"time_s", "trace_index"}:
            raise WorkloadValidationError(
                "Root arrivals require time_s and trace_index"
            )
        time = row["time_s"]
        if (
            isinstance(time, bool)
            or not isinstance(time, (int, float))
            or not math.isfinite(time)
            or time < previous
            or time < 0
        ):
            raise WorkloadValidationError(
                "Root times must be finite, nonnegative and sorted"
            )
        if type(row["trace_index"]) is not int or row["trace_index"] < 0:
            raise WorkloadValidationError("trace_index must be a nonnegative integer")
        previous = time
    return rows


class OpenRootReplay(AgentXReplay):
    def __init__(self, *args, root_arrivals, **kwargs):
        if kwargs.get("warmup", True):
            raise WorkloadValidationError(
                "Open root replay requires --no-agentx_warmup"
            )
        super().__init__(*args, **kwargs)
        self.root_arrivals = read_arrivals(root_arrivals)
        if any(row["trace_index"] >= len(self.graphs) for row in self.root_arrivals):
            raise WorkloadValidationError(
                "Root schedule trace_index is outside the corpus"
            )
        self._arrival_index = 0
        self.finished_plays = 0
        self.successful_plays = 0
        self.failed_plays = 0
        self.maximum_active_plays = 0
        self.initial_trace_ids = []
        self.system_idle_cap = None

    def capacity_graphs(self, *, sim_time=None):
        start = self.env.now
        end = start + self.profile_duration
        if sim_time is not None:
            end = min(end, sim_time)
        indices = dict.fromkeys(
            row["trace_index"]
            for row in self.root_arrivals
            if start + row["time_s"] < end
        )
        return (self.graphs[index] for index in indices)

    def _next_root_at(self):
        if self._arrival_index == len(self.root_arrivals):
            return math.inf
        return (
            self.profile_started_at + self.root_arrivals[self._arrival_index]["time_s"]
        )

    def _recycle(self):
        terminal = [play for play in self.trees if play.finished()]
        self.successful_plays += sum(not play.bad_outcome for play in terminal)
        self.failed_plays += sum(bool(play.bad_outcome) for play in terminal)
        active = [play for play in self.trees if not play.finished()]
        self.finished_plays += len(self.trees) - len(active)
        self.trees = active
        while (
            self.accepting
            and self.phase == "profiling"
            and self._next_root_at() <= self.env.now
            and self.env.now < self.issue_window_end_at
        ):
            row = self.root_arrivals[self._arrival_index]
            play = self._tree(self.graphs[row["trace_index"]], self._arrival_index)
            self.trees.append(play)
            self._arrival_index += 1
        self.maximum_active_plays = max(self.maximum_active_plays, len(self.trees))

    def _wait(self, boundary):
        if self.accepting and self.phase == "profiling":
            boundary = min(boundary, self._next_root_at())
        if not self.trees:
            yield self.env.timeout(max(0, boundary - self.env.now))
            return
        yield from super()._wait(boundary)

    def run(self):
        self.trees = []
        self.accepting = True
        self.phase = "profiling"
        self.profile_started_at = self.env.now
        self.issue_window_end_at = self.env.now + self.profile_duration
        while self.env.now < self.issue_window_end_at:
            self._idle_shifts()
            self._issue_ready()
            yield from self._wait(self.issue_window_end_at)
        self.phase = "grace"
        self.in_flight_at_window_end = len(self._in_flight)
        deadline = self.env.now + self.grace_period
        while self.env.now < deadline:
            self._idle_shifts()
            self._issue_ready()
            if not self.trees:
                break
            yield from self._wait(deadline)
        self.accepting = False
        self._epoch += 1
        yield from self._drain(self.env.now)
        self.phase = "complete"
        self.profile_finished_at = self.env.now
        self.requests = [
            r for r in self.requests if r.is_done and not r.cancel_requested
        ]

    def runtime_metadata(self):
        metadata = super().runtime_metadata()
        start = self.profile_started_at
        end = self.issue_window_end_at
        observed_end = min(end, self.env.now) if end is not None else None
        observed_seconds = observed_end - start if start is not None else 0.0
        in_window = [
            request
            for request in self.requests
            if observed_end is not None
            and request.completed_at is not None
            and request.completed_at < observed_end
            and not request.cancel_requested
        ]
        output_tokens = sum(request.generation_idx for request in in_window)
        return dict(
            metadata,
            load_model="open_root_arrivals",
            scheduled_roots=len(self.root_arrivals),
            issued_roots=self._arrival_index,
            issue_window_observed_s=observed_seconds,
            issue_window_completed_requests=len(in_window),
            issue_window_output_tokens=output_tokens,
            issue_window_completion_rps=(
                len(in_window) / observed_seconds if observed_seconds else None
            ),
            issue_window_output_token_tps=(
                output_tokens / observed_seconds if observed_seconds else None
            ),
            maximum_active_plays=self.maximum_active_plays,
            completed_plays=self.successful_plays
            + sum(play.finished() and not play.bad_outcome for play in self.trees),
            failed_plays=self.failed_plays
            + sum(play.finished() and bool(play.bad_outcome) for play in self.trees),
            terminal_plays=self.finished_plays
            + sum(play.finished() for play in self.trees),
            censored_plays=sum(not play.finished() for play in self.trees),
            grace_continues_existing_trees=True,
            concurrency_is_load_limit=False,
            root_arrival_schedule=self.root_arrivals,
        )
