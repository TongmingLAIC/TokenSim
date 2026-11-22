"""Online request timing aggregates and run-wide timing views."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RequestTime:
    """Per-request timing aggregates accumulated online."""

    id: int
    steps: int
    total_time: float = 0.0
    first_step_time: float = 0.0
    decode_time_sum: float = 0.0
    decode_time_max_value: float = 0.0
    first_step_service_time: float = 0.0
    decode_service_time_sum: float = 0.0
    first_step_batch: float = 0.0
    decode_batch_sum: float = 0.0

    @property
    def request_time(self):
        return self.total_time

    @property
    def prompt_time(self):
        return self.first_step_time

    @property
    def prefill_time(self):
        return self.first_step_time

    @property
    def decode_time(self):
        if self.steps <= 1:
            return 0.0
        return self.decode_time_sum / (self.steps - 1)

    @property
    def decode_max_time(self):
        if self.steps <= 1:
            return 0.0
        return self.decode_time_max_value

    @property
    def prompt_batch(self):
        return self.first_step_batch

    @property
    def decode_batch(self):
        if self.steps <= 1:
            return 0.0
        return self.decode_batch_sum / (self.steps - 1)

    @property
    def prompt_util(self):
        return self.first_step_service_time / self.first_step_time

    @property
    def prefill_util(self):
        return self.prompt_util

    @property
    def decode_util(self):
        if self.decode_time_sum <= 0:
            return 0.0
        return self.decode_service_time_sum / self.decode_time_sum

    @property
    def prompt_idle(self):
        return 1 - self.prompt_util

    @property
    def prefill_idle(self):
        return 1 - self.prefill_util

    @property
    def decode_idle(self):
        return 1 - self.decode_util


@dataclass(slots=True)
class LLMTime:
    time: list[RequestTime] = field(default_factory=list)

    @property
    def request_time(self):
        return [t.request_time for t in self.time]

    @property
    def prompt_time(self):
        return [t.prompt_time for t in self.time]

    @property
    def prefill_time(self):
        return [t.prefill_time for t in self.time]

    @property
    def decode_time(self):
        return [t.decode_time for t in self.time]

    @property
    def decode_max_time(self):
        return [t.decode_max_time for t in self.time]

    @property
    def prompt_util(self):
        return [t.prompt_util for t in self.time]

    @property
    def prefill_util(self):
        return [t.prefill_util for t in self.time]

    @property
    def decode_util(self):
        return [t.decode_util for t in self.time]

    @property
    def prompt_idle(self):
        return [t.prompt_idle for t in self.time]

    @property
    def prefill_idle(self):
        return [t.prefill_idle for t in self.time]

    @property
    def decode_idle(self):
        return [t.decode_idle for t in self.time]

    @property
    def prompt_batch(self):
        return [t.prompt_batch for t in self.time]

    @property
    def decode_batch(self):
        return [t.decode_batch for t in self.time]


class TimingRecorder:
    """Completion timing owned by one simulation engine."""

    def __init__(self, timing: LLMTime | None = None):
        self.timing = timing if timing is not None else LLMTime()

    def record_request(self, request_time: RequestTime) -> None:
        self.timing.time.append(request_time)

    def reset(self) -> LLMTime:
        self.timing.time.clear()
        return self.timing
