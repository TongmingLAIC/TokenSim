from TokenSim.llm.request_time import RequestTime, LLMTime, TimingRecorder

import simpy
from dataclasses import dataclass, field
from typing import Any, Sequence
from enum import Enum, auto

from TokenSim.block.block import PhysicalTokenBlock
from TokenSim.errors import ConfigurationError, SimulationStateError


class RequestStatus(Enum):
    """Status of a sequence."""

    WAITING = auto()
    RUNNING = auto()
    WAITING_FOR_KV = auto()
    WAITING_FOR_CONNECTOR_FREE = auto()
    FINISHED_STOPPED = auto()
    FINISHED_LENGTH_CAPPED = auto()
    CANCEL_PENDING = auto()
    FINISHED_CANCELLED = auto()


_ADMISSION_FIELDS = (
    "status",
    "num_computed_tokens",
    "scheduled_tokens",
    "scheduled_query_tokens",
    "scheduled_placeholder_tokens",
    "recompute_tokens",
    "cached_prefill_blocks",
    "cached_prefill_tokens",
    "gpu_cached_prefill_tokens",
    "external_cached_prefill_tokens",
    "gpu_cached_context_tokens",
    "external_cached_context_tokens",
    "effective_prefill_tokens",
    "reuse_hit_blocks",
    "reuse_miss_blocks",
    "input_cache_keys",
    "input_cache_committed",
    "first_prefill_source",
    "prefill_admission_count",
    "reloaded_gpu_tokens",
    "reloaded_cpu_tokens",
    "reloaded_generated_gpu_tokens",
    "reloaded_generated_cpu_tokens",
)


@dataclass(frozen=True)
class PrefillSourceStats:
    """Original prompt sources, independent of post-preemption KV recovery."""

    input_tokens: int
    gpu_hit_tokens: int
    cpu_hit_tokens: int

    @property
    def computed_tokens(self):
        return self.input_tokens - self.gpu_hit_tokens - self.cpu_hit_tokens


@dataclass
class Request:
    id: int
    prefill_len: int
    decode_len: int
    block_size: int = field(repr=False)
    cache_identity: object = field(
        default_factory=object, init=False, repr=False, compare=False
    )
    arrival_timestamp: float | None = None
    inter_arrival_time: float | None = None
    chat_id: str | None = None
    parent_chat_id: str | None = None
    turn: int | None = None
    hash_ids: list[str | int] | None = None
    output_hash_ids: list[str | int] | None = None
    expert_histogram: dict[int, int] | list | None = None
    prefill_expert_histogram: dict[int, int] | list | None = None
    decode_expert_histogram: dict[int, int] | list | None = None
    routing_assignments: list[dict] | None = None
    prefill_routing_assignments: list[dict] | None = None
    decode_routing_assignments: list[dict] | None = None
    af_a_owners: dict[str, str] = field(default_factory=dict, repr=False)
    cache_salt: str | None = None
    reuse_group: str | None = None
    session_key: Any = field(default=None, repr=False)
    cache_one_shot: bool = field(default=False, repr=False)
    control_prefetch: bool = field(default=False, repr=False)
    control_prefetch_callback: Any = field(default=None, repr=False)
    measurement_phase: str = "profile"
    record_timing: bool = True
    cached_prefill_blocks: int = 0
    cached_prefill_tokens: int = 0
    gpu_cached_prefill_tokens: int = 0
    external_cached_prefill_tokens: int = 0
    gpu_cached_context_tokens: int = 0
    external_cached_context_tokens: int = 0
    first_prefill_source: PrefillSourceStats | None = field(default=None, repr=False)
    trace_hash_ids: Sequence[int] | None = field(default=None, repr=False)
    trace_block_size: int | None = field(default=None, repr=False)
    shared_prefix_boundaries: tuple[int, ...] = ()
    prefill_admission_count: int = 0
    reloaded_gpu_tokens: int = 0
    reloaded_cpu_tokens: int = 0
    reloaded_generated_gpu_tokens: int = 0
    reloaded_generated_cpu_tokens: int = 0
    effective_prefill_tokens: int | None = None
    reuse_hit_blocks: int = 0
    reuse_miss_blocks: int = 0
    input_cache_committed: bool = False
    input_cache_keys: list = field(default_factory=list, repr=False)
    generation_idx: int = 0
    needs_recompute: bool = field(default=False, repr=False)
    recompute_tokens: int = 0
    recomputation_count: int = 0
    recomputed_tokens_total: int = 0
    recompute_service_time: float = 0.0
    status: RequestStatus = field(default=RequestStatus.WAITING, repr=False)
    arrival_at: float | None = field(default=None, repr=False)
    completed_at: float | None = field(default=None, repr=False)
    total_time: float = field(default=0.0, repr=False)
    prefill_latency: float = field(default=0.0, repr=False)
    decode_time_sum: float = field(default=0.0, repr=False)
    decode_time_max: float = field(default=0.0, repr=False)
    prefill_service_time: float = field(default=0.0, repr=False)
    decode_service_time_sum: float = field(default=0.0, repr=False)
    prefill_batch_size: float = field(default=0.0, repr=False)
    decode_batch_sum: float = field(default=0.0, repr=False)
    num_computed_tokens: int = field(default=0, repr=False)
    scheduled_tokens: int = field(default=0, repr=False)
    scheduled_query_tokens: int = field(default=0, repr=False)
    scheduled_accepted_tokens: int = field(default=0, repr=False)
    decode_round: int = field(default=0, repr=False)
    scheduled_placeholder_tokens: int = field(default=0, repr=False)
    prefill_chunk_pending: bool = field(default=False, repr=False)
    max_model_len: int | None = field(default=None, repr=False)
    cancel_requested: bool = field(default=False, repr=False)
    cancel_requested_at: float | None = field(default=None, repr=False)
    cancelled_at: float | None = field(default=None, repr=False)
    owner_worker_ids: set[int] = field(default_factory=set, repr=False)
    cancel_pending_worker_ids: set[int] = field(default_factory=set, repr=False)
    prefill_chunks: int = 0
    dp_domain: str | None = field(default=None, repr=False)
    dp_rank: int | None = field(default=None, repr=False)
    routing_worker_id: int | None = field(default=None, repr=False)
    source: Any = field(default=None, repr=False)
    tqdm_submit_func = None

    def __post_init__(self):
        self._physical_token_blocks: list[PhysicalTokenBlock] = []
        self._logical_tokens = 0
        self._append_tokens(self.prefill_len)
        self._last_event_at: float = 0.0
        self.completion_event: simpy.Event | None = None
        self.cancellation_event: simpy.Event | None = None
        if self.effective_prefill_tokens is None:
            self.effective_prefill_tokens = self.prefill_len

    @property
    def is_measured(self) -> bool:
        return self.measurement_phase != "warmup"

    @property
    def prefill_compute_len(self) -> int:
        if self.effective_prefill_tokens is None:
            return self.prefill_len
        return self.effective_prefill_tokens

    @property
    def is_prefill(self) -> bool:
        return self.generation_idx == 0

    @property
    def context_len(self) -> int:
        return self.prefill_len + self.generation_idx

    @property
    def is_done(self) -> bool:
        return not self.cancel_requested and (
            self.generation_idx >= self.decode_len
            or (
                self.max_model_len is not None
                and self.context_len >= self.max_model_len
            )
        )

    def request_cancel(self, env: simpy.Environment) -> simpy.Event:
        if self.cancellation_event is None:
            self.cancellation_event = env.event()
        if self.is_done:
            if not self.cancellation_event.triggered:
                self.cancellation_event.succeed(self)
            return self.cancellation_event
        if not self.cancel_requested:
            self.cancel_requested = True
            self.cancel_requested_at = env.now
            self.status = RequestStatus.CANCEL_PENDING
            self.cancel_pending_worker_ids = set(self.owner_worker_ids)
        return self.cancellation_event

    def acknowledge_cancel(
        self, env: simpy.Environment, worker_id: int | None = None
    ) -> None:
        if not self.cancel_requested:
            return
        if worker_id is not None:
            self.cancel_pending_worker_ids.discard(worker_id)
        if self.cancel_pending_worker_ids or self.cancelled_at is not None:
            return
        self.cancelled_at = env.now
        self.status = RequestStatus.FINISHED_CANCELLED
        if self.completion_event is not None and not self.completion_event.triggered:
            self.completion_event.succeed(None)
        if (
            self.cancellation_event is not None
            and not self.cancellation_event.triggered
        ):
            self.cancellation_event.succeed(self)

    @property
    def remaining_context_tokens(self) -> int:
        return max(0, self.context_len - self.num_computed_tokens)

    @property
    def is_context_build(self) -> bool:
        return self.is_prefill or self.needs_recompute

    @property
    def is_prompt_prefilling(self) -> bool:
        return self.is_context_build and self.num_computed_tokens < self.prefill_len

    @property
    def decode_graph_eligible(self) -> bool:
        return not self.is_prompt_prefilling or (
            self.step_tokens == 1 and self.num_computed_tokens > 0
        )

    @property
    def logits_tokens(self) -> int:
        return (
            1 + self.scheduled_placeholder_tokens
            if self.is_context_build
            else self.query_tokens
        )

    @property
    def step_tokens(self) -> int:
        return self.scheduled_tokens or max(1, self.remaining_context_tokens)

    @property
    def query_tokens(self) -> int:
        return self.scheduled_query_tokens or self.step_tokens

    def start_context_build(self, cached_tokens: int) -> None:
        self.num_computed_tokens = min(
            max(0, cached_tokens), max(0, self.context_len - 1)
        )
        self.scheduled_tokens = 0
        self.scheduled_query_tokens = 0
        self.scheduled_placeholder_tokens = 0

    def record_prefill_admission(self, gpu_tokens, cpu_tokens):
        if (
            min(gpu_tokens, cpu_tokens) < 0
            or gpu_tokens + cpu_tokens > self.prefill_len
        ):
            raise SimulationStateError("prompt-source tokens exceed the original input")
        if self.first_prefill_source is None:
            self.first_prefill_source = PrefillSourceStats(
                self.prefill_len, gpu_tokens, cpu_tokens
            )
        else:
            self.reloaded_gpu_tokens += gpu_tokens
            self.reloaded_cpu_tokens += cpu_tokens
        self.prefill_admission_count += 1

    def observe_uniform_sources(self, gpu_tokens, cpu_tokens) -> None:
        gpu = min(gpu_tokens, max(0, self.context_len - 1))
        cpu = min(cpu_tokens, max(0, self.context_len - 1 - gpu))
        self.gpu_cached_context_tokens = gpu
        self.external_cached_context_tokens = cpu
        self.gpu_cached_prefill_tokens = min(self.prefill_len, gpu)
        self.external_cached_prefill_tokens = min(
            cpu, max(0, self.prefill_len - self.gpu_cached_prefill_tokens)
        )
        self.record_prefill_admission(
            self.gpu_cached_prefill_tokens, self.external_cached_prefill_tokens
        )

    def record_context_admission(self, gpu_tokens, cpu_tokens):
        """Record a whole-context hit without expanding the input denominator."""
        if (
            min(gpu_tokens, cpu_tokens) < 0
            or gpu_tokens + cpu_tokens >= self.context_len
        ):
            raise SimulationStateError("context-source tokens must leave a real query")
        self.gpu_cached_context_tokens = gpu_tokens
        self.external_cached_context_tokens = cpu_tokens
        prompt_gpu = min(gpu_tokens, self.prefill_len)
        prompt_cpu = min(cpu_tokens, max(0, self.prefill_len - prompt_gpu))
        self.gpu_cached_prefill_tokens = prompt_gpu
        self.external_cached_prefill_tokens = prompt_cpu
        self.cached_prefill_tokens = prompt_gpu + prompt_cpu
        self.cached_prefill_blocks = self.cached_prefill_tokens // self.block_size
        self.effective_prefill_tokens = self.prefill_len - self.cached_prefill_tokens
        self.reloaded_generated_gpu_tokens += gpu_tokens - prompt_gpu
        self.reloaded_generated_cpu_tokens += cpu_tokens - prompt_cpu
        self.record_prefill_admission(prompt_gpu, prompt_cpu)

    def admission_state(self) -> tuple:
        return tuple(getattr(self, name) for name in _ADMISSION_FIELDS)

    def restore_admission_state(self, state: tuple) -> None:
        for name, value in zip(_ADMISSION_FIELDS, state):
            setattr(self, name, value)

    @property
    def num_physical_token_blocks(self):
        return len(self._physical_token_blocks)

    @property
    def num_logical_token_blocks(self):
        return -(-self._logical_tokens // self.block_size)

    def _append_physical_block(self, block) -> None:
        self._physical_token_blocks.append(block)

    def _append_tokens(self, num_tokens: int) -> None:
        if num_tokens > 0:
            self._logical_tokens += num_tokens

    @property
    def arrival_time(self):
        return self.arrival_at

    def validate_prompt_length(self, limit):
        if limit is not None and self.prefill_len >= limit:
            raise ConfigurationError(
                f"request {self.id} prompt must leave room for an output token "
                f"within max_model_len={limit}"
            )

    def arrive(self, env: simpy.Environment):
        if self.cancel_requested:
            return
        self.arrival_at = env.now
        self._last_event_at = env.now
        event_factory = getattr(env, "event", None)
        self.completion_event = event_factory() if event_factory is not None else None

    def advance(
        self, env: simpy.Environment, latency: float, batch: int, timing_recorder=None
    ):
        """Apply one scheduler step and emit RequestTime once sampling finishes."""
        if self.cancel_requested:
            return None
        context_build = self.is_prefill or self.needs_recompute
        if not context_build:
            self.decode_round += 1
        tokens = self.step_tokens
        self.scheduled_tokens = 0
        self.scheduled_query_tokens = 0
        self.scheduled_placeholder_tokens = 0
        self.num_computed_tokens += tokens
        if self.is_prefill:
            self.prefill_chunks += 1
        if self.num_computed_tokens < self.context_len:
            if self.needs_recompute:
                self.recompute_service_time += latency
            else:
                self.prefill_service_time += latency
            return None
        if self.needs_recompute:
            self.finish_recompute(latency)
            latency = 0.0
        return self.step(
            env,
            latency,
            batch,
            timing_recorder,
            output_tokens=(1 if context_build else tokens),
        )

    def step(
        self,
        env: simpy.Environment,
        latency: float,
        batch: int,
        timing_recorder=None,
        *,
        output_tokens: int = 1,
    ):
        """Sample output tokens and update online timing aggregates."""
        if self.cancel_requested:
            return None
        output_tokens = max(
            1, min(output_tokens, self.decode_len - self.generation_idx)
        )
        if self.max_model_len is not None:
            output_tokens = min(
                output_tokens, max(0, self.max_model_len - self.context_len)
            )
        self.generation_idx += output_tokens
        self._append_tokens(output_tokens)
        step_time = env.now - self._last_event_at
        self._last_event_at = env.now
        self.total_time += step_time
        if self.generation_idx == 1:
            self.prefill_latency = step_time
            self.prefill_service_time += latency
            self.prefill_batch_size = batch
        else:
            self.decode_time_sum += step_time
            if step_time > self.decode_time_max:
                self.decode_time_max = step_time
            self.decode_service_time_sum += latency
            self.decode_batch_sum += batch
        if self.is_done:
            if (
                self.max_model_len is not None
                and self.context_len >= self.max_model_len
            ):
                self.status = RequestStatus.FINISHED_LENGTH_CAPPED
            self.completed_at = env.now
            request_time = self.complete_timing()
            if self.record_timing and timing_recorder is not None:
                timing_recorder.record_request(request_time)
            if (
                self.completion_event is not None
                and not self.completion_event.triggered
            ):
                self.completion_event.succeed(request_time)
            if self.tqdm_submit_func:
                self.tqdm_submit_func(1)
            return request_time
        return None

    def release_logical_blocks(self) -> None:
        self._logical_tokens = 0
        self.input_cache_keys.clear()

    def prepare_recompute(self) -> None:
        self.needs_recompute = True
        self.recompute_tokens = self.context_len
        self.num_computed_tokens = 0
        self.scheduled_tokens = 0
        self.scheduled_query_tokens = 0
        self.scheduled_placeholder_tokens = 0

    def finish_recompute(self, latency: float) -> None:
        self.needs_recompute = False
        self.recomputation_count += 1
        self.recomputed_tokens_total += self.recompute_tokens
        self.recompute_service_time += latency
        self.recompute_tokens = 0

    def complete_timing(self) -> RequestTime:
        return RequestTime(
            id=self.id,
            steps=self.generation_idx,
            total_time=self.total_time,
            first_step_time=self.prefill_latency,
            decode_time_sum=self.decode_time_sum,
            decode_time_max_value=self.decode_time_max,
            first_step_service_time=self.prefill_service_time,
            decode_service_time_sum=self.decode_service_time_sum,
            first_step_batch=self.prefill_batch_size,
            decode_batch_sum=self.decode_batch_sum,
        )
