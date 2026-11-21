"""Engine-owned recording sessions installed only when tracing is enabled."""

from dataclasses import replace
from itertools import chain
import math
from pathlib import Path

from TokenSim.errors import SimulationStateError
from TokenSim.trace.archive import ArchivedTraceRecorder
from TokenSim.trace.af_scope import AFTraceObserver
from TokenSim.trace.pd_scope import EndpointTransferObserver
from TokenSim.trace.composition import Segment
from TokenSim.trace.forward_capture import ForwardCapture
from TokenSim.trace.events import TraceEvent
from TokenSim.trace.formats import write_trace
from TokenSim.trace.transient import ValidationTraceRecorder
from TokenSim.trace.transfers import TransferObserver


class TraceSession:
    def __init__(self, engine):
        self.engine = engine
        self.options = engine.config.trace
        self.recorder = (
            ValidationTraceRecorder()
            if self.options.mode == "validate"
            else ArchivedTraceRecorder()
        )
        self.pending = {}
        self.request_tails = {}
        self.iteration = 0
        self.run_phase = "profiling"
        self.completed = 0
        self.closed = False
        self.active_budgets = {}
        self.transfers = TransferObserver(self)
        self.af = AFTraceObserver(self)
        self.endpoint_transfers = EndpointTransferObserver(self)
        self.forward_capture = {}
        for worker in engine.request_workers:
            self._attach_worker(worker)
        engine.attach_trace(self)

    def close(self):
        """Release private archives once trace inspection or export is complete."""
        if self.closed:
            return
        self.closed = True
        self.transfers.close()
        self.recorder.close()
        for capture in self.forward_capture.values():
            capture.close()
        if self.engine.trace_session is self:
            self.engine.trace_session = None
        if self.engine.af_executor is not None:
            executor = self.engine.af_executor
            executor.trace_observer = None
            executor.trace.event_transform = None
            executor.trace.enabled = False
            executor.trace.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def mark_submitted(self, worker_id):
        if worker_id in self.pending:
            self.pending[worker_id]["submitted"] = True

    def mark_stopped(self, worker_id, now):
        if worker_id in self.pending:
            self.pending[worker_id]["stopped_at"] = now

    def _attach_worker(self, worker):
        self.forward_capture[worker.id] = ForwardCapture(worker)
        self.transfers.attach(worker)

    def set_run_phase(self, phase):
        self.endpoint_transfers.sync()
        self.run_phase = phase

    def estimate_forward(self, worker, requests):
        # Import completed handoffs before the receiving forward captures its tail.
        self.endpoint_transfers.sync()
        phases = {"prefill" if req.is_context_build else "decode" for req in requests}
        info = dict(
            ready=worker.env.now,
            requests=tuple(req.id for req in requests),
            tokens=sum(req.query_tokens for req in requests),
            phase=next(iter(phases)) if len(phases) == 1 else "mixed",
            run_phase=self.run_phase,
            iteration=self.iteration,
        )
        self.iteration += 1
        duration, segments = self.forward_capture[worker.id].estimate(requests)
        info.update(duration=duration, segments=segments)
        self.pending[worker.id] = info
        return duration

    def step_completed(self, worker, requests, save_duration):
        if save_duration > 0:
            plans = tuple(worker.connector._metadata.saves)
            own = getattr(worker.connector, "_own_plans", None)
            if own is not None:
                plans = tuple(own(plans))
            if plans:
                self._record_load(worker, plans, save_duration, kind="save")
        for request in requests:
            if request.is_done:
                self.request_tails.pop(request.id, None)

    def record_connector_load(self, worker, duration):
        self.endpoint_transfers.sync()
        plans = tuple(worker.connector._metadata.loads)
        own = getattr(worker.connector, "_own_plans", None)
        if own is not None:
            plans = tuple(own(plans))
        if plans:
            self._record_load(worker, plans, duration)

    def _record_load(self, worker, plans, duration, *, kind="load"):
        now = worker.env.now
        ids = tuple(request_id for plan in plans for request_id in plan.request_ids)
        dependencies = tuple(
            dict.fromkeys(
                self.request_tails[r][0] for r in ids if r in self.request_tails
            )
        )
        node = f"transfer:{self.iteration}"
        self.active_budgets = {
            key: end for key, end in self.active_budgets.items() if end > now
        }
        self.active_budgets[node] = now + duration
        self.recorder.record(
            TraceEvent(
                iteration_id=self.iteration,
                block_id=0,
                layer_id=-1,
                phase="prefill",
                event_type="transfer",
                resource_id=f"kv:{kind}:{worker.id}",
                ready_at=now,
                start_at=now,
                end_at=now + duration,
                queue_duration=0,
                service_duration=duration,
                request_count=len(ids),
                token_count=0,
                source_id=worker.connector.name,
                node_id=node,
                predecessor_ids=dependencies,
                worker_id=worker.id,
                operation="kv_transfer" if kind == "load" else "kv_save",
                run_phase=self.run_phase,
                payload_bytes=sum(plan.bytes for plan in plans),
                details={
                    "request_ids": ids,
                    "direction": kind,
                    "granularity": "module",
                    "cost_scope": "blocking_connector_invocation",
                    "run_phase": self.run_phase,
                    "source_workers": tuple(plan.source_worker_id for plan in plans),
                },
            )
        )
        self._complete(self.iteration)
        self.iteration += 1
        for request_id in ids:
            self.request_tails[request_id] = node, now + duration

    def mark_started(self, worker_id, start, elapsed):
        info = self.pending.get(worker_id)
        if info is not None:
            info.update(start=start, elapsed=elapsed, submitted=True)

    def _forward_events(self, worker_id, info, start, elapsed):
        worker = self.engine.workers[worker_id]
        requests = info["requests"]
        predecessors = tuple(
            dict.fromkeys(
                self.request_tails[r][0] for r in requests if r in self.request_tails
            )
        )
        segments = list(info["segments"])
        if elapsed > info["duration"] + 1e-12:
            segments.append(
                Segment(
                    "rank_synchronize", elapsed - info["duration"], -1, "synchronize"
                )
            )
        participants = worker.parallel_communicator.tp_group() or (worker_id,)
        for rank_index, physical_worker in enumerate(participants):
            time, offset, previous = start, 0.0, predecessors
            for index, segment in enumerate(segments):
                node = f"forward:{worker_id}:{info['iteration']}:rank:{physical_worker}:{index}"
                details = dict(
                    segment.details,
                    device_id=worker.device.device_id,
                    request_ids=requests,
                    run_phase=info["run_phase"],
                    placement_assumption=(
                        "symmetric_tp" if len(participants) > 1 else "local"
                    ),
                )
                yield TraceEvent(
                    iteration_id=info["iteration"],
                    block_id=worker_id,
                    layer_id=segment.layer,
                    phase=info["phase"],
                    event_type=segment.category,
                    resource_id=f"worker:{physical_worker}",
                    ready_at=info["ready"] if index == 0 else time,
                    start_at=time,
                    end_at=start + (offset + segment.seconds),
                    queue_duration=time - info["ready"] if index == 0 else 0,
                    service_duration=segment.seconds,
                    request_count=len(requests),
                    token_count=info["tokens"],
                    source_id=segment.source,
                    match_type=segment.match,
                    operation=segment.operation,
                    node_id=node,
                    predecessor_ids=previous,
                    worker_id=physical_worker,
                    tp_rank=rank_index,
                    dp_rank=worker.dp_rank,
                    pp_rank=worker.pp_rank,
                    run_phase=info["run_phase"],
                    details=details,
                )
                offset += segment.seconds
                time = start + offset
                previous = (node,)
            if abs(time - (start + elapsed)) > max(1e-10, 4 * math.ulp(time)):
                raise SimulationStateError(
                    "expanded trace must end at the executed forward boundary"
                )

    def finish(self, worker_id, start, elapsed):
        info = self.pending.pop(worker_id, None)
        if info is None:
            return
        external = dict(self.request_tails.values())
        terminal = None
        for event in self._forward_events(worker_id, info, start, elapsed):
            self.recorder.record(event)
            if event.worker_id == worker_id:
                terminal = event.node_id, event.end_at
        if terminal is not None:
            for request_id in info["requests"]:
                self.request_tails[request_id] = terminal
        self._complete(info["iteration"], external)

    def _pending_events(self):
        for worker_id, info in self.pending.items():
            if "start" not in info:
                continue
            cutoff = min(
                self.engine.env.now, info.get("stopped_at", self.engine.env.now)
            )
            for event in self._forward_events(
                worker_id, info, info["start"], info["elapsed"]
            ):
                if event.start_at < cutoff:
                    end = min(event.end_at, cutoff)
                    yield replace(
                        event,
                        end_at=end,
                        service_duration=end - event.start_at,
                        incomplete=True,
                        details=dict(event.details, incomplete=True),
                    )

    def _complete(self, iteration, external=None):
        self.recorder.completed_predecessors = (
            dict(self.request_tails.values()) if external is None else external
        )
        if self.options.validate:
            self.recorder.validate_invariants(iteration)
        self.recorder.complete_iteration(iteration)
        self.completed += 1

    def events(self):
        if self.closed:
            raise SimulationStateError("trace session is closed")
        from TokenSim.trace.views import expand_af_events

        self.endpoint_transfers.sync()
        af = self.engine.af_executor
        histories = [
            self.recorder.events,
            self._pending_events(),
            self.transfers.partial_events(),
            self.endpoint_transfers.partial_events(self.engine.env.now),
        ]
        if af is not None:
            histories.append(expand_af_events(af.trace.events))
        cutoff = self.engine.env.now
        tolerance = max(1e-12, 4 * math.ulp(cutoff))
        for event in chain.from_iterable(histories):
            if event.start_at >= cutoff and event.end_at > cutoff:
                continue
            if event.end_at > cutoff + tolerance:
                yield replace(
                    event,
                    end_at=cutoff,
                    service_duration=cutoff - event.start_at,
                    incomplete=True,
                    details=dict(event.details, incomplete=True),
                )
            elif event.end_at > cutoff:
                yield replace(
                    event,
                    end_at=cutoff,
                    service_duration=max(0, cutoff - event.start_at),
                )
            else:
                yield event

    def _worker_parallelism(self):
        layout = {
            str(worker.id): dict(
                domain=worker.execution_domain,
                tensor_parallel_size=worker.parallel_config.tensor_parallel_size,
            )
            for worker in self.engine.request_workers
            if getattr(worker, "execution_domain", None) is not None
        }
        for worker in self.engine.workers:
            bindings = {
                (binding.domain, len(binding.tp_group))
                for binding in getattr(worker, "af_runtime_bindings", ())
            }
            if len(bindings) == 1:
                domain, tp = next(iter(bindings))
                layout[str(worker.id)] = dict(domain=domain, tensor_parallel_size=tp)
        return layout

    def export(self, directory, *, termination="completed", result_name=None):
        if self.closed:
            raise SimulationStateError("trace session is closed")
        self.endpoint_transfers.sync()
        active_budgets = sum(
            end > self.engine.env.now for end in self.active_budgets.values()
        )
        descriptor = dict(
            mode=self.options.mode,
            schema_version=2,
            completed_scopes=self.completed - active_budgets,
            incomplete=any(info.get("submitted") for info in self.pending.values())
            or bool(self.transfers.pending)
            or bool(self.endpoint_transfers.pending)
            or bool(active_budgets)
            or termination != "completed",
        )
        if self.options.mode == "full":
            worker_parallelism = self._worker_parallelism()
            tp = self.engine.parallel_config.tensor_parallel_size
            if worker_parallelism:
                sizes = {
                    item["tensor_parallel_size"] for item in worker_parallelism.values()
                }
                tp = next(iter(sizes)) if len(sizes) == 1 else None
            path = (
                Path(self.options.output)
                if self.options.output
                else Path(directory)
                / (
                    f"{Path(result_name).stem}.trace.json"
                    if result_name
                    else "operator_trace.json"
                )
            )
            path = write_trace(
                path,
                self.events(),
                metadata=dict(
                    schema_version=2,
                    termination_reason=termination,
                    incomplete=descriptor["incomplete"],
                    clock="simulation",
                    internal_time_unit="seconds",
                    model_id=self.engine.model_spec.model_id,
                    dtype=self.engine.model_spec.activation_dtype,
                    hidden_size=self.engine.model_spec.hidden_size,
                    num_layers=self.engine.model_spec.num_layers,
                    num_attention_heads=self.engine.model_spec.num_attention_heads,
                    tensor_parallel_size=tp,
                    **(
                        {"worker_parallelism": worker_parallelism}
                        if worker_parallelism
                        else {}
                    ),
                ),
                trace_config=self.options,
                trace_format=self.options.format,
            )
            descriptor["path"] = str(path)
        return descriptor
