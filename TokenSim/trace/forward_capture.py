"""Capture the existing operator estimates while a worker plans a forward."""

from TokenSim.config.cache_config import stage_layer_range
from TokenSim.trace.composition import expand_composition


class ForwardCapture:
    def __init__(self, worker):
        self.backend = worker.latency_backend
        self.communicator = worker.parallel_communicator
        self.captured = []
        self.communications = []
        self.collecting = False
        self.backends = []
        self.communicator.collective_observer = self._record_collective
        self._configure(self.backend)

    def _record_collective(self, kind, nbytes, duration, group, estimate):
        if self.collecting:
            self.communications.append(
                dict(
                    kind=kind,
                    bytes=nbytes,
                    seconds=duration,
                    group=tuple(group),
                    estimate=estimate,
                )
            )

    def _configure(self, current):
        self.backends.append(current)
        current.trace_communication_start = self.communications.clear
        current.trace_composition = lambda estimator, values: self.captured.extend(
            expand_composition(estimator, values, self.communications)
        )
        if current.native_attention:
            current.native_attention.trace_layer_costs = {}
        for child in current._layer_backends:
            owned = stage_layer_range(
                current.model.num_layers,
                current.parallel_config.pipeline_parallel_size,
                current.rank_info.pp_rank,
                current.pipeline_layer_counts,
            )
            child.trace_layer_indices = tuple(
                layer
                for layer in owned
                if current.model.attention_window(layer) == child.model.sliding_window
                and current.model.attention_geometry(layer)
                == child.model.attention_geometry(layer)
            )
            self._configure(child)

    def estimate(self, requests):
        self.captured.clear()
        self.communications.clear()
        if self.backend.native_attention:
            self.backend.native_attention.trace_layer_costs.clear()
        self.collecting = True
        try:
            duration = self.backend.estimate_step_latency(requests)
        finally:
            self.collecting = False
        return duration, sorted(self.captured, key=_segment_order)

    def close(self):
        self.communicator.collective_observer = None
        for backend in self.backends:
            backend.trace_composition = None
            backend.trace_communication_start = None
            if backend.native_attention:
                backend.native_attention.trace_layer_costs = None


def _segment_order(segment):
    if segment.layer >= 0:
        return 1, segment.layer
    leading = {"step_overhead", "embedding_lookup", "embedding_scale"}
    first_stage = (
        segment.operation in leading
        or segment.details.get("collective_scope") == "tp_embedding"
    )
    return (0 if first_stage else 2), segment.layer
