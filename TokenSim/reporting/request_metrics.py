"""The successful requests and measured latencies used by result and SLO reports."""

from dataclasses import dataclass


def request_tpot(request):
    if request.generation_idx <= 1:
        return None
    return request.decode_time_sum / (request.generation_idx - 1)


@dataclass(frozen=True)
class RequestLatencies:
    end_to_end: float
    first_token: float
    tpot: float | None
    max_gap: float | None

    @classmethod
    def finished(cls, requests):
        return [
            cls(
                req.total_time,
                req.prefill_latency,
                request_tpot(req),
                req.decode_time_max if req.generation_idx > 1 else None,
            )
            for req in requests
            if req.is_done and not req.cancel_requested
        ]
