"""Recover per-request I/O exposure from local forward and storage intervals."""

import numpy as np
from TokenSim.kv_transfer.busy_intervals import BusyIndex, append_union


def request_record(req, now, outcome):
    source = req.first_prefill_source
    return dict(
        request_id=req.id,
        phase=req.measurement_phase,
        outcome=outcome,
        start_s=req.arrival_time,
        end_s=now,
        input_tokens=req.prefill_len,
        output_tokens=req.generation_idx,
        ttft_s=req.prefill_latency,
        mean_itl_s=(
            req.decode_time_sum / (req.generation_idx - 1)
            if req.generation_idx > 1
            else None
        ),
        latency_s=now - req.arrival_time,
        dp_rank=req.dp_rank,
        prefill_service_s=req.prefill_service_time,
        prefill_wait_s=(
            req.prefill_latency - req.prefill_service_time
            if req.generation_idx
            else None
        ),
        decode_service_s=req.decode_service_time_sum,
        decode_wait_s=req.decode_time_sum - req.decode_service_time_sum,
        mean_decode_batch=(
            req.decode_batch_sum / (req.generation_idx - 1)
            if req.generation_idx > 1
            else None
        ),
        gpu_hit_tokens=source.gpu_hit_tokens if source else None,
        cpu_hit_tokens=source.cpu_hit_tokens if source else None,
        computed_prompt_tokens=source.computed_tokens if source else None,
    )


def request_metrics(records):
    from TokenSim.reporting.metrics import _percentile_summary

    rows = [
        row
        for row in records
        if row["phase"] != "warmup" and row["outcome"] == "successful"
    ]

    def distribution(name):
        return _percentile_summary([row[name] for row in rows if row[name] is not None])

    duration = (
        max(row["end_s"] for row in rows) - min(row["start_s"] for row in rows)
        if rows
        else 0
    )
    output = sum(row["output_tokens"] for row in rows)
    return dict(
        request_count=len(rows),
        output_tokens=output,
        observation_duration_s=duration,
        output_token_throughput_tps=output / duration if duration else None,
        ttft_s=distribution("ttft_s"),
        tpot_s=distribution("mean_itl_s"),
        e2e_s=distribution("latency_s"),
        prefill_service_s=distribution("prefill_service_s"),
        prefill_wait_s=distribution("prefill_wait_s"),
        decode_service_s=distribution("decode_service_s"),
        decode_wait_s=distribution("decode_wait_s"),
    )


def uncovered(start, end, busy):
    return BusyIndex(busy).uncovered(start, end)


class IOObservation:
    def __init__(self, worker):
        self.worker = worker
        self.compute = []
        self.loads = []

    def forward(self, start, end):
        append_union(self.compute, start, end)

    def read(self, req, start, end, *, prefetch):
        self.loads.append(
            dict(
                request_id=req.id,
                start=start,
                end=end,
                prefetch=prefetch,
                cancelled=req.cancel_requested,
                phase=req.measurement_phase,
            )
        )

    def summary(self):
        eligible = [
            row
            for row in self.loads
            if not row["prefetch"] and not row["cancelled"] and row["phase"] != "warmup"
        ]
        busy = BusyIndex(self.compute)
        exposed = [busy.uncovered(row["start"], row["end"]) for row in eligible]
        return dict(
            local_forward_busy_seconds=sum(end - start for start, end in self.compute),
            demand_reads=len(eligible),
            demand_io_exposed_seconds=sum(exposed),
            demand_io_exposed_p90_s=(
                float(np.percentile(exposed, 90)) if exposed else None
            ),
            definition="Per-request admitted load interval outside local forward execution; parallel requests are not summed into a system makespan",
        )
