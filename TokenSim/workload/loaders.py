from __future__ import annotations

import json
import logging
import math
import random
from pathlib import Path
from typing import Any

import numpy as np

from TokenSim.config.workload_config import WorkloadConfig
from TokenSim.errors import WorkloadValidationError
from TokenSim.moe.routing import normalize_expert_histogram
from TokenSim.utils import get_decode_lens, get_prefill_lens
from TokenSim.workload.user_request import (
    _timestamp,
    UserRequest,
    UserRequestList,
    nonnegative_int,
)

logger = logging.getLogger(__name__)


def load_workload(
    args: Any, workload_config: WorkloadConfig, *, num_experts: int = 0
) -> UserRequestList:
    scale = getattr(args, "trace_timestamp_scale", None)
    target_qps = getattr(args, "trace_target_qps", None)
    if workload_config.distribution == "burst" and (
        scale is not None or target_qps is not None
    ):
        raise WorkloadValidationError(
            "trace timestamp scaling requires non-burst arrivals"
        )
    workload_type = args.workload_type
    if args.dataset_path and workload_type == "synthetic":
        workload_type = "json_pairs"

    if workload_type == "synthetic":
        user_requests = load_synthetic_workload(args, workload_config)
    elif workload_type == "json_pairs":
        if not args.dataset_path:
            raise WorkloadValidationError(
                "--dataset_path is required for json_pairs workloads"
            )
        user_requests = load_json_pairs_workload(
            dataset_path=args.dataset_path,
            request_count=args.request_count,
            dataset_skip_count=args.dataset_skip_count,
            num_experts=num_experts,
        )
    elif workload_type == "qwen_jsonl":
        if not args.dataset_path:
            raise WorkloadValidationError(
                "--dataset_path is required for qwen_jsonl workloads"
            )
        user_requests = load_qwen_jsonl_workload(
            dataset_path=args.dataset_path,
            request_count=args.request_count,
            dataset_skip_count=args.dataset_skip_count,
            num_experts=num_experts,
        )
    else:
        raise WorkloadValidationError(f"unknown workload_type {workload_type!r}")

    return apply_trace_timestamp_scaling(
        user_requests,
        trace_timestamp_scale=scale,
        trace_target_qps=target_qps,
    )


def load_synthetic_workload(
    args: Any, workload_config: WorkloadConfig
) -> UserRequestList:
    count = _validate_request_count(args.request_count)
    if count is None:
        raise WorkloadValidationError("synthetic workloads require request_count")
    prefill_lens = get_prefill_lens(
        len_mean=workload_config.prefill_mean_len,
        len_range=workload_config.prefill_range_len,
        request_count=count,
        rng=random.Random(args.random_seed),
    )
    decode_lens = get_decode_lens(
        distribution=workload_config.decode_len_distribution,
        len_mean=workload_config.decode_mean_len,
        len_range=workload_config.decode_range_len,
        request_count=count,
        rng=random.Random(f"{args.random_seed}:decode"),
        numpy_rng=np.random.RandomState(args.random_seed % (1 << 32)),
    )
    return UserRequestList(
        UserRequest(
            request_id=request_id,
            prefill_len=prefill_len,
            decode_len=decode_len,
        )
        for request_id, (prefill_len, decode_len) in enumerate(
            zip(prefill_lens, decode_lens)
        )
    )


def _build_user_request(
    *,
    request_id: int,
    prefill_len: Any,
    decode_len: Any,
    record: dict[str, Any],
    num_experts: int,
    arrival_time: float | None = None,
    include_inter_arrival: bool = False,
) -> UserRequest:
    return UserRequest(
        request_id=request_id,
        prefill_len=prefill_len,
        decode_len=decode_len,
        arrival_time=(
            record.get("arrival_time") if arrival_time is None else arrival_time
        ),
        inter_arrival_time=(
            record.get("inter_arrival_time") if include_inter_arrival else None
        ),
        chat_id=_optional_str(record.get("chat_id")),
        parent_chat_id=_optional_str(record.get("parent_chat_id")),
        turn=record.get("turn"),
        hash_ids=record.get("hash_ids"),
        output_hash_ids=record.get("output_hash_ids"),
        expert_histogram=_histogram_or_none(
            record.get("expert_histogram"),
            num_experts=num_experts,
            request_id=request_id,
        ),
        prefill_expert_histogram=_histogram_or_none(
            record.get("prefill_expert_histogram"),
            num_experts=num_experts,
            request_id=request_id,
        ),
        decode_expert_histogram=_histogram_or_none(
            record.get("decode_expert_histogram"),
            num_experts=num_experts,
            request_id=request_id,
        ),
        routing_assignments=_assignments_or_none(
            record.get("routing_assignments"), request_id=request_id
        ),
        prefill_routing_assignments=_assignments_or_none(
            record.get("prefill_routing_assignments"), request_id=request_id
        ),
        decode_routing_assignments=_assignments_or_none(
            record.get("decode_routing_assignments"), request_id=request_id
        ),
        cache_salt=_optional_str(record.get("cache_salt")),
        reuse_group=_optional_str(record.get("reuse_group")),
    )


def load_json_pairs_workload(
    dataset_path: str,
    request_count: int | None,
    dataset_skip_count: int = 0,
    num_experts: int = 0,
) -> UserRequestList:
    path = _validate_dataset_path(dataset_path)
    with path.open("r") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError as exc:
            raise WorkloadValidationError(
                f"invalid JSON in json_pairs dataset {path}: {exc}"
            ) from exc

    if not isinstance(data, list):
        raise WorkloadValidationError(f"json_pairs workload {path} must be a JSON list")

    data = _slice_dataset(data, request_count, dataset_skip_count)
    user_requests = UserRequestList()
    for request_id, item in enumerate(data):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            raise WorkloadValidationError(
                f"json_pairs record {request_id} in {path} must be "
                + "[prefill_len, decode_len]"
            )
        extra = item[2] if len(item) > 2 else {}
        if extra is None:
            extra = {}
        if not isinstance(extra, dict):
            raise WorkloadValidationError(
                f"json_pairs record {request_id} in {path} third field must be an object"
            )
        user_requests.append(
            _build_user_request(
                request_id=request_id,
                prefill_len=item[0],
                decode_len=item[1],
                record=extra,
                num_experts=num_experts,
                include_inter_arrival=True,
            )
        )

    _print_dataset_summary(path, user_requests)
    return user_requests


def load_qwen_jsonl_workload(
    dataset_path: str,
    request_count: int | None,
    dataset_skip_count: int = 0,
    num_experts: int = 0,
) -> UserRequestList:
    path = _validate_dataset_path(dataset_path)
    rows: list[dict[str, Any]] = []
    with path.open("r") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise WorkloadValidationError(
                    f"invalid JSONL record at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise WorkloadValidationError(
                    f"qwen_jsonl record at {path}:{line_number} must be a JSON object"
                )
            rows.append(row)

    rows = _slice_dataset(rows, request_count, dataset_skip_count)
    timestamps = [
        (
            _timestamp(row["timestamp"], f"timestamp of record {index} in {path}")
            if row.get("timestamp") is not None
            else None
        )
        for index, row in enumerate(rows)
    ]
    first_timestamp = min(
        (time for time in timestamps if time is not None), default=None
    )

    user_requests = UserRequestList()
    seen_request_ids: set[int] = set()
    for request_id, row in enumerate(rows):
        timestamp = timestamps[request_id]
        arrival_time = None
        if timestamp is not None and first_timestamp is not None:
            arrival_time = timestamp - first_timestamp
        trace_request_id = row.get("request_id")
        unique_request_id = request_id
        if trace_request_id is not None:
            unique_request_id = nonnegative_int(
                trace_request_id, f"request_id of record {request_id} in {path}"
            )
            if unique_request_id in seen_request_ids:
                unique_request_id = request_id
        while unique_request_id in seen_request_ids:
            unique_request_id += 1
        seen_request_ids.add(unique_request_id)
        try:
            prefill_len = row["input_length"]
            decode_len = row["output_length"]
        except KeyError as exc:
            raise WorkloadValidationError(
                f"qwen_jsonl record {request_id} in {path} is missing {exc.args[0]!r}"
            ) from exc
        row_record = {k: v for k, v in row.items() if k != "arrival_time"}
        user_requests.append(
            _build_user_request(
                request_id=unique_request_id,
                prefill_len=prefill_len,
                decode_len=decode_len,
                record=row_record,
                num_experts=num_experts,
                arrival_time=arrival_time,
            )
        )

    _print_dataset_summary(path, user_requests)
    return user_requests


def _validate_dataset_path(dataset_path: str) -> Path:
    path = Path(dataset_path)
    if not path.exists() or not path.is_file():
        raise WorkloadValidationError(f"dataset file not found: {dataset_path}")
    return path


def apply_trace_timestamp_scaling(
    user_requests: UserRequestList,
    *,
    trace_timestamp_scale: float | None = None,
    trace_target_qps: float | None = None,
) -> UserRequestList:
    if trace_timestamp_scale is None and trace_target_qps is None:
        return user_requests
    if trace_timestamp_scale is not None and trace_target_qps is not None:
        raise WorkloadValidationError(
            "trace_timestamp_scale and trace_target_qps are mutually exclusive"
        )

    target_qps = None
    if trace_timestamp_scale is not None:
        scale = _positive_finite_float(
            trace_timestamp_scale,
            "trace_timestamp_scale",
        )
    else:
        target_qps = _positive_finite_float(trace_target_qps, "trace_target_qps")

    arrival_times = [
        request.arrival_time
        for request in user_requests
        if request.arrival_time is not None
    ]
    if not arrival_times:
        raise WorkloadValidationError(
            "trace timestamp scaling requires dataset records with arrival_time or timestamp"
        )

    if target_qps is not None:
        if len(arrival_times) < 2:
            raise WorkloadValidationError(
                "trace_target_qps requires at least two timestamped arrivals"
            )
        span = max(arrival_times) - min(arrival_times)
        if span <= 0:
            raise WorkloadValidationError(
                "trace_target_qps requires arrivals that span a positive interval"
            )
        scale = _positive_finite_float(
            ((len(arrival_times) - 1) / span) / target_qps, "derived timestamp scale"
        )

    # Check all transformed timestamps before changing any request.
    scaled = [_timestamp(time * scale, "scaled arrival_time") for time in arrival_times]
    arrivals = iter(scaled)
    for request in user_requests:
        if request.arrival_time is not None:
            request.arrival_time = next(arrivals)
    return user_requests


def _positive_finite_float(value: Any, field_name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise WorkloadValidationError(f"{field_name} must be a positive finite number")
    try:
        numeric_value = float(value)
    except OverflowError as error:
        raise WorkloadValidationError(
            f"{field_name} must be a positive finite number"
        ) from error
    if not math.isfinite(numeric_value) or numeric_value <= 0:
        raise WorkloadValidationError(
            f"{field_name} must be a positive finite number, got {value!r}"
        )
    return numeric_value


def _validate_request_count(count: int | None) -> int | None:
    if count is None:
        return None
    count = nonnegative_int(count, "request_count")
    if not count:
        raise WorkloadValidationError("request_count must be positive")
    return count


def _slice_dataset(
    data: list[Any], request_count: int | None, dataset_skip_count: int
) -> list[Any]:
    count = _validate_request_count(request_count)
    start = nonnegative_int(dataset_skip_count, "dataset_skip_count")
    selected = data[start : None if count is None else start + count]
    if not selected:
        raise WorkloadValidationError(
            "dataset is empty after applying dataset_skip_count"
        )
    return selected


def _print_dataset_summary(path: Path, user_requests: UserRequestList) -> None:
    prefill_lens = user_requests.prefill_lens()
    decode_lens = user_requests.decode_lens()
    logger.info(
        "read %d requests from %s; mean prompt %.1f, mean output %.1f tokens",
        len(user_requests),
        path,
        sum(prefill_lens) / len(prefill_lens),
        sum(decode_lens) / len(decode_lens),
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _histogram_or_none(
    value: Any,
    *,
    num_experts: int,
    request_id: int,
) -> dict[int, int] | None:
    if value is None:
        return None
    if num_experts <= 0:
        raise WorkloadValidationError(
            f"expert histogram for request {request_id} requires a MoE model config"
        )
    return normalize_expert_histogram(
        value,
        num_experts=num_experts,
        request_id=request_id,
    )


def _assignments_or_none(
    value: Any,
    *,
    request_id: int,
) -> list[dict[str, Any]] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(
        isinstance(record, dict) for record in value
    ):
        raise WorkloadValidationError(
            f"routing assignments for request {request_id} must be a list of objects"
        )
    return [dict(record) for record in value]
