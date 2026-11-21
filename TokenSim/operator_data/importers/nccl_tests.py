"""Parse ``nccl-tests`` (``all_reduce_perf`` etc.) output into collective rows.

Example line (out-of-place columns first)::

    #       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
    #        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
        33554432       8388608     float     sum      -1   491.6   68.26  119.46      0    489.9   68.49  119.87      0
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from TokenSim.errors import ConfigurationError

_TYPE_MAP = {
    "float": "fp32",
    "float32": "fp32",
    "half": "fp16",
    "float16": "fp16",
    "bfloat16": "bf16",
    "bf16": "bf16",
    "int8": "int8",
    "uint8": "int8",
    "int32": "fp32",
    "fp8e4m3": "fp8",
    "fp8e5m2": "fp8",
}

_OPERATION_FROM_BINARY = {
    "all_reduce_perf": "all_reduce",
    "all_gather_perf": "all_gather",
    "reduce_scatter_perf": "reduce_scatter",
    "alltoall_perf": "all_to_all",
    "sendrecv_perf": "send_recv",
    "broadcast_perf": "broadcast",
}


_DEVICE_ROW = re.compile(
    r"^#\s*Rank\s+(\d+)(?:\s+Group\s+(-?\d+))?\s+Pid\s+\d+"
    r"\s+on\s+(\S+)\s+device\s+\d+\b"
)


def parse_nccl_tests_output(
    text_or_path: str | Path,
    *,
    operation: str,
    group_size: int,
    nodes: int,
    source_id: str,
    use_in_place: bool = False,
) -> list[dict[str, Any]]:
    path = (
        Path(text_or_path)
        if isinstance(text_or_path, Path)
        or (
            isinstance(text_or_path, str)
            and "\n" not in text_or_path
            and Path(text_or_path).is_file()
        )
        else None
    )
    text = (
        path.read_text(encoding="utf-8", errors="replace")
        if path
        else str(text_or_path)
    )
    rows: list[dict[str, Any]] = []
    groups: dict[str | None, dict[int, str]] = {}
    selection = 1 if use_in_place else 0
    mode = "in-place" if use_in_place else "out-of-place"
    time_columns, wrong_columns = (5, 9), (8, 12)
    for line_number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            device = _DEVICE_ROW.match(stripped)
            if device:
                rank, group, host = device.groups()
                hosts = groups.setdefault(group, {})
                if hosts.setdefault(int(rank), host) != host:
                    raise ConfigurationError(
                        "nccl-tests rank reported on multiple hosts"
                    )
            columns = stripped[1:].lower().split()
            if columns[:3] == ["size", "count", "type"]:
                time_columns = tuple(
                    i for i, name in enumerate(columns) if name == "time"
                )
                wrong_columns = tuple(
                    i for i, name in enumerate(columns) if name == "#wrong"
                )
            continue
        parts = stripped.split()
        if len(parts) < 3 or not all(value.isdigit() for value in parts[:2]):
            continue
        size_bytes = int(parts[0])
        dtype = _TYPE_MAP.get(parts[2].lower())
        if dtype is None:
            continue
        where = f"nccl-tests line {line_number} ({mode})"
        try:
            time = parts[time_columns[selection]]
            wrong = parts[wrong_columns[selection]]
        except IndexError as exc:
            raise ConfigurationError(f"{where}: missing time or #wrong column") from exc
        try:
            time_us = float(time)
            wrong_elements = float(wrong)
        except ValueError as exc:
            raise ConfigurationError(
                f"{where}: expected numeric time and #wrong, got {time!r} and {wrong!r}"
            ) from exc
        if wrong_elements != 0:
            raise ConfigurationError(f"{where}: expected #wrong=0, got {wrong}")
        if not math.isfinite(time_us) or time_us < 0:
            raise ConfigurationError(f"{where}: time must be finite and nonnegative")
        rows.append(
            {
                "dtype": dtype,
                "operation": operation,
                "group_size": int(group_size),
                "nodes": int(nodes),
                "message_bytes": size_bytes,
                "latency_us": time_us,
                "source_id": source_id,
                "kernel": "nccl-tests",
            }
        )
    for group, hosts in groups.items():
        observed_ranks, observed_nodes = len(hosts), len(set(hosts.values()))
        if (observed_ranks, observed_nodes) != (group_size, nodes):
            label = f" group {group}" if group is not None else ""
            raise ConfigurationError(
                f"nccl-tests{label} reports {observed_ranks} ranks on {observed_nodes} nodes; "
                f"expected {group_size} ranks on {nodes} nodes"
            )
    if not rows:
        raise ConfigurationError("no nccl-tests result rows found in input")
    return rows


def operation_from_binary_name(name: str) -> str:
    try:
        return _OPERATION_FROM_BINARY[Path(name).name]
    except KeyError as exc:
        raise ConfigurationError(f"unknown nccl-tests binary {name!r}") from exc
