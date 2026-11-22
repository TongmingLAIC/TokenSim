"""Workload, AgentX replay, and cache-policy CLI arguments."""

from __future__ import annotations

import argparse

from TokenSim.config.cache_policy import PRESETS


def add_workload_and_cache_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Request dataset or AgentX trace file.",
    )
    parser.add_argument(
        "--dataset_skip_count",
        type=int,
        default=0,
        help="Skip this many leading dataset records or traces.",
    )
    parser.add_argument(
        "--trace_timestamp_scale",
        type=float,
        default=None,
        help="Scale loaded timestamped trace arrivals as new_ts = old_ts * scale.",
    )
    parser.add_argument(
        "--trace_target_qps",
        type=float,
        default=None,
        help="Scale trace times so (arrivals - 1) / (last - earliest) matches this QPS.",
    )
    parser.add_argument(
        "--workload_type",
        choices=["synthetic", "json_pairs", "qwen_jsonl", "agentx_weka"],
        default="synthetic",
        help="Request source: synthetic lengths, a dataset format, or AgentX trace replay.",
    )
    parser.add_argument(
        "--agentx_concurrency",
        type=int,
        default=1,
        help="Concurrent AgentX trace trees (replay lanes).",
    )
    parser.add_argument(
        "--agentx_trace_count",
        type=int,
        default=None,
        help="Use at most this many traces from the dataset; default all.",
    )
    parser.add_argument(
        "--agentx_profile_duration",
        type=float,
        default=3600.0,
        help="Simulated seconds of the measured AgentX issue window.",
    )
    parser.add_argument(
        "--agentx_issue_mode",
        choices=("closed_loop", "recorded"),
        default="closed_loop",
        help="Issue requests when their dependencies finish (closed loop) or at recorded trace times.",
    )
    parser.add_argument(
        "--agentx_idle_gap_cap",
        type=float,
        default=10.0,
        help="Compress system-wide idle gaps longer than this many seconds; 0 disables.",
    )
    parser.add_argument(
        "--agentx_warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warm caches with sampled trace frontiers before the measured window.",
    )
    parser.add_argument(
        "--agentx_workload_seed",
        type=int,
        default=None,
        help="Trace warmup sampling seed; defaults to random_seed.",
    )
    parser.add_argument(
        "--agentx_grace_period",
        type=float,
        default=30.0,
        help="Seconds to allow issued requests to finish.",
    )
    parser.add_argument(
        "--agentx_warmup_requests_per_lane",
        type=int,
        default=10,
        help="Warmup requests issued per replay lane before the measured window.",
    )
    parser.add_argument(
        "--agentx_warmup_grace_period",
        type=float,
        default=1800.0,
        help="Seconds allowed for warmup requests to finish before measurement.",
    )
    parser.add_argument(
        "--agentx_tree_idle_cap",
        type=float,
        default=300.0,
        help="Skip idle gaps longer than this within one trace tree; 0 disables.",
    )
    parser.add_argument(
        "--agentx_cancellation_timeout",
        type=float,
        default=10.0,
        help="Seconds to drain cancellation and transfer ownership.",
    )
    parser.add_argument(
        "--agentx_root_arrivals",
        help="JSONL root arrival times and corpus trace indices.",
    )
    parser.add_argument(
        "--sticky_sessions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Retain bounded chat affinity with the ordinary DP placement policy.",
    )
    parser.add_argument(
        "--kv_cache_memory_bytes",
        type=int,
        default=None,
        help="Physical KV pool bytes per rank, bounded by available hardware memory.",
    )
    parser.add_argument(
        "--cache_policy",
        choices=tuple(PRESETS),
        default="apc",
    )
    for name in (
        "cache_queue_window_enabled",
        "cache_child_write_admission",
        "cache_child_reclamation",
    ):
        parser.add_argument(
            "--" + name, action=argparse.BooleanOptionalAction, default=True
        )
    parser.add_argument("--cache_retention_priority", type=int, default=50)
    parser.add_argument("--cache_retention_seconds", type=float, default=30.0)
    parser.add_argument("--cache_quota_fraction", type=float, default=0.5)
    parser.add_argument("--cache_quota_max_wait_seconds", type=float, default=5.0)
    parser.add_argument("--cache_prefetch_lead_seconds", type=float, default=0.1)
    parser.add_argument("--cache_prefetch_budget_bytes", type=int, default=17179869184)
