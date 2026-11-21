"""Arguments for implemented cache-control mechanisms."""

import argparse


def add_cache_policy_arguments(parser):
    parser.add_argument(
        "--kv_cache_memory_bytes",
        type=int,
        default=None,
        help="Physical KV pool bytes per rank, bounded by available hardware memory.",
    )
    parser.add_argument(
        "--cache_policy",
        choices=[
            "apc",
            "noop",
            "retain",
            "quota",
            "movement",
            "retain_movement",
            "all",
            "forced_movement",
            "enhanced_coop",
        ],
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
