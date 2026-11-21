"""Root arrival schedules preserve complete agent trees."""

import argparse


def add_open_root_arguments(parser):
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
