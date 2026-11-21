"""Use the real CLI parser with small CPU-only simulation inputs."""

from contextlib import redirect_stdout
import io
from pathlib import Path

from TokenSim.cli import build_parser, main

ROOT = Path(__file__).resolve().parents[2]
MOE_CLUSTER = ROOT / "data/clusters/8_a100/moe_tp2_dp4.json"


def simulation_args(directory, *extra, **overrides):
    options = dict(
        latency_backend="analytical",
        verbose="none",
        distribution="burst",
        request_count=8,
        prefill_mean_len=16,
        decode_mean_len=3,
        max_num_seqs=2,
        results_path=directory,
    )
    options.update(overrides)
    flags = []
    for name, value in options.items():
        if value is None:
            continue
        if type(value) is bool:
            flags.append(f"--{name}" if value else f"--no-{name}")
        else:
            flags.extend((f"--{name}", str(value)))
    return build_parser().parse_args([*flags, *extra])


def run_quiet(args, **kwargs):
    with redirect_stdout(io.StringIO()):
        return main(args, **kwargs)
