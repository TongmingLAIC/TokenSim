"""Resolve catalog and default cluster paths after parsing user overrides."""

from argparse import ArgumentParser
from pathlib import Path

from TokenSim.config.paths import default_data_root


DATA_ROOT = default_data_root()
_DEFAULT_CLUSTER = Path("clusters/1_a100/h1.json")


class SimulationArgumentParser(ArgumentParser):
    def parse_known_args(self, args=None, namespace=None):
        parsed, remaining = super().parse_known_args(args, namespace)
        root = (
            Path(parsed.data_root).expanduser().resolve()
            if parsed.data_root is not None
            else default_data_root()
        )
        parsed.data_root = str(root)
        if parsed.cluster is None:
            parsed.cluster = str(root / _DEFAULT_CLUSTER)
        return parsed, remaining
