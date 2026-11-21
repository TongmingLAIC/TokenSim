"""Catalog paths and backend selection from the installed data inventory."""

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from TokenSim.cli import build_parser
from TokenSim.hardware.context import HardwareContext


class DataPathsTest(unittest.TestCase):
    def test_default_cluster_uses_the_selected_catalog(self):
        parser = build_parser()
        with patch.dict(os.environ, {"TOKENSIM_DATA_ROOT": "env-catalog"}):
            env = parser.parse_args([])
            cli = parser.parse_args(["--data_root", "cli-catalog"])
            explicit = parser.parse_args(["--cluster", "custom.json"])
        suffix = Path("clusters/1_a100/h1.json")
        self.assertEqual(Path(env.cluster), Path(env.data_root) / suffix)
        self.assertEqual(Path(cli.cluster), Path(cli.data_root) / suffix)
        self.assertEqual(Path(cli.data_root), Path("cli-catalog").resolve())
        self.assertEqual(Path(env.data_root), Path("env-catalog").resolve())
        self.assertEqual(explicit.cluster, "custom.json")

    def test_cli_accepts_a_backend_from_the_selected_device_catalog(self):
        args = build_parser().parse_args(["--operator_backend", "nccl"])
        hardware = HardwareContext.load(args.data_root)
        package = hardware.operator_package("v100_sxm2", args.operator_backend)
        self.assertEqual(package.backend, "nccl")
        self.assertTrue(package.tables["collective"])
