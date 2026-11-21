"""Validate collective measurements before adding them to a package."""

import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest

from TokenSim.errors import ConfigurationError
from TokenSim.operator_data.importers.nccl_tests import parse_nccl_tests_output


ROOT = Path(__file__).resolve().parents[1]
OUT_COLUMNS = "33554432 8388608 float sum -1 491.6 68.26 119.46"
IN_COLUMNS = "489.9 68.49 119.87"


def parse(text, *, nodes=1, use_in_place=False):
    return parse_nccl_tests_output(
        text,
        operation="all_reduce",
        group_size=2,
        nodes=nodes,
        source_id="test",
        use_in_place=use_in_place,
    )


class NCCLCollectionTest(unittest.TestCase):
    def test_selected_columns_control_correctness(self):
        valid = f"{OUT_COLUMNS} 0 {IN_COLUMNS} 0"
        for mode, wrong in (
            ("out-of-place", f"{OUT_COLUMNS} 2 {IN_COLUMNS} 0"),
            ("in-place", f"{OUT_COLUMNS} 0 {IN_COLUMNS} 2"),
        ):
            with self.subTest(mode=mode):
                in_place = mode == "in-place"
                with self.assertRaisesRegex(
                    ConfigurationError, f"line 2 \\({mode}\\): expected #wrong=0"
                ):
                    parse(f"{valid}\n{wrong}", use_in_place=in_place)
                self.assertEqual(
                    parse(wrong, use_in_place=not in_place)[0]["latency_us"],
                    491.6 if in_place else 489.9,
                )

    def test_missing_or_unchecked_columns_fail(self):
        for text, use_in_place in (
            (OUT_COLUMNS, False),
            (f"{OUT_COLUMNS} 0", True),
            (f"{OUT_COLUMNS} N/A", False),
        ):
            with self.subTest(text=text, use_in_place=use_in_place):
                with self.assertRaisesRegex(ConfigurationError, "#wrong"):
                    parse(text, use_in_place=use_in_place)

    def test_printed_columns_account_for_iteration_statistics(self):
        header = (
            "# size count type redop root time algbw busbw #wrong i_min i_max i_p99 i_cv% "
            "time algbw busbw #wrong i_min i_max i_p99 i_cv%"
        )
        text = (
            f"{header}\n{OUT_COLUMNS} 0 480 500 497 1.2 {IN_COLUMNS} 0 480 500 497 1.2"
        )
        self.assertEqual(parse(text, use_in_place=True)[0]["latency_us"], 489.9)

    def test_checks_reported_rank_and_node_counts(self):
        for group in ("", " Group 0"):
            with self.subTest(group=group):
                rank0 = f"# Rank 0{group} Pid 100 on node-a device 0 [0000:17:00]"
                rank1 = f"# Rank 1{group} Pid 200 on node-b device 0 [0000:17:00]"
                text = f"# Using devices\n{rank0}\n{rank1}\n{OUT_COLUMNS} 0"
                self.assertEqual(parse(text, nodes=2)[0]["nodes"], 2)
                local = text.replace("node-b", "node-a")
                with self.assertRaisesRegex(
                    ConfigurationError, "reports 2 ranks on 1 nodes"
                ):
                    parse(local, nodes=2)
                with self.assertRaisesRegex(
                    ConfigurationError, "reports 1 ranks on 1 nodes"
                ):
                    parse(text.replace(rank1 + "\n", ""))

    def test_multinode_launch_checks_before_creating_output(self):
        script = ROOT / "scripts/collect/run_nccl_tests.sh"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            log_dir, package = root / "logs", root / "package"
            for launcher in ("", " \t"):
                with self.subTest(launcher=launcher):
                    result = subprocess.run(
                        ["bash", str(script), "2", "2"],
                        env=dict(
                            os.environ,
                            DEVICE="test-device",
                            NCCL_TESTS=str(root / "no-collection-binaries"),
                            LAUNCHER=launcher,
                            OUT=str(log_dir),
                            PACKAGE=str(package),
                        ),
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertIn("requires LAUNCHER", result.stderr)
                    self.assertFalse(log_dir.exists())
                    self.assertFalse(package.exists())
