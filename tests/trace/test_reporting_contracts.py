"""Request metrics and persisted JSON share the same observation population."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from TokenSim.cli import build_parser, main


class ReportingContractTest(unittest.TestCase):
    def test_burst_results_use_strict_json_and_no_gap_has_no_tpot(self):
        def fail_on_constant(value):
            raise AssertionError(f"unexpected non-JSON value: {value}")

        with TemporaryDirectory() as directory:
            path = Path(directory)
            data = path / "requests.json"
            data.write_text(json.dumps([[16, 1], [16, 1], [16, 2]]))
            workload = path / "workload.json"
            workload.write_text(
                json.dumps({"decode_token_latency": {"p50": 0.0, "max": 0.0}})
            )
            args = build_parser().parse_args(
                [
                    "--distribution",
                    "burst",
                    "--latency_backend",
                    "analytical",
                    "--request_count",
                    "3",
                    "--workload_type",
                    "json_pairs",
                    "--dataset_path",
                    str(data),
                    "--workload_config",
                    str(workload),
                    "--verbose",
                    "none",
                    "--results_path",
                    directory,
                ]
            )
            with redirect_stdout(io.StringIO()) as stdout:
                main(args)
            result = json.loads(
                (path / "result_inf.json").read_text(), parse_constant=fail_on_constant
            )
            self.assertIsNone(result["qps"])
            self.assertEqual(result["arrival_distribution"], "burst")
            slo = result["slo"]
            decode = slo["decode_token_latency"]
            self.assertEqual(decode["sample_count"], 1)
            self.assertGreater(decode["observed"]["p50"], 0.0)
            self.assertEqual(result["decode_time"]["p50"], decode["observed"]["p50"])
            self.assertEqual(slo["goodput_requests"], 2)
            self.assertNotIn("WARNING analytical share", stdout.getvalue())
