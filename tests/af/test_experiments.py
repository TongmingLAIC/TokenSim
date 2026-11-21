"""The reusable runner keeps fixed-batch results and CLI compatibility intact."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from TokenSim.af.experiments import AFExperimentRunner
from TokenSim.af.factory import simulate_decode
from TokenSim.trace.formats import TRACE_SUFFIXES
from TokenSim.trace.export import write_chrome_trace
from TokenSim.errors import ConfigurationError
from TokenSim.operator_data.package import (
    OperatorDataPackage,
    PackageMeta,
    SourceRecord,
)
from scripts import run_deepseek_mla as cli
from scripts.deepseek_v3_runtime import configuration


class AFExperimentRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = configuration(a_tp=2, f_tp=2, af_batch_size=1)

    def test_runner_matches_factory_timings_statistics_and_full_trace(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {"experiment": "comparison"}
            runner = AFExperimentRunner(
                self.runtime,
                root / "points",
                fallback="analytical_only",
                profile_metadata=metadata,
            )
            point = runner.run(2, 8, steps=3)
            executor, rows, compute, transfer = simulate_decode(
                self.runtime,
                2,
                8,
                steps=3,
                fallback="analytical_only",
            )
            try:
                path = root / "reference.json"
                write_chrome_trace(path, executor.trace.events, metadata)
                profile = point.directory / point.result["summary"]["profile"]
                self.assertEqual(path.read_bytes(), profile.read_bytes())
                self.assertEqual(point.result["iterations"], rows)
                self.assertEqual(
                    point.result["summary"]["operator_stats"], compute.stats_dict()
                )
                self.assertEqual(
                    point.result["summary"]["transfer"],
                    json.loads(json.dumps(transfer.describe())),
                )
            finally:
                executor.trace.close()
            with patch(
                "TokenSim.af.factory.simulate_decode",
                side_effect=AssertionError("reran"),
            ):
                self.assertTrue(runner.run(2, 8, steps=3).reused)
                restarted = AFExperimentRunner(
                    self.runtime,
                    root / "points",
                    fallback="analytical_only",
                    profile_metadata=metadata,
                )
                self.assertTrue(restarted.run(2, 8, steps=3).reused)

    def test_invalid_case_does_not_create_a_checkpoint(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "absent"
            runner = AFExperimentRunner(self.runtime, output)
            for case in (
                dict(batch_size=0),
                dict(seed=-1),
                dict(steps=True),
                dict(fallback="bad"),
                dict(trace_config={"fields": ["bad"]}),
            ):
                values = dict(batch_size=1, context_len=8)
                values.update(case)
                with self.subTest(case=case), self.assertRaises(ConfigurationError):
                    runner.run(**values)
            self.assertFalse(output.exists())

    def test_registered_table_content_invalidates_even_with_the_same_version(self):
        runtime = configuration()
        source = SourceRecord("measured:runner-test", "A", "measured")

        def package(latency):
            return OperatorDataPackage.from_rows(
                PackageMeta(
                    "same-version",
                    "a100_sxm_80g",
                    "vllm",
                    sources={source.source_id: source},
                ),
                {
                    "gemm": [
                        dict(
                            dtype="bf16",
                            m=1,
                            n=1,
                            k=1,
                            latency_us=latency,
                            source_id=source.source_id,
                        )
                    ]
                },
            )

        with TemporaryDirectory() as directory:
            runner = AFExperimentRunner(
                runtime, directory, trace_config={"mode": "off"}
            )
            runtime.hardware.register_package(package(1.0))
            first = runner.run(1, 8)
            self.assertTrue(runner.run(1, 8).reused)
            runtime.hardware.register_package(package(2.0))
            changed = runner.run(1, 8)
            self.assertFalse(changed.reused)
            self.assertNotEqual(first.key, changed.key)


class ExperimentCLITest(unittest.TestCase):
    def arguments(self, output):
        return [
            "--batches",
            "1",
            "2",
            "--context",
            "8",
            "--fallback",
            "analytical_only",
            "--output",
            str(output),
        ]

    def test_restart_skips_completed_batch_and_retries_interrupted_batch(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            simulate = cli.simulate
            calls = []

            def interrupt(config, batch, *values, **options):
                calls.append(batch)
                if batch == 2:
                    raise KeyboardInterrupt()
                return simulate(config, batch, *values, **options)

            with (
                patch.object(cli, "simulate", side_effect=interrupt),
                self.assertRaises(KeyboardInterrupt),
            ):
                cli.main(self.arguments(output))
            self.assertEqual(calls, [1, 2])
            completed_trace = output / "decode-layers1-b1.trace.json"
            expected = completed_trace.read_bytes()
            calls.clear()

            def observe(config, batch, *values, **options):
                calls.append(batch)
                return simulate(config, batch, *values, **options)

            with patch.object(cli, "simulate", side_effect=observe):
                cli.main(self.arguments(output))
            self.assertEqual(calls, [2])
            self.assertEqual(completed_trace.read_bytes(), expected)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(len(summary["results"]), 2)

    def test_json_trace_configuration_can_select_exact_fields(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "trace-config.json"
            config.write_text(
                '{"event_types":["compute"],"fields":["node_id","payload_bytes"]}'
            )
            output = root / "output"
            cli.main([*self.arguments(output), "--trace-config", str(config)])
            for path in output.glob("*.trace.json"):
                rows = json.loads(path.read_text())["traceEvents"]
                self.assertTrue(rows)
                self.assertTrue(
                    all(
                        set(row["args"]) == {"node_id", "payload_bytes"} for row in rows
                    )
                )


FORMATS = ("json", "archive") + (
    ("msgpack",) if importlib.util.find_spec("msgpack") else ()
)


class ExperimentTraceFormatsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = configuration(a_tp=2, f_tp=2, af_batch_size=1)

    def test_invalid_format_and_missing_dependency_fail_before_output_creation(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "absent"
            with self.assertRaises(ConfigurationError):
                AFExperimentRunner(self.runtime, output, trace_format="unknown")
            runner = AFExperimentRunner(self.runtime, output)
            with self.assertRaises(ConfigurationError):
                runner.run(1, 8, trace_format="unknown")
            with patch.dict(sys.modules, {"msgpack": None}):
                with self.assertRaisesRegex(ConfigurationError, r"tokensim\[trace\]"):
                    runner.run(1, 8, trace_format="msgpack")
            self.assertFalse(output.exists())

    def test_disabled_modes_keep_results_without_binary_dependency(self):
        with TemporaryDirectory() as directory:
            runner = AFExperimentRunner(
                self.runtime, directory, fallback="analytical_only"
            )
            full = runner.run(1, 8)
            expected = {**full.result["summary"], "profile": None}
            with patch.dict(sys.modules, {"msgpack": None}):
                for mode in ("off", "validate"):
                    with self.subTest(mode=mode):
                        point = runner.run(
                            1, 8, trace_format="msgpack", trace_config={"mode": mode}
                        )
                        self.assertEqual(
                            point.result["iterations"], full.result["iterations"]
                        )
                        self.assertEqual(point.result["summary"], expected)
                        self.assertFalse(list(point.directory.glob("*.trace.*")))
                        with self.assertRaisesRegex(
                            ConfigurationError, "no saved trace"
                        ):
                            runner.export_trace_json(
                                point, Path(directory) / "trace.json"
                            )

    def test_cli_uses_selected_suffix_and_preserves_default_public_summary(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            traces, summaries = [], []
            for trace_format in FORMATS:
                output = root / trace_format
                arguments = [
                    "--batches",
                    "1",
                    "--context",
                    "8",
                    "--steps",
                    "2",
                    "--fallback",
                    "analytical_only",
                    "--output",
                    str(output),
                ]
                if trace_format != "json":
                    arguments.extend(("--trace-format", trace_format))
                cli.main(arguments)
                summary = json.loads((output / "summary.json").read_text())
                if trace_format == "json":
                    self.assertNotIn("trace_format", summary)
                else:
                    self.assertEqual(summary.pop("trace_format"), trace_format)
                profile = summary["results"][0].pop("profile")
                self.assertEqual(Path(profile).suffix, TRACE_SUFFIXES[trace_format])
                converted = root / f"{trace_format}-converted.json"
                subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "import sys; from TokenSim.trace.formats import convert_trace_to_json; "
                        "convert_trace_to_json(sys.argv[1], sys.argv[2])",
                        str(output / profile),
                        str(converted),
                    ],
                    check=True,
                )
                traces.append(converted.read_bytes())
                summaries.append(summary)
                with patch.object(cli, "simulate", side_effect=AssertionError("reran")):
                    cli.main(arguments)
            self.assertTrue(all(trace == traces[0] for trace in traces))
            self.assertTrue(all(summary == summaries[0] for summary in summaries))
