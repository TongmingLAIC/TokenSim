"""Committed points survive failures and never reuse changed experiment inputs."""

import json
from multiprocessing import get_context
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from TokenSim.af.experiments import AFExperimentRunner
from TokenSim.af.factory import simulate_decode
from TokenSim.errors import ConfigurationError
from TokenSim.experiments import ExperimentRunner
from TokenSim.operator_data.package import (
    OperatorDataPackage,
    PackageMeta,
    SourceRecord,
)
from TokenSim.trace.export import write_chrome_trace
from scripts.deepseek_v3_runtime import configuration


def _shared_point(output, code_path, queue):
    runner = ExperimentRunner(output, code_paths=[code_path])

    def execute(directory):
        with (Path(output) / "executions").open("a") as stream:
            stream.write("executed\n")
        time.sleep(0.1)
        (directory / "trace").write_bytes(b"complete")
        return {"value": 1}

    result = runner.run({"case": "shared"}, execute)
    queue.put(result.reused)


def _killable_point(output, code_path, queue, pause):
    def execute(directory):
        (directory / "trace").write_bytes(b"partial")
        queue.put("staged")
        if pause:
            time.sleep(60)
        (directory / "trace").write_bytes(b"complete")
        return {"value": 1}

    point = ExperimentRunner(output, code_paths=[code_path]).run(
        {"case": "killed"}, execute
    )
    queue.put(point.reused)


class ExperimentRunnerTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.code = self.root / "runner_code.py"
        self.code.write_text("version = 1\n")
        self.out = self.root / "points"
        self.calls = 0

    def runner(self, output=None, **kwargs):
        kwargs.setdefault("code_paths", [self.code])
        return ExperimentRunner(output or self.out, **kwargs)

    def execute(self, directory):
        self.calls += 1
        (directory / "trace").write_bytes(b"complete")
        return {"value": 3, "tuple": (1, 2)}

    def test_fresh_and_new_runner_reuse_have_identical_results(self):
        self.assertEqual(
            ExperimentRunner(self.out).code_paths,
            (Path(__file__).resolve().parents[2] / "TokenSim",),
        )
        first = self.runner().run({"seed": 42}, self.execute)
        saved = self.runner().run({"seed": 42}, self.execute)
        self.assertEqual(self.calls, 1)
        self.assertFalse(first.reused)
        self.assertTrue(saved.reused)
        self.assertEqual(first.result, saved.result)
        self.assertEqual(saved.result["tuple"], [1, 2])
        repeated = self.runner().run({"seed": 42}, self.execute, resume=False)
        self.assertFalse(repeated.reused)
        self.assertEqual(self.calls, 2)

    def test_parameters_dirty_code_and_added_deleted_data_invalidate(self):
        data, code = self.root / "inputs", self.root / "source.py"
        data.mkdir()
        item = data / "table"
        item.write_text("one")
        code.write_text("version = 1")
        runner = ExperimentRunner(self.root / "runs", inputs=[data], code_paths=[code])
        points = [runner.run({"seed": 1}, self.execute)]
        points.append(runner.run({"seed": 2}, self.execute))
        item.write_text("two")
        points.append(runner.run({"seed": 1}, self.execute))
        added = data / "new-table"
        added.write_text("new")
        points.append(runner.run({"seed": 1}, self.execute))
        item.unlink()
        points.append(runner.run({"seed": 1}, self.execute))
        code.write_text("version = 2")
        points.append(runner.run({"seed": 1}, self.execute))
        self.assertEqual(len({point.key for point in points}), 6)
        self.assertTrue(all(not point.reused for point in points))

    def test_different_callbacks_in_the_same_source_file_are_distinct(self):
        runner = self.runner()

        def other(directory):
            return {"value": 4}

        first = runner.run({}, self.execute)
        second = runner.run({}, other)
        self.assertNotEqual(first.key, second.key)
        self.assertEqual(second.result, {"value": 4})
        self.assertTrue(runner.run({}, other).reused)

    def test_failed_point_retries_without_losing_other_committed_points(self):
        runner = self.runner()
        completed = runner.run({"case": 1}, self.execute)

        def interrupted(directory):
            (directory / "partial").write_text("partial")
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            runner.run({"case": 2}, interrupted)
        self.assertTrue(runner.run({"case": 1}, self.execute).reused)
        self.assertTrue((completed.directory / "manifest.json").is_file())
        self.assertFalse(list(self.out.glob(".*-*")))
        self.assertFalse(runner.run({"case": 2}, self.execute).reused)

    def test_missing_changed_and_corrupt_outputs_rerun(self):
        runner = self.runner()
        for damage in ("missing", "changed", "result", "manifest", "status"):
            point = runner.run({"damage": damage}, self.execute)
            if damage == "missing":
                (point.directory / "trace").unlink()
            elif damage == "changed":
                (point.directory / "trace").write_bytes(b"bad")
            elif damage == "status":
                path = point.directory / "manifest.json"
                manifest = json.loads(path.read_text())
                manifest["status"] = "running"
                path.write_text(json.dumps(manifest))
            else:
                (point.directory / f"{damage}.json").write_text("broken")
            with self.subTest(damage=damage):
                self.assertFalse(runner.run({"damage": damage}, self.execute).reused)

    def test_full_hash_and_stronger_verification_rebuild_checkpoints(self):
        self.runner().run({}, self.execute)
        runner = self.runner(verify="sha256")
        point = runner.run({}, self.execute)
        self.assertFalse(point.reused)
        self.assertTrue(runner.run({}, self.execute).reused)
        path = point.directory / "trace"
        stat = path.stat()
        path.write_bytes(b"modified")
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertFalse(runner.run({}, self.execute).reused)
        self.assertTrue(runner.run({}, self.execute).reused)

    def test_input_change_during_execution_is_not_committed(self):
        source = self.root / "data"
        source.write_text("initial")
        runner = self.runner(self.root / "runs", inputs=[source])

        def execute(directory):
            source.write_text("changed")
            return {}

        with self.assertRaisesRegex(ConfigurationError, "changed during execution"):
            runner.run({}, execute)
        self.assertFalse(list((self.root / "runs").glob("*/manifest.json")))

    def test_rejects_invalid_inputs_reserved_names_and_symlinks(self):
        for values in (dict(resume=1), dict(verify="unknown")):
            with self.assertRaises(ConfigurationError):
                self.runner(**values)
        runner = self.runner()
        for parameters in ([], {"bad": float("nan")}, {"bad": object()}):
            with self.assertRaises(ConfigurationError):
                runner.run(parameters, self.execute)
        for name in ("result.json", "manifest.json", "link"):

            def execute(directory):
                if name == "link":
                    (directory / name).symlink_to(self.root / "outside")
                else:
                    (directory / name).write_text("reserved")
                return {}

            with self.subTest(name=name), self.assertRaises(ConfigurationError):
                runner.run({"name": name}, execute)

    def test_same_point_in_two_processes_executes_once(self):
        context = get_context("spawn")
        queue = context.Queue()
        workers = [
            context.Process(
                target=_shared_point,
                args=(str(self.out), str(self.code), queue),
            )
            for _ in range(2)
        ]
        for worker in workers:
            worker.start()
        try:
            observations = sorted(queue.get(timeout=20) for _ in workers)
            for worker in workers:
                worker.join(20)
                self.assertEqual(worker.exitcode, 0)
            self.assertEqual(observations, [False, True])
            self.assertEqual((self.out / "executions").read_text(), "executed\n")
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                    worker.join()
            queue.close()

    def test_killed_process_releases_lock_and_stale_staging_is_replaced(self):
        context = get_context("spawn")
        for pause in (True, False):
            queue = context.Queue()
            worker = context.Process(
                target=_killable_point,
                args=(str(self.out), str(self.code), queue, pause),
            )
            worker.start()
            try:
                self.assertEqual(queue.get(timeout=20), "staged")
                if pause:
                    worker.kill()
                else:
                    self.assertFalse(queue.get(timeout=20))
                worker.join(20)
                self.assertEqual(worker.exitcode, -9 if pause else 0)
                self.assertEqual(bool(list(self.out.glob(".*-*"))), pause)
            finally:
                if worker.is_alive():
                    worker.kill()
                    worker.join()
                queue.close()
        completed = list(self.out.glob("*/manifest.json"))
        self.assertEqual(len(completed), 1)
        self.assertEqual((completed[0].parent / "trace").read_bytes(), b"complete")


class AFExperimentRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = configuration(a_tp=2, f_tp=2, af_batch_size=1)
        cls.code = (Path(__file__),)

    def test_runner_matches_factory_timings_statistics_and_full_trace(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {"experiment": "comparison"}
            runner = AFExperimentRunner(
                self.runtime,
                root / "points",
                fallback="analytical_only",
                profile_metadata=metadata,
                code_paths=self.code,
            )
            point = runner.run(2, 8, steps=3)
            executor, rows, compute, transfer = simulate_decode(
                self.runtime, 2, 8, steps=3, fallback="analytical_only"
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
                    code_paths=self.code,
                )
                self.assertTrue(restarted.run(2, 8, steps=3).reused)

    def test_invalid_case_and_trace_format_fail_before_checkpoint(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "absent"
            with self.assertRaises(ConfigurationError):
                AFExperimentRunner(
                    self.runtime, output, trace_format="unknown", code_paths=self.code
                )
            runner = AFExperimentRunner(self.runtime, output, code_paths=self.code)
            for case in (
                dict(batch_size=0),
                dict(seed=-1),
                dict(steps=True),
                dict(fallback="bad"),
                dict(trace_format="unknown"),
                dict(trace_config={"fields": ["bad"]}),
            ):
                with self.subTest(case=case), self.assertRaises(ConfigurationError):
                    runner.run(**{**dict(batch_size=1, context_len=8), **case})
            with patch.dict(sys.modules, {"msgpack": None}):
                with self.assertRaisesRegex(ConfigurationError, r"tokensim\[trace\]"):
                    runner.run(1, 8, trace_format="msgpack")
            self.assertFalse(output.exists())

    def test_disabled_trace_modes_and_registered_table_invalidation(self):
        with TemporaryDirectory() as directory:
            runner = AFExperimentRunner(
                self.runtime,
                directory,
                fallback="analytical_only",
                code_paths=self.code,
            )
            full = runner.run(1, 8)
            expected = {**full.result["summary"], "profile": None}
            with patch.dict(sys.modules, {"msgpack": None}):
                for mode in ("off", "validate"):
                    point = runner.run(
                        1, 8, trace_format="msgpack", trace_config={"mode": mode}
                    )
                    self.assertEqual(point.result["summary"], expected)
                    self.assertFalse(list(point.directory.glob("*.trace.*")))
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
                runtime, directory, trace_config={"mode": "off"}, code_paths=self.code
            )
            runtime.hardware.register_package(package(1.0))
            first = runner.run(1, 8)
            self.assertTrue(runner.run(1, 8).reused)
            runtime.hardware.register_package(package(2.0))
            changed = runner.run(1, 8)
            self.assertFalse(changed.reused)
            self.assertNotEqual(first.key, changed.key)
