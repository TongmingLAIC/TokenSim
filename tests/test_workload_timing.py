"""Trace selection and simulation time boundaries."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest

import simpy

from TokenSim.cli.lifecycle import request_observation_duration, run_simulation
from TokenSim.config.workload_config import WorkloadConfig
from TokenSim.errors import WorkloadValidationError
from TokenSim.llm.llm_request import Request
from TokenSim.workload.loaders import (
    apply_trace_timestamp_scaling,
    load_synthetic_workload,
    load_workload,
)
from TokenSim.workload.source import LLMSource
from TokenSim.workload.user_request import UserRequest


class WorkloadTimingTest(unittest.TestCase):
    def test_selected_trace_span_sets_qps_without_repeating_rows(self):
        with TemporaryDirectory() as directory:
            trace = Path(directory) / "requests.json"
            trace.write_text(
                json.dumps([[8, 2, {"arrival_time": t}] for t in (1, 10, 20)])
            )
            args = SimpleNamespace(
                workload_type="json_pairs",
                dataset_path=str(trace),
                dataset_skip_count=1,
                request_count=100,
                trace_target_qps=1,
            )
            selected = load_workload(args, WorkloadConfig())
            self.assertEqual([r.arrival_time for r in selected], [1.0, 2.0])
            with self.assertRaisesRegex(WorkloadValidationError, "non-burst"):
                load_workload(args, WorkloadConfig(distribution="burst"))
            for count in (0, -1):
                with (
                    self.subTest(count=count),
                    self.assertRaisesRegex(WorkloadValidationError, "request_count"),
                ):
                    args.request_count = count
                    load_workload(args, WorkloadConfig())

    def test_scaling_rejects_a_nonfinite_result_without_mutating_input(self):
        records = [UserRequest(i, 8, 2, arrival_time=t) for i, t in enumerate((1, 2))]
        with self.assertRaisesRegex(WorkloadValidationError, "scaled arrival_time"):
            apply_trace_timestamp_scaling(records, trace_timestamp_scale=1e308)
        self.assertEqual([r.arrival_time for r in records], [1, 2])
        records[1].arrival_time = 1
        with self.assertRaisesRegex(WorkloadValidationError, "positive interval"):
            apply_trace_timestamp_scaling(records, trace_target_qps=1)

    def test_uniform_input_and_output_lengths_use_independent_seeded_streams(self):
        args = SimpleNamespace(request_count=100, random_seed=1)
        config = WorkloadConfig(
            prefill_mean_len=20,
            decode_mean_len=12,
            prefill_range_len=8,
            decode_range_len=8,
        )
        first = load_synthetic_workload(args, config)
        self.assertEqual(first, load_synthetic_workload(args, config))
        self.assertGreater(len({r.prefill_len - r.decode_len for r in first}), 1)

    def test_ordinary_source_stops_after_last_arrival(self):
        env = simpy.Environment()
        requests = [Request(i, 8, 1, 16) for i in range(2)]
        arrival_times = []
        engine = SimpleNamespace(
            add_requests=lambda batch: arrival_times.append(env.now)
        )
        env.process(LLMSource(env, engine, requests, 2, "uniform"))
        env.run()
        self.assertEqual(arrival_times, [0, 0.5])
        self.assertEqual(env.now, 0.5)


class LifecycleTest(unittest.TestCase):
    def test_hard_cutoff_applies_to_accepted_writes_and_post_workload_audit(self):
        for mode in ("write", "audit"):
            with self.subTest(mode=mode):
                env = simpy.Environment()
                connector = SimpleNamespace(
                    async_offload=True, loads=[], pending_writes=[], write_jobs=0
                )
                engine = SimpleNamespace(
                    env=env,
                    failure_event=env.event(),
                    raise_if_failed=lambda: None,
                    request_workers=[
                        SimpleNamespace(
                            connector=connector, scheduler=SimpleNamespace()
                        )
                    ],
                    cache_controller=None,
                )
                req = Request(0, 8, 1, 16)
                finished = []

                def storage():
                    yield env.timeout(5)
                    finished.append(env.now)
                    connector.write_jobs = 0

                def simulate():
                    req.arrive(env)
                    yield env.timeout(1)
                    req.step(env, 1, 1)
                    if mode == "write":
                        connector.write_jobs = 1
                        env.process(storage())

                if mode == "audit":
                    connector.post_workload_drain = storage
                replay = SimpleNamespace(
                    cancellation_timeout=10, grace_period=0, issue_window_end_at=1
                )
                status = run_simulation(
                    engine, env.process(simulate()), replay=replay, sim_time=2
                )
                self.assertEqual(status, "sim_time_limit")
                self.assertEqual(env.now, 2)
                self.assertEqual(finished, [])
                self.assertEqual(request_observation_duration([req], env.now), 1)
