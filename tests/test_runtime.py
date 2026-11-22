"""Scheduling order, independent timing, lockstep DP, deadlines and result semantics."""

from contextlib import redirect_stdout
import heapq
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import simpy

from TokenSim.cli.lifecycle import request_observation_duration, run_simulation
from TokenSim.config.simulation import SimulationConfig
from TokenSim.config.workload_config import WorkloadConfig
from TokenSim.errors import (
    ConfigurationError,
    SimulationStateError,
    WorkloadValidationError,
)
from TokenSim.llm.llm_engine import LLMWorker, Message, Task, Worker
from TokenSim.llm.llm_request import Request
from TokenSim.llm.llm_scheduler import LLMDynamicScheduler, LLMStaticScheduler
from TokenSim.workload.loaders import (
    apply_trace_timestamp_scaling,
    load_qwen_jsonl_workload,
    load_synthetic_workload,
    load_workload,
)
from TokenSim.workload.source import LLMSource
from TokenSim.workload.user_request import UserRequest
from tests.support.cli import MOE_CLUSTER, run_quiet, simulation_args

ROOT = Path(__file__).resolve().parents[1]


class RuntimeTest(unittest.TestCase):
    args = staticmethod(simulation_args)
    run_main = staticmethod(run_quiet)

    def test_synthetic_generation_is_repeatable_with_independent_streams(self):
        args = SimpleNamespace(random_seed=7, request_count=100)
        config = WorkloadConfig(
            prefill_mean_len=20,
            prefill_range_len=8,
            decode_mean_len=1,
            decode_range_len=4,
            decode_len_distribution="exponential",
        )
        py_state, np_state = random.getstate(), np.random.get_state()
        first = load_synthetic_workload(args, config)
        self.assertEqual(random.getstate(), py_state)
        np.testing.assert_equal(np.random.get_state(), np_state)
        second = load_synthetic_workload(args, config)
        self.assertEqual(first, second)
        self.assertTrue(all(1 <= n <= 4 for n in first.decode_lens()))
        uniform = load_synthetic_workload(
            args,
            WorkloadConfig(
                prefill_mean_len=20,
                decode_mean_len=12,
                prefill_range_len=8,
                decode_range_len=8,
            ),
        )
        self.assertGreater(len({r.prefill_len - r.decode_len for r in uniform}), 1)

    def test_trace_scaling_and_ordinary_source_arrivals(self):
        with tempfile.TemporaryDirectory() as directory:
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
        records = [UserRequest(i, 8, 2, arrival_time=t) for i, t in enumerate((1, 2))]
        with self.assertRaisesRegex(WorkloadValidationError, "scaled arrival_time"):
            apply_trace_timestamp_scaling(records, trace_timestamp_scale=1e308)
        self.assertEqual([r.arrival_time for r in records], [1, 2])
        env = simpy.Environment()
        arrivals = []
        env.process(
            LLMSource(
                env,
                SimpleNamespace(add_requests=lambda _: arrivals.append(env.now)),
                [Request(i, 8, 1, 16) for i in range(2)],
                2,
                "uniform",
            )
        )
        env.run()
        self.assertEqual(arrivals, [0, 0.5])

    def test_mooncake_cache_is_shared_within_engine_and_isolated_between_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            cluster = path / "cluster.json"
            cluster.write_text(
                json.dumps(
                    dict(
                        num_workers=1,
                        networks={"node": "ethernet100Gb"},
                        worker_groups=[
                            dict(
                                role="hybrid",
                                hardware="a100_sxm_80g",
                                num_workers=1,
                                network="node",
                            )
                        ],
                        kv_cache_capacity_tokens_total=80,
                    )
                )
            )
            connector = path / "kv.json"
            connector.write_text(
                json.dumps(
                    dict(
                        kv_connector="MooncakeStoreConnector",
                        engine_id="reused-id",
                        kv_connector_extra_config={"memory_capacity_blocks": 128},
                    )
                )
            )
            dataset = path / "requests.json"
            prefixes = ([1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4])
            dataset.write_text(
                json.dumps([[64, 2, {"hash_ids": ids}] for ids in prefixes])
            )
            services = []
            for mode in ("off", "full"):
                target = path / mode
                engine = self.run_main(
                    self.args(
                        target,
                        distribution="uniform",
                        qps=1,
                        batching="paged-attn",
                        trace_mode=mode,
                        request_count=3,
                        max_num_batched_tokens=32,
                        cluster=cluster,
                        kv_transfer_config=connector,
                        dataset_path=dataset,
                        workload_type="json_pairs",
                    )
                )
                result = json.loads((target / "result_1.0.json").read_text())
                self.assertEqual(result["request_count"], 3)
                store = engine.workers[0].connector
                self.assertGreater(store.mooncake_stats.mooncake_memory_hit_tokens, 0)
                self.assertEqual(store.mooncake_stats.store_hit_count, 1)
                if mode == "full":
                    trace = json.loads(Path(result["trace"]["path"]).read_text())
                    self.assertTrue(
                        any(
                            row.get("args", {}).get("operation") == "kv_transfer"
                            for row in trace["traceEvents"]
                        )
                    )
                services.append(store.service)
            self.assertIsNot(services[0], services[1])

    def test_model_geometry_routing_and_dp_lockstep(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            workload = path / "workload.json"
            workload.write_text(
                json.dumps(
                    {
                        "qps": 9,
                        "prefill_mean_len": 16,
                        "decode_mean_len": 2,
                        "first_token_latency": {"max": 7},
                    }
                )
            )
            engine = self.run_main(
                simulation_args(
                    path,
                    model="moe-toy",
                    cluster=MOE_CLUSTER,
                    workload_config=workload,
                    request_count=2,
                    prefill_mean_len=None,
                    decode_mean_len=None,
                    moe_routing_distribution="hot",
                    moe_hot_experts="0,1",
                    moe_hot_expert_fraction=0.8,
                )
            )
            self.assertEqual(engine.moe_config.routing.hot_experts, [0, 1])
            self.assertEqual(engine.workload_config.qps, 9)
            workload.write_text(json.dumps({"hidden_size": 256}))
            with self.assertRaisesRegex(ConfigurationError, "hidden_size"):
                WorkloadConfig.from_file(workload)
            lockstep_dir = path / "lockstep"
            dp_engine = self.run_main(
                simulation_args(
                    lockstep_dir,
                    model="moe-toy",
                    cluster=MOE_CLUSTER,
                    batching="paged-attn",
                    dp_lockstep=True,
                )
            )
            dp_result = json.loads((lockstep_dir / "result_inf.json").read_text())
            self.assertEqual(dp_result["termination_reason"], "completed")
            self.assertEqual(len(dp_engine.timing_recorder.timing.time), 8)
            self.assertIsNotNone(dp_result["dp_step_stats"])

    def test_equal_priority_messages_and_scheduler_limits(self):
        messages = [Message(Task.ADD, sender_id=i) for i in range(40)]
        heapq.heapify(messages)
        messages.append(Message(Task.CANCEL, sender_id=-1))
        heapq.heapify(messages)
        self.assertEqual(heapq.heappop(messages).task, Task.CANCEL)
        self.assertEqual(
            [heapq.heappop(messages).sender_id for _ in range(40)], list(range(40))
        )
        for cls in (LLMDynamicScheduler, LLMStaticScheduler):
            scheduler = cls(max_num_seqs=2)
            scheduler.add_requests([Request(i, 16 + i, 3, 16) for i in range(5)])
            self.assertEqual([r.id for r in scheduler.schedule()[0]], [0, 1])
            scheduler.running.clear()
            self.assertEqual([r.id for r in scheduler.schedule()[0]], [2, 3])
        with self.assertRaises(ConfigurationError):
            SimulationConfig(max_num_seqs=0)

    def test_main_batch_modes_count_output_tokens_and_keep_timing_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            engines = []
            for batching in ("dynamic", "static", "paged-attn"):
                path = Path(directory) / batching
                engine = self.run_main(self.args(path, "--batching", batching))
                result = json.loads((path / "result_inf.json").read_text())
                self.assertEqual(result["request_count"], 8)
                self.assertEqual(result["notdone"], [])
                self.assertAlmostEqual(
                    result["output_token_ps"] * result["duration"], 24
                )
                engines.append(engine)
            self.assertIsNot(
                engines[0].timing_recorder.timing, engines[1].timing_recorder.timing
            )

    def test_pd_dynamic_batching_charges_every_transfer_plan(self):
        issued, charged = [], []
        original_send, original_prepare = Worker.send_task, LLMWorker.prepare_step

        def send(worker, target, task, requests=None, metadata=None):
            if task == Task.ADD and metadata is not None:
                issued.extend(plan.latency for plan in metadata.loads)
            return original_send(worker, target, task, requests, metadata)

        def prepare(worker):
            running, latency = original_prepare(worker)
            charged.append(latency)
            return running, latency

        with tempfile.TemporaryDirectory() as directory:
            args = self.args(
                directory,
                "--request_count",
                "48",
                "--max_num_seqs",
                "4",
                "--cluster",
                str(ROOT / "data/clusters/8_a100/p2d5.json"),
            )
            with (
                patch.object(Worker, "send_task", send),
                patch.object(LLMWorker, "prepare_step", prepare),
            ):
                engine = self.run_main(args)
            self.assertGreater(sum(issued), 0)
            self.assertAlmostEqual(sum(issued), sum(charged))
            self.assertEqual(len(engine.timing_recorder.timing.time), 48)

    def test_latency_targets_are_reported_only_when_configured(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            engine = self.run_main(self.args(path))
            self.assertNotIn("slo", json.loads((path / "result_inf.json").read_text()))
            first_token = engine.timing_recorder.timing.prefill_time
            bound = sorted(first_token)[len(first_token) // 2]
            targets = path / "workload.json"
            targets.write_text(
                json.dumps(
                    {
                        "first_token_latency": {"max": bound, "p50": 1e9},
                        "decode_token_latency": {"p99": 0.0},
                    }
                )
            )
            self.run_main(self.args(path, "--workload_config", str(targets)))
            result = json.loads((path / "result_inf.json").read_text())
            slo = result["slo"]
            self.assertEqual(
                slo["first_token_latency"]["met"], {"max": False, "p50": True}
            )
            self.assertEqual(
                slo["goodput_requests"], sum(t <= bound for t in first_token)
            )

    def test_cutoffs_failures_and_post_workload_drains(self):
        for mode in ("write", "audit"):
            env = simpy.Environment()
            connector = SimpleNamespace(
                async_offload=True, loads=[], pending_writes=[], write_jobs=0
            )
            engine = SimpleNamespace(
                env=env,
                failure_event=env.event(),
                raise_if_failed=lambda: None,
                request_workers=[
                    SimpleNamespace(connector=connector, scheduler=SimpleNamespace())
                ],
                cache_controller=None,
            )
            req = Request(0, 8, 1, 16)

            def storage(env=env, connector=connector):
                yield env.timeout(5)
                connector.write_jobs = 0

            def simulate(env=env, req=req, connector=connector, mode=mode):
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
            self.assertEqual((status, env.now), ("sim_time_limit", 2))
            self.assertEqual(request_observation_duration([req], env.now), 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.run_main(self.args(path, "--sim_time", "0.000001"))
            result = json.loads((path / "result_inf.json").read_text())
            self.assertEqual(result["termination_reason"], "sim_time_limit")
            self.assertTrue(result["notdone"])
            with patch(
                "TokenSim.llm.llm_engine.LLMEngine.raise_if_failed",
                side_effect=SimulationStateError("conservation failed"),
            ):
                with self.assertRaises(SimulationStateError):
                    self.run_main(self.args(path))
            self.assertTrue((path / "failure.json").is_file())

    def test_agentx_cutoff_and_qwen_jsonl_unique_request_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(
                directory,
                "--batching",
                "paged-attn",
                "--workload_type",
                "agentx_weka",
                "--dataset_path",
                str(ROOT / "tests/fixtures/agentx_weka.jsonl"),
                "--block_size",
                "64",
                "--agentx_profile_duration",
                "2",
                "--sim_time",
                "0.001",
            )
            self.run_main(args)
            result = json.loads((Path(directory) / "agentx_c1.json").read_text())
            self.assertEqual(result["termination_reason"], "sim_time_limit")
            path = Path(directory) / "trace.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps({"request_id": i, "input_length": 8, "output_length": 2})
                    for i in (1, 1, 2, 0, 0)
                )
            )
            with redirect_stdout(io.StringIO()):
                requests = load_qwen_jsonl_workload(str(path), 5)
            self.assertEqual(len({r.request_id for r in requests}), 5)
