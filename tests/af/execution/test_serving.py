"""Serving integration, decode-only AF, mixed batches, budgets and CLI export."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import simpy

from tests.support.cli import run_quiet, simulation_args
from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.config.config import KVTransferConfig, ParallelConfig
from TokenSim.config.simulation import SimulationConfig
from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.llm.engine.af_worker import run_worker_compute
from TokenSim.llm.llm_engine import LLMEngine, LLMWorker
from TokenSim.llm.llm_request import Request
from TokenSim.reporting import get_af_stats
from tests.af.support.config import _cluster, _cluster_dict, _with_prefill_pool
from tests.af.support.execution import (
    FixedComputeProvider,
    FixedTransferProvider,
    _DTypeCheckingComputeProvider,
    _build_engine,
    _disabled_cluster,
    _run_request,
    _typed_cluster,
    af_test_hardware,
)
from tests.af.support.model import _moe


class AFEngineIntegrationTest(unittest.TestCase):
    def test_decode_only_connects_runtime_plan_dtypes_and_providers(self):
        compute = _DTypeCheckingComputeProvider()
        transfer = FixedTransferProvider(0.0001, 0.0001)
        env, engine, roofline = _build_engine(_typed_cluster(), compute, transfer)

        plan = engine.af_runtime_plan
        self.assertIsNotNone(plan)
        self.assertIs(engine.af_executor.runtime_plan, plan)
        self.assertEqual(
            {
                "weight_dtype": plan.effective_dtypes.weight_dtype,
                "activation_dtype": plan.effective_dtypes.activation_dtype,
                "kv_cache_dtype": plan.effective_dtypes.kv_cache_dtype,
                "a_to_f_dtype": plan.effective_dtypes.a_to_f_dtype,
                "f_to_a_dtype": plan.effective_dtypes.f_to_a_dtype,
            },
            {
                "weight_dtype": "fp8",
                "activation_dtype": "bf16",
                "kv_cache_dtype": "fp32",
                "a_to_f_dtype": "fp8",
                "f_to_a_dtype": "fp16",
            },
        )
        self.assertEqual(
            [(layer.layer_id, layer.layer_type) for layer in plan.layers],
            [(0, "dense")],
        )
        self.assertEqual(len(compute.validated_contracts), 1)
        self.assertEqual(compute.validated_contracts[0][0], "TestModel")

        decode_manager = engine.decode_workers.workers[0].scheduler.block_manager
        with patch.object(
            decode_manager,
            "commit_input_cache",
            wraps=decode_manager.commit_input_cache,
        ) as commit_input_cache:
            request = _run_request(env, engine)

        self.assertTrue(request.is_done)
        self.assertEqual(request.generation_idx, 3)
        self.assertEqual(request.step_calls, 3)
        self.assertEqual(commit_input_cache.call_count, 2)
        self.assertEqual(engine.af_iteration_count, 2)
        self.assertEqual({query.phase for query in compute.queries}, {"decode"})
        self.assertEqual({query.phase for query in transfer.queries}, {"decode"})
        self.assertTrue({query.stage for query in compute.queries})
        for query in compute.queries:
            self.assertEqual(query.weight_dtype, "fp8")
            self.assertEqual(query.activation_dtype, "bf16")
            self.assertEqual(query.kv_cache_dtype, "fp32")
            self.assertEqual(query.a_to_f_dtype, "fp8")
            self.assertEqual(query.f_to_a_dtype, "fp16")
        for query in transfer.queries:
            self.assertEqual(query.a_to_f_dtype, "fp8")
            self.assertEqual(query.f_to_a_dtype, "fp16")
            if query.direction == "a_to_f":
                self.assertEqual(query.dtype, "fp8")
            elif query.direction == "f_to_a":
                self.assertEqual(query.dtype, "fp16")
        self.assertEqual(
            get_af_stats(engine)["af_phase_fidelity"],
            {
                "prefill": "prefill_aggregate",
                "decode": "decode_af_rank_local",
            },
        )

    def test_prefill_uses_complete_backend_and_never_submits_af(self):
        compute, transfer = _DTypeCheckingComputeProvider(), FixedTransferProvider(
            0.0001, 0.0001
        )
        env, engine, _ = _build_engine(_typed_cluster(), compute, transfer)
        original = LLMWorker.dynamic_batch
        prefill_calls = []

        def observe(worker, requests):
            prefill_calls.append(tuple(request.is_prefill for request in requests))
            return original(worker, requests)

        with patch.object(LLMWorker, "dynamic_batch", observe):
            request = _run_request(env, engine, decode_len=2)
        self.assertTrue(request.is_done)
        self.assertEqual(prefill_calls, [(True,)])
        self.assertEqual(engine.af_iteration_count, 1)
        self.assertEqual({query.phase for query in transfer.queries}, {"decode"})

    def test_warmup_iterations_are_excluded_from_profile_count(self):
        env, engine, _ = _build_engine(
            _typed_cluster(),
            FixedComputeProvider(0.001, 0.001),
            FixedTransferProvider(0.0001, 0.0001),
        )
        for index, phase in enumerate(("warmup", "profile")):
            req = Request(index, 8, 2, 4, measurement_phase=phase)
            req.arrive(env)
            engine.add_requests_burst([req])
            env.run(until=req.completion_event)
            engine.raise_if_failed()
            self.assertTrue(req.is_done)
            if phase == "warmup":
                engine.reset_profile_stats()
        stats = get_af_stats(engine)
        self.assertEqual(stats["af_warmup_iteration_count"], 1)
        self.assertEqual(stats["af_iteration_count"], 1)
        self.assertIsNone(stats["af_trace"])

    def test_af_disabled_keeps_complete_backend_and_creates_no_af_state(self):
        env, engine, roofline = _build_engine(_disabled_cluster())

        request = _run_request(env, engine, decode_len=2)

        self.assertTrue(request.is_done)
        self.assertEqual(request.step_calls, 2)
        self.assertIsNone(engine.af_runtime_plan)
        self.assertIsNone(engine.af_executor)
        self.assertEqual(
            get_af_stats(engine),
            {
                "af_enabled": False,
                "af_iteration_count": 0,
                "af_trace": None,
                "af_runtime_state": None,
                "af_phase_fidelity": None,
                "af_latency_providers": None,
                "af_operator_stats": None,
            },
        )


class AFServingCorrectnessTest(unittest.TestCase):
    def make_engine(self, *, utilization=1.0, capacity=None, total=None, replicas=1):
        hardware = af_test_hardware(capacity_gib=0.1)
        cluster = _with_prefill_pool(
            _cluster(a_tp=1, a_dp=replicas, f_tp=1, f_dp=1, f_ep=1)
        )
        cluster.kv_cache_capacity_tokens_per_dp_rank = capacity
        cluster.kv_cache_capacity_tokens_total = total
        return LLMEngine(
            env=simpy.Environment(),
            kv_transfer_config=KVTransferConfig.default(),
            model_spec=hardware.model("TestModel"),
            cluster_config=cluster,
            hardware=hardware,
            config=SimulationConfig(
                block_size=4, batching="paged-attn", gpu_memory_utilization=utilization
            ),
            af_compute_provider=FixedComputeProvider(0.001, 0.001),
            af_transfer_provider=FixedTransferProvider(0.001, 0.001),
        )

    def test_memory_utilization_limits_every_physical_budget(self):
        full = self.make_engine()
        half = self.make_engine(utilization=0.5)
        self.assertLess(
            half.request_workers[0].cache_config.num_gpu_blocks,
            full.request_workers[0].cache_config.num_gpu_blocks // 2,
        )
        for original, limited in zip(
            full.af_runtime_plan.physical_states, half.af_runtime_plan.physical_states
        ):
            self.assertEqual(limited.capacity_bytes, original.capacity_bytes // 2)

    def test_kv_limit_cannot_manufacture_physical_memory(self):
        full = self.make_engine()
        capped = self.make_engine(capacity=10**12)
        self.assertEqual(
            full.request_workers[0].cache_config.num_gpu_blocks,
            capped.request_workers[0].cache_config.num_gpu_blocks,
        )

    def test_lockstep_is_rejected_before_starting_workers(self):
        for ranks in (1, 2):
            with self.subTest(ranks=ranks):
                env = simpy.Environment()
                hardware = af_test_hardware(capacity_gib=0.1)
                with self.assertRaisesRegex(ConfigurationError, "dp_lockstep.*AF"):
                    LLMEngine(
                        env=env,
                        kv_transfer_config=KVTransferConfig.default(),
                        model_spec=hardware.model("TestModel"),
                        cluster_config=_cluster(
                            a_tp=1, a_dp=ranks, f_tp=1, f_dp=1, f_ep=1
                        ),
                        hardware=hardware,
                        parallel_config=ParallelConfig(data_parallel_size=ranks),
                        config=SimulationConfig(dp_lockstep=True),
                    )
                self.assertEqual(len(env._queue), 0)


class MixedServingTest(unittest.TestCase):
    def engine(self, *, capacity=None, compute=None, moe=False, af_block_size=8):
        cluster = _with_prefill_pool(
            ClusterConfig(**_cluster_dict(af_block_size=af_block_size))
        )
        cluster.kv_cache_capacity_tokens_per_dp_rank = capacity
        hardware = af_test_hardware(capacity_gib=0.1, moe=_moe() if moe else None)
        env = simpy.Environment()
        compute = compute or FixedComputeProvider(0.001, 0.002)
        engine = LLMEngine(
            env=env,
            kv_transfer_config=KVTransferConfig.default(),
            model_spec=hardware.model("TestModel"),
            cluster_config=cluster,
            hardware=hardware,
            config=SimulationConfig(
                block_size=4, batching="paged-attn", max_num_seqs=2
            ),
            af_compute_provider=compute,
            af_transfer_provider=FixedTransferProvider(0.001, 0.001),
        )
        return env, engine, compute

    def test_mixed_batch_prefill_uses_complete_backend_and_decode_uses_af(self):
        env, engine, compute = self.engine(moe=True)
        worker = engine.decode_workers.workers[0]
        prefill, decode = Request(1, 8, 4, 4), Request(2, 8, 3, 4)
        decode.generation_idx = 1
        with patch.object(worker, "dynamic_batch", return_value=0.001) as complete:
            env.run(until=env.process(run_worker_compute(worker, [prefill, decode])))
        self.assertEqual(complete.call_count, 1)
        self.assertEqual(complete.call_args.args[0], [prefill])
        self.assertEqual({query.phase for query in compute.queries}, {"decode"})
        self.assertEqual(engine.af_iteration_count, 1)

    def test_memory_pressure_recomputes_decode_context_through_af(self):
        env, engine, compute = self.engine(capacity=16, af_block_size=16)
        requests = [Request(i, 8, 8, 4) for i in (1, 2)]
        original = LLMWorker.dynamic_batch

        def complete(worker, running):
            self.assertTrue(all(request.is_prefill for request in running))
            return original(worker, running)

        with patch.object(LLMWorker, "dynamic_batch", complete):
            from TokenSim.llm.engine.messages import Task

            worker = engine.decode_workers.workers[0]
            for req in requests:
                req.generation_idx = 1
                req.arrive(env)
            worker.send_task(worker, Task.ADD, requests)
            env.run()
        engine.raise_if_failed()
        self.assertTrue(all(req.is_done for req in requests))
        self.assertGreater(sum(req.recomputed_tokens_total for req in requests), 0)
        self.assertTrue(
            any(
                q.phase == "prefill" and q.context_lengths[0] > 8
                for q in compute.queries
            )
        )
        self.assertGreater(engine.af_iteration_count, 0)
        self.assertEqual(engine.af_executor.active_iterations, {})

    def test_failed_decode_cancels_complete_prefill_and_releases_resources(self):
        class FailDecode(FixedComputeProvider):
            def estimate(self, query):
                raise ValueError("decode failed")

        env, engine, _ = self.engine(compute=FailDecode(0.001, 0.002))
        decode, prefill = Request(1, 8, 4, 4), Request(2, 8, 4, 4)
        decode.generation_idx = 1
        worker = engine.decode_workers.workers[0]
        with patch.object(worker, "dynamic_batch", return_value=0.1):
            process = env.process(run_worker_compute(worker, [decode, prefill]))
            with self.assertRaisesRegex(SimulationStateError, "decode failed"):
                env.run(until=process)
        self.assertEqual(engine.af_executor.active_iterations, {})
        for resource in engine.af_resources.compute_slots.values():
            self.assertEqual(resource.count, 0)
            self.assertFalse(resource.queue)


class AFCLIIntegrationTest(unittest.TestCase):
    def test_current_cli_runs_af_and_exports_runtime_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            cluster_path = path / "cluster.json"
            from pydantic import TypeAdapter

            cluster_path.write_text(
                TypeAdapter(ClusterConfig)
                .dump_json(
                    _with_prefill_pool(ClusterConfig(**_cluster_dict())),
                    exclude_none=True,
                )
                .decode()
            )
            args = simulation_args(
                path / "results",
                cluster=cluster_path,
                model="TestModel",
                batching="paged-attn",
                block_size=4,
                qps=1,
                request_count=2,
                prefill_mean_len=8,
                decode_mean_len=2,
                prefill_range_len=0,
                decode_range_len=0,
            )
            args.af_compute_provider = FixedComputeProvider(0.001, 0.001)
            args.af_transfer_provider = FixedTransferProvider(0.001, 0.001)
            with patch(
                "TokenSim.cli.run.HardwareContext.load", return_value=af_test_hardware()
            ):
                engine = run_quiet(args)
            result = json.loads(
                next((path / "results").glob("result_*.json")).read_text()
            )
            self.assertTrue(result["af_enabled"])
            self.assertGreater(result["af_iteration_count"], 0)
            self.assertIsNotNone(result["af_runtime_state"])
            self.assertEqual(result["notdone"], [])
            self.assertEqual(len(engine.timing_recorder.timing.request_time), 2)
