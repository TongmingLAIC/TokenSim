from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import simpy

from TokenSim.af.execution.model import ModelAFExecutor
from TokenSim.af.factory import build_model_runtime, simulate_decode
from TokenSim.af.planning.batching import AFBlockBuilder
from TokenSim.af.planning.planner import ParallelExecutionPlanner
from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.config.config import KVTransferConfig
from TokenSim.config.simulation import SimulationConfig
from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.hardware import HardwareContext
from TokenSim.llm.llm_engine import LLMEngine
from TokenSim.llm.llm_request import Request
from scripts.run_deepseek_mla import configuration, simulate
from tests.af.support.config import _cluster_dict, _with_prefill_pool
from tests.af.support.execution import _run_request, af_test_hardware
from tests.af.support.model import _moe
from tests.af.support.observations import observe_services


class ModelExecutorTest(unittest.TestCase):
    def test_script_trace_off_matches_services(self):
        config = configuration(a_tp=2, f_tp=2, af_batch_size=1)
        results = []
        for trace in (True, False):
            with observe_services() as services:
                executor, rows, _, _ = simulate(
                    config, 2, 8, steps=2, trace=trace, fallback="analytical_only"
                )
            results.append((rows, services))
            self.assertEqual(bool(executor.trace.events), trace)
        self.assertEqual(results[0], results[1])

    def test_boundaries_use_declared_lanes_and_separated_timing(self):
        config = configuration(a_tp=2, f_tp=2, af_batch_size=1)
        config.doc["af"]["transfer_topology_mode"] = "flat_link"
        config.doc["af"]["communication_timing"] = dict(
            enabled=True, host_submit_us=5, host_completion_us=2
        )
        config = build_model_runtime(
            config.doc, hardware=config.hardware, model=config.model
        )
        executor, _, _, _ = simulate_decode(config, 2, 8, fallback="analytical_only")
        events = [e for e in executor.trace.events if e.layer_id == -1]
        collectives = [e for e in events if e.event_type == "collective"]
        self.assertEqual(
            {e.operation: e.resource_id for e in collectives},
            {
                "all_reduce": "link:af-fabric:a_to_f",
                "all_gather": "link:af-fabric:f_to_a",
            },
        )
        self.assertTrue(any(e.event_type == "host_submit" for e in events))
        self.assertTrue(any(e.event_type == "synchronize" for e in events))
        executor.trace.validate_invariants(0)

    def test_f_router_metadata_uses_endpoint_provider(self):
        config = configuration(a_tp=2, f_tp=2, af_batch_size=2)
        config.doc["af"]["router_location"] = "f"
        config = build_model_runtime(
            config.doc, hardware=config.hardware, model=config.model
        )
        executor, _, _, _ = simulate_decode(config, 2, 8, fallback="analytical_only")
        metadata = [
            event
            for event in executor.trace.events
            if event.operation == "af_router_metadata"
        ]
        self.assertTrue(metadata)
        self.assertTrue(
            all(
                event.details["timing_scope"] == "transport_service"
                for event in metadata
            )
        )
        executor.trace.validate_invariants(0)

    def test_collective_hosts_submit_independently_under_contention(self):
        config = configuration(a_tp=2, f_tp=2, af_batch_size=1)
        config.doc["af"]["communication_timing"] = dict(
            enabled=True, host_submit_us=5, host_completion_us=2
        )
        config = build_model_runtime(
            config.doc, hardware=config.hardware, model=config.model
        )
        executor, _, _, _ = simulate_decode(config, 1, 8, fallback="analytical_only")
        executor.trace.clear()
        env = executor.env
        start = env.now
        block = SimpleNamespace(
            iteration_id=99, block_id=0, phase="decode", request_count=1, token_count=1
        )
        step = SimpleNamespace(
            operation="all_reduce",
            direction="collective",
            logical_stage_id=None,
            leg_id=None,
            source_domain="a-domain",
            target_domain="a-domain",
            layer_type=None,
        )
        host = executor.resources.host_resources_for(("a-pool:1",))[0][1]
        held = host.request()

        def release():
            yield env.timeout(10e-6)
            host.release(held)

        env.process(release())
        env.process(
            executor._run_communication_host_stage(
                block,
                -1,
                step,
                "test-submit",
                (),
                ("a-pool:0", "a-pool:1"),
                5e-6,
                "host_submit",
            )
        )
        env.run()
        events = [
            event
            for event in executor.trace.events
            if event.event_type == "host_submit"
        ]
        self.assertEqual(len(events), 2)
        self.assertAlmostEqual(events[0].start_at, start)
        self.assertAlmostEqual(events[0].end_at, start + 5e-6)
        self.assertAlmostEqual(events[1].start_at, start + 10e-6)
        self.assertAlmostEqual(env.now, start + 15e-6)

    def test_boundary_failure_cancels_other_blocks_and_releases_resources(self):
        executor, _, _, _ = simulate(
            configuration(a_tp=2, f_tp=2, af_batch_size=1),
            1,
            8,
            fallback="analytical_only",
        )
        original = executor.compute_provider.estimate

        def fail(query):
            if query.operation == "lm_head" and query.block_id == 0:
                raise ValueError("missing boundary shape")
            return original(query)

        requests = [
            SimpleNamespace(
                id=i, is_prefill=False, prefill_len=8, context_len=8, generation_idx=0
            )
            for i in range(3)
        ]
        iteration = executor.submit(0, requests)
        with patch.object(executor.compute_provider, "estimate", fail):
            executor.env.process(executor.run_iteration(iteration))
            with self.assertRaisesRegex(
                SimulationStateError, "lm_head.*missing boundary shape"
            ):
                executor.env.run()
        self.assertIsNone(iteration.end_at)
        self.assertFalse(executor.active_iterations)
        for resource in (
            *executor.resources.compute_slots.values(),
            *executor.resources.communication_resources.values(),
        ):
            self.assertEqual(resource.count, 0)
            self.assertFalse(resource.queue)

    def test_standard_engine_runs_full_model_and_latent_kv(self):
        hardware = HardwareContext.load("data")
        doc = json.loads(Path("data/clusters/deepseek_v3/1a32f.json").read_text())
        doc["af"]["communication_domains"] = {
            "node_size": 8,
            "node_count": 5,
            "placement_strategy": "dedicated",
        }
        config = build_model_runtime(
            doc, hardware=hardware, model=hardware.model("deepseek-v3-671b-mla")
        )
        from dataclasses import replace
        from TokenSim.hardware.device.sourced_value import SourcedValue

        cluster = _with_prefill_pool(config.cluster)
        base = hardware.device(cluster.worker_groups[-1].hardware)
        hardware.devices.add(
            replace(
                base,
                device_id="test-prefill-memory",
                display_name="test-prefill-memory",
                aliases=(),
                memory_capacity_bytes=SourcedValue(
                    2048 * 2**30, "test:virtual-prefill", "D"
                ),
            )
        )
        cluster.worker_groups[-1].hardware = "test-prefill-memory"
        env = simpy.Environment()
        engine = LLMEngine(
            env=env,
            kv_transfer_config=KVTransferConfig.default(),
            model_spec=config.model,
            cluster_config=cluster,
            hardware=hardware,
            config=SimulationConfig(
                trace={"mode": "full"},
                block_size=16,
                batching="paged-attn",
                prefill_worker_pool_type="round_robin",
                decode_worker_pool_type="round_robin",
                max_num_seqs=8,
                latency_fallback="analytical_only",
            ),
        )
        request = _run_request(env, engine, decode_len=2)
        self.assertTrue(request.is_done)
        self.assertIsInstance(engine.af_executor, ModelAFExecutor)
        events = engine.af_executor.trace.events
        self.assertEqual(
            {e.layer_id for e in events if e.layer_id >= 0}, set(range(61))
        )
        self.assertEqual(
            {
                e.operation
                for e in events
                if e.layer_id == -1 and e.event_type == "compute"
            },
            {"embedding_lookup", "final_norm", "lm_head", "token_selection"},
        )
        self.assertTrue(
            all(layer.mla.cache_width == 576 for layer in config.plan.layers)
        )
        self.assertFalse(any(e.stage == "router" and e.layer_id < 3 for e in events))
        self.assertTrue(any(e.stage == "shared_expert" for e in events))

    def test_non_v3_mla_metadata_uses_the_same_execution_path(self):
        from dataclasses import replace

        config = configuration()
        model = replace(
            config.model,
            model_id="other-mla",
            num_layers=2,
            moe=replace(config.model.moe, first_k_dense_replace=1, num_moe_layers=1),
        )
        config = build_model_runtime(config.doc, hardware=config.hardware, model=model)
        executor, _, _, _ = simulate_decode(config, 1, 8, fallback="analytical_only")
        events = executor.trace.events
        self.assertFalse(any(e.layer_id == 0 and e.stage == "router" for e in events))
        self.assertTrue(any(e.layer_id == 1 and e.stage == "router" for e in events))
        attention = next(e for e in events if e.stage == "attention")
        self.assertGreater(len(attention.details["components"]), 1)
        self.assertAlmostEqual(
            attention.service_duration,
            sum(item["latency_us"] for item in attention.details["components"]) / 1e6,
        )


class ModelBoundaryAndQueryGeometryTest(unittest.TestCase):
    def engine(self, *, layers=1, moe=None):
        cluster = _with_prefill_pool(ClusterConfig(**_cluster_dict()))
        cluster.af.latency_provider = "operator_table"
        hardware = af_test_hardware(layers=layers, moe=moe, capacity_gib=0.1)
        env = simpy.Environment()
        return env, LLMEngine(
            env=env,
            kv_transfer_config=KVTransferConfig.default(),
            model_spec=hardware.model("TestModel"),
            cluster_config=cluster,
            hardware=hardware,
            config=SimulationConfig(
                trace={"mode": "full"}, latency_fallback="analytical_only"
            ),
        )

    def test_decode_layer_kinds_boundaries_and_prefill_rejection(self):
        env, engine = self.engine(
            layers=2, moe=replace(_moe(), first_k_dense_replace=1, num_shared_experts=1)
        )
        req = Request(1, 8, 3, 4)
        req.generation_idx = 1
        executor = engine.af_executor
        iteration = executor.submit(0, [req])
        env.run(until=env.process(executor.run_iteration(iteration)))
        events = executor.trace.events
        dense, experts = (
            {
                e.stage
                for e in events
                if e.layer_id == layer and e.event_type == "compute"
            }
            for layer in (0, 1)
        )
        self.assertIn("dense_ffn", dense)
        self.assertTrue(
            {"router", "routed_expert", "shared_expert", "weighted_combine"} <= experts
        )
        boundaries = [
            e.operation
            for e in events
            if e.layer_id == -1 and e.event_type == "compute"
        ]
        self.assertEqual(
            boundaries, ["embedding_lookup", "final_norm", "lm_head", "token_selection"]
        )
        prefill = Request(2, 16, 3, 4)
        prefill.scheduled_tokens = 8
        with self.assertRaisesRegex(ConfigurationError, "only accepts decode"):
            executor.submit(0, [prefill])

    def test_recompute_chunk_and_cached_suffix_query_geometry(self):
        link = ClusterConfig(**_cluster_dict()).af.links[0]
        req = Request(1, 64, 3, 4)
        req.generation_idx = 1
        req.prepare_recompute()
        req.scheduled_tokens = 8
        first = AFBlockBuilder().build(0, [req], 8, 0, link).blocks[0]
        self.assertEqual(
            (first.context_lengths, first.query_offsets, first.logits_rows),
            ((8,), (0,), 0),
        )
        self.assertEqual(
            ParallelExecutionPlanner._block_tokens(first),
            tuple((1, i) for i in range(8)),
        )

        suffix = Request(2, 64, 3, 4)
        suffix.start_context_build(56)
        suffix.generation_idx = 1
        suffix.prepare_recompute()
        suffix.num_computed_tokens = 57
        suffix.scheduled_tokens = 8
        second = AFBlockBuilder().build(0, [suffix], 8, 0, link).blocks[0]
        self.assertEqual(
            (second.context_lengths, second.query_offsets, second.logits_rows),
            ((65,), (57,), 1),
        )
        self.assertEqual(
            ParallelExecutionPlanner._block_tokens(second),
            tuple((2, i) for i in range(57, 65)),
        )
