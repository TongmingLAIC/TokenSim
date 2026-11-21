"""Model boundaries, layer selection and request geometry."""

from dataclasses import replace
import unittest

import simpy

from TokenSim.af.planning.batching import AFBlockBuilder
from TokenSim.af.planning.planner import ParallelExecutionPlanner
from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.config.config import KVTransferConfig
from TokenSim.config.simulation import SimulationConfig
from TokenSim.errors import ConfigurationError
from TokenSim.llm.llm_engine import LLMEngine
from TokenSim.llm.llm_request import Request
from tests.af.support.config import _cluster_dict, _with_prefill_pool
from tests.af.support.execution import af_test_hardware
from tests.af.support.model import _moe


class ModelBoundaryTest(unittest.TestCase):
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

    def decode_trace(self, **geometry):
        env, engine = self.engine(**geometry)
        req = Request(1, 8, 3, 4)
        req.generation_idx = 1
        executor = engine.af_executor
        iteration = executor.submit(0, [req])
        env.run(until=env.process(executor.run_iteration(iteration)))
        executor.trace.validate_invariants(iteration.iteration_id)
        return executor.trace.events, iteration.makespan

    def test_decode_executes_each_layer_kind(self):
        moe = replace(_moe(), first_k_dense_replace=1, num_shared_experts=1)
        events, _ = self.decode_trace(layers=2, moe=moe)
        dense, experts = (
            {
                e.stage
                for e in events
                if e.layer_id == layer and e.event_type == "compute"
            }
            for layer in (0, 1)
        )
        self.assertIn("dense_ffn", dense)
        self.assertFalse({"router", "routed_expert"} & dense)
        self.assertTrue(
            {"router", "routed_expert", "shared_expert", "weighted_combine"} <= experts
        )

    def test_ordinary_serving_includes_embedding_head_and_sampling(self):
        events, makespan = self.decode_trace()
        boundaries = [
            e for e in events if e.layer_id == -1 and e.event_type == "compute"
        ]
        self.assertEqual(
            [e.operation for e in boundaries],
            ["embedding_lookup", "final_norm", "lm_head", "token_selection"],
        )
        self.assertTrue(all(e.service_duration > 0 for e in boundaries))
        self.assertGreater(makespan, sum(e.service_duration for e in boundaries))

    def test_prefill_is_rejected_before_ownership_or_iteration_changes(self):
        env, engine = self.engine()
        for scheduled in (8, 16):
            req = Request(1, 16, 3, 4)
            req.scheduled_tokens = scheduled
            before = engine.af_executor._next_iteration_id
            with self.assertRaisesRegex(ConfigurationError, "only accepts decode"):
                engine.af_executor.submit(0, [req])
            self.assertEqual(req.af_a_owners, {})
            self.assertEqual(engine.af_executor._next_iteration_id, before)
        self.assertFalse(engine.af_executor.trace.events)


class QueryGeometryTest(unittest.TestCase):
    def test_initial_chunk_does_not_read_uncomputed_prompt_keys(self):
        req = Request(1, 64, 3, 4)
        req.generation_idx = 1
        req.prepare_recompute()
        req.scheduled_tokens = 8
        link = ClusterConfig(**_cluster_dict()).af.links[0]
        block = AFBlockBuilder().build(0, [req], 8, 0, link).blocks[0]
        self.assertEqual(block.context_lengths, (8,))
        self.assertEqual(block.query_lengths, (8,))
        self.assertEqual(block.query_offsets, (0,))
        self.assertEqual(block.logits_rows, 0)
        self.assertEqual(
            ParallelExecutionPlanner._block_tokens(block),
            tuple((1, i) for i in range(8)),
        )

    def test_cached_suffix_uses_actual_query_positions_and_full_kv(self):
        req = Request(1, 64, 3, 4)
        req.start_context_build(56)
        req.generation_idx = 1
        req.prepare_recompute()
        req.num_computed_tokens = 57
        req.scheduled_tokens = 8
        link = ClusterConfig(**_cluster_dict()).af.links[0]
        block = AFBlockBuilder().build(0, [req], 8, 0, link).blocks[0]
        self.assertEqual(block.context_lengths, (65,))
        self.assertEqual(block.query_offsets, (57,))
        self.assertEqual(block.logits_rows, 1)
        self.assertEqual(
            ParallelExecutionPlanner._block_tokens(block),
            tuple((1, i) for i in range(57, 65)),
        )
