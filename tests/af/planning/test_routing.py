"""Router residency, replica selection, expert grouping, coalescing, and casts."""

from dataclasses import asdict, replace
from types import SimpleNamespace
import unittest

import simpy

from TokenSim.af.execution.executor import AFExecutor
from TokenSim.af.planning.coalesce import coalesce_ready_transfers
from TokenSim.af.planning.compute import group_routed_expert_nodes
from TokenSim.af.planning.nodes import ExecutionNode, RankLocalWork
from TokenSim.af.planning.planner import IndependentRoundRobinAFReplicaSelector
from TokenSim.af.providers.queries import (
    AFRoutingEdge,
    AFServiceEstimate,
    AFTensorSlice,
)
from TokenSim.config.af.domains import (
    DomainParallelConfig,
    DomainRankMapping,
    ExecutionDomainConfig,
    LogicalInstanceConfig,
    PhysicalWorkerRef,
)
from TokenSim.errors import ConfigurationError, WorkloadValidationError
from TokenSim.llm.llm_request import Request
from TokenSim.moe.routing import RoutingAssignment
from tests.af.support.config import _cluster
from tests.af.support.execution import (
    FixedComputeProvider,
    FixedTransferProvider,
    af_test_hardware,
    af_test_resources,
    execution_plan,
)
from tests.af.support.model import _moe, _runtime_plan


class ConstantProvider:
    def estimate(self, query):
        return AFServiceEstimate(
            0.001, "placement-test", payload_bytes=getattr(query, "payload_bytes", None)
        )


class RouterPlacementAndContinuousRoutingTest(unittest.TestCase):
    def run_case(self, side):
        base = _cluster(a_tp=1, a_dp=1, f_tp=1, f_dp=1, f_ep=1)
        cluster = replace(
            base,
            af=replace(
                base.af,
                router_location=side,
                af_block_size=2,
                transport_router_logits=True,
            ),
        )
        moe = replace(_moe(top_k=1, shared=1), num_shared_experts=0)
        plan = _runtime_plan(cluster, moe_config=moe, dtype="bf16")
        env = simpy.Environment()
        executor = AFExecutor(
            env,
            cluster,
            cluster.workers(),
            ConstantProvider(),
            ConstantProvider(),
            "TestModel",
            "bf16",
            total_layers=2,
            hidden_size=16,
            moe_config=moe,
            random_seed=42,
            runtime_plan=plan,
            resources=af_test_resources(env, cluster, cluster.workers()),
        )
        requests = [
            SimpleNamespace(
                id=i, is_prefill=False, prefill_len=8, context_len=9, generation_idx=1
            )
            for i in range(2)
        ]
        iteration = executor.submit(0, requests)
        env.process(executor.run_iteration(iteration))
        env.run()
        executor.trace.validate_invariants(iteration.iteration_id)
        return plan, executor.trace.events

    def test_ownership_dependencies_and_work_are_preserved(self):
        by_side = {}
        for side in ("a", "f"):
            plan, events = self.run_case(side)
            by_side[side] = events
            routers = [e for e in events if e.stage == "router"]
            self.assertTrue(routers)
            for event in routers:
                self.assertEqual(event.physical_endpoints, (f"{side}-pool:0",))
                layer = [e for e in events if e.layer_id == event.layer_id]
                hidden = next(e for e in layer if e.operation == "af_a_to_f")
                if side == "f":
                    self.assertGreaterEqual(event.start_at, hidden.end_at)
                else:
                    self.assertGreaterEqual(hidden.start_at, event.end_at)
            if side == "a":
                self.assertEqual(
                    len([e for e in events if e.operation == "af_router_logits"]),
                    len(routers),
                )
            else:
                metadata = [e for e in events if e.operation == "af_router_metadata"]
                self.assertEqual(len(metadata), len(routers))
                for event in metadata:
                    self.assertEqual(
                        event.payload_bytes,
                        2 * plan.layer_plan(event.layer_id).top_k * 8,
                    )
        workload = lambda es: sorted(
            (e.layer_id, e.token_count, str(e.routing_edges))
            for e in es
            if e.stage == "routed_expert"
        )
        self.assertEqual(workload(by_side["a"]), workload(by_side["f"]))

    def test_synthetic_and_external_continuous_routing(self):
        cluster = _cluster(a_tp=1, a_dp=1, f_tp=1, f_dp=1, f_ep=1)
        cluster.af.af_block_size = 8
        hardware = af_test_hardware(layers=1, moe=_moe())
        plan = _runtime_plan(cluster, hardware=hardware, moe_config=_moe())
        env = simpy.Environment()
        executor = AFExecutor(
            env,
            cluster,
            cluster.workers(),
            ConstantProvider(),
            ConstantProvider(),
            "TestModel",
            "fp16",
            1,
            16,
            moe_config=_moe(),
            runtime_plan=plan,
            resources=af_test_resources(env, cluster, cluster.workers()),
        )
        requests = [Request(i, 8, 8, 4) for i in (11, 22)]
        for req, step in zip(requests, (1, 3)):
            req.generation_idx = step
        iteration = executor.submit(0, requests)
        snapshot = iteration.blocks[0].routing_snapshot_for_layer(0)
        self.assertEqual(dict(snapshot.request_decode_steps), {11: 1, 22: 3})
        env.run(until=env.process(executor.run_iteration(iteration)))
        executor.trace.validate_invariants(iteration.iteration_id)

        ext_reqs = [Request(i, 8, 8, 4) for i in (11, 22)]
        for req, step in zip(ext_reqs, (1, 3)):
            req.generation_idx = step
            req.decode_routing_assignments = [
                asdict(
                    RoutingAssignment(
                        0, "decode", step, 0, req.id, req.context_len - 1, k, k, w, 0, 0
                    )
                )
                for k, w in enumerate((0.7, 0.3))
            ]
        ext_snap = executor.submit(0, ext_reqs).blocks[0].routing_snapshot_for_layer(0)
        self.assertEqual(ext_snap.assignment_count, 4)
        wrong = replace(ext_snap.assignments[0], decode_step=99)
        with self.assertRaisesRegex(WorkloadValidationError, "different.*step"):
            replace(ext_snap, assignments=(wrong, *ext_snap.assignments[1:]))


class ReplicaSelectorAndGroupingTest(unittest.TestCase):
    @staticmethod
    def _domain(name, capability, *, dp_size, ep_size=1):
        instances = [
            LogicalInstanceConfig(
                name=f"{name}-dp{dp}",
                rank_mapping=[
                    DomainRankMapping(
                        domain_rank=dp,
                        tp_rank=0,
                        pp_rank=0,
                        dp_rank=dp,
                        ep_rank=(dp if ep_size > 1 else 0),
                        physical_worker=PhysicalWorkerRef(f"{name}-pool", dp),
                    )
                ],
            )
            for dp in range(dp_size)
        ]
        return ExecutionDomainConfig(
            name=name,
            capability=capability,
            worker_group=f"{name}-pool",
            parallel=DomainParallelConfig(1, 1, dp_size, ep_size),
            logical_instances=instances,
        )

    @staticmethod
    def _expert_node(expert, *, worker="f:0", rank=0, tp_group=(0,), prefix="i0:b0:l0"):
        edge = AFRoutingEdge(
            iteration_id=0,
            phase="decode",
            decode_step=0,
            layer_id=0,
            request_id=0,
            token_id=0,
            top_k_index=expert,
            expert_id=expert,
            routing_weight=0.5,
            a_dp_rank=0,
            a_tp_rank=0,
        )
        work = RankLocalWork(
            token_count=1,
            tensor_shape=(("tokens", 1), ("hidden_size", 8)),
            expert_ids=(expert,),
            expert_workload=((expert, 1),),
            original_token_count=1,
            assignment_count=1,
            output_token_count=1,
            routing_edges=(edge,),
        )
        return ExecutionNode(
            node_id=f"{prefix}:routed:e{expert}:r{rank}",
            operation="routed_expert",
            kind="compute",
            predecessor_ids=(),
            stage="routed_expert",
            direction=None,
            source_domain="f",
            target_domain="f",
            logical_participants=("f0",),
            domain_ranks=(rank,),
            physical_workers=(worker,),
            dp_rank=0,
            tp_rank=rank,
            tp_group=tp_group,
            local_work=work,
            routing_edges=(edge,),
            layer_type="moe",
        )

    def test_independent_replica_rotation_and_ep_scopes(self):
        a_domain = self._domain("a", "af_attention", dp_size=2)
        f_domain = self._domain("f", "af_ffn", dp_size=3)
        selector = IndependentRoundRobinAFReplicaSelector()
        selections = [selector.select_for_block(a_domain, f_domain) for _ in range(6)]
        self.assertEqual([s.a_owner.name for s in selections], ["a-dp0", "a-dp1"] * 3)
        self.assertEqual(
            [s.f_service.name for s in selections], ["f-dp0", "f-dp1", "f-dp2"] * 2
        )

        f_reused = self._domain("f", "af_ffn", dp_size=3, ep_size=3)
        reused = IndependentRoundRobinAFReplicaSelector().select_for_block(
            a_domain, f_reused
        )
        self.assertEqual(reused.expert_parallel_mode, "rank_reuse")
        self.assertEqual(
            tuple(i.name for i in reused.routed_expert_instances),
            ("f-dp0", "f-dp1", "f-dp2"),
        )

    def test_physical_expert_grouping_and_coalescing(self):
        first, second = self._expert_node(0), self._expert_node(1)
        consumer = replace(
            first,
            node_id="combine",
            stage="weighted_combine",
            predecessor_ids=(first.node_id, second.node_id),
        )
        grouped = group_routed_expert_nodes((first, second, consumer))
        self.assertEqual(len(grouped), 2)
        self.assertEqual(grouped[0].local_work.expert_ids, (0, 1))
        self.assertEqual(grouped[1].predecessor_ids, (grouped[0].node_id,))
        for distinct in (
            self._expert_node(1, worker="f:1"),
            self._expert_node(1, tp_group=(0, 1)),
            self._expert_node(1, prefix="i0:b0:l1"),
        ):
            self.assertEqual(len(group_routed_expert_nodes((first, distinct))), 2)
        with self.assertRaisesRegex(ConfigurationError, "dependent"):
            group_routed_expert_nodes(
                (first, replace(second, predecessor_ids=(first.node_id,)))
            )

        t1 = ExecutionNode(
            node_id="one",
            operation="af_f_to_a",
            kind="communication",
            predecessor_ids=("producer",),
            stage=None,
            direction="f_to_a",
            source_domain="f",
            target_domain="a",
            logical_participants=("f", "a"),
            domain_ranks=(0, 1),
            physical_workers=("f:0", "a:0"),
            dp_rank=0,
            network="net",
            payload_bytes=32,
            tensor_slices=(
                AFTensorSlice("first", (0, 16), "partial", payload_bytes=32),
            ),
        )
        t2 = replace(
            t1,
            node_id="two",
            payload_bytes=64,
            tensor_slices=(
                AFTensorSlice("second", (0, 32), "partial", payload_bytes=64),
            ),
        )
        t3 = replace(t1, node_id="three", predecessor_ids=("later",))
        nodes = coalesce_ready_transfers((t1, t2, t3))
        self.assertEqual((len(nodes), sum(n.payload_bytes for n in nodes)), (2, 128))


class ConversionBatchTest(unittest.TestCase):
    def test_block_converts_once_per_direction_and_shares_quantization(self):
        for f_ranks, coalesce in ((1, False), (1, True), (2, False)):
            cluster = _cluster(a_tp=1, a_dp=1, f_tp=f_ranks, f_dp=1, f_ep=f_ranks)
            cluster = replace(
                cluster,
                af=replace(
                    cluster.af,
                    af_block_size=8,
                    activation_dtype="bf16",
                    coalesce_ready_transfers=coalesce,
                    links=[
                        replace(
                            cluster.af.links[0], a_to_f_dtype="fp8", f_to_a_dtype="fp8"
                        )
                    ],
                ),
            )
            env = simpy.Environment()
            compute = FixedComputeProvider(1e-6, 1e-6)
            transfer = FixedTransferProvider(0, 0)
            executor = AFExecutor(
                env,
                cluster,
                cluster.workers(),
                compute,
                transfer,
                "TestModel",
                "bf16",
                1,
                16,
                runtime_plan=execution_plan(cluster, moe=_moe()),
                moe_config=_moe(),
                resources=af_test_resources(env, cluster),
            )
            requests = [
                SimpleNamespace(
                    id=i,
                    is_prefill=False,
                    prefill_len=8,
                    context_len=8,
                    generation_idx=0,
                )
                for i in range(8)
            ]
            iteration = executor.submit(0, requests)
            env.process(executor.run_iteration(iteration))
            env.run()
            encoded = [
                q
                for q in compute.queries
                if q.stage == "encode" and q.physical_worker == "a-pool:0"
            ]
            self.assertEqual(len(encoded), 1)
            self.assertEqual(encoded[0].local_shape["elements"], 8 * 16)
            executor.trace.validate_invariants(0)
