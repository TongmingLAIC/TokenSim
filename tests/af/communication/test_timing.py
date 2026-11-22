"""Host/transport timing, subgroup readiness, and collective contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import simpy

from TokenSim.af.communication.resources import AFResourceRegistry
from TokenSim.af.execution.executor import AFExecutor
from TokenSim.af.planning.batching import AFBlock
from TokenSim.af.planning.dag import AFExecutionDAG
from TokenSim.af.planning.nodes import ExecutionNode, RankLocalWork
from TokenSim.af.providers.operator_table import TopologyAFTransferProvider
from TokenSim.af.providers.queries import AFServiceEstimate
from TokenSim.config.af.communication import AFCommunicationTimingConfig
from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.hardware.domains import CommunicationDomainConfig
from TokenSim.hardware.placement import build_topology_placement
from tests.af.support.config import _cluster
from tests.af.support.execution import (
    af_test_hardware,
    af_test_resources,
    execution_plan,
)
from tests.af.support.model import _transfer_query


class ZeroCompute:
    def estimate(self, query):
        return AFServiceEstimate(0.0, "test:zero-compute")


class WireOnly:
    def __init__(self, duration_us=3.0, scope="transport_service"):
        self.duration_us, self.scope = duration_us, scope

    def estimate(self, query):
        return AFServiceEstimate(
            self.duration_us / 1e6,
            "test:wire",
            payload_bytes=query.payload_bytes,
            details={"timing_scope": self.scope},
        )


def run_case(
    submit_us=5.0,
    complete_us=0.0,
    wire_us=3.0,
    scope="per_worker",
    enabled=True,
    provider_scope="transport_service",
):
    cluster = _cluster(a_tp=1, a_dp=1, f_tp=1, f_dp=1)
    cluster.af.communication_timing = AFCommunicationTimingConfig(
        enabled=enabled,
        host_submit_us=submit_us,
        host_completion_us=complete_us,
        host_resource_scope=scope,
    )
    cluster.af.transfer_topology_mode = "flat_link"
    env = simpy.Environment()
    executor = AFExecutor(
        env,
        cluster,
        cluster.workers(),
        ZeroCompute(),
        WireOnly(wire_us, provider_scope),
        "test",
        "bf16",
        total_layers=1,
        hidden_size=16,
        runtime_plan=execution_plan(cluster),
        resources=af_test_resources(env, cluster, cluster.workers()),
    )
    requests = [
        SimpleNamespace(
            id=i, is_prefill=False, prefill_len=1, prefill_compute_len=1, context_len=1
        )
        for i in range(2)
    ]
    iteration = executor.submit(0, requests)
    env.process(executor.run_iteration(iteration))
    env.run()
    executor.trace.validate_invariants(iteration.iteration_id)
    return executor, iteration


class CommunicationTimingTest(unittest.TestCase):
    def test_host_submission_overlap_wire_queue_and_completion_cost(self):
        executor, iteration = run_case()
        sends = [
            e
            for e in executor.trace.events
            if e.event_type == "transfer" and e.direction == "a_to_f"
        ]
        submits = [
            e
            for e in executor.trace.events
            if e.event_type == "host_submit" and e.direction == "a_to_f"
        ]
        self.assertEqual((len(sends), len(submits)), (2, 2))
        self.assertAlmostEqual(submits[1].queue_duration, 5e-6)
        self.assertAlmostEqual(iteration.makespan, 21e-6)
        queued, _ = run_case(submit_us=1.0, wire_us=10.0)
        qsends = [
            e
            for e in queued.trace.events
            if e.event_type == "transfer" and e.direction == "a_to_f"
        ]
        self.assertAlmostEqual(qsends[1].queue_duration, 9e-6)
        completed, _ = run_case(complete_us=2.0)
        syncs = [e for e in completed.trace.events if e.event_type == "synchronize"]
        self.assertTrue(
            all(
                e.resource_id.startswith("host:")
                and abs(e.service_duration - 2e-6) < 1e-12
                for e in syncs
            )
        )

    def test_shared_scope_legacy_mode_and_validation(self):
        _, per_it = run_case()
        _, shared_it = run_case(scope="shared")
        self.assertGreater(shared_it.makespan, per_it.makespan)
        legacy, old = run_case(enabled=False, provider_scope="legacy_unspecified")
        _, new = run_case(submit_us=0.0)
        self.assertAlmostEqual(old.makespan, new.makespan)
        self.assertFalse(
            any(e.event_type == "host_submit" for e in legacy.trace.events)
        )
        for scope in ("legacy_unspecified", "eager_interval"):
            with self.assertRaisesRegex(SimulationStateError, "timing_scope"):
                run_case(provider_scope=scope)
        for value in (-1.0, float("nan"), float("inf"), True, "5"):
            with self.assertRaises((ConfigurationError, ValueError)):
                AFCommunicationTimingConfig(host_submit_us=value)

    def test_resource_registry_lane_resolution(self):
        root = Path(__file__).resolve().parents[3]
        cluster = ClusterConfig.from_file(
            root / "data/clusters/af/decode_tp2_dp2_ep4_rank_reuse.json"
        )
        cluster.af.transfer_topology_mode = "flat_link"
        registry = AFResourceRegistry(simpy.Environment(), cluster, cluster.workers())
        link = cluster.af_link
        a_to_f = registry.communication_resources_for_step(link, "a_to_f")
        f_to_a = registry.communication_resources_for_step(link, "f_to_a")
        self.assertIsNot(a_to_f[0][1], f_to_a[0][1])
        plan = cluster.af_domains[0].communication_plan[0]
        collective = registry.communication_resources_for_step(
            link, plan.operation, plan.resource_group, plan.resource_lane
        )
        self.assertEqual(collective[0][0], (plan.resource_group, plan.resource_lane))
        with self.assertRaisesRegex(ConfigurationError, "explicit resource_lane"):
            registry.communication_resources_for_step(
                link, "all_reduce", "collective-fabric"
            )


class InputCompute:
    def __init__(self, late=4, delay=100e-6):
        self.late, self.delay = late, delay

    def estimate(self, query):
        return AFServiceEstimate(
            self.delay if query.worker_id == self.late else 0,
            "test:input",
            details={"timing_scope": "device_service"},
        )


def subgroup_fixture(*, flat=False, cache=True, timing=False, compute=None):
    cluster = _cluster(a_tp=8, a_dp=1, f_tp=8, f_dp=1)
    cluster.af.communication_domains = CommunicationDomainConfig(
        node_size=8,
        node_count=4,
        local_latency_us=1,
        inter_latency_us=2,
        device_indices={
            f"{pool}:{rank}": offset + (rank // 4) * 8 + rank % 4
            for pool, offset in (("a-pool", 0), ("f-pool", 16))
            for rank in range(8)
        },
    )
    cluster.af.transfer_topology_mode = "flat_link" if flat else "physical"
    cluster.af.service_estimate_cache = cache
    cluster.af.communication_timing = AFCommunicationTimingConfig(
        enabled=timing, host_submit_us=1, host_completion_us=1
    )
    cluster.af.trace_enabled = True
    workers = cluster.workers()
    hardware = af_test_hardware()
    placement = build_topology_placement(cluster, workers, hardware)
    provider = TopologyAFTransferProvider(
        hardware, cluster, workers, placement, fallback="analytical_only"
    )
    executor = AFExecutor(
        simpy.Environment(),
        cluster,
        workers,
        compute or InputCompute(),
        provider,
        "TestModel",
        "bf16",
        1,
        16,
    )
    return executor, provider, hardware, workers, placement


def dag_for(executor, ranks=tuple(range(8))):
    identities = tuple(
        executor.resources.worker(rank).physical_worker_ref.identity for rank in ranks
    )
    common = dict(
        source_domain="a-domain",
        target_domain="a-domain",
        logical_participants=("a-domain:dp0",),
        dp_rank=0,
    )
    producers = tuple(
        ExecutionNode(
            node_id=f"p:{rank}",
            operation="attention",
            kind="compute",
            predecessor_ids=(),
            stage="a",
            direction=None,
            domain_ranks=(rank,),
            physical_workers=(identity,),
            local_work=RankLocalWork(
                1, (("tokens", 1), ("hidden_size", 16), ("tp_size", 1))
            ),
            **common,
        )
        for rank, identity in zip(ranks, identities)
    )
    collective = ExecutionNode(
        node_id="test:collective",
        operation="all_reduce",
        kind="communication",
        predecessor_ids=tuple(n.node_id for n in producers),
        stage=None,
        direction="collective",
        domain_ranks=ranks,
        physical_workers=identities,
        payload_bytes=800,
        resource_lane="a_to_f",
        network=executor.cluster_config.af.links[0].network,
        logical_stage_id="test:collective",
        input_producer_ids=tuple(zip(identities, (n.node_id for n in producers))),
        **common,
    )
    consumer = replace(
        producers[0], node_id="test:consumer", predecessor_ids=("test:collective",)
    )
    return AFExecutionDAG((*producers, collective, consumer))


class SubgroupAndContractTests(unittest.TestCase):
    def test_subgroup_phase_dependencies_and_cancellation_cleanup(self):
        executor, *_ = subgroup_fixture(cache=True, timing=True)
        block = AFBlock(1, 0, (0,), 1, 1, (1,), 0, "af-link")
        executor.env.process(
            executor._run_dag(
                block, -1, executor.cluster_config.af.links[0], dag_for(executor)
            )
        )
        executor.env.run()
        phases = {
            e.details["phase_id"]: e
            for e in executor.trace.events
            if "phase_id" in e.details
        }
        self.assertLess(phases["local_rs:0"].end_at, 100e-6)
        self.assertGreaterEqual(phases["local_rs:1"].start_at, 100e-6)
        executor.trace.validate_invariants(1)
        ex2, *_ = subgroup_fixture()
        op = ex2.env.process(
            ex2._run_dag(block, -1, ex2.cluster_config.af.links[0], dag_for(ex2))
        )

        def cancel():
            yield ex2.env.timeout(2e-6)
            op.interrupt("cancel")

        ex2.env.process(cancel())
        with self.assertRaises(simpy.Interrupt):
            ex2.env.run()
        ex2.env.run()
        self.assertFalse(
            ex2.resources.domain_scheduler.owners
            or ex2.resources.domain_scheduler.waiters
        )

    def test_measurement_applicability_and_all_to_all_contracts(self):
        executor, base, hardware, workers, placement = subgroup_fixture()
        query = replace(
            _transfer_query(payload_bytes=100, endpoints=("a-pool:0", "a-pool:1")),
            dtype="bf16",
        )
        phase = base.communication_plan(query).phases[0]
        record = dict(
            model_version="node-domain-fifo-v1",
            algorithm="direct",
            phase_kind="transfer",
            operation="send_recv",
            dtype=phase.dtype,
            participants=2,
            nodes=1,
            logical_input_bytes=phase.logical_input_bytes,
            wire_bytes=phase.wire_bytes,
            wire_schedule=(((0, 0, phase.wire_bytes),),),
            domain_parameters=tuple(
                (name, getattr(base.domain_config, name))
                for name in (
                    "local_bandwidth_bytes_per_s",
                    "inter_tx_bandwidth_bytes_per_s",
                    "inter_rx_bandwidth_bytes_per_s",
                    "local_latency_us",
                    "inter_latency_us",
                )
            ),
            timing_scope="transport_service",
            latency_us=7,
            source_id="test:A",
            transport_backend="nccl",
        )
        package = SimpleNamespace(
            meta=SimpleNamespace(
                dataset_version="test-v1",
                extra={"communication_phase_services": (record,)},
            )
        )
        with patch.object(
            hardware,
            "operator_package",
            side_effect=lambda d, b: package if d == "a-device" else None,
        ):
            provider = TopologyAFTransferProvider(
                hardware, executor.cluster_config, workers, placement
            )
        self.assertEqual(
            provider.estimate(replace(query, transport_backend="nccl")).match_type,
            "exact",
        )
        a2a = replace(
            _transfer_query(
                "all_to_all",
                payload_bytes=400,
                endpoints=tuple(f"a-pool:{r}" for r in range(8)),
            ),
            group_size=8,
            remote_fraction=0.75,
            load_imbalance=2,
            destination_redistribution=True,
        )
        plan = base.communication_plan(a2a)
        self.assertEqual(sum(size for _, size in plan.expected_output_bytes), 8 * 400)
        for bad in (
            dict(payload_bytes=-1),
            dict(local_fraction=0.3, remote_fraction=0.6),
            dict(load_imbalance=0.9),
            dict(group_size=3),
            dict(physical_endpoints=()),
        ):
            with self.assertRaises(ConfigurationError):
                base.estimate(replace(a2a, **bad))
