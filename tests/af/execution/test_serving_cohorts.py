"""Independent request streams use native PP resources and resident KV leases."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import simpy

from TokenSim.af.communication.endpoints import EndpointResources
from TokenSim.af.factory import build_model_runtime
from TokenSim.af.runtime.compiler import compile_af_runtime_plan
from TokenSim.af.serving.handoff import execute_handoff, plan_handoff
from TokenSim.af.serving.metrics import steady_metrics
from TokenSim.af.serving.service import AFService
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.placement import build_topology_placement
from TokenSim.latency.operator_table import OperatorTableLatencyBackend

from tests.af.support.config import _cluster
from tests.af.support.execution import (
    FixedComputeProvider,
    FixedTransferProvider,
    af_test_hardware,
)
from tests.af.support.model import _moe, _runtime_hardware
from tests.af.support.pipeline import cluster_document, pipeline_cluster


class CohortServingTest(unittest.TestCase):
    def test_production_provider_reserves_endpoint_injection_atomically(self):
        cluster = pipeline_cluster(resource_scope="endpoint")
        hardware = _runtime_hardware()
        model = replace(hardware.model("TestModel"), moe=_moe(top_k=1, shared=1))
        plan = compile_af_runtime_plan(
            cluster_config=cluster,
            worker_configs=cluster.workers(),
            hardware=hardware,
            model=model,
            block_size=4,
        )
        placement = build_topology_placement(cluster, cluster.workers(), hardware)
        runtime = SimpleNamespace(
            hardware=hardware,
            model=model,
            cluster=cluster,
            workers=cluster.workers(),
            plan=plan,
            placement=placement,
        )
        env = simpy.Environment()
        service = AFService(env, runtime)
        sessions = [
            service.admit(i, 0, maximum_context=16, current_context=8) for i in range(2)
        ]
        for session in sessions:
            env.process(service.forward([session]))
        env.run()
        self.assertFalse(service.executor.resources.endpoint_resources.scheduler.owners)
        self.assertFalse(
            service.executor.resources.endpoint_resources.scheduler.waiters
        )
        self.assertTrue(
            any(event.event_type == "collective" for event in service.trace.events)
        )
        self.assertTrue(
            any("link:endpoint:" in event.resource_id for event in service.trace.events)
        )

    def test_independent_cohorts_overlap_and_report_utilization(self):
        cluster = pipeline_cluster()
        hardware = _runtime_hardware()
        model = replace(hardware.model("TestModel"), moe=_moe(top_k=1, shared=1))
        plan = compile_af_runtime_plan(
            cluster_config=cluster,
            worker_configs=cluster.workers(),
            hardware=hardware,
            model=model,
            block_size=4,
        )
        placement = build_topology_placement(cluster, cluster.workers(), hardware)
        runtime = SimpleNamespace(
            hardware=hardware,
            model=model,
            cluster=cluster,
            workers=cluster.workers(),
            plan=plan,
            placement=placement,
        )
        env = simpy.Environment()
        transfer = FixedTransferProvider(0, 0)
        transfer.operation_durations["af_pipeline_handoff"] = 0.1
        transfer.placement = placement
        service = AFService(
            env, runtime, providers=(FixedComputeProvider(1, 1), transfer)
        )
        events = []

        def stream(session):
            for _ in range(3):
                yield from service.forward([session])
                session.generated += 1
                events.append(dict(time=env.now, tokens=1, phase="decode"))

        sessions = [
            service.admit(i, 0, maximum_context=20, current_context=8) for i in range(4)
        ]
        for session in sessions:
            env.process(stream(session))
        env.run()
        metrics = steady_metrics(
            events,
            service.trace.events,
            start=1,
            end=env.now + 1e-6,
            capacities=service.executor.resources.resource_capacities(),
        )
        self.assertEqual(metrics["decode_tokens"], 12)
        self.assertTrue(
            all(
                value <= 1 + 1e-12 for value in metrics["resource_utilization"].values()
            )
        )
        self.assertGreater(sum(metrics["resource_queue_s"].values()), 0)
        self.assertEqual(service.summary()["passes"], 12)
        for session in sessions:
            service.release(session)
        self.assertFalse(service.memory.leases)


class CompletePrefillTest(unittest.TestCase):
    def service(self, colocated=True):
        hardware = af_test_hardware(layers=4, moe=_moe(top_k=1), capacity_gib=0.1)
        doc = cluster_document(_cluster(a_tp=2, a_dp=1, f_tp=2, f_dp=1, f_ep=2))
        doc["af"].update(
            latency_provider="operator_table",
            transfer_topology_mode="flat_link",
            communication_resource_scope="endpoint",
        )
        if colocated:
            group = doc["worker_groups"][0]
            group["capabilities"] = ["af_attention", "af_ffn"]
            doc["worker_groups"] = [group]
            doc["num_workers"] = group["num_workers"]
            domain = doc["af"]["execution_domains"][1]
            domain["worker_group"] = group["name"]
            for instance in domain["logical_instances"]:
                for rank in instance["rank_mapping"]:
                    rank["physical_worker"]["group_name"] = group["name"]
            link = doc["af"]["links"][0]
            link.update(ffn_group=group["name"], colocated=True)
        runtime = build_model_runtime(
            doc, hardware=hardware, model=hardware.model("TestModel")
        )
        env = simpy.Environment()
        service = AFService(env, runtime)
        session = service.admit(0, 0, maximum_context=24, current_context=16)
        return env, service, session

    def test_chunks_skip_output_head_and_use_no_af_providers(self):
        env, service, session = self.service()
        logits_queries = []
        original = OperatorTableLatencyBackend.query

        def query(backend, table, key):
            if table == "gemm" and key["n"] == backend.vocab_local:
                logits_queries.append(dict(key))
            return original(backend, table, key)

        with (
            patch.object(
                service.executor.compute_provider,
                "estimate",
                side_effect=AssertionError("AF compute used for prefill"),
            ),
            patch.object(OperatorTableLatencyBackend, "query", query),
        ):
            env.run(
                until=env.process(
                    service.forward(
                        [session], phase="prefill", query_tokens=2, emit_logits=False
                    )
                )
            )
            self.assertTrue(service.prefill.backends)
            self.assertTrue(
                all(
                    backend.stats.component_seconds["lm_head"] > 0
                    for backend in service.prefill.backends.values()
                )
            )
            self.assertFalse(logits_queries)
            env.run(
                until=env.process(
                    service.forward([session], phase="prefill", query_tokens=2)
                )
            )
        self.assertEqual(session.cached_context, 20)
        self.assertTrue(logits_queries)
        self.assertGreater(
            sum(
                backend.stats.component_seconds["lm_head"]
                for backend in service.prefill.backends.values()
            ),
            0,
        )
        self.assertEqual(service.executor._next_iteration_id, 0)

    def test_cancellation_releases_compute_and_queued_collectives(self):
        for delay in (1e-9, 0.001):
            with self.subTest(delay=delay):
                env, service, session = self.service()
                registry = service.executor.resources.endpoint_resources
                link = service.runtime.cluster.af_link
                endpoints = tuple(
                    worker.physical_worker_ref.identity
                    for worker in service.runtime.workers
                )
                keys = tuple(
                    registry.format_key(key)
                    for key, _ in registry.for_step(
                        link.network, "collective", endpoints
                    )
                )

                def block():
                    yield from registry.run_transfer(keys, 1, 0, ("block",))

                blocker = env.process(block())
                forward = env.process(
                    service.forward([session], phase="prefill", query_tokens=2)
                )
                env.run(until=delay)
                if delay > 1e-9:
                    self.assertTrue(registry.scheduler.waiters)
                forward.interrupt()
                with self.assertRaises(simpy.Interrupt):
                    env.run(until=forward)
                self.assertEqual(session.cached_context, 16)
                self.assertEqual(service.memory.leases[0].current_context, 16)
                self.assertFalse(registry.scheduler.waiters)
                for resource in service.executor.resources.compute_slots.values():
                    self.assertEqual(resource.count, 0)
                    self.assertFalse(resource.queue)
                blocker.interrupt()
                with self.assertRaises(simpy.Interrupt):
                    env.run(until=blocker)
                self.assertFalse(registry.scheduler.owners)


class HandoffTest(unittest.TestCase):
    def services(self, kv_heads=None):
        hardware = _runtime_hardware()
        model = replace(hardware.model("TestModel"), attention_windows=(4, 0))
        if kv_heads is not None:
            model = replace(model, num_key_value_heads=kv_heads)
        env = simpy.Environment()
        runtimes = []
        for namespace, pp in (("source", 1), ("target", 2)):
            cluster = pipeline_cluster(
                pp, pp, resource_scope="endpoint", namespace=namespace
            )
            workers = cluster.workers()
            plan = compile_af_runtime_plan(
                cluster_config=cluster,
                worker_configs=workers,
                hardware=hardware,
                model=model,
                block_size=4,
            )
            runtimes.append(
                SimpleNamespace(
                    hardware=hardware,
                    model=model,
                    cluster=cluster,
                    workers=workers,
                    plan=plan,
                    placement=build_topology_placement(cluster, workers, hardware),
                )
            )
        identities = [
            worker.physical_worker_ref.identity
            for runtime in runtimes
            for worker in runtime.workers
        ]
        resources = EndpointResources(
            env, {"af-net": "test", "cross": "test", "replication": "test"}, identities
        )
        source, target = [
            AFService(env, runtime, communication=resources) for runtime in runtimes
        ]
        return env, hardware, resources, source, target

    def test_warm_suffix_replication_ordering_and_cancellation(self):
        env, hardware, resources, source, target = self.services()
        before = source.admit(1, 0, maximum_context=24, current_context=20)
        after = target.admit(1, 0, maximum_context=24, current_context=16)
        incremental = plan_handoff(
            source,
            before,
            target,
            after,
            context=20,
            mode="incremental",
            cross_network="cross",
            replication_network="replication",
        )
        full = plan_handoff(
            source,
            before,
            target,
            after,
            context=20,
            mode="full",
            cross_network="cross",
            replication_network="replication",
        )
        bytes_per_token = (
            source.runtime.model.kv_elements_per_token()
            * source.runtime.plan.kv_dtype_size_bytes
        )
        self.assertEqual(sum(c.bytes for c in incremental), 2 * 4 * bytes_per_token)
        self.assertEqual(sum(c.bytes for c in full), (4 + 20) * bytes_per_token)
        events = []
        env.process(
            execute_handoff(
                env,
                incremental,
                hardware=hardware,
                resources=resources,
                identity=(1,),
                target=target,
                target_session=after,
                context=20,
                events=events,
            )
        )
        self.assertEqual(after.cached_context, 16)
        env.run()
        self.assertEqual(after.cached_context, 20)
        self.assertFalse(resources.scheduler.owners)

        env2, hw2, res2, src2, dst2 = self.services(kv_heads=1)
        s_session = src2.admit(1, 0, maximum_context=24, current_context=20)
        t_session = dst2.admit(1, 0, maximum_context=24, current_context=16)
        copies = plan_handoff(
            src2,
            s_session,
            dst2,
            t_session,
            context=20,
            mode="incremental",
            cross_network="cross",
            replication_network="replication",
        )
        rep_events = []
        env2.process(
            execute_handoff(
                env2,
                copies,
                hardware=hw2,
                resources=res2,
                identity=(1,),
                target=dst2,
                target_session=t_session,
                context=20,
                events=rep_events,
            )
        )
        env2.run()
        for layer in (0, 1):
            cross = next(
                e for e in rep_events if e["layer"] == layer and e["network"] == "cross"
            )
            local = next(
                e
                for e in rep_events
                if e["layer"] == layer and e["network"] == "replication"
            )
            self.assertGreaterEqual(local["start_s"], cross["end_s"])
        with self.assertRaises(ConfigurationError):
            plan_handoff(
                src2,
                s_session,
                dst2,
                t_session,
                context=25,
                mode="full",
                cross_network="cross",
                replication_network="replication",
            )
