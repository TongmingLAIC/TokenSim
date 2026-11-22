"""Cross-owner FFN invocations preserve readiness, ownership and cancellation."""

from dataclasses import replace
from types import SimpleNamespace
import unittest

import simpy

from TokenSim.af.execution.executor import AFExecutor
from TokenSim.af.serving.metrics import ResourceMetrics, steady_metrics
from TokenSim.af.serving.work import ledger
from TokenSim.af.trace.metrics import AFTraceEvent
from TokenSim.config.af.settings import AFConfig
from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.trace.occupancy import unique_executions, validate_resource_capacity
from tests.af.support.batching import batch_runtime, submit_owner
from tests.af.support.config import _cluster
from tests.af.support.execution import (
    FixedComputeProvider,
    FixedTransferProvider,
    af_test_resources,
    execution_plan,
)
from tests.support.hardware import make_device, make_hardware


class FFNBatchingTest(unittest.TestCase):
    def test_multiple_owners_form_one_exact_expert_batch(self):
        env, executor, provider = batch_runtime()
        submissions = [submit_owner(executor, owner) for owner in range(4)]
        for iteration, _ in submissions:
            env.process(executor.run_iteration(iteration))
        env.run()
        queries = [q for q in provider.queries if q.stage == "routed_expert"]
        self.assertEqual(len(queries), 1)
        self.assertEqual(
            (queries[0].original_token_count, queries[0].assignment_count), (8, 16)
        )
        self.assertEqual(sum(n for _, n in queries[0].expert_workload), 16)
        self.assertEqual(
            {edge.a_dp_rank for edge in queries[0].routing_edges}, set(range(4))
        )
        for owner, (iteration, requests) in enumerate(submissions):
            self.assertEqual(iteration.blocks[0].placement.a_dp_rank, owner)
            self.assertTrue(
                all(r.af_a_owners == {"a-domain": f"a{owner}"} for r in requests)
            )
            executor.trace.validate_invariants(iteration.iteration_id)
        events = executor.trace.events
        physical = [e for e in unique_executions(events) if e.stage == "routed_expert"]
        self.assertEqual(len(physical), 1)
        validate_resource_capacity(events, executor.trace.resource_capacities)
        metrics = steady_metrics(
            [],
            events,
            start=0,
            end=env.now,
            capacities=executor.trace.resource_capacities,
        )
        f_worker = physical[0].resource_id
        self.assertAlmostEqual(
            metrics["resource_utilization"][f_worker] * env.now, 20e-6
        )
        online = ResourceMetrics(dict(start=0, end=env.now))
        for event in events:
            online.observe(event)
        self.assertEqual(online.summary([]), metrics)

    def test_limit_and_same_owner_keep_batches_separate(self):
        for owners, limit, count, expected in ((4, 4, 2, 2), (1, 0, 4, 2)):
            with self.subTest(owners=owners):
                env, executor, provider = batch_runtime(owners=owners, max_tokens=limit)
                for owner in range(owners):
                    iteration, _ = submit_owner(executor, owner, count=count)
                    env.process(executor.run_iteration(iteration))
                env.run()
                self.assertEqual(
                    len([q for q in provider.queries if q.stage == "routed_expert"]),
                    expected,
                )

    def test_late_owner_is_not_assumed_ready(self):
        env, executor, provider = batch_runtime(owners=2)
        left, _ = submit_owner(executor, 0)
        right, _ = submit_owner(executor, 1)
        env.process(executor.run_iteration(left))

        def later():
            yield env.timeout(50e-6)
            yield from executor.run_iteration(right)

        env.process(later())
        env.run()
        self.assertEqual(
            len([q for q in provider.queries if q.stage == "routed_expert"]), 2
        )
        self.assertLess(left.end_at, right.start_at)

    def test_expert_tp_and_ep_preserve_each_assignment_on_its_ranks(self):
        for ep in (1, 2):
            with self.subTest(ep=ep):
                env, executor, provider = batch_runtime(owners=2, f_tp=2, f_ep=ep)
                for owner in range(2):
                    iteration, _ = submit_owner(executor, owner)
                    env.process(executor.run_iteration(iteration))
                env.run()
                queries = [q for q in provider.queries if q.stage == "routed_expert"]
                self.assertEqual(len(queries), 2)
                self.assertEqual(
                    sum(q.assignment_count for q in queries), 16 if ep == 1 else 8
                )
                validate_resource_capacity(
                    executor.trace.events, executor.trace.resource_capacities
                )

    def test_batch_failure_reaches_all_consumers_and_releases_device(self):
        class Failing(FixedComputeProvider):
            def estimate(self, query):
                if query.stage == "routed_expert":
                    raise ValueError("batch kernel unavailable")
                return super().estimate(query)

        env, executor, _ = batch_runtime(owners=2, provider=Failing(1e-6, 0))
        failures = []

        def guarded(iteration):
            try:
                yield from executor.run_iteration(iteration)
            except SimulationStateError as error:
                failures.append(str(error))

        for owner in range(2):
            env.process(guarded(submit_owner(executor, owner)[0]))
        env.run()
        self.assertEqual(len(failures), 2)
        self.assertTrue(all("batch kernel unavailable" in e for e in failures))
        self.assertFalse(executor.active_iterations)
        self.assertTrue(
            all(
                not r.count and not r.queue
                for r in executor.resources.compute_slots.values()
            )
        )

    def test_cancel_one_batch_consumer_preserves_other_output(self):
        env, executor, provider = batch_runtime(owners=2)
        left, _ = submit_owner(executor, 0)
        right, _ = submit_owner(executor, 1)
        left_task = env.process(executor.run_iteration(left))
        env.process(executor.run_iteration(right))

        def cancel():
            yield env.timeout(10e-6)
            left_task.interrupt()

        def swallow():
            try:
                yield left_task
            except simpy.Interrupt:
                pass

        env.process(cancel())
        env.process(swallow())
        env.run()
        self.assertIsNone(left.end_at)
        self.assertIsNotNone(right.end_at)
        self.assertEqual(
            len([q for q in provider.queries if q.stage == "routed_expert"]), 1
        )
        executor.trace.validate_invariants(right.iteration_id)
        self.assertTrue(
            all(
                not r.count and not r.queue
                for r in executor.resources.compute_slots.values()
            )
        )

    def test_host_submission_is_charged_once_for_a_shared_batch(self):
        env, executor, _ = batch_runtime(owners=2, submit_us=2)
        for owner in range(2):
            env.process(executor.run_iteration(submit_owner(executor, owner)[0]))
        env.run()
        submits = [
            e
            for e in executor.trace.events
            if e.details.get("cost_scope") == "per_ffn_batch"
        ]
        self.assertEqual(len(submits), 1)
        self.assertAlmostEqual(submits[0].service_duration, 2e-6)

    def test_disabling_batching_preserves_independent_queries(self):
        env, executor, provider = batch_runtime(owners=2, enabled=False)
        for owner in range(2):
            env.process(executor.run_iteration(submit_owner(executor, owner)[0]))
        env.run()
        self.assertEqual(
            len([q for q in provider.queries if q.stage == "routed_expert"]), 2
        )
        self.assertFalse(any(e.execution_group_id for e in executor.trace.events))

    def test_all_consumers_cancel_before_window_releases_queue(self):
        env, executor, _ = batch_runtime(owners=2, wait_us=100)
        tasks = []

        def guarded(iteration):
            try:
                yield from executor.run_iteration(iteration)
            except simpy.Interrupt:
                pass

        for owner in range(2):
            tasks.append(env.process(guarded(submit_owner(executor, owner)[0])))

        def cancel():
            yield env.timeout(10e-6)
            for task in tasks:
                task.interrupt()

        env.process(cancel())
        env.run(until=env.all_of(tasks))
        self.assertFalse(executor.active_stages)
        self.assertTrue(
            all(
                not r.count and not r.queue
                for r in executor.resources.compute_slots.values()
            )
        )

    def test_shared_execution_intervals_cannot_disagree(self):
        env, executor, _ = batch_runtime(owners=2)
        for owner in range(2):
            env.process(executor.run_iteration(submit_owner(executor, owner)[0]))
        env.run()
        members = [e for e in executor.trace.events if e.execution_group_id]
        self.assertEqual(len(members), 2)
        with self.assertRaisesRegex(
            SimulationStateError, "inconsistent resource intervals"
        ):
            validate_resource_capacity(
                [members[0], replace(members[1], end_at=members[1].end_at + 1e-6)], {}
            )


class MicrobatchAndOwnerWindowTests(unittest.TestCase):
    @staticmethod
    def _dense_runtime(limit):
        cluster = _cluster(a_tp=1, a_dp=1, f_tp=1, f_dp=1)
        cluster = replace(
            cluster, af=replace(cluster.af, af_block_size=2, max_inflight_blocks=limit)
        )
        env = simpy.Environment()
        executor = AFExecutor(
            env,
            cluster,
            cluster.workers(),
            FixedComputeProvider(1.0, 3.0),
            FixedTransferProvider(0, 0, timing_scope="transport_service"),
            "TestModel",
            "bf16",
            1,
            16,
            runtime_plan=execution_plan(cluster),
            resources=af_test_resources(env, cluster),
        )
        requests = [
            SimpleNamespace(
                id=i, is_prefill=False, prefill_len=8, context_len=8, generation_idx=0
            )
            for i in range(4)
        ]
        return env, executor, executor.submit(0, requests)

    def test_serial_window_changes_overlap_and_releases_on_cancel(self):
        outcomes = []
        for limit in (0, 1, 2):
            env, executor, iteration = self._dense_runtime(limit)
            env.process(executor.run_iteration(iteration))
            env.run()
            executor.trace.validate_invariants(iteration.iteration_id)
            compute = [e for e in executor.trace.events if e.event_type == "compute"]
            outcomes.append(
                (
                    env.now,
                    [(e.operation, e.token_count, e.service_duration) for e in compute],
                )
            )
        self.assertGreater(outcomes[1][0], outcomes[0][0])
        self.assertEqual(outcomes[0][0], outcomes[2][0])
        self.assertEqual(sorted(outcomes[0][1]), sorted(outcomes[1][1]))

        env, executor, iteration = self._dense_runtime(1)
        process = env.process(executor.run_iteration(iteration))

        def cancel():
            yield env.timeout(0.25)
            process.interrupt("cancel window")

        env.process(cancel())
        with self.assertRaises(simpy.Interrupt):
            env.run()
        self.assertFalse(
            executor.active_iterations
            or executor.active_stages
            or executor.ffn_batcher.tasks
        )

    def test_owner_credit_scope_and_cancellation(self):
        for scope in ("iteration", "owner"):
            env, executor, _ = batch_runtime(
                owners=1, max_inflight=1, inflight_scope=scope
            )
            first, _ = submit_owner(executor, 0, offset=0)
            second, _ = submit_owner(executor, 0, offset=100)
            env.process(executor.run_iteration(first))
            env.process(executor.run_iteration(second))
            env.run()
            starts = [
                e.start_at
                for e in executor.trace.events
                if e.iteration_id == second.iteration_id
                and e.event_type == "compute"
                and e.source_domain == executor.cluster_config.af_domains[0].name
            ]
            if scope == "owner":
                self.assertGreaterEqual(min(starts), first.end_at)
            else:
                self.assertLess(min(starts), first.end_at)
        with self.assertRaises((ValueError, ConfigurationError)):
            AFConfig(max_inflight_scope="global_guess")


class AFWorkLedgerTest(unittest.TestCase):
    @staticmethod
    def _event(node, group=None):
        return AFTraceEvent(
            iteration_id=0,
            block_id=0,
            layer_id=0,
            phase="decode",
            event_type="compute",
            resource_id="f0",
            ready_at=0,
            start_at=0,
            end_at=1,
            queue_duration=0,
            service_duration=1,
            request_count=2,
            token_count=2,
            source_id="fixture",
            node_id=node,
            execution_group_id=group,
            details=dict(
                device_id="TestGPU",
                components=[
                    dict(
                        name="ffn",
                        table="gemm",
                        key=dict(m=2, n=4, k=8, dtype="bf16"),
                        latency_us=1e6,
                    )
                ],
            ),
        )

    def test_physical_deduplication_mixed_precision_and_partial_rejection(self):
        hardware = make_hardware(make_device(peak_flops=128))
        spans = [self._event("a", "shared"), self._event("b", "shared")]
        value = ledger(iter(spans), hardware, {"TestGPU": 2}, start=0, end=1)
        self.assertEqual(value, ledger(spans, hardware, {"TestGPU": 2}, start=0, end=1))
        self.assertEqual(
            (
                value["useful_flops"]["TestGPU"],
                value["physical_compute_invocations"],
                value["logical_compute_spans"],
                value["system_model_utilization"],
            ),
            (128, 1, 2, 0.5),
        )
        span = self._event("packed")
        span.details["components"][0].update(
            work_flops_by_dtype={"bf16": 128},
            analytical_bounds={"compute_us": 1e6, "memory_us": 2e6},
        )
        mixed = ledger(iter([span]), hardware, {"TestGPU": 1}, start=0, end=1)
        self.assertEqual(mixed["components"][0]["analytical_memory_seconds"], 2)
        with self.assertRaises(ConfigurationError):
            ledger([self._event("partial")], hardware, {"TestGPU": 1}, start=0.5, end=1)
