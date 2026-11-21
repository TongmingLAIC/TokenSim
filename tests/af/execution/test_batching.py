"""Cross-owner FFN invocations preserve readiness, ownership and cancellation."""

from dataclasses import replace
import unittest

import simpy

from TokenSim.af.serving.metrics import ResourceMetrics, steady_metrics
from TokenSim.trace.occupancy import unique_executions, validate_resource_capacity
from TokenSim.errors import SimulationStateError
from tests.af.support.execution import FixedComputeProvider


from tests.af.support.batching import batch_runtime, submit_owner


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
