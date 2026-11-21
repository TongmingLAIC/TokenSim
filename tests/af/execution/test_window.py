"""Admission windows isolate overlap without changing block or routing work."""

from dataclasses import replace
from types import SimpleNamespace
import unittest

import simpy

from TokenSim.af.execution.executor import AFExecutor
from tests.af.support.batching import batch_runtime, submit_owner
from tests.af.support.config import _cluster
from tests.af.support.execution import (
    FixedComputeProvider,
    FixedTransferProvider,
    af_test_resources,
    execution_plan,
)


def dense_runtime(limit):
    cluster = _cluster(a_tp=1, a_dp=1, f_tp=1, f_dp=1)
    cluster = replace(
        cluster,
        af=replace(cluster.af, af_block_size=2, max_inflight_blocks=limit),
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


class MicrobatchWindowTests(unittest.TestCase):
    def test_serial_window_changes_overlap_without_changing_compute_work(self):
        outcomes = []
        for limit in (0, 1, 2):
            env, executor, iteration = dense_runtime(limit)
            env.process(executor.run_iteration(iteration))
            env.run()
            executor.trace.validate_invariants(iteration.iteration_id)
            events = executor.trace.events
            compute = [e for e in events if e.event_type == "compute"]
            outcomes.append(
                (
                    env.now,
                    [(e.operation, e.token_count, e.service_duration) for e in compute],
                )
            )
            admissions = [e for e in events if e.operation == "microbatch_admission"]
            self.assertEqual(len(admissions), 0 if limit == 0 else 2)
            if limit == 1:
                self.assertGreater(admissions[1].queue_duration, 0)
                self.assertEqual(
                    admissions[1].start_at, iteration.block_completed_at[0]
                )
                second_attention = next(
                    e for e in compute if e.block_id == 1 and e.stage == "attention"
                )
                self.assertGreaterEqual(second_attention.ready_at, admissions[1].end_at)
        self.assertGreater(outcomes[1][0], outcomes[0][0])
        self.assertEqual(outcomes[0][0], outcomes[2][0])
        self.assertEqual(sorted(outcomes[0][1]), sorted(outcomes[1][1]))

    def test_serial_window_retains_cross_owner_batching(self):
        env, executor, provider = batch_runtime(owners=2, max_inflight=1)
        for owner in range(2):
            iteration, _ = submit_owner(executor, owner, count=4)
            env.process(executor.run_iteration(iteration))
        env.run()
        queries = [q for q in provider.queries if q.stage == "routed_expert"]
        self.assertEqual(len(queries), 2)
        self.assertTrue(
            all({e.a_dp_rank for e in q.routing_edges} == {0, 1} for q in queries)
        )
        self.assertFalse(executor.active_iterations)

    def test_cancellation_releases_running_and_queued_blocks(self):
        env, executor, iteration = dense_runtime(1)
        process = env.process(executor.run_iteration(iteration))

        def cancel():
            yield env.timeout(0.25)
            process.interrupt("cancel window")

        env.process(cancel())
        with self.assertRaises(simpy.Interrupt):
            env.run()
        self.assertFalse(executor.active_iterations)
        self.assertFalse(executor.active_stages)
        self.assertFalse(executor.ffn_batcher.tasks)
