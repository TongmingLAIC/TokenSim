"""Owner credits span overlapping iterations and remain backward compatible."""

import unittest
from TokenSim.config.af.settings import AFConfig
from TokenSim.errors import ConfigurationError
from tests.af.support.batching import batch_runtime, submit_owner


class OwnerWindowTest(unittest.TestCase):
    def test_shared_credit_limits_overlapping_iterations(self):
        for scope in ("iteration", "owner"):
            env, executor, _ = batch_runtime(
                owners=1, max_inflight=1, inflight_scope=scope
            )
            first, _ = submit_owner(executor, 0, offset=0)
            second, _ = submit_owner(executor, 0, offset=100)
            a = env.process(executor.run_iteration(first))
            b = env.process(executor.run_iteration(second))
            env.run()
            starts = [
                e.start_at
                for e in executor.trace.events
                if e.iteration_id == second.iteration_id
                and e.event_type == "compute"
                and e.source_domain == executor.cluster_config.af_domains[0].name
            ]
            self.assertTrue(a.triggered and b.triggered)
            if scope == "owner":
                self.assertGreaterEqual(min(starts), first.end_at)
                self.assertTrue(
                    all(
                        not r.count and not r.queue
                        for r in executor.owner_windows.values()
                    )
                )
                self.assertTrue(
                    any(
                        e.details.get("scope") == "per_attention_owner"
                        for e in executor.trace.events
                    )
                )
            else:
                self.assertLess(min(starts), first.end_at)

    def test_cancelled_iteration_releases_shared_credit(self):
        import simpy

        env, executor, _ = batch_runtime(
            owners=1, max_inflight=1, inflight_scope="owner"
        )
        first, _ = submit_owner(executor, 0, offset=0)
        second, _ = submit_owner(executor, 0, offset=100)
        p = env.process(executor.run_iteration(first))

        def monitor():
            try:
                yield p
            except simpy.Interrupt:
                return

        env.process(monitor())
        q = env.process(executor.run_iteration(second))
        env.run(until=1e-6)
        p.interrupt("cancelled")
        env.run()
        self.assertTrue(q.triggered and q.ok)
        self.assertTrue(
            all(not r.count and not r.queue for r in executor.owner_windows.values())
        )

    def test_bad_credit_scope_is_rejected(self):
        with self.assertRaises((ValueError, ConfigurationError)):
            AFConfig(max_inflight_scope="global_guess")
