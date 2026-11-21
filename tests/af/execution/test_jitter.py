"""Synthetic perturbations and peer synchronization preserve explicit contracts."""

from dataclasses import replace
from types import SimpleNamespace
import unittest

from TokenSim.af.execution.barrier import PeerLayerBarrier
from TokenSim.af.execution.jitter import perturb_service, request_multiplier
from TokenSim.af.providers.queries import AFServiceEstimate
from TokenSim.config.af.settings import AFConfig
from TokenSim.config.af.jitter import AFJitterConfig
from TokenSim.errors import ConfigurationError, SimulationStateError
from tests.af.support.batching import batch_runtime, submit_owner


class JitterTest(unittest.TestCase):
    def test_zero_is_identity_and_seed_does_not_depend_on_call_order(self):
        block = SimpleNamespace(
            request_ids=(1, 2), phase="decode", context_lengths=(8, 8)
        )
        estimate = AFServiceEstimate(1, "test")
        self.assertIs(
            perturb_service(estimate, AFJitterConfig(), block, 0, "a"), estimate
        )
        cfg = AFJitterConfig(enabled=True, compute_cv=0.3, seed=7)
        a = perturb_service(estimate, cfg, block, 0, "a")
        perturb_service(estimate, cfg, block, 1, "a")
        self.assertEqual(a, perturb_service(estimate, cfg, block, 0, "a"))
        self.assertGreater(a.service_duration, 0)
        self.assertNotEqual(
            a, perturb_service(estimate, replace(cfg, seed=8), block, 0, "a")
        )
        self.assertIs(
            perturb_service(estimate, cfg, block, 0, "wire", network=True), estimate
        )

    def test_request_work_is_additive_across_batch_membership(self):
        cfg = AFJitterConfig(enabled=True, compute_cv=0.2)

        def service(ids):
            b = SimpleNamespace(
                request_ids=ids, phase="decode", context_lengths=(8,) * len(ids)
            )
            return perturb_service(
                AFServiceEstimate(len(ids), "test"), cfg, b, 0, "a"
            ).service_duration

        self.assertAlmostEqual(service((1, 2)), service((1,)) + service((2,)))
        values = [request_multiplier(0, "service", i, 0.3) for i in range(10000)]
        self.assertAlmostEqual(sum(values) / len(values), 1, delta=0.015)

    def test_tp_rank_jitter_is_not_averaged_over_requests(self):
        cfg = AFJitterConfig(enabled=True, tp_rank_cv=0.3)
        block = SimpleNamespace(
            request_ids=(0,), phase="decode", context_lengths=(8,), source_a_worker_id=0
        )
        query = SimpleNamespace(
            direction="collective",
            operation="all_reduce",
            physical_endpoints=("a:0", "a:1"),
        )
        estimate = AFServiceEstimate(1, "collective")
        small = perturb_service(
            estimate, cfg, block, 0, "all_reduce", network=True, query=query
        )
        block.request_ids = tuple(range(256))
        block.context_lengths = (8,) * 256
        large = perturb_service(
            estimate, cfg, block, 0, "all_reduce", network=True, query=query
        )
        self.assertEqual(small, large)
        self.assertNotEqual(small.service_duration, 1)
        query.direction = "a_to_f"
        query.operation = "af_a_to_f"
        self.assertIs(
            perturb_service(estimate, cfg, block, 0, "wire", network=True, query=query),
            estimate,
        )

    def test_invalid_config(self):
        for fields in (
            {"compute_cv": -1},
            {"network_cv": float("nan")},
            {"seed": True},
            {"enabled": 1},
            {"compute_cv": 3},
        ):
            with self.assertRaises((ValueError, ConfigurationError)):
                AFJitterConfig(**fields)
        with self.assertRaises((ValueError, ConfigurationError)):
            AFConfig(peer_layer_barrier=1)


class PeerLayerBarrierTest(unittest.TestCase):
    def runtime(self):
        env, ex, _ = batch_runtime(owners=2)
        # Construct a separate barrier for direct lifecycle tests.
        proxy = SimpleNamespace(
            env=env,
            cluster_config=SimpleNamespace(
                af=SimpleNamespace(peer_layer_barrier=True),
                af_domains=ex.cluster_config.af_domains,
            ),
            trace=ex.trace,
        )
        return env, ex, PeerLayerBarrier(proxy)

    def test_late_peer_releases_waiter_and_records_wait(self):
        env, ex, gate = self.runtime()
        iterations = [submit_owner(ex, owner)[0] for owner in range(2)]
        for it in iterations:
            gate.register(it)
        a = env.process(gate.wait(iterations[0].blocks[0], 0, ()))
        env.run()
        self.assertFalse(a.triggered)

        def late():
            yield env.timeout(3)
            yield from gate.wait(iterations[1].blocks[0], 0, ())

        b = env.process(late())
        env.run()
        self.assertTrue(a.triggered and b.triggered)
        waits = [
            e.queue_duration
            for e in ex.trace.events
            if e.operation == "peer_layer_barrier"
        ]
        self.assertEqual(sorted(waits), [0, 3])
        for it in iterations:
            gate.complete(it)
        self.assertFalse(gate.rounds)

    def test_mismatched_tail_aborts_and_wakes_peer(self):
        env, ex, gate = self.runtime()
        first = submit_owner(ex, 0, count=2)[0]
        gate.register(first)
        task = env.process(gate.wait(first.blocks[0], 0, ()))
        env.run()
        second = submit_owner(ex, 1, count=3)[0]
        with self.assertRaisesRegex(ConfigurationError, "matching microbatch"):
            gate.register(second)
        with self.assertRaises(SimulationStateError):
            env.run()
        self.assertTrue(task.triggered)

    def test_cancel_wakes_other_peers(self):
        env, ex, gate = self.runtime()
        first = submit_owner(ex, 0)[0]
        gate.register(first)
        env.process(gate.wait(first.blocks[0], 0, ()))
        env.run()
        gate.abort("cancelled")
        with self.assertRaisesRegex(SimulationStateError, "cancelled"):
            env.run()
