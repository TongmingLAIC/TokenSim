"""Packet maxima, bounded buffers and ordered receive have observable causes."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
import simpy

from TokenSim.af.execution.barrier import PeerLayerBarrier
from TokenSim.af.execution.jitter import perturb_service, request_multiplier
from TokenSim.af.execution.transport_effects import (
    AFTransportEffects,
    OrderedArrivalStream,
    ReceiverQueue,
    packet_tail,
)
from TokenSim.af.providers.queries import AFServiceEstimate
from TokenSim.config.af.jitter import AFJitterConfig
from TokenSim.config.af.settings import AFConfig
from TokenSim.config.af.transport_effects import AFTransportEffectsConfig
from TokenSim.errors import ConfigurationError, SimulationStateError
from tests.af.support.batching import batch_runtime, submit_owner


class PacketTailTest(unittest.TestCase):
    def test_probability_of_any_late_packet_grows_with_payload(self):
        cfg = AFTransportEffectsConfig(packet_tail_probability=0.01, tail_scale_us=100)
        small = [
            packet_tail(replace(cfg, seed=i), "wire", 4096, (0,)) for i in range(2000)
        ]
        large = [
            packet_tail(replace(cfg, seed=i), "wire", 4096 * 128, (0,))
            for i in range(2000)
        ]
        self.assertLess(sum(v > 0 for v in small) / len(small), 0.03)
        self.assertGreater(sum(v > 0 for v in large) / len(large), 0.65)
        self.assertEqual(packet_tail(AFTransportEffectsConfig(), "wire", 100, (0,)), 0)
        self.assertEqual(large[0], packet_tail(cfg, "wire", 4096 * 128, (0,)))

    def test_bad_packet_distribution_is_rejected(self):
        for kwargs in (
            {"packet_tail_probability": 2},
            {"packet_bytes": 0},
            {"tail_shape": 1},
            {"receive_cohorts": 0},
            {"retry_us": -1},
        ):
            with self.assertRaises((ValueError, ConfigurationError)):
                AFTransportEffectsConfig(**kwargs)

    def test_rank_disturbance_does_not_disappear_in_a_large_batch(self):
        cfg = AFJitterConfig(enabled=True, rank_compute_cv=0.3)
        b = SimpleNamespace(
            request_ids=tuple(range(256)),
            phase="decode",
            context_lengths=(8,) * 256,
            source_a_worker_id=0,
        )
        base = AFServiceEstimate(1, "device")
        price = perturb_service(base, cfg, b, 0, "attention")
        self.assertNotEqual(price.service_duration, 1)
        b.request_ids = (0,)
        self.assertEqual(price, perturb_service(base, cfg, b, 0, "attention"))


class ReceiveQueueTest(unittest.TestCase):
    def test_stream_order_releases_before_consumer_compute_finishes(self):
        env = simpy.Environment()
        stream = OrderedArrivalStream(env, True)
        stream.register(0)
        stream.register(1)
        deliveries = []

        def message(key, flight, compute):
            yield env.timeout(flight)
            yield from stream.arrived(key)
            deliveries.append((key, env.now))
            yield env.timeout(compute)

        env.process(message(0, 10, 100))
        env.process(message(1, 1, 1))
        env.run()
        self.assertEqual(deliveries, [(0, 10), (1, 10)])
        self.assertFalse(stream.order)

    def test_finite_receiver_backpressures_and_releases_cancelled_reservation(self):
        env = simpy.Environment()
        queue = ReceiverQueue(env, 1)
        first = env.process(queue.enter(0))
        second = env.process(queue.enter(1))
        env.run()
        self.assertTrue(first.triggered)
        self.assertFalse(second.triggered)
        self.assertEqual(queue.slots.count, 1)
        queue.release(0)
        env.run()
        self.assertTrue(second.triggered)
        queue.release(1)
        self.assertFalse(queue.entries)

    def test_input_slot_releases_after_all_f_consumers_before_return(self):
        env = simpy.Environment()
        cfg = AFTransportEffectsConfig(enabled=True, receive_cohorts=1)
        ex = SimpleNamespace(
            env=env,
            cluster_config=SimpleNamespace(
                af=SimpleNamespace(transport_effects=cfg),
                af_domains=(SimpleNamespace(name="a"), SimpleNamespace(name="f")),
            ),
        )
        effects = AFTransportEffects(ex)
        block = SimpleNamespace(iteration_id=0, block_id=0)
        dag = SimpleNamespace(
            nodes=tuple(
                SimpleNamespace(kind="compute", source_domain="f", node_id=f"f{rank}")
                for rank in range(2)
            )
        )
        effects.register_layer(block, 0, dag)
        env.process(effects.receiver.enter((0, 0, 0)))
        env.run()
        effects.complete_compute(block, 0, "f0")
        self.assertEqual(effects.receiver.slots.count, 1)
        effects.complete_compute(block, 0, "f1")
        self.assertEqual(effects.receiver.slots.count, 0)
        self.assertFalse(effects.consumers)

    def test_delayed_leg_cannot_post_ahead_of_a_reserved_cohort(self):
        env = simpy.Environment()
        cfg = AFTransportEffectsConfig(
            enabled=True, receive_cohorts=1, ordered_receive=True
        )
        ex = SimpleNamespace(
            env=env,
            cluster_config=SimpleNamespace(af=SimpleNamespace(transport_effects=cfg)),
        )
        effects = AFTransportEffects(ex)

        def block(i):
            return SimpleNamespace(
                iteration_id=i,
                block_id=0,
                request_ids=(i,),
                request_count=1,
                context_lengths=(8,),
                placement=SimpleNamespace(a_dp_rank=i),
            )

        b, a = block(0), block(1)

        def leg(owner, target, delay):
            yield env.timeout(delay)
            step = SimpleNamespace(
                direction="a_to_f",
                payload_bytes=4096,
                network="net",
                target_endpoint=target,
                physical_workers=("a", target),
                node_id=f"{owner.iteration_id}:{target}",
            )
            estimate = AFServiceEstimate(1e-6, "wire", payload_bytes=4096)

            def wire():
                start = env.now
                yield env.timeout(1e-6)
                return start, start, env.now, estimate

            yield from effects.run(owner, 0, step, estimate, wire)

        legs = [env.process(leg(b, "f1", 0)), env.process(leg(b, "f0", 2e-6))]

        def consume_b():
            yield env.all_of(legs)
            effects.release_layer(b, 0)

        b_done = env.process(consume_b())

        def consume_a():
            yield env.process(leg(a, "f0", 1.5e-6))
            effects.release_layer(a, 0)

        a_done = env.process(consume_a())
        env.run()
        self.assertTrue(b_done.triggered and a_done.triggered)
        self.assertEqual(effects.receiver.slots.count, 0)

    def test_basic_executor_releases_input_slots_between_layers(self):
        from tests.af.support.batching import batch_runtime, submit_owner

        env, executor, _ = batch_runtime(
            owners=2,
            layers=2,
            transport_effects=AFTransportEffectsConfig(
                enabled=True, receive_cohorts=1, ordered_receive=True
            ),
        )
        tasks = [
            env.process(executor.run_iteration(submit_owner(executor, owner)[0]))
            for owner in range(2)
        ]
        env.run()
        self.assertTrue(all(task.triggered and task.ok for task in tasks))
        self.assertFalse(executor.transport_effects.receiver.entries)

    def test_cancel_pending_reservation_leaves_no_reserve_process(self):
        env = simpy.Environment()
        receiver = ReceiverQueue(env, 1)
        env.process(receiver.enter(0))

        def waiter():
            try:
                yield from receiver.enter(1)
            except simpy.Interrupt:
                return

        task = env.process(waiter())
        env.run()
        reservation_process = receiver.entries[1]["process"]
        task.interrupt("cancelled")
        receiver.release(1)
        env.run()
        self.assertFalse(reservation_process.is_alive)
        self.assertFalse(receiver.slots.queue)
        receiver.release(0)
        env.run()
        self.assertEqual(receiver.slots.count, 0)

    def test_network_tail_does_not_hold_wire_resource(self):
        env = simpy.Environment()
        cfg = AFTransportEffectsConfig(
            enabled=True,
            packet_tail_probability=1,
            tail_scale_us=1000,
            receive_cohorts=2,
        )
        ex = SimpleNamespace(
            env=env,
            cluster_config=SimpleNamespace(af=SimpleNamespace(transport_effects=cfg)),
        )
        effects = AFTransportEffects(ex)
        occupied = simpy.Resource(env, 1)
        ends = []

        def packet(i):
            b = SimpleNamespace(
                iteration_id=i,
                block_id=0,
                request_ids=(i,),
                request_count=1,
                context_lengths=(8,),
                placement=SimpleNamespace(a_dp_rank=i),
            )
            step = SimpleNamespace(
                direction="a_to_f",
                payload_bytes=8192,
                network="net",
                target_endpoint="f",
                node_id=str(i),
            )

            def wire():
                ready = env.now
                with occupied.request() as reservation:
                    yield reservation
                    start = env.now
                    yield env.timeout(1e-6)
                return (
                    ready,
                    start,
                    env.now,
                    AFServiceEstimate(1e-6, "wire", payload_bytes=8192),
                )

            result = yield from effects.run(
                b, 0, step, AFServiceEstimate(1e-6, "wire", payload_bytes=8192), wire
            )
            ends.append(result)
            effects.release_layer(b, 0)

        env.process(packet(0))
        env.process(packet(1))
        env.run()
        self.assertEqual(effects.inflight_bytes[("net", "a_to_f")], 0)
        self.assertEqual(effects.receiver.slots.count, 0)
        self.assertAlmostEqual(
            max(r[3].details["resource_occupancy_end_at"] for r in ends), 2e-6
        )
        self.assertGreater(max(r[2] for r in ends), 2e-6)


class JitterAndPeerBarrierTest(unittest.TestCase):
    def test_jitter_seed_additivity_rank_cv_and_validation(self):
        block = SimpleNamespace(
            request_ids=(1, 2),
            phase="decode",
            context_lengths=(8, 8),
            source_a_worker_id=0,
        )
        estimate = AFServiceEstimate(1, "test")
        self.assertIs(
            perturb_service(estimate, AFJitterConfig(), block, 0, "a"), estimate
        )
        cfg = AFJitterConfig(enabled=True, compute_cv=0.3, seed=7)
        a = perturb_service(estimate, cfg, block, 0, "a")
        self.assertEqual(a, perturb_service(estimate, cfg, block, 0, "a"))
        self.assertNotEqual(
            a, perturb_service(estimate, replace(cfg, seed=8), block, 0, "a")
        )

        add_cfg = AFJitterConfig(enabled=True, compute_cv=0.2)

        def service(ids):
            b = SimpleNamespace(
                request_ids=ids, phase="decode", context_lengths=(8,) * len(ids)
            )
            return perturb_service(
                AFServiceEstimate(len(ids), "test"), add_cfg, b, 0, "a"
            ).service_duration

        self.assertAlmostEqual(service((1, 2)), service((1,)) + service((2,)))
        values = [request_multiplier(0, "service", i, 0.3) for i in range(4000)]
        self.assertAlmostEqual(sum(values) / len(values), 1, delta=0.025)

        tp_cfg = AFJitterConfig(enabled=True, tp_rank_cv=0.3)
        query = SimpleNamespace(
            direction="collective",
            operation="all_reduce",
            physical_endpoints=("a:0", "a:1"),
        )
        small = perturb_service(
            estimate, tp_cfg, block, 0, "all_reduce", network=True, query=query
        )
        block.request_ids = tuple(range(64))
        block.context_lengths = (8,) * 64
        self.assertEqual(
            small,
            perturb_service(
                estimate, tp_cfg, block, 0, "all_reduce", network=True, query=query
            ),
        )
        for fields in (
            {"compute_cv": -1},
            {"network_cv": float("nan")},
            {"seed": True},
            {"compute_cv": 3},
        ):
            with self.assertRaises((ValueError, ConfigurationError)):
                AFJitterConfig(**fields)
        with self.assertRaises((ValueError, ConfigurationError)):
            AFConfig(peer_layer_barrier=1)

    def test_peer_layer_barrier_wait_mismatch_and_abort(self):
        env, ex, _ = batch_runtime(owners=2)
        proxy = SimpleNamespace(
            env=env,
            cluster_config=SimpleNamespace(
                af=SimpleNamespace(peer_layer_barrier=True),
                af_domains=ex.cluster_config.af_domains,
            ),
            trace=ex.trace,
        )
        gate = PeerLayerBarrier(proxy)
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
        for it in iterations:
            gate.complete(it)
        self.assertFalse(gate.rounds)

        env2, ex2, _ = batch_runtime(owners=2)
        gate2 = PeerLayerBarrier(
            SimpleNamespace(
                env=env2,
                cluster_config=SimpleNamespace(
                    af=SimpleNamespace(peer_layer_barrier=True),
                    af_domains=ex2.cluster_config.af_domains,
                ),
                trace=ex2.trace,
            )
        )
        first = submit_owner(ex2, 0, count=2)[0]
        gate2.register(first)
        task = env2.process(gate2.wait(first.blocks[0], 0, ()))
        env2.run()
        with self.assertRaisesRegex(ConfigurationError, "matching microbatch"):
            gate2.register(submit_owner(ex2, 1, count=3)[0])
        with self.assertRaises(SimulationStateError):
            env2.run()
        self.assertTrue(task.triggered)
