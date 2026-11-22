"""Endpoint contention, DMA ownership and receiver capacity use native events."""

from collections import Counter
from fractions import Fraction
import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import simpy

from TokenSim.config.config import KVTransferConfig
from TokenSim.config.simulation import SimulationConfig
from TokenSim.errors import SimulationStateError
from TokenSim.reporting.stats import get_connector_stats
from TokenSim.kv_transfer.transport.fabric import EndpointFabric, fair_rates
from TokenSim.kv_transfer.transport.uniform import PDHandoff
from TokenSim.llm.llm_engine import LLMEngine
from TokenSim.llm.llm_request import Request
from TokenSim.workload.source import LLMSource
from tests.af.runtime import test_kv as kv_tests
from tests.af.support.config import _cluster as _af_cluster
from tests.af.support.model import _engine as _af_engine, _request as _af_request
from tests.support.hardware import make_hardware
from tests.support.pd import (
    _pd_cluster,
    _DeterministicBackend,
    _MODEL,
    _DEVICE,
    _workload_config,
    _expected_transfer_latency,
)


class EndpointFabricTest(unittest.TestCase):
    def test_shared_sender_and_disjoint_receivers_use_residual_capacity(self):
        rates = fair_rates(
            {
                0: ("s0", "r0", "fabric"),
                1: ("s0", "r1", "fabric"),
                2: ("s1", "r2", "fabric"),
            },
            {"s0": 100, "s1": 100, "r0": 100, "r1": 100, "r2": 100, "fabric": 180},
        )
        self.assertEqual(rates, {0: 50, 1: 50, 2: 80})

    def test_arrivals_rebalance_remaining_bytes_and_receiver_bandwidth(self):
        env = simpy.Environment()
        fabric = EndpointFabric(
            env,
            send_bytes_per_second=100,
            receive_bytes_per_second=60,
            fabric_bytes_per_second=200,
        )
        env.process(fabric.transfer(0, 0, 90))

        def later():
            yield env.timeout(0.5)
            yield env.process(fabric.transfer(1, 0, 60))

        env.process(later())
        env.run()
        self.assertEqual(len(fabric.completed), 2)
        self.assertAlmostEqual(fabric.completed[0]["end"], 2.5)
        self.assertAlmostEqual(fabric.completed[1]["end"], 2.5)
        self.assertAlmostEqual(
            sum(
                (interval["end"] - interval["start"]) * sum(interval["rates"].values())
                for interval in fabric.intervals
            ),
            150,
        )

    def test_late_high_bandwidth_and_zero_byte_flows_avoid_clock_livelock(self):
        rate = 50_000_000_000
        env = simpy.Environment(initial_time=1_000_000.0)
        fabric = EndpointFabric(
            env,
            send_bytes_per_second=rate,
            receive_bytes_per_second=rate,
            fabric_bytes_per_second=rate,
        )
        process = env.process(fabric.transfer(0, 1, 1_000_003))
        for _ in range(100):
            if process.triggered:
                break
            env.step()
        self.assertTrue(process.triggered)
        ideal = Fraction.from_float(1_000_000.0) + Fraction(1_000_003, rate)
        error = abs(float(Fraction.from_float(fabric.completed[0]["end"]) - ideal))
        self.assertLessEqual(error, 2 * math.ulp(float(ideal)))
        env.process(fabric.transfer(0, 1, 0))
        env.run()
        self.assertFalse(fabric.active)


def setup_transport(capacity=None, requests=None, *, transfer=None):
    env = simpy.Environment()
    cluster = _pd_cluster()
    cluster.kv_cache_capacity_tokens_per_dp_rank = capacity
    transfer = transfer or KVTransferConfig(
        kv_connector="P2PConnector",
        kv_connector_extra_config={
            "pd_endpoint_limits": {
                "send_bytes_per_second": 1280,
                "receive_bytes_per_second": 2560,
                "fabric_bytes_per_second": 5120,
            }
        },
    )
    with patch(
        "TokenSim.llm.engine.worker.build_latency_backend",
        return_value=_DeterministicBackend(),
    ):
        engine = LLMEngine(
            env=env,
            kv_transfer_config=transfer,
            model_spec=_MODEL,
            workload_config=_workload_config(),
            cluster_config=cluster,
            hardware=make_hardware(_DEVICE, models=[_MODEL]),
            config=SimulationConfig(
                block_size=16,
                batching="paged-attn",
                max_num_seqs=8,
                decode_worker_pool_type="round_robin",
            ),
        )
    req = Request(id=0, prefill_len=32, decode_len=3, block_size=16)
    env.process(LLMSource(env, engine, requests or [req], 1, "burst"))
    return env, engine, req


class PDTransportIntegrationTest(unittest.TestCase):
    def test_source_prefix_survives_a_completed_handoff(self):
        first = Request(
            id=0, prefill_len=32, decode_len=3, block_size=16, hash_ids=[1, 2]
        )
        env, engine, _ = setup_transport(requests=[first])
        env.run()
        engine.raise_if_failed()
        second = Request(
            id=1, prefill_len=32, decode_len=3, block_size=16, hash_ids=[1, 2]
        )
        second.arrive(env)
        engine.add_requests([second])
        env.run()
        engine.raise_if_failed()
        self.assertTrue(second.is_done)
        self.assertEqual(second.first_prefill_source.gpu_hit_tokens, 31)

    def test_capacity_backpressure_yields_until_dma_and_decode_release(self):
        requests = [
            Request(id=i, prefill_len=32, decode_len=3, block_size=16) for i in range(3)
        ]
        env, engine, _ = setup_transport(capacity=64, requests=requests)
        env.run()
        engine.raise_if_failed()
        self.assertTrue(all(r.is_done for r in requests))
        self.assertEqual(len(engine.pd_handoff.fabric.completed), 3)
        self.assertEqual(engine.pd_handoff.report()["receiver_reserved_pages"], 0)
        for worker in engine.request_workers:
            self.assertEqual(
                worker.scheduler.block_manager.get_num_allocated_blocks(), 0
            )

    def test_af_decode_replica_delivery_and_tp_shard_reservations(self):
        manager = kv_tests.ReplicaKVBlockManagerTest()._manager(gpu=(2, 2))
        request = _af_request(0, prefill_len=8)
        manager.transfer_reservations[request.id] = 2
        manager.pd_handoff = SimpleNamespace(admitted=Mock())
        manager.reserve_gpu_blocks(2)
        self.assertTrue(manager.can_allocate(request))
        manager.allocate(request)
        manager.pd_handoff.admitted.assert_called_once_with(request)
        manager.release_request_blocks(request)
        self.assertTrue(
            all(m.get_num_free_blocks() == 2 for m in manager.rank_managers.values())
        )

        env, engine = _af_engine(_af_cluster(a_tp=1, a_dp=4, f_tp=1, f_dp=1, f_ep=1))
        engine.pd_handoff = PDHandoff(
            engine,
            dict(
                send_bytes_per_second=1e6,
                receive_bytes_per_second=1e6,
                fabric_bytes_per_second=4e6,
            ),
        )
        requests = [Request(i, 8, 3, block_size=4) for i in range(4)]
        for r in requests:
            r.dp_rank = 0
        env.process(LLMSource(env, engine, requests, 1, "burst"))
        env.run()
        engine.raise_if_failed()
        starts = [
            row for row in engine.pd_handoff.events if row["kind"] == "transfer_start"
        ]
        self.assertEqual(
            Counter(row["target_dp_rank"] for row in starts), Counter(range(4))
        )
        self.assertEqual(engine.pd_handoff.report()["receiver_reserved_pages"], 0)

        env2, tp_engine = _af_engine(
            _af_cluster(a_tp=2, a_dp=1, f_tp=1, f_dp=1, f_ep=1)
        )
        tp_engine.pd_handoff = PDHandoff(
            tp_engine,
            dict(
                send_bytes_per_second=1e6,
                receive_bytes_per_second=1e6,
                fabric_bytes_per_second=2e6,
            ),
        )
        tp_req = Request(0, 8, 3, block_size=4)
        env2.process(LLMSource(env2, tp_engine, [tp_req], 1, "burst"))
        env2.run()
        tp_engine.raise_if_failed()
        self.assertTrue(tp_req.is_done)
        self.assertEqual(len(tp_engine.pd_handoff.fabric.completed), 2)

    def test_sources_pinned_and_target_reserved_until_wire_completion(self):
        env, engine, req = setup_transport()
        env.run(until=0.02)
        handoff = engine.pd_handoff
        self.assertIn(req.id, handoff.jobs)
        job = handoff.jobs[req.id]
        self.assertTrue(job["blocks"])
        self.assertTrue(all(block.ref_count > 0 for block in job["blocks"]))
        self.assertGreater(job["reserved"], 0)
        self.assertEqual(len(job["target"].scheduler.running), 0)
        self.assertFalse(req.is_done)
        env.run()
        engine.raise_if_failed()
        self.assertTrue(req.is_done)
        self.assertEqual(handoff.report()["receiver_reserved_pages"], 0)
        self.assertEqual(handoff.report()["pinned_source_pages"], 0)
        flow = handoff.fabric.completed[0]
        self.assertAlmostEqual(flow["end"] - flow["start"], flow["bytes"] / 1280)
        admitted = next(e for e in handoff.events if e["kind"] == "decode_admitted")
        self.assertGreaterEqual(admitted["time"], flow["end"])
        for worker in engine.request_workers:
            manager = worker.scheduler.block_manager
            self.assertEqual(manager.get_num_allocated_blocks(), 0)

    def test_failed_dma_releases_source_and_receiver_ownership(self):
        env, engine, req = setup_transport()
        handoff = engine.pd_handoff
        original = handoff.fabric.transfer

        def interrupted_peer(source, target, bytes_):
            if target == 100:
                yield env.timeout(0.02)
                raise ValueError("injected DMA failure")
            yield from original(source, target, bytes_)

        handoff.fabric.transfer = interrupted_peer
        payloads = handoff._payloads
        handoff._payloads = lambda job: [*payloads(job), (100, 1)]
        env.run()
        with self.assertRaisesRegex(SimulationStateError, "injected DMA failure"):
            engine.raise_if_failed()
        self.assertFalse(req.owner_worker_ids)
        self.assertEqual(handoff.events[-1]["kind"], "transfer_failed")
        report = handoff.report()
        self.assertEqual(report["network"]["active_flows"], 0)
        self.assertEqual(report["network"]["completed"], [])
        self.assertEqual(report["receiver_reserved_pages"], 0)
        self.assertEqual(report["pinned_source_pages"], 0)
        self.assertEqual(report["pending_requests"], 0)
        for worker in engine.request_workers:
            self.assertEqual(
                worker.scheduler.block_manager.get_num_allocated_blocks(), 0
            )

    def test_af_decode_replica_delivery_and_tp_shard_reservations(self):
        manager = kv_tests.ReplicaKVBlockManagerTest()._manager(gpu=(2, 2))
        request = _af_request(0, prefill_len=8)
        manager.transfer_reservations[request.id] = 2
        manager.pd_handoff = SimpleNamespace(admitted=Mock())
        manager.reserve_gpu_blocks(2)
        self.assertTrue(manager.can_allocate(request))
        manager.allocate(request)
        manager.pd_handoff.admitted.assert_called_once_with(request)
        manager.release_request_blocks(request)
        self.assertTrue(
            all(m.get_num_free_blocks() == 2 for m in manager.rank_managers.values())
        )

        env, engine = _af_engine(_af_cluster(a_tp=1, a_dp=4, f_tp=1, f_dp=1, f_ep=1))
        engine.pd_handoff = PDHandoff(
            engine,
            dict(
                send_bytes_per_second=1e6,
                receive_bytes_per_second=1e6,
                fabric_bytes_per_second=4e6,
            ),
        )
        requests = [Request(i, 8, 3, block_size=4) for i in range(4)]
        for r in requests:
            r.dp_rank = 0
        env.process(LLMSource(env, engine, requests, 1, "burst"))
        env.run()
        engine.raise_if_failed()
        starts = [
            row for row in engine.pd_handoff.events if row["kind"] == "transfer_start"
        ]
        self.assertEqual(
            Counter(row["target_dp_rank"] for row in starts), Counter(range(4))
        )
        self.assertEqual(engine.pd_handoff.report()["receiver_reserved_pages"], 0)

        env2, tp_engine = _af_engine(
            _af_cluster(a_tp=2, a_dp=1, f_tp=1, f_dp=1, f_ep=1)
        )
        tp_engine.pd_handoff = PDHandoff(
            tp_engine,
            dict(
                send_bytes_per_second=1e6,
                receive_bytes_per_second=1e6,
                fabric_bytes_per_second=2e6,
            ),
        )
        tp_req = Request(0, 8, 3, block_size=4)
        env2.process(LLMSource(env2, tp_engine, [tp_req], 1, "burst"))
        env2.run()
        tp_engine.raise_if_failed()
        self.assertTrue(tp_req.is_done)
        self.assertEqual(len(tp_engine.pd_handoff.fabric.completed), 2)

    def test_storage_connectors_do_not_absorb_legacy_pd_transfer(self):
        baseline, _, prior = setup_transport(
            transfer=KVTransferConfig(kv_connector="P2PConnector")
        )
        baseline.run()
        store = {
            "kv_connector": "MooncakeStoreConnector",
            "kv_connector_extra_config": {
                "mode": "embedded",
                "memory_capacity_blocks": 8,
                "local_buffer_size": 4096,
            },
        }
        for settings in (
            store,
            {
                **store,
                "kv_connector": "UniformAsyncStore",
            },
            {
                "kv_connector": "MultiConnector",
                "kv_connector_extra_config": {"connectors": [store]},
            },
        ):
            with self.subTest(connector=settings["kv_connector"]):
                env, engine, req = setup_transport(
                    transfer=KVTransferConfig.from_mapping(settings)
                )
                env.run()
                engine.raise_if_failed()
                self.assertTrue(req.is_done)
                self.assertAlmostEqual(req.total_time, prior.total_time)
                stats = get_connector_stats(engine)
                byte_count = (
                    32 * engine.decode_workers.workers[0].cache_config.size_per_token
                )
                self.assertEqual(stats["connector_transfer_bytes"], byte_count * 2)
                self.assertAlmostEqual(
                    stats["connector_load_wait_time"],
                    _expected_transfer_latency(byte_count),
                )

    def test_cancel_waits_for_dma_and_never_admits_decode(self):
        env, engine, req = setup_transport()
        env.run(until=0.02)
        event = engine.cancel_requests([req])[0]
        self.assertFalse(event.triggered)
        env.run()
        self.assertTrue(event.triggered)
        self.assertGreater(req.cancelled_at, 0.02)
        self.assertFalse(
            any(e["kind"] == "decode_admitted" for e in engine.pd_handoff.events)
        )
        self.assertEqual(engine.pd_handoff.report()["pending_requests"], 0)
        for worker in engine.request_workers:
            self.assertEqual(
                worker.scheduler.block_manager.get_num_allocated_blocks(), 0
            )

    def test_af_decode_replica_delivery_and_tp_shard_reservations(self):
        manager = kv_tests.ReplicaKVBlockManagerTest()._manager(gpu=(2, 2))
        request = _af_request(0, prefill_len=8)
        manager.transfer_reservations[request.id] = 2
        manager.pd_handoff = SimpleNamespace(admitted=Mock())
        manager.reserve_gpu_blocks(2)
        self.assertTrue(manager.can_allocate(request))
        manager.allocate(request)
        manager.pd_handoff.admitted.assert_called_once_with(request)
        manager.release_request_blocks(request)
        self.assertTrue(
            all(m.get_num_free_blocks() == 2 for m in manager.rank_managers.values())
        )

        env, engine = _af_engine(_af_cluster(a_tp=1, a_dp=4, f_tp=1, f_dp=1, f_ep=1))
        engine.pd_handoff = PDHandoff(
            engine,
            dict(
                send_bytes_per_second=1e6,
                receive_bytes_per_second=1e6,
                fabric_bytes_per_second=4e6,
            ),
        )
        requests = [Request(i, 8, 3, block_size=4) for i in range(4)]
        for r in requests:
            r.dp_rank = 0
        env.process(LLMSource(env, engine, requests, 1, "burst"))
        env.run()
        engine.raise_if_failed()
        starts = [
            row for row in engine.pd_handoff.events if row["kind"] == "transfer_start"
        ]
        self.assertEqual(
            Counter(row["target_dp_rank"] for row in starts), Counter(range(4))
        )
        self.assertEqual(engine.pd_handoff.report()["receiver_reserved_pages"], 0)

        env2, tp_engine = _af_engine(
            _af_cluster(a_tp=2, a_dp=1, f_tp=1, f_dp=1, f_ep=1)
        )
        tp_engine.pd_handoff = PDHandoff(
            tp_engine,
            dict(
                send_bytes_per_second=1e6,
                receive_bytes_per_second=1e6,
                fabric_bytes_per_second=2e6,
            ),
        )
        tp_req = Request(0, 8, 3, block_size=4)
        env2.process(LLMSource(env2, tp_engine, [tp_req], 1, "burst"))
        env2.run()
        tp_engine.raise_if_failed()
        self.assertTrue(tp_req.is_done)
        self.assertEqual(len(tp_engine.pd_handoff.fabric.completed), 2)


if __name__ == "__main__":
    unittest.main()
