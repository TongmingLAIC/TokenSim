from __future__ import annotations

from TokenSim.config.simulation import SimulationConfig

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import simpy

from TokenSim.config.config import ClusterConfig, KVTransferConfig, WorkerGroupConfig
from TokenSim.llm.llm_engine import LLMEngine, Task
from TokenSim.llm.llm_request import Request, RequestStatus
from tests.support.hardware import make_hardware
from TokenSim.workload.source import LLMSource
from TokenSim.reporting import (
    RunReport,
    export_result,
    get_connector_stats,
    get_mooncake_stats,
)

from tests.support.pd import (
    DECODE_LATENCY,
    PREFILL_LATENCY,
    _DeterministicBackend,
    _DEVICE,
    _MODEL,
    _expected_transfer_latency,
    _pd_cluster,
    _workload_config,
)


def _run_single_request_p2p() -> (
    tuple[LLMEngine, Request, _DeterministicBackend, float]
):
    env = simpy.Environment()
    cluster = _pd_cluster()
    backend = _DeterministicBackend()
    with patch(
        "TokenSim.llm.engine.worker.build_latency_backend", return_value=backend
    ):
        engine = LLMEngine(
            env=env,
            kv_transfer_config=cluster.effective_kv_transfer(),
            model_spec=_MODEL,
            workload_config=_workload_config(),
            cluster_config=cluster,
            hardware=make_hardware(_DEVICE, models=[_MODEL]),
            config=SimulationConfig(
                block_size=16,
                batching="paged-attn",
                prefill_worker_pool_type="round_robin",
                decode_worker_pool_type="round_robin",
                max_num_seqs=8,
                max_occupy_ratio=1,
            ),
        )
    request = Request(id=0, prefill_len=32, decode_len=2, block_size=16)

    env.process(
        LLMSource(
            env=env, engine=engine, requests=[request], qps=1, distribution="burst"
        )
    )

    def stop_when_done():
        deadline = 0.1
        while env.now < deadline:
            if request.is_done:
                duration = env.now
                engine.send_task(engine, Task.STOP)
                yield env.timeout(1e-6)
                return duration
            yield env.timeout(1e-6)
        raise AssertionError("P2P integration request did not finish")

    monitor = env.process(stop_when_done())
    env.run(until=monitor)
    return engine, request, backend, monitor.value


class KVTransferIntegrationTest(unittest.TestCase):
    def test_mooncake_master_shares_across_engines_but_respects_namespaces(self):
        env = simpy.Environment()
        cluster = ClusterConfig(
            num_workers=1,
            networks={"node": "ethernet-test"},
            worker_groups=[WorkerGroupConfig("hybrid", "TestGPU", 1, "node")],
        )
        hardware = make_hardware(_DEVICE, models=[_MODEL])

        def make_engine(identity, role, prefix="", options=None):
            return LLMEngine(
                env=env,
                model_spec=_MODEL,
                hardware=hardware,
                cluster_config=cluster,
                kv_transfer_config=KVTransferConfig(
                    kv_connector="MooncakeStoreConnector",
                    engine_id=identity,
                    kv_role=role,
                    kv_connector_extra_config={
                        "master_server_address": "shared-master",
                        "cache_prefix": prefix,
                        "global_segment_size": 4096,
                        "local_buffer_size": 64,
                        "host_available_bytes": 65536,
                        **(options or {}),
                    },
                ),
                config=SimulationConfig(
                    batching="paged-attn",
                    block_size=16,
                    latency_backend_type="analytical",
                    max_num_batched_tokens=16,
                ),
            )

        producer = make_engine("producer", "kv_producer")
        consumer = make_engine(
            "consumer",
            "kv_consumer",
            options={
                "global_segment_size": 2048,
                "local_buffer_size": 32,
                "admission_policy": "never",
                "load_async": True,
                "save_policy": "once",
            },
        )
        left, right = producer.workers[0].connector, consumer.workers[0].connector
        self.assertIs(left.service, right.service)
        self.assertEqual(left.service.store.memory_capacity_bytes, 6144)

        def run(engine, identity, hashes=(11, 22)):
            request = Request(identity, 32, 2, 16, hash_ids=list(hashes))
            request.arrive(env)
            engine.add_requests([request])
            env.run(until=request.completion_event)
            engine.raise_if_failed()
            return request

        run(producer, 0)
        producer_stats = get_mooncake_stats(producer)
        loaded = run(consumer, 1)
        self.assertGreater(loaded.cached_prefill_tokens, 0)
        self.assertEqual(get_mooncake_stats(producer), producer_stats)
        self.assertEqual(get_mooncake_stats(consumer)["mooncake_put_count"], 0)
        consumer.reset_profile_stats()
        self.assertEqual(get_mooncake_stats(producer), producer_stats)
        self.assertEqual(get_mooncake_stats(consumer)["mooncake_get_count"], 0)
        self.assertEqual(get_connector_stats(consumer)["connector_transfer_count"], 0)
        run(producer, 4, (33, 44))
        run(consumer, 3, (33, 44))
        self.assertGreater(get_connector_stats(consumer)["connector_transfer_count"], 0)
        run(producer, 5, (55, 66))
        cancelled = Request(6, 32, 2, 16, hash_ids=[55, 66])
        cancelled.arrive(env)
        consumer.add_requests([cancelled])

        def cancel_during_load():
            yield env.timeout(10e-6)
            yield consumer.cancel_requests([cancelled])[0]

        env.run(until=env.process(cancel_during_load()))
        consumer.raise_if_failed()
        self.assertTrue(cancelled.cancel_requested)
        self.assertFalse(right._pending_async_loads)
        self.assertFalse(right._pending_async_puts)
        isolated = make_engine("isolated", "kv_consumer", "other-prefix")
        self.assertIs(isolated.workers[0].connector.service, left.service)
        self.assertEqual(run(isolated, 2).cached_prefill_tokens, 0)

    def test_p2p_transfer_timing_and_exported_metrics(self):
        engine, request, backend, duration = _run_single_request_p2p()

        size_per_token = engine.workers[0].cache_config.size_per_token
        self.assertEqual(size_per_token, 4)
        expected_blocks = 2
        expected_bytes = expected_blocks * 16 * size_per_token
        expected_transfer_latency = _expected_transfer_latency(expected_bytes)

        stats = get_connector_stats(engine)

        self.assertEqual(request.status, RequestStatus.FINISHED_STOPPED)
        self.assertEqual(
            [row.id for row in engine.timing_recorder.timing.time], [request.id]
        )
        self.assertEqual(backend.calls, [("prefill", 1), ("decode", 1)])
        counts = dict(
            connector_transfer_count=2,
            connector_transfer_blocks=expected_blocks * 2,
            connector_transfer_bytes=expected_bytes * 2,
            connector_producer_count=1,
            connector_consumer_count=1,
            connector_placement_decision_count=2,
        )
        self.assertEqual({key: stats[key] for key in counts}, counts)
        wait = expected_transfer_latency
        for key, expected in (
            ("connector_transfer_latency", 2 * wait),
            ("connector_load_wait_time", wait),
            ("connector_save_wait_time", wait),
        ):
            self.assertAlmostEqual(stats[key], expected, delta=1e-15)
        self.assertEqual(request.prefill_batch_size, 1)
        self.assertEqual(request.decode_batch_sum, 1)
        self.assertAlmostEqual(
            request.prefill_service_time, PREFILL_LATENCY, delta=1e-15
        )
        self.assertAlmostEqual(
            request.decode_service_time_sum, DECODE_LATENCY, delta=1e-15
        )
        self.assertAlmostEqual(
            request.decode_time_sum, DECODE_LATENCY + wait, delta=1e-15
        )
        self.assertGreater(duration, PREFILL_LATENCY + DECODE_LATENCY)

        with tempfile.TemporaryDirectory() as tmpdir:
            export_result(
                run=RunReport(results_path=Path(tmpdir), qps=1, batching="paged-attn"),
                g_time=engine.timing_recorder.timing,
                engine=engine,
                cluster=_pd_cluster(),
                request_count=1,
                prefill_lens=[request.prefill_len],
                decode_lens=[request.decode_len],
                requests=[request],
                notdone=[],
                duration=duration,
            )
            result = json.loads((Path(tmpdir) / "result_1.json").read_text())

        for key, value in stats.items():
            self.assertEqual(result[key], value)


if __name__ == "__main__":
    unittest.main()
