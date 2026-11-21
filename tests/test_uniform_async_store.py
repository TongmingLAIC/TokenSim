"""Asynchronous uniform KV storage, prefetch, and request observations."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch
import simpy

from TokenSim.config.config import ClusterConfig, WorkerGroupConfig, KVTransferConfig
from TokenSim.config.simulation import SimulationConfig
from TokenSim.llm.llm_engine import LLMEngine, Task
from TokenSim.kv_transfer.busy_intervals import BusyIndex, append_union
from TokenSim.llm.llm_request import Request
from TokenSim.workload.source import LLMSource
from tests.support.hardware import make_hardware
from tests.test_kv_transfer_integration import _MODEL, _DEVICE, _DeterministicBackend


def setup(policy=None, annotated=False):
    env = simpy.Environment()
    cluster = ClusterConfig(
        num_workers=2,
        networks={"net1": "ethernet-test"},
        worker_groups=[
            WorkerGroupConfig(role="hybrid", hardware="TestGPU", num_workers=2)
        ],
    )
    transfer = KVTransferConfig(
        kv_connector="UniformAsyncStore",
        kv_connector_extra_config=dict(
            mode="standalone-store",
            memory_capacity_blocks=0,
            enable_offload=True,
            ssd_capacity_blocks=64,
            ssd_read_bandwidth_bytes_per_s=64,
            ssd_write_bandwidth_bytes_per_s=64,
            ssd_read_latency_us=0,
            ssd_write_latency_us=0,
            online_policy=policy or {},
        ),
    )
    with patch(
        "TokenSim.llm.engine.worker.build_latency_backend",
        return_value=_DeterministicBackend(),
    ):
        engine = LLMEngine(
            env,
            transfer,
            _MODEL,
            cluster,
            make_hardware(_DEVICE, models=[_MODEL]),
            config=SimulationConfig(block_size=16, batching="paged-attn"),
        )
    first = Request(0, 32, 2, 16, hash_ids=[1, 2])
    if annotated:
        first.chat_id = "parent"
        first.source = SimpleNamespace(trace_id="parent", play_id=0, lane=0)
    env.process(LLMSource(env, engine, [first], 1, "burst"))
    return env, engine, first


def prepare(budget=128):
    env, engine, parent = setup(
        dict(prefetch=True, prefetch_budget_bytes=budget), annotated=True
    )
    env.run()
    engine.raise_if_failed()
    manager = engine.workers[0].scheduler.block_manager
    manager.watermark_blocks = 0
    pressure = Request(999, manager.num_total_gpu_blocks * 16, 1, 16)
    manager.allocate(pressure)
    manager.release_request_blocks(pressure)
    controller = engine.cache_controller
    controller.tool_wait(parent)
    env.run(until=env.now + 1)
    observed = Request(1, 32, 2, 16, hash_ids=[1, 2], chat_id="parent")
    controller.client_resuming(observed)
    controller.tool_wait(parent)
    return env, engine, controller, parent


class UniformAsyncStoreTest(unittest.TestCase):
    def test_publication_occurs_after_io_and_other_replica_restores_prefix(self):
        env, engine, first = setup()
        env.run(until=0.01)
        connector = engine.workers[0].connector
        keys = connector._keys_for_request(first)
        self.assertEqual(
            connector.service.store.lookup(keys, now=env.now).hit_blocks, 0
        )
        self.assertTrue(engine.workers[0].scheduler.block_manager.transfer_references)
        env.run()
        engine.raise_if_failed()
        self.assertEqual(
            connector.service.store.lookup(keys, now=env.now).hit_blocks, 2
        )
        second = Request(1, 32, 2, 16, hash_ids=[1, 2])
        second.arrive(env)
        engine.add_requests([second])
        env.run()
        engine.raise_if_failed()
        self.assertTrue(second.is_done)
        self.assertEqual(second.first_prefill_source.cpu_hit_tokens, 16)
        self.assertEqual(engine.workers[1].connector.mooncake_stats.get_count, 1)
        self.assertEqual(connector.state.program_bytes, 128)
        self.assertEqual(connector.state.read_bytes, 64)
        self.assertEqual(connector.state.unread_program_bytes, 64)
        connector.store.validate()
        for worker in engine.request_workers:
            self.assertFalse(worker.scheduler.block_manager.transfer_references)
            self.assertEqual(
                worker.scheduler.block_manager.get_num_allocated_blocks(), 0
            )

    def test_cancelled_load_keeps_dma_pins_until_completion_without_publishing(self):
        env, engine, _ = setup()
        env.run()
        engine.raise_if_failed()
        second = Request(1, 32, 2, 16, hash_ids=[1, 2])
        second.arrive(env)
        engine.add_requests([second])
        env.run(until=env.now + 0.1)
        target = engine.workers[1].scheduler.block_manager
        self.assertTrue(target.transfer_references)
        engine.cancel_requests([second])
        env.run()
        engine.raise_if_failed()
        self.assertTrue(second.cancel_requested)
        probe = Request(2, 32, 2, 16, hash_ids=[1, 2])
        self.assertEqual(target.kv_cache_manager.plan_reuse(probe).hit_tokens, 0)
        self.assertFalse(target.transfer_references)
        self.assertEqual(target.get_num_allocated_blocks(), 0)
        engine.workers[1].connector.store.validate()


class UniformPrefetchTest(unittest.TestCase):
    def test_observed_tool_wait_can_admit_a_known_parent_prefix(self):
        env, engine, parent = setup(dict(reuse_threshold=2), annotated=True)
        env.run()
        engine.raise_if_failed()
        connector = engine.workers[0].connector
        self.assertEqual(connector.state.program_bytes, 0)
        engine.cache_controller.tool_wait(parent)
        env.run()
        engine.raise_if_failed()
        self.assertEqual(connector.state.program_bytes, 128)

    def test_control_copy_is_not_a_served_request_and_restores_known_pages(self):
        env, engine, controller, _ = prepare()
        env.run()
        engine.raise_if_failed()
        self.assertEqual(controller.prefetch.stats["completed"], 1)
        self.assertEqual(controller.prefetch.stats["copied_bytes"], 128)
        req = Request(2, 32, 2, 16, hash_ids=[1, 2], chat_id="parent")
        req.arrive(env)
        controller.client_resuming(req)
        engine.send_task(engine.workers[0], Task.ADD, [req])
        env.run()
        engine.raise_if_failed()
        self.assertEqual(req.first_prefill_source.gpu_hit_tokens, 31)
        self.assertEqual(controller.prefetch.stats["useful_bytes"], 128)
        self.assertEqual(controller.prefetch.stats["benefiting_requests"], 1)
        self.assertFalse(engine.workers[0].scheduler.loading)
        self.assertEqual(
            engine.workers[0].scheduler.block_manager.get_num_allocated_blocks(), 0
        )

    def test_close_preserves_active_io_but_never_publishes_cancelled_copy(self):
        env, engine, controller, _ = prepare()
        env.run(until=env.now + 0.1)
        self.assertTrue(engine.workers[0].connector.loads)
        controller.close()
        self.assertTrue(engine.workers[0].connector.loads)
        env.run()
        engine.raise_if_failed()
        self.assertEqual(controller.prefetch.stats["completed"], 0)
        self.assertEqual(controller.prefetch.stats["cancelled"], 1)
        self.assertFalse(controller.pending)
        manager = engine.workers[0].scheduler.block_manager
        probe = Request(3, 32, 2, 16, hash_ids=[1, 2], chat_id="parent")
        self.assertEqual(manager.kv_cache_manager.plan_reuse(probe).hit_tokens, 0)
        self.assertEqual(manager.get_num_allocated_blocks(), 0)
        engine.workers[0].connector.store.validate()

    def test_prefetch_budget_limits_actual_copied_prefix_bytes(self):
        env, engine, controller, _ = prepare(64)
        env.run()
        engine.raise_if_failed()
        self.assertEqual(controller.prefetch.stats["copied_bytes"], 64)


class UniformSourceObservationTest(unittest.TestCase):
    def test_full_prefix_records_hits_and_leaves_one_real_query(self):
        class ObservedBackend(_DeterministicBackend):
            def __init__(self):
                super().__init__()
                self.queries = []

            def estimate_step_latency(self, requests):
                self.queries.extend(
                    (r.id, r.query_tokens) for r in requests if r.is_prefill
                )
                return super().estimate_step_latency(requests)

        backend = ObservedBackend()
        env = simpy.Environment()
        cluster = ClusterConfig(
            num_workers=1,
            networks={"net1": "ethernet-test"},
            worker_groups=[
                WorkerGroupConfig(role="hybrid", hardware="TestGPU", num_workers=1)
            ],
        )
        with patch(
            "TokenSim.llm.engine.worker.build_latency_backend",
            return_value=backend,
        ):
            engine = LLMEngine(
                env,
                cluster.effective_kv_transfer(),
                _MODEL,
                cluster,
                make_hardware(_DEVICE, models=[_MODEL]),
                config=SimulationConfig(block_size=16, batching="paged-attn"),
            )
        first = Request(
            id=0, prefill_len=32, decode_len=2, block_size=16, hash_ids=[1, 2]
        )
        first.arrive(env)
        engine.add_requests([first])
        env.run()
        second = Request(
            id=1, prefill_len=32, decode_len=2, block_size=16, hash_ids=[1, 2]
        )
        second.arrive(env)
        engine.add_requests([second])
        env.run()
        engine.raise_if_failed()
        self.assertTrue(second.is_done)
        self.assertEqual(second.first_prefill_source.gpu_hit_tokens, 31)
        self.assertEqual(second.first_prefill_source.cpu_hit_tokens, 0)
        self.assertEqual(second.prefill_admission_count, 1)
        self.assertEqual(second.effective_prefill_tokens, 0)
        self.assertIn((1, 1), backend.queries)

    def test_io_window_excludes_overlapping_compute_and_preserves_gaps(self):
        spans = [(0, 2), (1, 3), (3, 4), (6, 7)]
        union = []
        for start, stop in spans:
            append_union(union, start, stop)
        self.assertEqual(union, [(0, 4), (6, 7)])
        index = BusyIndex(spans)
        for start, end, exposed in [(1, 8, 3), (4, 6, 2), (1, 3, 0), (7, 9, 2)]:
            with self.subTest(start=start, end=end):
                self.assertEqual(index.uncovered(start, end), exposed)
