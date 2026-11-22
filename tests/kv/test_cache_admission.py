"""Physical KV budgets, admission quotas, and waiting-queue progress."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

from TokenSim.block.block import Device, PhysicalTokenBlock
from TokenSim.block.pool.free_queue import FreeBlockQueue
from TokenSim.block.pool.priority_queue import PriorityFreeMap
from TokenSim.cli import build_parser
from TokenSim.cli.run import main
from TokenSim.config.cache_config import CacheConfig
from TokenSim.config.parallel_config import ParallelConfig
from TokenSim.errors import ConfigurationError
from TokenSim.hardware import HardwareContext
from TokenSim.config.cache_policy import PRESETS
from TokenSim.kv_transfer.policies.native import NativeCacheController
from TokenSim.kv_transfer.policies.quota import should_defer
from TokenSim.kv_transfer.policies.retention import RetentionRegistry
from TokenSim.llm.llm_request import Request
from TokenSim.llm.paged_scheduling import PagedSchedulingMixin

ROOT = Path(__file__).resolve().parents[2]


class ByteBudgetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hardware = HardwareContext.load(ROOT / "data")
        cls.model = hardware.models.resolve(ROOT / "data/models/deepseek-v4-pro.yaml")
        cls.device = hardware.device("b200_sxm")

    def cache(self, **kwargs):
        return CacheConfig(
            256,
            self.device,
            self.model,
            parallel_config=ParallelConfig(tensor_parallel_size=8),
            speculative_tokens=3,
            **kwargs,
        )

    def test_budget_is_rounded_to_physical_slots(self):
        cache = self.cache(kv_cache_memory_bytes=10 * (1 << 30))
        self.assertEqual(
            cache.num_gpu_blocks, 10 * (1 << 30) // cache.physical_slot_bytes
        )
        self.assertEqual(
            cache.memory_budget["available_kv_bytes"],
            cache.num_gpu_blocks * cache.physical_slot_bytes,
        )

    def test_decode_growth_does_not_reserve_the_global_prefill_envelope(self):
        from types import SimpleNamespace as NS
        from TokenSim.kv_transfer.policies.quota import maximum_slots

        cache = self.cache(kv_cache_memory_bytes=10 * (1 << 30))
        layout = cache.hybrid_layout
        scheduler = NS(
            cache_config=cache, block_manager=NS(grouped_cache=True, layout=layout)
        )
        req = NS(prefill_len=8192, decode_len=256)
        reserved = maximum_slots(scheduler, req)
        self.assertLess(reserved, layout.max_request_slots(8448))
        self.assertEqual(
            reserved, sum(g.max_request_slots(8448, 4) for g in layout.groups)
        )

    def test_budget_cannot_invent_memory_or_hide_another_override(self):
        for kwargs in [
            dict(kv_cache_memory_bytes=1),
            dict(kv_cache_memory_bytes=1000 * (1 << 30)),
            dict(
                kv_cache_memory_bytes=1 << 30, kv_cache_capacity_tokens_per_dp_rank=8192
            ),
        ]:
            with self.assertRaises(ConfigurationError):
                self.cache(**kwargs)


class CacheQuotaTest(unittest.TestCase):
    def request(self, ident, prompt=64, output=16, session="a"):
        return NS(
            id=ident,
            prefill_len=prompt,
            decode_len=output,
            cache_salt=session,
            chat_id=session,
            arrival_at=0.0,
        )

    def scheduler(self, running, owned, shared=(0, 1)):
        blocks = {i: NS(block_number=i) for i in range(16)}
        manager = NS(
            grouped_cache=False,
            block_size=16,
            num_total_gpu_blocks=16,
            block_table=NS(get_blocks=lambda ident: [blocks[i] for i in owned[ident]]),
            kv_cache_manager=NS(
                plan_reuse=lambda req: NS(hit_blocks=[blocks[i] for i in shared])
            ),
        )
        return NS(
            running=running,
            block_manager=manager,
            cache_config=NS(model_spec=NS(max_position_embeddings=4096)),
        )

    def test_shared_references_count_once(self):
        peers = [self.request(0, output=1), self.request(1, output=1)]
        scheduler = self.scheduler(peers, {0: [0, 1, 2, 3], 1: [0, 1, 2, 3]})
        incoming = self.request(2, prompt=32, output=1)
        self.assertFalse(
            should_defer(scheduler, incoming, now=1, fraction=0.5, max_wait_seconds=5)
        )

    def test_session_quota_defers_growth_until_timeout_and_isolates_other_sessions(
        self,
    ):
        peer = self.request(0, output=32)
        incoming = self.request(1)
        scheduler = self.scheduler([peer], {0: [0, 1, 2, 3]})
        self.assertTrue(
            should_defer(scheduler, incoming, now=1, fraction=0.5, max_wait_seconds=5)
        )
        self.assertFalse(
            should_defer(scheduler, incoming, now=6, fraction=0.5, max_wait_seconds=5)
        )
        incoming.cache_salt = "different"
        self.assertFalse(
            should_defer(scheduler, incoming, now=1, fraction=0.5, max_wait_seconds=5)
        )

    def test_play_identity_groups_parent_and_child_without_salt(self):
        from TokenSim.kv_transfer.policies.quota import session_id

        parent, child = self.request(0), self.request(1)
        for req in (parent, child):
            req.cache_salt = None
            req.chat_id = str(req.id)
            req.session_key = ("tree", 3, 2)
        self.assertEqual(session_id(parent), session_id(child))
        child.session_key = ("tree", 3, 4)
        self.assertNotEqual(session_id(parent), session_id(child))

    def test_global_growth_reservation_counts_running_and_loading_after_timeout(self):
        peer = self.request(0, prompt=128, output=64, session="other")
        incoming = self.request(1)
        for status in ("running", "loading"):
            with self.subTest(status=status):
                scheduler = self.scheduler([], {0: [0, 1, 2, 3]}, shared=())
                scheduler.running = [peer] if status == "running" else []
                scheduler.loading = {0: peer} if status == "loading" else {}
                self.assertTrue(
                    should_defer(
                        scheduler, incoming, now=100, fraction=1, max_wait_seconds=5
                    )
                )
        self.assertFalse(
            should_defer(
                self.scheduler([], {}),
                incoming,
                now=100,
                fraction=1,
                max_wait_seconds=5,
            )
        )

    def test_grouped_cache_accounts_for_shared_window_and_full_groups(self):
        scheduler = self.scheduler([self.request(0)], {0: [0, 1, 2, 3]})
        manager = scheduler.block_manager
        blocks = manager.block_table.get_blocks(0)
        manager.grouped_cache = True
        manager.layout = NS(max_request_slots=lambda length: 4)
        manager.kv_cache_manager.plan_reuse = lambda req: NS(
            hit_blocks=blocks[:1],
            group_hits=[(0, [(0, blocks[0])]), (1, list(enumerate(blocks[1:])))],
        )
        self.assertFalse(
            should_defer(
                scheduler, self.request(1), now=1, fraction=0.3, max_wait_seconds=5
            )
        )

    def test_long_hit_precedes_a_cold_head(self):
        cold, hot = self.request(0), self.request(1)
        scheduler = self.scheduler([], {})
        scheduler.waiting = [cold, hot]
        scheduler.block_manager.kv_cache_manager.plan_reuse = lambda req: NS(
            hit_tokens=64 if req is hot else 0, hit_blocks=[]
        )
        controller = NS(
            preset=PRESETS["quota"],
            env=NS(now=0),
            engine=NS(
                config=NS(cache_quota_fraction=0.45, cache_quota_max_wait_seconds=15)
            ),
        )
        NativeCacheController.reorder_waiting(controller, scheduler)
        self.assertEqual(scheduler.waiting, [hot, cold])
        cold.arrival_at = -60
        NativeCacheController.reorder_waiting(controller, scheduler)
        self.assertEqual(scheduler.waiting, [cold, hot])

    def test_deferred_head_does_not_block_an_admissible_request(self):
        peer = self.request(0, prompt=128, output=16)
        large, small = self.request(1, prompt=160), self.request(2, prompt=16)
        scheduler = self.scheduler([peer], {0: list(range(8))}, shared=())
        scheduler.waiting = [large, small]
        scheduler.block_manager.kv_cache_manager.plan_reuse = lambda req: NS(
            hit_tokens=0, hit_blocks=[]
        )
        controller = NS(
            preset=PRESETS["quota"],
            env=NS(now=1),
            engine=NS(
                config=NS(cache_quota_fraction=1, cache_quota_max_wait_seconds=15)
            ),
        )
        NativeCacheController.reorder_waiting(controller, scheduler)
        self.assertEqual(scheduler.waiting, [small, large])


class ProgrammableRetentionTest(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.blocks = [
            NS(block_number=i, block_hash=f"hash-{i}", ref_count=0) for i in range(3)
        ]
        self.registry = RetentionRegistry(lambda: self.now)
        self.queue = PriorityFreeMap(
            lambda key: self.registry.priority("pool", self.blocks[key]),
            lambda: self.now,
        )
        self.registry.bind("pool", self.queue.refresh)
        for i in range(3):
            self.queue[i] = None

    def retain(self, block, owner="a", revision=1, priority=50, ttl=10):
        return self.registry.declare(
            "pool",
            [self.blocks[block]],
            owner=owner,
            revision=revision,
            priority=priority,
            ttl=ttl,
        )

    def test_retention_priority_expiration_owners_and_recycling(self):
        self.retain(0, priority=100)
        self.retain(2, priority=100)
        self.assertEqual([self.blocks[i].ref_count for i in (0, 2)], [0, 0])
        self.assertEqual(self.queue.popitem(last=False)[0], 1)
        self.assertCountEqual(
            [self.queue.popitem(last=False)[0] for _ in range(2)], [0, 2]
        )
        for i in range(3):
            self.queue[i] = None
        self.retain(0, owner="a", revision=3, priority=25)
        self.assertFalse(self.retain(0, owner="a", revision=2, priority=0))
        self.retain(0, owner="b", priority=80)
        self.registry.release("b")
        self.assertEqual(self.registry.priority("pool", self.blocks[0])[0], 25)
        self.now = 11
        self.assertEqual(self.queue.popitem(last=False)[0], 0)
        self.setUp()
        self.retain(0)
        self.blocks[0].block_hash = "different-content"
        self.queue.refresh(0)
        self.assertEqual(self.queue.popitem(last=False)[0], 0)

    def test_uniform_queue_and_retain_cli_keep_blocks_evictable(self):
        queue = FreeBlockQueue()
        queue.enable_priority(lambda slot: (50, float("inf")), lambda: self.now)
        cached = [PhysicalTokenBlock(Device.GPU, i, 16) for i in range(2)]
        for block, stamp in zip(cached, (20, 10)):
            block.last_accessed = stamp
            queue.append_lru(block)
        queue.append(PhysicalTokenBlock(Device.GPU, 2, 16))
        self.assertEqual([queue.popleft().block_number for _ in range(3)], [2, 1, 0])

        released = []
        scheduler = NS(
            block_manager=NS(grouped_cache=False, can_allocate=lambda req: False),
            connector=NS(async_offload=False),
            defer_prefills=False,
            stats={"full_input_gate": 0},
            cache_controller=NS(
                resumed=lambda req: released.append(req),
                defer_admission=lambda scheduler, req: False,
            ),
        )
        self.assertFalse(
            PagedSchedulingMixin._admit_waiting(scheduler, Request(1, 16, 1, 16), None)
        )
        self.assertFalse(released)

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            trace = directory / "requests.jsonl"
            trace.write_text(
                "".join(
                    json.dumps(
                        dict(
                            request_id=i,
                            timestamp=i * 0.1,
                            input_length=64,
                            output_length=2,
                            chat_id="parent",
                            turn=i,
                            hash_ids=[1, 2, 3, 4],
                        )
                    )
                    + "\n"
                    for i in range(4)
                )
            )
            args = build_parser().parse_args(
                [
                    "--model",
                    str(ROOT / "data/models/llama-7b.yaml"),
                    "--cluster",
                    str(ROOT / "data/clusters/1_a100/h1.json"),
                    "--workload_type",
                    "qwen_jsonl",
                    "--request_count",
                    "4",
                    "--dataset_path",
                    str(trace),
                    "--batching",
                    "paged-attn",
                    "--latency_backend",
                    "analytical",
                    "--max_num_batched_tokens",
                    "128",
                    "--max_num_seqs",
                    "4",
                    "--verbose",
                    "none",
                    "--cache_policy",
                    "retain",
                    "--results_path",
                    str(directory / "result"),
                ]
            )
            with redirect_stdout(io.StringIO()):
                engine = main(args)
            result = json.loads(
                next((directory / "result").glob("result_*.json")).read_text()
            )
        self.assertEqual(result["termination_reason"], "completed")
        self.assertGreater(engine.cache_controller.stats["retained_blocks"], 0)
