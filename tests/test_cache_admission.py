"""Physical KV budgets, admission quotas, and waiting-queue progress."""

from pathlib import Path
from types import SimpleNamespace as NS
import unittest

from TokenSim.config.cache_config import CacheConfig
from TokenSim.config.parallel_config import ParallelConfig
from TokenSim.errors import ConfigurationError
from TokenSim.hardware import HardwareContext
from TokenSim.config.cache_policy import PRESETS
from TokenSim.kv_transfer.policies.native import NativeCacheController
from TokenSim.kv_transfer.policies.quota import should_defer

ROOT = Path(__file__).resolve().parents[1]


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
            req.source = NS(trace_id="tree", play_id=3, lane=2)
        self.assertEqual(session_id(parent), session_id(child))
        child.source.lane = 4
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
