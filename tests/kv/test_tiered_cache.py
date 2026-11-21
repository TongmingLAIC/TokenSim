"""Tiered copies preserve finite capacity, DMA pins and completion visibility."""

from types import SimpleNamespace as NS
import unittest
import simpy

from TokenSim.config.kv_transfer_config import KVTransferConfig
from TokenSim.errors import ConfigurationError
from TokenSim.kv_transfer.connectors.tiered import TieredKVConnector
from TokenSim.kv_transfer.storage.tiered_config import TieredConfig
from TokenSim.kv_transfer.storage.native import NativePageStore


def connector(policy="write_through", env=None, **options):
    env = env or simpy.Environment()
    extra = dict(
        l2_bytes_per_rank=128,
        l3_bytes=options.pop("l3_bytes", 256),
        staging_bytes_per_rank=16,
        write_policy=policy,
        network_bytes_per_second=100,
        backend_bytes_per_second=100,
        network_latency_us=0,
        backend_latency_us=0,
        transfer_utilization=1,
        **options,
    )
    cfg = KVTransferConfig(
        kv_connector="TieredKVConnector",
        kv_role="kv_both",
        kv_connector_extra_config=extra,
    )
    pins = []
    manager = NS(
        layout=NS(
            physical_slot_bytes=16,
            groups=[NS(group_id=0, group_bytes=8, block_tokens=16)],
        ),
        publish_callbacks=[],
        pin_transfer=lambda b: pins.append(b),
        unpin_transfer=lambda b: pins.remove(b),
    )
    link = NS(bandwidth_per_direction_bytes_per_s=100, latency_us=0)
    worker = NS(
        id=0,
        env=env,
        scheduler=NS(block_manager=manager),
        hardware_context=NS(link=lambda name: link),
        device=NS(host_link="pcie"),
        nettype="net",
        model_spec=NS(cache_signature=lambda: ("model",)),
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1),
        tp_rank=0,
        pp_rank=0,
    )
    result = TieredKVConnector(cfg, NS(hybrid_layout=manager.layout))
    result.bind_runtime(worker)

    def notify():
        if not result.changed.triggered:
            result.changed.succeed()
        result.changed = env.event()

    result.notify = notify
    result.pins = pins
    return result


def publish(c, key, one_shot=False):
    block = NS(block_hash=(0, key, 0))
    c._queue_write(NS(group_bytes=8), 0, block, NS(cache_one_shot=one_shot))
    c.wait_for_save()
    return block


class TieredCacheTest(unittest.TestCase):
    def fill_to_high_watermark(self, c):
        for i in range(6):
            view = c.store.reserve_write((0, str(i), 0), 8)
            c.store.finish_write(view)

    def test_visibility_waits_for_dma_and_remote_backend(self):
        c = connector()
        block = publish(c, "a")
        self.assertIsNone(c.store.lookup(block.block_hash))
        self.assertEqual(c.pins, [block])
        c.env.run(until=0.17)
        self.assertEqual(c.store.lookup(block.block_hash).tier, 2)
        self.assertFalse(c.store.has_remote(block.block_hash))
        self.assertEqual(c.pins, [])
        self.assertEqual(c.store.local.pages[block.block_hash].references, 1)
        c.env.run()
        self.assertTrue(c.store.has_remote(block.block_hash))
        self.assertEqual(c.store.local.pages[block.block_hash].references, 0)
        self.assertEqual(c.service.completed["write"], 8)
        c.store.validate()

    def test_write_back_preserves_source_until_flush_completion(self):
        c = connector("write_back")
        block = publish(c, "a")
        c.env.run()
        self.assertEqual(c.service.completed["write"], 0)
        page = c.store.local.pages[block.block_hash]
        self.assertFalse(c.store.before_evict(page))
        self.assertGreater(page.references, 0)
        self.assertIsNone(c.service.store.lookup(c.store.remote_key(block.block_hash)))
        c.env.run()
        self.assertTrue(c.store.before_evict(page))
        c.store.validate()

    def test_write_around_releases_gpu_after_staging_and_delays_visibility(self):
        c = connector("write_around")
        block = publish(c, "a")
        c.env.run(until=0.17)
        self.assertEqual(c.store.local.used_bytes, 0)
        self.assertEqual(c.pins, [])
        self.assertEqual(c.staging_lock.count, 1)
        self.assertIsNone(c.store.lookup(block.block_hash))
        c.env.run()
        self.assertEqual(c.store.lookup(block.block_hash).tier, 3)
        self.assertEqual(c.pins, [])
        self.assertEqual(c.write_link.bytes, 16)

    def test_shared_l3_deduplicates_equal_shards_and_pins_reads(self):
        env = simpy.Environment()
        left, right = connector("write_around", env), connector("write_around", env)
        a = publish(left, "a")
        env.run()
        self.assertIs(left.service, right.service)
        view = right.store.lookup(a.block_hash)
        self.assertEqual(view.tier, 3)
        right.store.acquire(view)
        self.assertEqual(view.page.references, 1)
        right.store.release(view)
        self.assertEqual(right.service.store.used_bytes, 8)
        right.store.promote(view)
        self.assertEqual(right.store.lookup(a.block_hash).tier, 2)
        right.store.validate()

    def test_tail_flush_exposes_deferred_bytes_and_selective_exclusion(self):
        for policy, expected in [("write_back", 8), ("selective_write_back", 0)]:
            c = connector(policy)
            publish(c, "a", one_shot=True)
            c.env.run()
            c.capture_end_of_workload()
            self.assertEqual(c.tail["before"]["storage"]["l3_missing_payload_bytes"], 8)
            c.env.run(until=c.env.process(c.post_workload_drain()))
            self.assertEqual(c.service.completed["write"], expected)
            c.store.validate()

    def test_profile_counter_baseline_and_unique_copy_coverage(self):
        c = connector()
        a = publish(c, "a")
        c.env.run()
        c.capture_profile_baseline()
        publish(c, "b")
        c.env.run()
        self.assertEqual(c.profile_baseline["shared_l3"]["completed_bytes"]["write"], 8)
        self.assertEqual(c.service.completed["write"], 16)
        remote_order = tuple(c.service.store.lru)
        c.store.describe()
        coverage = c.service.coverage()
        self.assertEqual(tuple(c.service.store.lru), remote_order)
        self.assertEqual(coverage["l2_l3_overlap_payload_bytes"], 16)
        self.assertEqual(coverage["all_tier_unique_payload_bytes"], 16)
        self.assertEqual(coverage["all_tier_unique_objects"], 2)
        self.assertTrue(c.store.has_remote(a.block_hash))

    def test_failed_admission_retries_when_a_pinned_remote_copy_finishes(self):
        c = connector("write_back", l3_bytes=16)
        remote = [c.service.store.reserve_write(("other", i), 8) for i in range(2)]
        for i in range(7):
            view = c.store.reserve_write((0, str(i), 0), 8)
            c.store.finish_write(view)
        self.assertIsNone(c.store.reserve_write((0, "new", 0), 8))
        attempts = c.refused_l3_writes
        self.assertIsNone(c.store.reserve_write((0, "newer", 0), 8))
        self.assertEqual(c.refused_l3_writes, attempts)
        c.service.store.finish_write(remote[0])
        self.assertIsNone(c.store.reserve_write((0, "newer", 0), 8))
        c.env.run()
        view = c.store.reserve_write((0, "newest", 0), 8)
        self.assertIsNotNone(view)
        c.store.finish_write(view)
        c.store.validate()

    def test_watermark_accounts_for_pending_demotions_and_reaches_low(self):
        c = connector("write_back")
        self.fill_to_high_watermark(c)
        c.store.pressure_flush()
        promised = c.pending_eviction_bytes
        self.assertEqual(promised, 32)
        c.store.pressure_flush()
        self.assertEqual(c.pending_eviction_bytes, promised)
        c.env.run()
        self.assertEqual(c.pending_eviction_bytes, 0)
        self.assertGreater(c.service.completed["write"], 0)
        self.assertEqual(c.store.used_bytes, c.store.local.used_bytes)
        self.assertLessEqual(
            c.store.local.used_bytes,
            c.options.low_watermark * c.store.local.capacity_bytes,
        )
        c.store.validate()

    def test_read_touch_cancels_a_pending_watermark_eviction(self):
        c = connector("write_back")
        self.fill_to_high_watermark(c)
        c.store.pressure_flush()
        c.store.lookup((0, "0", 0))
        c.env.run()
        self.assertIn((0, "0", 0), c.store.local.pages)
        self.assertNotIn((0, "1", 0), c.store.local.pages)
        self.assertEqual(c.pending_eviction_bytes, 0)
        c.store.validate()

    def test_copy_only_compatibility_option_keeps_local_residency(self):
        c = connector("write_back", watermark_eviction=False)
        self.fill_to_high_watermark(c)
        c.store.pressure_flush()
        c.env.run()
        self.assertEqual(c.store.local.used_bytes, 96)
        self.assertEqual(c.pending_eviction_bytes, 0)
        self.assertGreater(c.service.completed["write"], 0)
        c.store.validate()

    def test_invalid_capacity_and_policy_are_rejected(self):
        for raw in [
            dict(l2_bytes_per_rank=1),
            dict(write_policy="unknown"),
            dict(low_watermark=0.9, high_watermark=0.8),
            dict(backend="magic"),
        ]:
            with self.assertRaises(ConfigurationError):
                TieredConfig(**raw)


class PageStoreEvictionTest(unittest.TestCase):
    def populated(self):
        store = NativePageStore(24)
        for key in ["a", "b", "c"]:
            store.finish_write(store.reserve_write(key, 8))
        return store

    def test_nested_pins_and_eviction_preserve_the_oldest_eligible_page(self):
        store = self.populated()
        page = store.lookup("b")
        store.acquire(page)
        store.acquire(page)
        store.release(page)
        store.validate()
        store.release(page)
        replacement = store.reserve_write("d", 8)
        self.assertNotIn("a", store.pages)
        self.assertIn("b", store.pages)
        store.finish_write(replacement)
        store.validate()

    def test_veto_can_pin_and_move_the_candidate_before_another_victim(self):
        store = self.populated()

        def before(page):
            if page.key == "a":
                store.acquire(page)
                return False
            return True

        store.before_evict = before
        replacement = store.reserve_write("d", 8)
        self.assertNotIn("b", store.pages)
        self.assertIn("a", store.pages)
        store.release(store.pages["a"])
        store.finish_write(replacement)
        store.validate()

    def test_write_completion_order_does_not_replace_read_recency(self):
        store = NativePageStore(16)
        old = store.reserve_write("old", 8)
        newer = store.reserve_write("newer", 8)
        self.assertIsNone(store.reserve_write("blocked", 8))
        store.finish_write(newer)
        store.finish_write(old)
        store.finish_write(store.reserve_write("replacement", 8))
        self.assertNotIn("old", store.pages)
        self.assertIn("newer", store.pages)
        for _ in range(3000):
            store.lookup("newer")
        store.finish_write(store.reserve_write("replacement2", 8))
        self.assertIn("newer", store.pages)
        store.validate()

    def test_all_vetoes_keep_pages_retryable_and_all_pins_refuse_admission(self):
        store = self.populated()
        visited = []
        store.before_evict = lambda page: visited.append(page.key) or False
        self.assertIsNone(store.reserve_write("d", 8))
        self.assertEqual(visited, ["a", "b", "c"])
        store.validate()
        store.before_evict = None
        for page in store.pages.values():
            store.acquire(page)
        self.assertIsNone(store.reserve_write("d", 8))
        for page in list(store.pages.values()):
            store.release(page)
        store.finish_write(store.reserve_write("d", 8))
        self.assertNotIn("a", store.pages)
        store.validate()
