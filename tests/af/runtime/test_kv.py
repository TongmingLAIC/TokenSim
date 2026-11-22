"""AF attention replica KV cache allocation, rollback, reuse, and preemption."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from TokenSim.af.runtime.topology import AFCacheConfig
from TokenSim.block.replica_kv import ReplicaKVBlockManager
from TokenSim.config.config import KVTransferConfig
from TokenSim.errors import OutOfBlocksError, SimulationStateError
from TokenSim.kv_transfer import P2PConnector
from TokenSim.llm.llm_request import Request
from TokenSim.llm.llm_scheduler import LLMPagedAttnScheduler
from tests.af.support.config import _cluster
from tests.af.support.model import _request, _runtime_plan


class ReplicaKVBlockManagerTest(unittest.TestCase):
    def _manager(self, gpu=(8, 8), cpu=(8, 8)):
        cluster = _cluster(a_tp=2, a_dp=1, f_tp=1, f_dp=1, f_ep=1)
        plan = _runtime_plan(cluster)
        replica = plan.replica_for_instance("a-domain", "a0")
        return ReplicaKVBlockManager(
            block_size=4,
            model="TestModel",
            replica=replica,
            gpu_blocks_by_rank=dict(zip(replica.rank_keys, gpu)),
            cpu_blocks_by_rank=dict(zip(replica.rank_keys, cpu)),
        )

    def test_allocate_append_reserve_and_free_across_ranks(self):
        manager = self._manager()
        req = _request(1)
        manager.allocate(req)
        self.assertEqual(manager.get_num_blocks(req), 2)
        self.assertEqual(
            [manager.rank_status(key)["gpu_used"] for key in manager.rank_keys], [2, 2]
        )

        req.generation_idx = 1
        req._append_tokens(1)
        manager.append_slot(req)
        self.assertEqual(manager.get_num_blocks(req), 3)
        self.assertEqual(
            [manager.rank_status(key)["gpu_used"] for key in manager.rank_keys], [3, 3]
        )

        manager.reserve_gpu_blocks(1)
        self.assertEqual(
            [manager.rank_status(key)["gpu_used"] for key in manager.rank_keys], [4, 4]
        )
        manager.release_reserved_blocks(1)
        self.assertEqual(
            [manager.rank_status(key)["gpu_used"] for key in manager.rank_keys], [3, 3]
        )

        manager.free(req)
        self.assertEqual(
            [manager.rank_status(key)["gpu_used"] for key in manager.rank_keys], [0, 0]
        )

    def test_partial_rank_allocation_rolls_back_every_rank_and_request(self):
        manager = self._manager()
        req = _request(2)
        failing_rank = manager.rank_keys[1]

        def fail(_request, *, plan=None):
            raise OutOfBlocksError("injected follower exhaustion")

        manager.rank_managers[failing_rank].allocate = fail
        with self.assertRaisesRegex(OutOfBlocksError, "injected"):
            manager.allocate(req)

        self.assertEqual(req.num_physical_token_blocks, 0)
        self.assertNotIn(req.id, manager._request_views)
        self.assertEqual(
            [manager.rank_status(key)["gpu_used"] for key in manager.rank_keys], [0, 0]
        )

    def test_logical_capacity_uses_smallest_rank_and_owner_conflict_fails(self):
        manager = self._manager(gpu=(100, 80), cpu=(40, 30))
        self.assertEqual(manager.num_total_gpu_blocks, 80)
        self.assertEqual(manager.num_total_cpu_blocks, 30)
        self.assertEqual(manager.get_gpu_status(), (80, 0, 80))

        req = _request(6)
        manager.allocate(req)
        req.af_a_owners[manager.domain] = "another-replica"
        with self.assertRaisesRegex(SimulationStateError, "owner conflict"):
            manager.can_append_slot(req)

    def test_scheduler_preemption_releases_all_rank_shards(self):
        manager = self._manager(gpu=(2, 2))
        cache = SimpleNamespace(
            block_size=4,
            num_gpu_blocks=2,
            num_cpu_blocks=8,
            model="TestModel",
        )
        scheduler = LLMPagedAttnScheduler(
            id=0,
            cache_config=cache,
            max_num_seqs=8,
            max_occupy_ratio=1,
            block_manager=manager,
        )
        req0 = _request(7, prefill_len=4, decode_len=4)
        req1 = _request(8, prefill_len=4, decode_len=4)
        scheduler.add_requests([req0, req1])
        scheduler.schedule_output()
        for req in (req0, req1):
            req.generation_idx = 4
            req._append_tokens(4)
        output = scheduler.schedule_output()
        self.assertTrue(output.preempted)
        victim = output.preempted[0]
        self.assertNotIn(victim.id, manager._request_views)
        self.assertEqual(
            [manager.rank_status(key)["gpu_used"] for key in manager.rank_keys], [2, 2]
        )

    def test_prefix_reuse_keeps_logical_hits_and_rank_refcounts_aligned(self):
        manager = self._manager()
        original = Request(
            10, 8, 5, block_size=4, hash_ids=["a", "b"], output_hash_ids=["c"]
        )
        original.arrive(SimpleNamespace(now=0.0))
        manager.allocate(original)
        for i in range(original.decode_len):
            manager.append_slot(original)
            original.advance(SimpleNamespace(now=i + 1), 1, 1)
        self.assertTrue(original.is_done)
        manager.free(original)

        reused = Request(11, 12, 1, block_size=4, hash_ids=["a", "b", "c"])
        reused.arrive(SimpleNamespace(now=6.0))
        manager.allocate(reused)
        self.assertEqual(reused.reuse_hit_blocks, 3)
        self.assertEqual(manager.get_num_blocks(reused), 3)
        for rank_key in manager.rank_keys:
            rank = manager.rank_managers[rank_key]
            self.assertEqual(
                rank.kv_cache_manager.plan_reuse(reused).hit_block_count, 3
            )
            self.assertTrue(
                all(b.ref_count == 1 for b in rank.block_table.get_blocks(reused.id))
            )

    def test_replica_recomputes_suffix_missing_from_one_tp_rank(self):
        manager = self._manager(gpu=(8, 5))
        for index, hashes in enumerate((["a", "b"], ["c"], ["d"], ["e"], ["a"], ["f"])):
            req = Request(index, 4 * len(hashes), 1, 4, hash_ids=hashes)
            manager.allocate(req)
            req.num_computed_tokens = req.prefill_len
            manager.commit_input_cache(req)
            manager.release_request_blocks(req)

        scheduler = LLMPagedAttnScheduler(
            id=0,
            cache_config=SimpleNamespace(block_size=4),
            block_manager=manager,
        )
        requests = [Request(i, 8, 1, 4, hash_ids=["a", "b"]) for i in (7, 8)]
        self.assertEqual(
            [
                rank.kv_cache_manager.plan_reuse(requests[0]).hit_tokens
                for rank in manager.rank_managers.values()
            ],
            [8, 4],
        )
        scheduler.add_requests(requests)
        self.assertEqual(scheduler.schedule_output().running, requests)
        for req in requests:
            self.assertEqual((req.num_computed_tokens, req.scheduled_tokens), (4, 4))
        for rank in manager.rank_managers.values():
            left, right = (rank.block_table.get_blocks(req.id) for req in requests)
            self.assertIs(left[0], right[0])
            self.assertIsNot(left[1], right[1])

    def test_connector_counts_logical_blocks_once_and_release_clears_all_shards(self):
        cluster = _cluster(a_tp=2, a_dp=1, f_tp=1, f_dp=1, f_ep=1)
        plan = _runtime_plan(cluster)
        replica = plan.replica_for_instance("a-domain", "a0")
        manager = ReplicaKVBlockManager.from_plan(plan, replica)
        cache = AFCacheConfig.from_plan(plan, replica)
        connector = P2PConnector(KVTransferConfig(kv_connector="P2PConnector"), cache)
        req = _request(12)
        manager.allocate(req)

        transfer = connector.build_transfer_plan(
            requests=[req],
            source_worker_id=0,
            target_worker_id=1,
            kind="load",
            latency=0.1,
            request_block_counts=manager.get_request_block_counts([req]),
        )
        self.assertEqual((transfer.blocks, transfer.request_block_counts), (2, {12: 2}))
        manager.release_request_blocks(req)
        self.assertEqual(
            [manager.rank_status(key)["gpu_used"] for key in manager.rank_keys], [0, 0]
        )
