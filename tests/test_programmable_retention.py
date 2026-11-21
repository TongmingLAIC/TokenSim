"""KV retention, eviction order, and CLI behavior with cache control."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from TokenSim.block.block import Device, PhysicalTokenBlock
from TokenSim.block.free_queue import FreeBlockQueue
from TokenSim.block.priority_queue import PriorityFreeMap
from TokenSim.cli import build_parser
from TokenSim.cli.run import main
from TokenSim.kv_transfer.retention import RetentionRegistry
from TokenSim.llm.paged_scheduling import PagedSchedulingMixin

ROOT = Path(__file__).resolve().parents[1]


class ProgrammableRetentionTest(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.blocks = [
            SimpleNamespace(block_number=i, block_hash=f"hash-{i}", ref_count=0)
            for i in range(3)
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

    def test_retention_changes_eviction_order_without_pinning(self):
        self.retain(0, priority=100)
        self.retain(2, priority=100)
        self.assertEqual([self.blocks[i].ref_count for i in (0, 2)], [0, 0])
        self.assertEqual(self.queue.popitem(last=False)[0], 1)
        self.assertCountEqual(
            [self.queue.popitem(last=False)[0] for _ in range(2)], [0, 2]
        )

    def test_expiration_restores_original_lru_position(self):
        self.retain(0)
        self.now = 11
        self.assertEqual(self.queue.popitem(last=False)[0], 0)

    def test_shared_owners_release_independently(self):
        self.retain(0, owner="a", priority=25)
        self.retain(0, owner="b", priority=80)
        self.registry.release("b")
        self.assertEqual(self.registry.priority("pool", self.blocks[0])[0], 25)
        self.registry.release("a")
        self.assertEqual(self.queue.popitem(last=False)[0], 0)

    def test_stale_retry_cannot_replace_newer_claim(self):
        self.retain(0, revision=3)
        self.assertFalse(self.retain(0, revision=2, priority=0))
        self.assertEqual(self.registry.priority("pool", self.blocks[0])[0], 50)

    def test_recycled_content_drops_old_priority(self):
        self.retain(0)
        self.blocks[0].block_hash = "different-content"
        self.queue.refresh(0)
        self.assertEqual(self.queue.popitem(last=False)[0], 0)

    def test_reset_releases_claims(self):
        self.retain(0)
        self.registry.reset()
        self.assertEqual(self.queue.popitem(last=False)[0], 0)
        self.assertFalse(self.registry.claims)

    def test_uniform_queue_prefers_uncached_blocks_then_uses_cached_recency(self):
        queue = FreeBlockQueue()
        queue.enable_priority(lambda slot: (50, float("inf")), lambda: self.now)
        cached = [PhysicalTokenBlock(Device.GPU, i, 16) for i in range(2)]
        for block, stamp in zip(cached, (20, 10)):
            block.last_accessed = stamp
            queue.append_lru(block)
        fresh = PhysicalTokenBlock(Device.GPU, 2, 16)
        queue.append(fresh)
        self.assertEqual([queue.popleft().block_number for _ in range(3)], [2, 1, 0])


class CacheControlRuntimeTest(unittest.TestCase):
    def test_rejected_admission_preserves_retention_declaration(self):
        released = []
        scheduler = SimpleNamespace(
            block_manager=SimpleNamespace(
                grouped_cache=False, can_allocate=lambda req: False
            ),
            connector=SimpleNamespace(async_offload=False),
            defer_prefills=False,
            stats={"full_input_gate": 0},
            cache_controller=SimpleNamespace(
                resumed=lambda req: released.append(req),
                defer_admission=lambda scheduler, req: False,
            ),
        )
        self.assertFalse(PagedSchedulingMixin._admit_waiting(scheduler, object(), None))
        self.assertFalse(released)

    def simulate(self, directory, mode):
        directory.mkdir()
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
                mode,
                "--results_path",
                str(directory / "result"),
            ]
        )
        with redirect_stdout(io.StringIO()):
            engine = main(args)
        path = next((directory / "result").glob("result_*.json"))
        return engine, json.loads(path.read_text())

    def test_empty_controller_preserves_simulated_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, baseline = self.simulate(Path(tmp) / "apc", "apc")
            b, empty = self.simulate(Path(tmp) / "noop", "noop")
        for result in [baseline, empty]:
            result.pop("simulator_wall_time", None)
            for worker in result["scheduler_stats"]:
                worker["counts"].pop("kv_control_completions", None)
        self.assertEqual(baseline, empty)
        self.assertIsNone(a.cache_controller)
        self.assertGreater(b.cache_controller.stats["completions"], 0)

    def test_retention_keeps_blocks_evictable_and_releases_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, result = self.simulate(Path(tmp) / "retain", "retain")
        self.assertEqual(result["termination_reason"], "completed")
        self.assertFalse(result["notdone"])
        self.assertGreater(engine.cache_controller.stats["retained_blocks"], 0)
        for worker in engine.workers:
            manager = getattr(worker.scheduler, "block_manager", None)
            if manager:
                self.assertEqual(manager.get_num_allocated_blocks(), 0)
