"""DSpark storage budgets yield integral, capacity-conserving physical slots."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest

from TokenSim.block.hybrid.dspark_memory import persistent_allocations
from TokenSim.block.hybrid.kv_layout import hybrid_kv_layout
from TokenSim.block.hybrid.manager import HybridKVBlockManager
from TokenSim.config.cache_config import CacheConfig
from TokenSim.config.parallel_config import ParallelConfig
from TokenSim.config.weight_memory import packed_weight_memory
from TokenSim.errors import ConfigurationError
from TokenSim.hardware import HardwareContext
from tests.test_dspark import model
from tests.support.kv import layout

ROOT = Path(__file__).resolve().parents[1]


class DSparkCapacityTest(unittest.TestCase):
    def test_probabilistic_and_greedy_buffers_use_integer_bytes(self):
        value = model()
        for method in ("probabilistic", "greedy"):
            candidate = replace(
                value, dspark=replace(value.dspark, draft_sample_method=method)
            )
            buffers = persistent_allocations(candidate, 8240, 8, 5)
            with self.subTest(method=method):
                self.assertTrue(all(type(size) is int for size in buffers.values()))
                self.assertEqual(
                    buffers["dspark_draft_logits"],
                    8 * 5 * value.vocab_size * 2 if method == "probabilistic" else 0,
                )

    def test_packed_element_sizes_round_up_to_whole_storage_bytes(self):
        candidate = SimpleNamespace(
            dspark=SimpleNamespace(
                target_layer_ids=(0,),
                draft_sample_method="probabilistic",
                confidence_head=True,
            ),
            hidden_size=2,
            vocab_size=3,
            operator_dtype=lambda _: "int4",
        )
        self.assertEqual(
            persistent_allocations(candidate, 1, 1, 1)["dspark_draft_logits"], 2
        )

    def test_grouped_cache_rejects_unsupported_compression_geometry(self):
        value = model()
        for ratio, reason in (
            (1, "at least one full compressed layer"),
            (3, "must be divisible"),
            (512, "must be divisible"),
        ):
            candidate = replace(
                value,
                components=replace(
                    value.components,
                    compress_ratios=(ratio,) * value.num_layers
                    + (1,) * value.dspark.num_layers,
                ),
            )
            with (
                self.subTest(ratio=ratio),
                self.assertRaisesRegex(ConfigurationError, reason),
            ):
                hybrid_kv_layout(candidate, block_size=256)

    def test_weight_budget_uses_selected_moe_execution_layout(self):
        hardware = HardwareContext.load(ROOT / "data")
        value = model()
        parallel = ParallelConfig(
            tensor_parallel_size=2,
            data_parallel_size=4,
            enable_expert_parallel=True,
            expert_parallel_scope="global",
            expert_parallel_size=8,
        )
        cache = CacheConfig(
            256,
            hardware.device("gb200"),
            value,
            parallel_config=parallel,
            attention_backend="module",
            moe_execution={"backend": "generic"},
        )
        generic = packed_weight_memory(
            value, parallel, include_mtp=False, layout_backend="fused"
        )
        mega = packed_weight_memory(
            value, parallel, include_mtp=False, layout_backend="mega_moe"
        )
        self.assertNotEqual(generic["total_bytes"], mega["total_bytes"])
        self.assertEqual(
            cache.memory_budget["deductions"]["weights"], generic["total_bytes"]
        )

    def test_original_gb200_decode_settings_produce_floor_conserving_slots(self):
        hardware = HardwareContext.load(ROOT / "data")
        parallel = ParallelConfig(
            data_parallel_size=8,
            data_parallel_size_local=4,
            enable_expert_parallel=True,
            expert_parallel_scope="global",
            expert_parallel_size=8,
        )
        cache = CacheConfig(
            256,
            hardware.device("gb200"),
            model(),
            parallel_config=parallel,
            usable_memory_fraction=0.9,
            max_num_batched_tokens=8240,
            max_num_seqs=8,
            speculative_tokens=5,
            attention_backend="module",
            moe_execution={"backend": "deep_gemm_amxf4_mega_moe"},
        )
        available = cache.memory_budget["available_kv_bytes"]
        slots, stride = cache.num_gpu_blocks, cache.physical_slot_bytes
        self.assertIs(type(available), int)
        self.assertIs(type(slots), int)
        self.assertGreaterEqual(slots, 2)
        self.assertLessEqual(slots * stride, available)
        self.assertGreater((slots + 1) * stride, available)

    def test_allocator_rejects_noninteger_counts_at_its_boundary(self):
        for slots in (3.0, 3.5, True):
            with self.subTest(slots=slots), self.assertRaises(ConfigurationError):
                HybridKVBlockManager(layout(), slots, "fixture")


if __name__ == "__main__":
    unittest.main()
