"""DSpark draft work, persistent tensors and grouped KV follow the measured image."""

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import yaml

from TokenSim.block.hybrid.dspark_memory import persistent_allocations
from TokenSim.block.hybrid.kv_layout import hybrid_kv_layout
from TokenSim.config.dspark import DSparkConfig
from TokenSim.config.model_config import ModelSpec
from TokenSim.config.weight_memory import packed_weight_memory
from TokenSim.errors import ConfigurationError
from TokenSim.hardware import HardwareContext
from TokenSim.latency.operator_table import OperatorTableLatencyBackend
from TokenSim.llm.llm_request import Request

ROOT = Path(__file__).resolve().parents[1]


def model():
    return ModelSpec.from_mapping(
        yaml.safe_load(
            (ROOT / "data/models/deepseek-v4-pro-0813-dspark.yaml").read_text()
        )
    )


class DSparkTest(unittest.TestCase):
    def test_three_layer_draft_has_shared_vocab_tensors_and_replicated_markov_heads(
        self,
    ):
        value = model()
        tensors = {t["name"]: t for t in packed_weight_memory(value)["tensors"]}
        self.assertEqual(
            tensors["dspark.main_proj"]["shape"],
            [value.hidden_size, 3 * value.hidden_size],
        )
        self.assertEqual(tensors["dspark.markov_w1"]["shape"], [value.vocab_size, 512])
        self.assertEqual(tensors["dspark.markov_w2"]["shape"], [value.vocab_size, 512])
        self.assertEqual(tensors["dspark.confidence.weight"]["dtype"], "fp32")
        self.assertNotIn("dspark.confidence.bias", tensors)
        self.assertFalse(
            any(
                name.endswith(".e_proj") or name.endswith(".h_proj") for name in tensors
            )
        )
        self.assertEqual(sum(name == "embedding" for name in tensors), 1)
        self.assertEqual(sum(name == "lm_head" for name in tensors), 1)
        layout = hybrid_kv_layout(value, speculative_tokens=5)
        draft_pages = [
            layer for group in layout.groups for layer in group.layers if layer.draft
        ]
        self.assertEqual(len(draft_pages), 3)
        self.assertTrue(
            all(layer.window_tokens == value.sliding_window for layer in draft_pages)
        )
        without = packed_weight_memory(value, include_mtp=False)
        self.assertFalse(
            any(t["name"].startswith(("dspark.", "mtp.")) for t in without["tensors"])
        )

    def test_one_block_forward_runs_three_layers_and_five_markov_steps(self):
        value = model()
        hardware = HardwareContext.load(ROOT / "data")
        backend = OperatorTableLatencyBackend(
            device=hardware.device("gb200"),
            model=value,
            fallback="analytical_only",
            expected_decode_query_len=6,
            max_num_seqs=8,
        )
        request = Request(1, 64, 64, 256)
        request.generation_idx = 1
        request.num_computed_tokens = 64
        request.scheduled_tokens = 3
        request.scheduled_query_tokens = 6
        draft = backend.native_mtp
        with patch.object(draft, "_moe", wraps=draft._moe) as moe:
            costs = draft.estimate([request])
        self.assertEqual(moe.call_count, 3)
        self.assertEqual(draft.counts["block_forwards"], 1)
        self.assertEqual(draft.counts["markov_steps"], 5)
        self.assertEqual(draft.counts["context_kv_layer_rows"], 18)
        self.assertEqual(draft.counts["backbone_layer_rows"], 30)
        self.assertEqual(draft.counts["query_rows"], 10)
        self.assertTrue(all(cost >= 0 for cost in costs.values()))
        with patch(
            "TokenSim.latency.dspark.roofline_latency_us", return_value=1
        ) as estimator:
            draft._attention([request])
        self.assertEqual(estimator.call_args.args[1], 5 * (67 + 5) * value.head_dim)

    def test_probabilistic_cache_survives_synthetic_rejection_and_geometry_validates(
        self,
    ):
        value = model()
        buffers = persistent_allocations(value, 8192, 8, 5)
        self.assertEqual(buffers["dspark_draft_logits"], 8 * 5 * value.vocab_size * 2)
        self.assertEqual(persistent_allocations(value, 8192, 8, 0), {})
        greedy = replace(
            value, dspark=replace(value.dspark, draft_sample_method="greedy")
        )
        self.assertEqual(
            persistent_allocations(greedy, 8192, 8, 5)["dspark_draft_logits"], 0
        )
        with self.assertRaises(ConfigurationError):
            replace(value, dspark=DSparkConfig(num_layers=2))
        with self.assertRaises(ConfigurationError):
            DSparkConfig(target_layer_ids=(1, 1))
        legacy = ModelSpec.from_mapping(
            yaml.safe_load((ROOT / "data/models/deepseek-v4-pro.yaml").read_text())
        )
        self.assertNotIn("dspark", legacy.to_dict())


if __name__ == "__main__":
    unittest.main()
