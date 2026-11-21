"""Physical-coordinate interpolation preserves alpha-beta growth between anchors."""

import unittest

from TokenSim.errors import ConfigurationError
from TokenSim.operator_data.lookup import LookupPolicy, OperatorLookup
from TokenSim.operator_data.package import (
    OperatorDataPackage,
    PackageMeta,
    SourceRecord,
)


class AnalyticalInterpolationTests(unittest.TestCase):
    def setUp(self):
        self.key = dict(dtype="fp32", operation="all_reduce", group_size=16, nodes=2)
        rows = [
            {
                **self.key,
                "message_bytes": size,
                "latency_us": 1 + size,
                "source_id": "anchors",
            }
            for size in (1, 1024)
        ]
        meta = PackageMeta(
            "test",
            "device",
            "nccl",
            sources={"anchors": SourceRecord("anchors", "A", "measured")},
        )
        self.package = OperatorDataPackage.from_rows(meta, {"collective": rows})

    def test_sparse_alpha_beta_anchors_recover_linear_byte_growth(self):
        lookup = OperatorLookup(
            self.package, LookupPolicy(interpolation_space="analytical")
        )
        result = lookup.lookup(
            "collective",
            {**self.key, "message_bytes": 32},
            scaler=lambda _, key: 7 + 2 * key["message_bytes"],
        )
        self.assertAlmostEqual(result.latency_us, 33)
        self.assertEqual(result.match_type, "interpolated")
        self.assertIn("analytical_interpolated", result.detail["flags"])

    def test_default_axis_interpolation_remains_available(self):
        result = OperatorLookup(self.package).lookup(
            "collective", {**self.key, "message_bytes": 32}
        )
        self.assertAlmostEqual(result.latency_us, (2 + 1025) / 2)

    def test_missing_or_nonmonotonic_physics_falls_back_explicitly(self):
        lookup = OperatorLookup(
            self.package, LookupPolicy(interpolation_space="analytical")
        )
        for scaler in (None, lambda *_: None, lambda _, key: -key["message_bytes"]):
            result = lookup.lookup(
                "collective", {**self.key, "message_bytes": 32}, scaler=scaler
            )
            self.assertAlmostEqual(result.latency_us, (2 + 1025) / 2)
            self.assertIn("axis_interpolation_fallback", result.detail["flags"])

    def test_exact_anchor_and_invalid_policy(self):
        lookup = OperatorLookup(
            self.package, LookupPolicy(interpolation_space="analytical")
        )
        result = lookup.lookup("collective", {**self.key, "message_bytes": 1024})
        self.assertEqual(result.latency_us, 1025)
        self.assertEqual(result.match_type, "exact")
        with self.assertRaises(ConfigurationError):
            LookupPolicy(interpolation_space="unrecognized")


if __name__ == "__main__":
    unittest.main()
