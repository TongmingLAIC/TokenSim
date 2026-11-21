"""Operator inputs and package overlays retain their original meaning."""

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from TokenSim.errors import ConfigurationError
from TokenSim.hardware.device import SourcedValue
from TokenSim.hardware.links import LinkClass
from TokenSim.hardware import HardwareContext
from TokenSim.latency.operator_table import OperatorTableLatencyBackend
from TokenSim.latency.operator_table.resolver import OperatorResolver
from TokenSim.operator_data.generator import merge_packages
from TokenSim.operator_data.importers import import_aiconfigurator_system
from TokenSim.operator_data.lookup import MissingOperatorDataError, OperatorLookup
from TokenSim.operator_data.manifest import ShapeManifest
from TokenSim.operator_data.package import (
    OperatorDataPackage,
    OperatorDataValidationError,
    PackageMeta,
    SourceRecord,
)


GEMM = dict(dtype="fp16", m=3, n=64, k=128)
ROOT = Path(__file__).resolve().parents[1]


def package(key, *, method="measured", backend="vllm", reference="original"):
    source = SourceRecord(method, "A", method, reference)
    latency = 4 if method == "measured" else 10
    return OperatorDataPackage.from_rows(
        PackageMeta(method, "test-device", backend, sources={source.source_id: source}),
        {"gemm": [{**key, "latency_us": latency, "source_id": source.source_id}]},
        (
            {"gemm": [{**key, "latency_us": latency, "source_id": source.source_id}]}
            if method == "analytical"
            else {}
        ),
    )


class OperatorDataIntegrityTest(unittest.TestCase):
    def test_shape_accepts_exact_integer_representations_at_every_boundary(self):
        measured = package({**GEMM, "m": "03", "dtype": "half"})
        lookup = OperatorLookup(measured)
        manifest = ShapeManifest("test-shapes")
        for m in (3, 3.0, "3"):
            with self.subTest(m=m):
                key = {**GEMM, "m": m}
                self.assertEqual(lookup.lookup("gemm", key).latency_us, 4)
                self.assertEqual(
                    measured.describe_sources("gemm", key)[0]["grade"], "A"
                )
                manifest.add("gemm", key)
        self.assertEqual(manifest.keys["gemm"], [GEMM])
        for m in (3.9, "3.9", True, float("inf"), float("nan")):
            with self.subTest(m=m):
                key = {**GEMM, "m": m}
                with self.assertRaisesRegex(OperatorDataValidationError, "m:"):
                    package(key)
                with self.assertRaisesRegex(MissingOperatorDataError, "m:") as error:
                    lookup.lookup("gemm", key)
                self.assertEqual(error.exception.kind, "malformed_key")
                with self.assertRaisesRegex(ConfigurationError, "m:"):
                    manifest.add("gemm", key)

    def test_malformed_key_does_not_hit_warm_cache_or_fall_back(self):
        hardware = HardwareContext.load(ROOT / "data")
        device = hardware.device("a100_sxm_80g")
        key = {**GEMM, "m": 1}
        measured = package(key)
        for engine in (
            OperatorResolver(device=device, package=measured, fallback="table_first"),
            OperatorTableLatencyBackend(
                device=device,
                model=hardware.model("llama-7b"),
                package=measured,
                fallback="table_first",
            ),
        ):
            with self.subTest(engine=type(engine).__name__):
                self.assertEqual(engine.query("gemm", key).latency_us, 4)
                with self.assertRaises(MissingOperatorDataError) as error:
                    engine.query("gemm", {**key, "m": True})
                self.assertEqual(error.exception.kind, "malformed_key")
                with self.assertRaises(MissingOperatorDataError):
                    engine.query("gemm", {**key, "m": 1.9})

    def test_measured_overlay_keeps_source_and_only_winning_analysis(self):
        measured, analytical = package(GEMM), package(GEMM, method="analytical")
        for base, extra, preference in (
            (analytical, measured, "extra"),
            (measured, analytical, "base"),
        ):
            with self.subTest(preference=preference):
                combined = merge_packages(base, extra, prefer=preference)
                self.assertEqual(combined.tables["gemm"], measured.tables["gemm"])
                self.assertEqual(
                    combined.describe_sources("gemm")[0]["method"], "measured"
                )
                self.assertFalse(combined.analysis)
        winning_formula = merge_packages(measured, analytical, prefer="extra")
        self.assertEqual(winning_formula.analysis, analytical.analysis)
        self.assertEqual(winning_formula.tables, analytical.tables)

    def test_merge_requires_matching_targets_and_consistent_source_ids(self):
        measured = package(GEMM)
        with self.assertRaisesRegex(ConfigurationError, "different device/backend"):
            merge_packages(measured, package(GEMM, backend="sglang"))
        with self.assertRaisesRegex(ConfigurationError, "conflicting metadata"):
            merge_packages(measured, package(GEMM, reference="another capture"))
        self.assertEqual(merge_packages(measured, measured).tables, measured.tables)

    def test_aic_stable_release_supersedes_prerelease_and_post_release_supersedes_stable(
        self,
    ):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for version, rows in {
                "1.3.0rc23": "half,3,64,128,0.001\nhalf,4,64,128,0.002\n",
                "1.3.0": "half,3,64,128,0.003\nhalf,5,64,128,0.005\n",
                "1.3.0.post1": "half,3,64,128,0.004\n",
                "scratch": "half,3,64,128,0.010\n",
            }.items():
                path = root / "gemm" / "vllm" / version / "gemm_perf.txt"
                path.parent.mkdir(parents=True)
                path.write_text("gemm_dtype,m,n,k,latency\n" + rows)
            package = import_aiconfigurator_system(
                root, device_id="test-device", backend="vllm"
            )
        self.assertEqual(package.meta.framework_version, "1.3.0.post1")
        self.assertEqual(
            {row["m"]: row["latency_us"] for row in package.tables["gemm"]},
            {3: 4, 4: 2, 5: 5},
        )

    def test_flat_sglang_import_uses_explicit_gpus_per_node(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "wideep_deepep_ll_perf.txt").write_text(
                "node_num,hidden_size,num_topk,num_experts,num_token,"
                "dispatch_avg_t_us,combine_avg_t_us,version\n"
                "2,64,2,16,8,3,4,0.5.14\n"
            )
            with self.assertRaisesRegex(ConfigurationError, "--gpus-per-node"):
                import_aiconfigurator_system(
                    root, device_id="test-device", backend="sglang"
                )
            output = root / "package"
            result = subprocess.run(
                [
                    "bash",
                    str(ROOT / "scripts/collect/run_aiconfigurator.sh"),
                    "--import-only",
                ],
                env=dict(
                    os.environ,
                    DEVICE="test-device",
                    BACKEND="sglang",
                    RUN_DIR=str(root),
                    PACKAGE=str(output),
                    GPUS_PER_NODE="4",
                    PATH=str(Path(sys.executable).parent)
                    + os.pathsep
                    + os.environ["PATH"],
                ),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            rows = OperatorDataPackage.load(output).tables["ep_all2all"]
            self.assertEqual(
                {(r["phase"], r["ep_size"], r["latency_us"]) for r in rows},
                {("dispatch", 8, 3), ("combine", 8, 4)},
            )

    def test_nonfinite_or_negative_hardware_values_fail_before_estimation(self):
        link = dict(
            bandwidth_per_direction_bytes_per_s=1e9,
            latency_us=0,
            half_bandwidth_bytes=0,
        )
        self.assertEqual(
            LinkClass.from_mapping("zero-startup", link).transfer_us(1000), 1
        )
        for field, value in (
            ("bandwidth_per_direction_bytes_per_s", float("inf")),
            ("latency_us", float("nan")),
            ("half_bandwidth_bytes", -1),
            ("half_bandwidth_bytes", float("inf")),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaises(ConfigurationError):
                    LinkClass.from_mapping("invalid", {**link, field: value})
                with self.assertRaises(ConfigurationError):
                    LinkClass("invalid", "invalid", **{**link, field: value})
        for value in (float("inf"), float("nan"), True):
            with self.subTest(value=value), self.assertRaises(ConfigurationError):
                SourcedValue.parse_optional_zero(
                    {"value": value, "source_id": "test"},
                    "reserved_bytes",
                    "test",
                    "default",
                )
