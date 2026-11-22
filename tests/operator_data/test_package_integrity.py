"""Operator package validation, overlay precedence, and collective parser checks."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from TokenSim.errors import ConfigurationError
from TokenSim.hardware import HardwareContext
from TokenSim.hardware.device import SourcedValue
from TokenSim.hardware.links import LinkClass
from TokenSim.latency.operator_table import OperatorTableLatencyBackend
from TokenSim.latency.operator_table.resolver import OperatorResolver
from TokenSim.operator_data.generator import merge_packages
from TokenSim.operator_data.importers import import_aiconfigurator_system
from TokenSim.operator_data.importers.nccl_tests import parse_nccl_tests_output
from TokenSim.operator_data.lookup import MissingOperatorDataError, OperatorLookup
from TokenSim.operator_data.manifest import ShapeManifest
from TokenSim.operator_data.package import (
    OperatorDataPackage,
    OperatorDataValidationError,
    PackageMeta,
    SourceRecord,
)

GEMM = dict(dtype="fp16", m=3, n=64, k=128)
ROOT = Path(__file__).resolve().parents[2]
OUT_COLUMNS = "33554432 8388608 float sum -1 491.6 68.26 119.46"
IN_COLUMNS = "489.9 68.49 119.87"


def _package(key, *, method="measured", backend="vllm", reference="original"):
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


def _parse_nccl(text, *, nodes=1, use_in_place=False):
    return parse_nccl_tests_output(
        text,
        operation="all_reduce",
        group_size=2,
        nodes=nodes,
        source_id="test",
        use_in_place=use_in_place,
    )


class OperatorDataIntegrityTest(unittest.TestCase):
    def test_shape_accepts_exact_integer_representations_at_every_boundary(self):
        measured = _package({**GEMM, "m": "03", "dtype": "half"})
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
                    _package(key)
                with self.assertRaisesRegex(MissingOperatorDataError, "m:") as error:
                    lookup.lookup("gemm", key)
                self.assertEqual(error.exception.kind, "malformed_key")
                with self.assertRaisesRegex(ConfigurationError, "m:"):
                    manifest.add("gemm", key)

    def test_malformed_key_does_not_hit_warm_cache_or_fall_back(self):
        hardware = HardwareContext.load(ROOT / "data")
        device = hardware.device("a100_sxm_80g")
        key = {**GEMM, "m": 1}
        measured = _package(key)
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

    def test_measured_overlay_and_merge_target_validation(self):
        measured, analytical = _package(GEMM), _package(GEMM, method="analytical")
        for base, extra, preference in (
            (analytical, measured, "extra"),
            (measured, analytical, "base"),
        ):
            combined = merge_packages(base, extra, prefer=preference)
            self.assertEqual(combined.tables["gemm"], measured.tables["gemm"])
            self.assertEqual(combined.describe_sources("gemm")[0]["method"], "measured")
            self.assertFalse(combined.analysis)
        winning_formula = merge_packages(measured, analytical, prefer="extra")
        self.assertEqual(winning_formula.analysis, analytical.analysis)
        with self.assertRaisesRegex(ConfigurationError, "different device/backend"):
            merge_packages(measured, _package(GEMM, backend="sglang"))
        with self.assertRaisesRegex(ConfigurationError, "conflicting metadata"):
            merge_packages(measured, _package(GEMM, reference="another capture"))

    def test_aic_version_precedence_and_flat_gpus_per_node(self):
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
        with TemporaryDirectory() as directory:
            flat = Path(directory)
            (flat / "wideep_deepep_ll_perf.txt").write_text(
                "node_num,hidden_size,num_topk,num_experts,num_token,"
                "dispatch_avg_t_us,combine_avg_t_us,version\n"
                "2,64,2,16,8,3,4,0.5.14\n"
            )
            with self.assertRaisesRegex(ConfigurationError, "--gpus-per-node"):
                import_aiconfigurator_system(
                    flat, device_id="test-device", backend="sglang"
                )
            imported = import_aiconfigurator_system(
                flat, device_id="test-device", backend="sglang", gpus_per_node=4
            )
            self.assertEqual(
                {
                    (r["phase"], r["ep_size"], r["latency_us"])
                    for r in imported.tables["ep_all2all"]
                },
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
        ):
            with self.assertRaises(ConfigurationError):
                LinkClass.from_mapping("invalid", {**link, field: value})
        for value in (float("inf"), float("nan"), True):
            with self.assertRaises(ConfigurationError):
                SourcedValue.parse_optional_zero(
                    {"value": value, "source_id": "test"},
                    "reserved_bytes",
                    "test",
                    "default",
                )


class NCCLCollectionTest(unittest.TestCase):
    def test_selected_columns_and_iteration_stats(self):
        valid = f"{OUT_COLUMNS} 0 {IN_COLUMNS} 0"
        for mode, wrong in (
            ("out-of-place", f"{OUT_COLUMNS} 2 {IN_COLUMNS} 0"),
            ("in-place", f"{OUT_COLUMNS} 0 {IN_COLUMNS} 2"),
        ):
            in_place = mode == "in-place"
            with self.assertRaisesRegex(
                ConfigurationError, f"line 2 \\({mode}\\): expected #wrong=0"
            ):
                _parse_nccl(f"{valid}\n{wrong}", use_in_place=in_place)
            self.assertEqual(
                _parse_nccl(wrong, use_in_place=not in_place)[0]["latency_us"],
                491.6 if in_place else 489.9,
            )
        header = (
            "# size count type redop root time algbw busbw #wrong i_min i_max i_p99 i_cv% "
            "time algbw busbw #wrong i_min i_max i_p99 i_cv%"
        )
        text = (
            f"{header}\n{OUT_COLUMNS} 0 480 500 497 1.2 {IN_COLUMNS} 0 480 500 497 1.2"
        )
        self.assertEqual(_parse_nccl(text, use_in_place=True)[0]["latency_us"], 489.9)

    def test_missing_columns_and_reported_rank_node_counts(self):
        for text, use_in_place in (
            (OUT_COLUMNS, False),
            (f"{OUT_COLUMNS} 0", True),
            (f"{OUT_COLUMNS} N/A", False),
        ):
            with self.assertRaisesRegex(ConfigurationError, "#wrong"):
                _parse_nccl(text, use_in_place=use_in_place)
        rank0 = "# Rank 0 Group 0 Pid 100 on node-a device 0 [0000:17:00]"
        rank1 = "# Rank 1 Group 0 Pid 200 on node-b device 0 [0000:17:00]"
        text = f"# Using devices\n{rank0}\n{rank1}\n{OUT_COLUMNS} 0"
        self.assertEqual(_parse_nccl(text, nodes=2)[0]["nodes"], 2)
        with self.assertRaisesRegex(ConfigurationError, "reports 2 ranks on 1 nodes"):
            _parse_nccl(text.replace("node-b", "node-a"), nodes=2)
