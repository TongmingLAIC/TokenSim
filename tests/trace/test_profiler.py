"""Correlation ownership, collective waits, ambiguity, and paired trace binding."""

from copy import deepcopy
from dataclasses import replace
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from TokenSim.errors import ConfigurationError
from TokenSim.profiler.collection import bind_captures
from TokenSim.profiler.collectives import analyze_collectives, service_metrics
from TokenSim.profiler.compatibility import validate_inputs
from TokenSim.profiler.compare import compare_traces
from TokenSim.profiler.importer import ImportedTrace, read_trace
from TokenSim.profiler.matching import match_spans
from TokenSim.profiler.records import Span
from TokenSim.profiler.reporting import write_comparison
from TokenSim.profiler.timeline import prepare_timeline
from tests.support.profiler import fixtures


def span(name, index, *, duration=1, shape=None):
    return Span(
        str(index),
        name,
        "0",
        0,
        0,
        "prefill",
        index * 10,
        index * 10 + duration,
        "compute",
        "fixture",
        "fixture",
        shape=shape or {},
    )


def collective_experiment(*, proof=True):
    identity = dict(hostname="host", boot_id="boot", time_namespace="time:[1]")
    manifest = (
        dict(
            measurement_protocol="paired_structure_gpu_timing_v1",
            worker_identity=[dict(identity, rank=rank) for rank in (0, 1)],
        )
        if proof
        else {}
    )
    inputs = [
        dict(
            path=f"rank{rank}",
            sha256="hash",
            manifest=manifest,
            metadata=dict(host_name="host", baseTimeNanoseconds=1000),
        )
        for rank in (0, 1)
    ]
    pairs = []
    for rank, start, end in ((0, 100, 116), (1, 110, 117)):
        left = Span(
            f"sim:{rank}",
            "all_reduce",
            str(rank),
            0,
            0,
            "decode",
            0,
            6,
            "collective",
            "sim",
            "simulation",
            busy_us=6,
            details=dict(participants=[0, 1]),
        )
        right = Span(
            f"gpu:{rank}",
            "all_reduce",
            str(rank),
            0,
            0,
            "decode",
            start,
            end,
            "collective",
            "gpu",
            f"clock:{rank}",
            busy_us=end - start,
            details=dict(
                service_intervals=[(start, end)],
                communication_intervals=[(start, end)],
                compute_intervals=[],
                collective_calls=[
                    dict(
                        operation="all_reduce",
                        start_us=start,
                        end_us=end,
                        wait_model="peer_start_barrier",
                    )
                ],
            ),
        )
        pairs.append((left, right))
    observed = ImportedTrace([right for _, right in pairs], {}, inputs, {})
    return pairs, observed


def paired_fixture(directory):
    root = Path(directory)
    _, structure = fixtures(root)
    with gzip.open(structure, "rt") as stream:
        document = json.load(stream)
    rows = document["traceEvents"] if isinstance(document, dict) else document
    kernels = [dict(row) for row in rows if row.get("cat") == "kernel"]
    for index, row in enumerate(kernels):
        row.update(ts=100000 + index * 20, dur=7 + index)
    timing = root / "timing" / "profile_tp0.json"
    timing.parent.mkdir()
    timing.write_text(json.dumps(kernels))
    manifest = dict(
        batch_size=1,
        input_tokens=1,
        output_tokens=1,
        paired_captures=bind_captures([structure], [timing], root),
    )
    (root / "manifest.json").write_text(json.dumps(manifest))
    return structure, timing, manifest


class ProfilerComparisonTest(unittest.TestCase):
    def test_partial_service_exclusion_and_real_qwen3_fixtures(self):
        with tempfile.TemporaryDirectory() as directory:
            simulated, profile = fixtures(directory)
            content = json.loads(simulated.read_text())
            content["traceEvents"][0]["args"]["incomplete"] = True
            simulated.write_text(json.dumps(content))
            report = compare_traces(simulated, [profile], Path(directory) / "report")
            self.assertEqual(report["coverage"]["matched_units"], 1)

        fixtures_dir = Path(__file__).resolve().parents[1] / "fixtures/profiler"
        layer = read_trace(fixtures_dir / "vllm_qwen3_layer.json")
        self.assertEqual({item.operation for item in layer.spans}, {"qkv", "rope"})
        row_parallel = fixtures_dir / "vllm_qwen3_row_parallel.json"
        imported = read_trace(row_parallel)
        self.assertEqual(
            imported.spans[0].details["simulated_members"], ["output", "all_reduce"]
        )
        with tempfile.TemporaryDirectory() as directory:
            simulated, _ = fixtures(directory)
            content = json.loads(simulated.read_text())
            for row, op in zip(content["traceEvents"], ("output", "all_reduce")):
                row["args"]["operation"] = op
            simulated.write_text(json.dumps(content))
            report = compare_traces(
                simulated, [row_parallel], Path(directory) / "report"
            )
            self.assertEqual(report["coverage"]["matched_units"], 1)

    def test_fusion_shape_ambiguity_and_unique_order_matching(self):
        left = [span("ffn_residual", 0), span("attention_norm", 1)]
        left[1].layer = 1
        right = span("attention_norm", 2)
        right.layer = 1
        right.details["simulated_members"] = ["ffn_residual", "attention_norm"]
        pairs, unmatched, _ = match_spans(left, [right])
        self.assertEqual(pairs[0][0].members, ("0", "1"))
        self.assertFalse(unmatched)

        for amb_left, amb_right in (
            ([span("qkv", 0), span("qkv", 1)], [span("qkv", 2, duration=100)]),
            ([span("qkv", 0), span("output", 1)], [span("output", 2), span("qkv", 3)]),
        ):
            pairs, unmatched, _ = match_spans(amb_left, amb_right)
            self.assertEqual(pairs, [])
            self.assertTrue(all(item["status"] == "ambiguous" for item in unmatched))

        pairs, unmatched, _ = match_spans(
            [span("all_reduce", 0), span("all_reduce", 1)],
            [span("all_reduce", 2), span("all_reduce", 3)],
        )
        self.assertEqual(len(pairs), 2)
        self.assertFalse(unmatched)
        pairs, unmatched, _ = match_spans(
            [span("qkv", 0, shape={"m": 1})], [span("qkv", 1, shape={"m": 4})]
        )
        self.assertFalse(pairs)

    def test_cpu_ownership_model_identity_and_unattributed_kernels(self):
        with tempfile.TemporaryDirectory() as directory:
            simulated, profile = fixtures(directory)
            report = compare_traces(simulated, [profile], Path(directory) / "report")
            self.assertEqual(report["coverage"]["matched_units"], 2)
            self.assertEqual(report["matches"][1]["start_offset_us"], -2)
            (Path(directory) / "manifest.json").write_text('{"model":"Different"}')
            with self.assertRaisesRegex(ConfigurationError, "model_id"):
                compare_traces(simulated, [profile], Path(directory) / "bad")

    def test_paired_structure_and_gpu_timing_binding_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            structure, timing, _ = paired_fixture(directory)
            spans = read_trace(timing).spans
            self.assertEqual([s.operation for s in spans], ["qkv", "output"])
            self.assertEqual([s.duration_us for s in spans], [7, 8])
            with self.assertRaisesRegex(ConfigurationError, "forward count"):
                bind_captures(
                    [structure], [timing], Path(directory), expected_forwards=2
                )
            rows = json.loads(timing.read_text())
            rows[0]["name"] = "another_kernel"
            timing.write_text(json.dumps(rows))
            with self.assertRaises(ConfigurationError):
                read_trace(timing)
            with self.assertRaises(ConfigurationError):
                bind_captures([structure], [timing], Path(directory))


class CollectiveWaitTest(unittest.TestCase):
    def test_arrival_wait_overlap_and_shared_clock_alignment(self):
        pairs, observed = collective_experiment()
        observed.diagnostics["collective_waits"] = analyze_collectives(pairs, observed)
        first = service_metrics(*pairs[0])
        self.assertEqual(first["profiled_collective_residence_us"], 16)
        self.assertEqual(first["profiled_peer_wait_us"], 10)
        self.assertEqual(first["profiled_device_service_us"], 6)
        simulated = ImportedTrace([left for left, _ in pairs], {}, [], {})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            extra = replace(
                observed.spans[0],
                id="gpu:extra",
                operation="attention",
                rank="2",
                clock="clock:2",
                start_us=108,
                end_us=120,
                category="compute",
                details={},
            )
            observed.spans.append(extra)
            report = write_comparison(
                simulated,
                observed,
                pairs,
                [dict(extra.as_dict(), status="unmatched", side="profiled")],
                [],
                output,
                mapping_hash="hash",
            )
            chrome = json.loads((output / "comparison.trace.json").read_text())
        row = next(
            r for r in chrome["traceEvents"] if r["args"].get("id") == "gpu:extra"
        )
        bar = next(
            b
            for b in prepare_timeline(report, detail="module")["bars"]
            if b.span_id == "gpu:extra"
        )
        self.assertEqual(row["ts"], bar.start_us + bar.translation_us)

    def test_unqualified_waits_and_domain_rank_mapping(self):
        for case in ("clock", "peer", "barrier", "disjoint", "namespace"):
            pairs, observed = collective_experiment(proof=case != "clock")
            if case == "peer":
                pairs.pop()
            elif case == "barrier":
                pairs[0][1].details["collective_calls"][0]["wait_model"] = "unresolved"
            elif case == "disjoint":
                pairs[0][1].details["collective_calls"][0]["end_us"] = 105
            elif case == "namespace":
                observed.inputs = deepcopy(observed.inputs)
                observed.inputs[1]["manifest"]["worker_identity"][1][
                    "time_namespace"
                ] = "another"
            self.assertEqual(analyze_collectives(pairs, observed)["analyzed_calls"], 0)

        pairs, _ = collective_experiment()
        simulated = ImportedTrace(
            [left for left, _ in pairs],
            {
                "worker_parallelism": {
                    "0": {"domain": "prefill", "tensor_parallel_size": 4},
                    "1": {"domain": "decode", "tensor_parallel_size": 2},
                }
            },
            [],
            {},
        )
        observed = ImportedTrace(
            [replace(pairs[0][1], rank="local", source="trace.json")],
            {},
            [{"path": "trace.json", "manifest": {"tensor_parallel_size": 2}}],
            {},
        )
        validate_inputs(simulated, observed, {"local": "1"})
        with self.assertRaisesRegex(
            ConfigurationError, "incompatible tensor_parallel_size"
        ):
            validate_inputs(simulated, observed, {"local": "0"})
