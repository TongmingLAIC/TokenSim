"""Peer arrival accounting requires complete groups and a proven clock basis."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from TokenSim.errors import ConfigurationError
from TokenSim.profiler.collectives import analyze_collectives, service_metrics
from TokenSim.profiler.compatibility import validate_inputs
from TokenSim.profiler.importer import ImportedTrace
from TokenSim.profiler.matching import match_spans
from TokenSim.profiler.records import Span
from TokenSim.profiler.reporting import write_comparison
from TokenSim.profiler.timeline import prepare_timeline


def experiment(*, proof=True):
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


class CollectiveWaitTest(unittest.TestCase):
    def test_arrival_wait_preserves_residence_and_shared_clock_alignment(self):
        pairs, observed = experiment()
        before = [(right.start_us, right.end_us, right.busy_us) for _, right in pairs]
        observed.diagnostics["collective_waits"] = analyze_collectives(pairs, observed)
        self.assertEqual(observed.diagnostics["collective_waits"]["analyzed_calls"], 2)
        first = service_metrics(*pairs[0])
        self.assertEqual(first["profiled_collective_residence_us"], 16)
        self.assertEqual(first["profiled_peer_wait_us"], 10)
        self.assertEqual(first["profiled_device_service_us"], 6)
        self.assertEqual(service_metrics(*pairs[1])["profiled_peer_wait_us"], 0)
        self.assertEqual(
            before,
            [(right.start_us, right.end_us, right.busy_us) for _, right in pairs],
        )
        simulated = ImportedTrace([left for left, _ in pairs], {}, [], {})
        with tempfile.TemporaryDirectory() as directory:
            report = write_comparison(
                simulated, observed, pairs, [], [], Path(directory), mapping_hash="hash"
            )
        self.assertEqual(
            [a["translation_us"] for a in report["alignment"]["anchors"].values()],
            [-100, -100],
        )
        self.assertEqual(report["matches"][1]["start_offset_us"], -10)

    def test_compute_overlapping_a_wait_is_preserved_in_device_work(self):
        pairs, observed = experiment()
        right = pairs[0][1]
        right.details["service_intervals"].append((103, 108))
        right.details["compute_intervals"] = [(103, 108)]
        analyze_collectives(pairs, observed)
        self.assertEqual(service_metrics(*pairs[0])["profiled_device_service_us"], 11)

    def test_unqualified_collective_waits_remain_unknown(self):
        for case in ("clock", "peer", "barrier", "disjoint", "namespace"):
            with self.subTest(case=case):
                pairs, observed = experiment(proof=case != "clock")
                if case == "peer":
                    pairs.pop()
                elif case == "barrier":
                    pairs[0][1].details["collective_calls"][0][
                        "wait_model"
                    ] = "unresolved"
                elif case == "disjoint":
                    pairs[0][1].details["collective_calls"][0]["end_us"] = 105
                elif case == "namespace":
                    observed.inputs = deepcopy(observed.inputs)
                    observed.inputs[1]["manifest"]["worker_identity"][1][
                        "time_namespace"
                    ] = "another"
                result = analyze_collectives(pairs, observed)
                self.assertEqual(result["analyzed_calls"], 0)
                self.assertIsNone(
                    service_metrics(*pairs[0])["profiled_device_service_us"]
                )
                self.assertEqual(pairs[0][1].busy_us, 16)

    def test_unmatched_shared_clock_aligns_the_same_in_chrome_and_timeline(self):
        pairs, observed = experiment()
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
        observed.diagnostics["collective_waits"] = analyze_collectives(pairs, observed)
        simulated = ImportedTrace([left for left, _ in pairs], {}, [], {})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
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
            row for row in chrome["traceEvents"] if row["args"].get("id") == extra.id
        )
        bars = prepare_timeline(report, detail="module")["bars"]
        bar = next(bar for bar in bars if bar.span_id == extra.id)
        self.assertEqual(row["ts"], 8)
        self.assertEqual(row["ts"], bar.start_us + bar.translation_us)

    def test_repeated_collectives_do_not_pair_across_missing_callsites(self):
        pairs, observed = experiment()
        simulated = []
        for left, right in pairs:
            simulated.extend(
                [
                    replace(
                        left,
                        id=f"{left.id}:{site}",
                        shape={"m": site},
                        start_us=site * 6,
                        end_us=site * 6 + 6,
                    )
                    for site in range(2)
                ]
            )
            right.shape = {"m": int(right.rank)}
        matched, _, _ = match_spans(simulated, observed.spans)
        self.assertEqual(
            [left.details["collective_callsite"] for left, _ in matched], [0, 1]
        )
        result = analyze_collectives(matched, observed)
        self.assertEqual(result["analyzed_calls"], 0)

    def test_rank_mapping_selects_the_declared_worker_domain_layout(self):
        pairs, _ = experiment()
        worker_parallelism = {
            "0": {"domain": "prefill", "tensor_parallel_size": 4},
            "1": {"domain": "decode", "tensor_parallel_size": 2},
        }
        simulated = ImportedTrace(
            [left for left, _ in pairs],
            {"worker_parallelism": worker_parallelism},
            [],
            {},
        )
        observed = ImportedTrace(
            [replace(pairs[0][1], rank="local", source="trace.json")],
            {},
            [{"path": "trace.json", "manifest": {"tensor_parallel_size": 2}}],
            {},
        )
        evidence = validate_inputs(simulated, observed, {"local": "1"})
        self.assertIn(
            {"source": "trace.json", "field": "tensor_parallel_size"},
            evidence["checked"],
        )
        with self.assertRaisesRegex(
            ConfigurationError, "incompatible tensor_parallel_size"
        ):
            validate_inputs(simulated, observed, {"local": "0"})
