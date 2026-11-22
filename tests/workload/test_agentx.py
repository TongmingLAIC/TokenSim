"""AgentX open-root replay arrivals and slow-tail metric contracts."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from TokenSim.cli import build_parser
from TokenSim.cli.run import main
from TokenSim.errors import WorkloadValidationError
from TokenSim.llm.llm_request import Request
from TokenSim.reporting.metrics import RunReport, get_agentx_metrics
from TokenSim.workload.open_agentx import read_arrivals

ROOT = Path(__file__).resolve().parents[2]


class OpenRootsTest(unittest.TestCase):
    def test_arrivals_and_within_tree_waits_survive_native_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            trace = tmp / "trace.jsonl"
            graph = dict(
                id="open-root",
                models=["fixture"],
                block_size=16,
                hash_id_scope="local",
                requests=[
                    dict(
                        t=0.0,
                        model="fixture",
                        type="s",
                        **{"in": 32, "out": 2},
                        hash_ids=[1, 2],
                        api_time=0.1,
                    ),
                    dict(
                        t=0.5,
                        model="fixture",
                        type="s",
                        **{"in": 64, "out": 2},
                        hash_ids=[1, 2, 3, 4],
                        api_time=0.1,
                    ),
                ],
            )
            oversized = dict(
                graph,
                id="outside-window",
                requests=[{**graph["requests"][0], "in": 1_000_000, "hash_ids": []}],
            )
            trace.write_text(json.dumps(graph) + "\n" + json.dumps(oversized) + "\n")
            arrivals = tmp / "arrivals.jsonl"
            arrivals.write_text(
                "\n".join(
                    json.dumps(dict(time_s=t, trace_index=i))
                    for t, i in [(0.0, 0), (0.05, 0), (9.95, 0), (10, 1)]
                )
                + "\n"
            )
            output = tmp / "output"
            args = build_parser().parse_args(
                [
                    "--model",
                    str(ROOT / "data/models/llama-7b.yaml"),
                    "--cluster",
                    str(ROOT / "data/clusters/1_a100/h1.json"),
                    "--workload_type",
                    "agentx_weka",
                    "--dataset_path",
                    str(trace),
                    "--agentx_root_arrivals",
                    str(arrivals),
                    "--no-agentx_warmup",
                    "--agentx_profile_duration",
                    "10",
                    "--batching",
                    "paged-attn",
                    "--block_size",
                    "16",
                    "--latency_backend",
                    "analytical",
                    "--verbose",
                    "none",
                    "--results_path",
                    str(output),
                ]
            )
            with redirect_stdout(io.StringIO()):
                main(args)
            events = [
                json.loads(s)
                for s in (output / "replay/events.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [e["time_s"] for e in events if e["kind"] == "tree_started"],
                [0.0, 0.05, 9.95],
            )
            requests = [
                json.loads(s)
                for s in (output / "replay/requests.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(requests), 6)
            for lane in [0, 1, 2]:
                pair = sorted(
                    [r for r in requests if r["lane"] == lane],
                    key=lambda r: r["turn_index"],
                )
                self.assertGreaterEqual(
                    pair[1]["start_s"] - pair[0]["end_s"], 0.4 - 1e-9
                )
            result = json.loads(next(output.glob("agentx_c*.json")).read_text())[
                "agentx_metrics"
            ]
            self.assertEqual(result["scheduled_roots"], 4)
            self.assertEqual(result["issued_roots"], 3)
            self.assertEqual(result["issue_window_observed_s"], 10)
            self.assertEqual(result["issue_window_completed_requests"], 5)
            self.assertEqual(result["issue_window_output_tokens"], 10)
            self.assertEqual(result["issue_window_completion_rps"], 0.5)
            self.assertEqual(result["issue_window_output_token_tps"], 1.0)
            self.assertGreaterEqual(result["maximum_active_plays"], 2)
            self.assertEqual(result["completed_plays"], 3)
            self.assertEqual(result["censored_plays"], 0)
            self.assertIsNone(result["concurrency"])

    def test_unsorted_and_nonfinite_arrivals_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "arrivals.jsonl"
            for times in [[2.0, 1.0], [float("nan")]]:
                path.write_text(
                    "\n".join(json.dumps(dict(time_s=t, trace_index=0)) for t in times)
                )
                with self.assertRaises(WorkloadValidationError):
                    read_arrivals(path)


class AgentXMetricContractTest(unittest.TestCase):
    def report(self, intervals):
        requests = []
        for index, interval in enumerate(intervals):
            req = Request(index, 8, 2, 16)
            req.generation_idx = 2
            req.prefill_latency = 1.0
            req.decode_time_sum = interval
            req.total_time = 1.0 + interval
            req.record_prefill_admission(0, 0)
            requests.append(req)
        run = RunReport(
            Path("."),
            1.0,
            "paged-attn",
            agentx_concurrency=4,
            agentx_metadata={"replay_protocol": "aiperf-49634d1c-native"},
        )
        return get_agentx_metrics(run, requests, 2.0)

    def test_interactivity_and_normalized_percentiles(self):
        intervals = [0.01, 0.02, 0.04, 0.08]
        report = self.report(intervals)
        result = report["interactivity_tps"]
        norm = report["e2e_normalized_interactivity_tps"]
        seconds_per_token = [(1.0 + value) / 2 for value in intervals]
        for percentile in (50, 75, 90, 95, 99):
            with self.subTest(percentile=percentile):
                self.assertAlmostEqual(
                    result[f"p{percentile}"], 1 / np.percentile(intervals, percentile)
                )
                self.assertAlmostEqual(
                    norm[f"p{percentile}"],
                    1 / np.percentile(seconds_per_token, percentile),
                )
        self.assertAlmostEqual(result["mean"], 1 / np.mean(intervals))
        self.assertAlmostEqual(result["max"], 1 / min(intervals))
        self.assertAlmostEqual(norm["mean"], 1 / np.mean(seconds_per_token))

    def test_empty_zero_and_nonfinite_intervals_are_filtered(self):
        empty = self.report([])
        for name in ("interactivity_tps", "e2e_normalized_interactivity_tps"):
            self.assertEqual((empty[name]["count"], empty[name]["mean"]), (0, None))
        zero_filtered = self.report([0.0, 0.01, 0.04])
        self.assertEqual(zero_filtered["tpot_s"]["count"], 2)
        self.assertEqual(zero_filtered["interactivity_tps"]["count"], 2)
        self.assertAlmostEqual(
            zero_filtered["interactivity_tps"]["p90"],
            1 / zero_filtered["tpot_s"]["p90"],
        )
        nonfinite = self.report([float("nan"), float("inf"), -1.0, 0.04])
        self.assertEqual(nonfinite["tpot_s"]["count"], 1)
        self.assertEqual(nonfinite["interactivity_tps"]["mean"], 25)
