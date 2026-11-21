"""Root arrivals remain open while native turn dependencies and tool gaps hold."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from TokenSim.cli import build_parser
from TokenSim.cli.run import main
from TokenSim.workload.open_agentx import read_arrivals
from TokenSim.errors import WorkloadValidationError

ROOT = Path(__file__).resolve().parents[1]


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


if __name__ == "__main__":
    unittest.main()
