"""Synchronized data-parallel workers consume messages and finish requests."""

import json
from pathlib import Path
import tempfile
import unittest

from tests.support.cli import MOE_CLUSTER, run_quiet, simulation_args


class LockstepRuntimeTest(unittest.TestCase):
    def test_parallel_workers_complete_synchronized_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = run_quiet(
                simulation_args(
                    directory,
                    model="moe-toy",
                    cluster=MOE_CLUSTER,
                    batching="paged-attn",
                    dp_lockstep=True,
                )
            )
            result = json.loads((Path(directory) / "result_inf.json").read_text())
            self.assertEqual(result["termination_reason"], "completed")
            self.assertEqual(result["notdone"], [])
            self.assertEqual(len(engine.timing_recorder.timing.time), 8)
            self.assertAlmostEqual(result["output_token_ps"] * result["duration"], 24)
            self.assertIsNotNone(result["dp_step_stats"])
