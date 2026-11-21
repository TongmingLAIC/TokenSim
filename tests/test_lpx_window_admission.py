"""Fixed decode admission uses the same bounded-window byte model as serving."""

from dataclasses import replace
from pathlib import Path
import unittest

from TokenSim.af.factory import simulate_decode
from TokenSim.hardware import HardwareContext
from TokenSim.errors import ConfigurationError
from scripts.lpx_runtime import Layout, configuration

ROOT = Path(__file__).resolve().parents[1]


class LPXWindowAdmissionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hardware = HardwareContext.load(ROOT / "data")
        model = replace(
            hardware.models.resolve("gemma-4-31b-text"),
            dtype="bf16",
            kv_cache_dtype="fp8",
        )
        layout = Layout(
            a_device="groq_lp30",
            f_device="groq_lp30",
            a_tp=16,
            f_tp=16,
            f_ep=1,
            a_pp=16,
            f_pp=16,
        )
        cls.runtime = configuration(
            hardware, model, layout, namespace="window-admission"
        )

    def test_100k_mixed_windows_fit_and_execute_one_native_step(self):
        owner = next(iter(self.runtime.plan.request_owners))
        replica = self.runtime.plan.replica_for_instance(owner.domain, owner.instance)
        self.assertGreater((100001 + 15) // 16, replica.logical_gpu_blocks)
        executor, rows, _, _ = simulate_decode(
            self.runtime, 1, 100000, steps=1, trace=False, fallback="analytical_only"
        )
        self.assertEqual(len(rows), 1)
        self.assertGreater(rows[0]["makespan_us"], 0)
        executor.trace.close()

    def test_large_cohort_still_rejects_physical_sram_overcommit(self):
        with self.assertRaises(ConfigurationError):
            simulate_decode(
                self.runtime,
                32,
                100000,
                steps=1,
                trace=False,
                fallback="analytical_only",
            )


if __name__ == "__main__":
    unittest.main()
