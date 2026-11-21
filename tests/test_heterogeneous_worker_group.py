"""Mixed device groups preserve physical identities and per-worker pricing."""

from dataclasses import asdict, replace
import unittest
from pydantic import TypeAdapter

from TokenSim.config.cluster_config import WorkerGroupConfig
from TokenSim.errors import ConfigurationError
from TokenSim.af.factory import build_model_runtime, simulate_decode
from TokenSim.af.providers.operator_table import OperatorTableAFComputeProvider
from tests.af.support.config import _cluster
from tests.af.support.model import _compute_query
from tests.support.hardware import make_device, make_hardware, make_model


class HeterogeneousWorkerGroupTest(unittest.TestCase):
    def group(self, hardware, count=2):
        return WorkerGroupConfig(
            name="mixed", role="decode", hardware=hardware, num_workers=count
        )

    def test_mixed_group_assigns_each_device_to_its_physical_identity(self):
        group = self.group(["a100_sxm_80g", "b200_sxm"])
        workers = group.workers({"net1": "ethernet400Gb"})
        self.assertEqual([w.hardware for w in workers], group.hardware)
        self.assertEqual(
            [w.physical_worker_ref.identity for w in workers],
            ["mixed:0", "mixed:1"],
        )
        self.assertEqual(asdict(group)["hardware"], ["a100_sxm_80g", "b200_sxm"])

    def test_homogeneous_string_keeps_its_serialization_and_worker_devices(self):
        group = self.group("b200_sxm")
        self.assertEqual(asdict(group)["hardware"], "b200_sxm")
        self.assertEqual(
            [w.hardware for w in group.workers({"net1": "ethernet400Gb"})],
            ["b200_sxm", "b200_sxm"],
        )

    def test_invalid_device_inventory_fails_before_simulation(self):
        for value in [
            "",
            " ",
            [],
            ["b200_sxm"],
            ["b200_sxm", ""],
            ["b200_sxm", 1],
            ("b200_sxm", "b200_sxm"),
            None,
            True,
        ]:
            with self.subTest(hardware=value), self.assertRaises(ConfigurationError):
                self.group(value)

    def test_omitted_device_inventory_cannot_use_the_empty_default(self):
        with self.assertRaisesRegex(ConfigurationError, "non-empty string"):
            WorkerGroupConfig(role="decode", num_workers=2)

    def test_native_plan_and_services_use_each_worker_device(self):
        slow = make_device("slow", memory_gib=16, peak_flops=1e12, bandwidth=1e10)
        fast = replace(
            make_device("fast", memory_gib=32), scale_up_link="ethernet-test"
        )
        model = make_model("TestModel", hidden_size=16, intermediate_size=64)
        hardware = make_hardware(slow, fast, models=[model])
        base = _cluster(a_tp=1, a_dp=2, f_tp=1, f_dp=2, f_ep=1)
        doc = TypeAdapter(type(base)).dump_python(base, exclude_none=True)
        for group in doc["worker_groups"]:
            group["hardware"] = ["slow", "fast"]
        doc["topology"] = "test-topology"
        doc["networks"] = {"af-net": "ethernet-test"}
        doc["af"]["transfer_topology_mode"] = "physical"
        runtime = build_model_runtime(doc, hardware=hardware, model=model)
        self.assertEqual(
            [worker.hardware for worker in runtime.workers],
            ["slow", "fast", "slow", "fast"],
        )
        for state in runtime.plan.physical_states:
            self.assertEqual(
                state.capacity_bytes,
                hardware.device(state.hardware).memory_capacity_bytes.value,
            )
        provider = OperatorTableAFComputeProvider(
            hardware, model, fallback="analytical_only"
        )
        query = _compute_query("a")
        slow_service = provider.estimate(replace(query, device_type="slow"))
        fast_service = provider.estimate(replace(query, device_type="fast"))
        self.assertGreater(slow_service.service_duration, fast_service.service_duration)
        executor, steps, _, _ = simulate_decode(
            runtime, 2, 16, steps=2, fallback="analytical_only"
        )
        try:
            self.assertEqual(len(steps), 2)
            self.assertTrue(all(step["makespan_us"] > 0 for step in steps))
        finally:
            executor.trace.close()


if __name__ == "__main__":
    unittest.main()
