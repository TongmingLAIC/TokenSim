"""Useful work follows physical invocations and allocated device-time."""

import unittest

from TokenSim.af.serving.work import ledger
from TokenSim.errors import ConfigurationError
from TokenSim.af.trace.metrics import AFTraceEvent
from tests.support.hardware import make_device, make_hardware


def event(node, group=None):
    return AFTraceEvent(
        iteration_id=0,
        block_id=0,
        layer_id=0,
        phase="decode",
        event_type="compute",
        resource_id="f0",
        ready_at=0,
        start_at=0,
        end_at=1,
        queue_duration=0,
        service_duration=1,
        request_count=2,
        token_count=2,
        source_id="fixture",
        node_id=node,
        execution_group_id=group,
        details=dict(
            device_id="TestGPU",
            components=[
                dict(
                    name="ffn",
                    table="gemm",
                    key=dict(m=2, n=4, k=8, dtype="bf16"),
                    latency_us=1e6,
                )
            ],
        ),
    )


class AFWorkTest(unittest.TestCase):
    def test_one_pass_stream_retains_logical_count_and_physical_deduplication(self):
        hardware = make_hardware(make_device(peak_flops=128))
        spans = [event("a", "shared"), event("b", "shared")]
        expected = ledger(spans, hardware, {"TestGPU": 2}, start=0, end=1)
        observed = ledger(iter(spans), hardware, {"TestGPU": 2}, start=0, end=1)
        self.assertEqual(observed, expected)

    def test_mixed_precision_components_remain_supported_in_stream(self):
        hardware = make_hardware(make_device(peak_flops=128))
        span = event("packed")
        span.details["components"][0].update(
            work_flops_by_dtype={"bf16": 128},
            analytical_bounds={"compute_us": 1e6, "memory_us": 2e6},
        )
        result = ledger(iter([span]), hardware, {"TestGPU": 1}, start=0, end=1)
        self.assertEqual(result["useful_flops"], {"TestGPU": 128})
        self.assertEqual(result["system_model_utilization"], 1)
        self.assertEqual(result["components"][0]["analytical_memory_seconds"], 2)

    def test_partial_compute_cannot_be_counted_as_complete_work(self):
        hardware = make_hardware(make_device(peak_flops=128))
        with self.assertRaises(ConfigurationError):
            ledger([event("partial")], hardware, {"TestGPU": 1}, start=0.5, end=1)

    def test_shared_members_count_once_and_idle_allocations_enter_denominator(self):
        hardware = make_hardware(make_device(peak_flops=128))
        value = ledger(
            [event("member-a", "shared"), event("member-b", "shared")],
            hardware,
            {"TestGPU": 2},
            start=0,
            end=1,
        )
        self.assertEqual(value["useful_flops"]["TestGPU"], 128)
        self.assertEqual(value["physical_compute_invocations"], 1)
        self.assertEqual(value["logical_compute_spans"], 2)
        self.assertEqual(value["system_model_utilization"], 0.5)


if __name__ == "__main__":
    unittest.main()
