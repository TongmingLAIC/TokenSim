"""Sub-ULP transfer tails cannot livelock the native endpoint clock."""

from fractions import Fraction
import math
import unittest

import simpy

from TokenSim.kv_transfer.endpoint_fabric import EndpointFabric


class EndpointClockTest(unittest.TestCase):
    def test_late_high_bandwidth_flows_complete_with_bounded_clock_rounding(self):
        rate = 50_000_000_000
        for start in (600.0, 1_000_000.0):
            for nbytes in (1_000_003, 1_000_000_007):
                with self.subTest(start=start, nbytes=nbytes):
                    env = simpy.Environment(initial_time=start)
                    fabric = EndpointFabric(
                        env,
                        send_bytes_per_second=rate,
                        receive_bytes_per_second=rate,
                        fabric_bytes_per_second=rate,
                    )
                    process = env.process(fabric.transfer(0, 1, nbytes))
                    for _ in range(100):
                        if process.triggered:
                            break
                        env.step()
                    self.assertTrue(
                        process.triggered, "Endpoint clock did not progress"
                    )
                    self.assertFalse(fabric.active)
                    flow = fabric.completed[0]
                    self.assertEqual(flow["bytes"], nbytes)
                    ideal = Fraction.from_float(start) + Fraction(nbytes, rate)
                    error = abs(float(Fraction.from_float(flow["end"]) - ideal))
                    self.assertLessEqual(error, 2 * math.ulp(float(ideal)))
                    served = sum(
                        (row["end"] - row["start"]) * sum(row["rates"].values())
                        for row in fabric.intervals
                    )
                    self.assertLessEqual(
                        abs(served - nbytes), 2 * math.ulp(start) * rate
                    )

    def test_zero_byte_flow_preserves_its_zero_duration(self):
        env = simpy.Environment(initial_time=1_000_000.0)
        fabric = EndpointFabric(
            env,
            send_bytes_per_second=1e12,
            receive_bytes_per_second=1e12,
            fabric_bytes_per_second=1e12,
        )
        env.process(fabric.transfer(0, 1, 0))
        env.run()
        self.assertEqual(fabric.completed[0]["end"], 1_000_000.0)
        self.assertFalse(fabric.active)


if __name__ == "__main__":
    unittest.main()
