"""Global Ring preserves the explicit rank order and shared NIC budgets."""

import unittest

from TokenSim.comm.collectives import CollectiveModel
from TokenSim.comm.collective_types import CollectiveQuery
from TokenSim.comm.global_ring import route_loads
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.links import LinkCatalog, LinkClass
from TokenSim.hardware.topology import TopologyLevel, TopologySpec


class GlobalRingTests(unittest.TestCase):
    def setUp(self):
        self.links = LinkCatalog(
            [
                LinkClass(
                    "local",
                    "local",
                    300e9,
                    1,
                    half_bandwidth_bytes=0,
                    collective_efficiency=1,
                ),
                LinkClass(
                    "nic",
                    "nic",
                    25e9,
                    2,
                    half_bandwidth_bytes=0,
                    collective_efficiency=1,
                ),
            ]
        )
        self.topology = TopologySpec(
            "two_nodes",
            (
                TopologyLevel("node", 8, "local"),
                TopologyLevel("cluster", None, "nic", links_per_endpoint=2),
            ),
        )
        self.model = CollectiveModel(
            self.topology, self.links, launch_us=8, fallback="analytical_only"
        )

    def query(self, ranks, size, operation="all_reduce", algorithm="ring"):
        return CollectiveQuery(
            operation,
            size,
            self.topology.group_layout(ranks),
            "fp32",
            algorithm,
            "global_ring",
        )

    def test_contiguous_ring_uses_two_nics_once_per_node(self):
        result = self.model.estimate(self.query(tuple(range(16)), 16_000_000))
        self.assertEqual(route_loads(self.topology, tuple(range(16))), (1, 1))
        self.assertAlmostEqual(result.latency_us, 8 + 30 * 2 + 30e6 / 50e9 * 1e6)
        self.assertEqual(result.bottleneck_level, "cluster")

    def test_interleaving_shares_the_node_budget_across_eight_flows(self):
        ranks = tuple(rank for pair in zip(range(8), range(8, 16)) for rank in pair)
        self.assertEqual(route_loads(self.topology, ranks), (0, 8))
        result = self.model.estimate(self.query(ranks, 16_000_000))
        self.assertAlmostEqual(result.latency_us, 8 + 60 + 30e6 * 8 / 50e9 * 1e6)

    def test_single_node_matches_ring_bytes_and_rounds(self):
        result = self.model.estimate(self.query(tuple(range(8)), 8_000_000))
        self.assertAlmostEqual(result.latency_us, 8 + 14 + 14e6 / 300e9 * 1e6)

    def test_allgather_uses_existing_complete_buffer_convention(self):
        result = self.model.estimate(
            self.query(tuple(range(8)), 8_000_000, "all_gather")
        )
        self.assertAlmostEqual(result.latency_us, 8 + 7 + 7e6 / 300e9 * 1e6)

    def test_default_hierarchical_result_is_unchanged(self):
        layout = self.topology.group_layout(tuple(range(16)))
        query = CollectiveQuery("all_reduce", 16_000_000, layout, "fp32", "ring")
        result = self.model.estimate(query)
        self.assertEqual(query.schedule, "hierarchical")
        self.assertTrue(result.source_id.startswith("analytical:collective:"))

    def test_incompatible_algorithm_and_duplicate_ranks_are_rejected(self):
        with self.assertRaises(ConfigurationError):
            self.model.estimate(self.query(tuple(range(8)), 1024, algorithm="tree"))
        with self.assertRaises(ConfigurationError):
            self.model.estimate(self.query((0, 0), 1024))

    def test_zero_work_has_zero_latency(self):
        self.assertEqual(self.model.estimate(self.query((0,), 1024)).latency_us, 0)
        self.assertEqual(
            self.model.estimate(self.query(tuple(range(8)), 0)).latency_us, 0
        )


if __name__ == "__main__":
    unittest.main()
