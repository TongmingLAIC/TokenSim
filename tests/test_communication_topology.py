"""Check physical placement and collective matching against explicit inventories."""

from dataclasses import replace
import unittest

from TokenSim.comm.collectives import CollectiveModel
from TokenSim.comm.collective_types import CollectiveQuery
from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.links import LinkCatalog, LinkClass
from TokenSim.hardware.placement import build_topology_placement
from TokenSim.hardware.topology import TopologyLevel, TopologySpec
from TokenSim.operator_data.lookup import (
    LookupPolicy,
    MissingOperatorDataError,
    OperatorLookup,
)
from TokenSim.operator_data.package import (
    OperatorDataPackage,
    PackageMeta,
    SourceRecord,
)
from tests.support.hardware import make_hardware


def measurements(rows, *, group_size=4, scopes=None):
    sources = {
        source: SourceRecord(source, "A", "measured") for _, _, _, source in rows
    }
    meta = PackageMeta(
        "test-collective-v1",
        "TestGPU",
        "nccl",
        sources=sources,
        extra={"collective_measurement_scopes": scopes} if scopes else {},
    )
    return OperatorLookup(
        OperatorDataPackage.from_rows(
            meta,
            {
                "collective": [
                    dict(
                        dtype="fp16",
                        operation="all_reduce",
                        group_size=group_size,
                        nodes=nodes,
                        message_bytes=size,
                        latency_us=cost,
                        source_id=source,
                    )
                    for nodes, size, cost, source in rows
                ]
            },
        ),
        LookupPolicy(extrapolate="none"),
    )


class PhysicalCollectiveTest(unittest.TestCase):
    def setUp(self):
        self.topology = TopologySpec(
            "two-racks",
            (
                TopologyLevel("node", 8, "nic"),
                TopologyLevel("rack", 2, "nic"),
                TopologyLevel("cluster", 2, "nic"),
            ),
        )
        self.links = LinkCatalog(
            [
                LinkClass(
                    "nic",
                    "test",
                    10e9,
                    1,
                    half_bandwidth_bytes=0,
                    collective_efficiency=1,
                )
            ]
        )

    def test_measured_key_counts_actual_nodes_in_sparse_racks(self):
        layout = self.topology.group_layout((0, 1, 8, 16))
        lookup = measurements(((3, 1024, 17, "three"), (4, 1024, 99, "four")))
        model = CollectiveModel(self.topology, self.links, measured_lookup=lookup)
        result = model.estimate(CollectiveQuery("all_reduce", 1024, layout))
        self.assertEqual((result.latency_us, result.source_id), (17, "three"))
        self.assertEqual(layout.node_count, 3)
        self.assertEqual(layout.fan, (2, 2, 2))

    def test_finite_topology_validates_singleton_and_multirank_layouts(self):
        for slots in ((32,), (0, 32)):
            with self.subTest(slots=slots):
                with self.assertRaisesRegex(
                    ConfigurationError, "outside the inventory"
                ):
                    self.topology.group_layout(slots)
                with self.assertRaisesRegex(
                    ConfigurationError, "outside the inventory"
                ):
                    self.topology.lowest_common_level(slots)
        self.assertEqual(self.topology.capacity, 32)
        self.assertEqual(self.topology.lowest_common_level((0, 31)), 2)

    def test_auto_uses_same_effective_budget_as_chosen_algorithm(self):
        for ports, oversubscription, scope, expected in (
            (1, 1, "link", "ring"),
            (4, 1, "link", "tree"),
            (4, 2, "link", "ring"),
            (4, 1, "endpoint", "ring"),
        ):
            with self.subTest(
                ports=ports, oversubscription=oversubscription, scope=scope
            ):
                topology = TopologySpec(
                    "single-node",
                    (
                        TopologyLevel(
                            "node",
                            8,
                            "nic",
                            links_per_endpoint=ports,
                            oversubscription=oversubscription,
                        ),
                    ),
                )
                links = LinkCatalog(
                    [replace(self.links.get("nic"), bandwidth_scope=scope)]
                )
                model = CollectiveModel(
                    topology, links, fallback="analytical_only", launch_us=0
                )
                query = CollectiveQuery(
                    "all_reduce", 1_000_000, topology.group_layout(range(8))
                )
                auto = model.estimate(query)
                explicit = [
                    model.estimate(replace(query, algorithm=name))
                    for name in ("ring", "tree")
                ]
                self.assertEqual(
                    auto, min(explicit, key=lambda estimate: estimate.latency_us)
                )
                self.assertEqual(auto.levels[0].algorithm, expected)

    def test_global_ring_filters_sources_before_interpolating(self):
        ranks = (0, 1, 8, 9)
        scope = {
            "ring": {
                "schedule": "global_ring",
                "algorithm": "ring",
                "topology_id": self.topology.topology_id,
                "ranks": list(ranks),
            },
        }
        lookup = measurements(
            (
                (2, 1024, 100, "ring"),
                (2, 2048, 999, "unscoped"),
                (2, 4096, 400, "ring"),
            ),
            scopes=scope,
        )
        query = CollectiveQuery(
            "all_reduce",
            2048,
            self.topology.group_layout(ranks),
            schedule="global_ring",
        )
        model = CollectiveModel(self.topology, self.links, measured_lookup=lookup)
        result = model.estimate(query)
        self.assertEqual(
            (result.match_type, result.source_id), ("interpolated", "ring")
        )
        self.assertGreater(result.latency_us, 100)
        self.assertLess(result.latency_us, 400)
        self.assertEqual(
            model.estimate(replace(query, message_bytes=1024)).latency_us, 100
        )

        hierarchical = replace(query, schedule="hierarchical")
        self.assertEqual(model.estimate(hierarchical).source_id, "unscoped")
        self.assertEqual(
            model.estimate(replace(hierarchical, message_bytes=1024)).match_type,
            "analytical",
        )
        for topology, query in (
            (
                self.topology,
                replace(query, layout=self.topology.group_layout((0, 8, 1, 9))),
            ),
            (replace(self.topology, topology_id="different"), query),
        ):
            with self.subTest(topology=topology.topology_id, ranks=query.layout.ranks):
                model = CollectiveModel(topology, self.links, measured_lookup=lookup)
                self.assertEqual(model.estimate(query).match_type, "analytical")
                strict = CollectiveModel(
                    topology, self.links, measured_lookup=lookup, fallback="table_only"
                )
                with self.assertRaisesRegex(
                    MissingOperatorDataError, "Global Ring requires measurements"
                ):
                    strict.estimate(query)


class PhysicalPlacementTest(unittest.TestCase):
    @staticmethod
    def cluster(groups, *, topology):
        size = sum(group["num_workers"] for group in groups)
        return ClusterConfig(
            num_workers=size,
            networks={"a": "ethernet-test", "b": "ethernet-test"},
            worker_groups=[
                dict(hardware="TestGPU", role="hybrid", network="a") | group
                for group in groups
            ],
            parallel_config={"tensor_parallel_size": size},
            topology=topology,
        )

    def test_explicit_slots_are_reserved_before_implicit_ranks(self):
        hardware = make_hardware()
        hardware.topologies.add(
            TopologySpec("finite", (TopologyLevel("node", 2, "nvlink-test"),))
        )
        for pins, implicit_workers, expected in (
            ([1], 1, {0: 1, 1: 0}),
            ([2], 1, "outside the inventory"),
            ([1], 2, "exceeds topology inventory"),
        ):
            with self.subTest(pins=pins, implicit_workers=implicit_workers):
                cluster = self.cluster(
                    [
                        {"num_workers": 1, "device_indices": pins},
                        {"num_workers": implicit_workers},
                    ],
                    topology="finite",
                )
                if isinstance(expected, str):
                    with self.assertRaisesRegex(ConfigurationError, expected):
                        build_topology_placement(cluster, cluster.workers(), hardware)
                else:
                    result = build_topology_placement(
                        cluster, cluster.workers(), hardware
                    )
                    self.assertEqual(result.device_index_by_worker, expected)

    def test_synthesized_topology_preserves_explicit_network_node(self):
        hardware = make_hardware()
        cluster = self.cluster(
            [
                {"num_workers": 1},
                {"num_workers": 1, "device_indices": [3]},
                {"num_workers": 1, "network": "b"},
            ],
            topology=None,
        )
        placement = build_topology_placement(cluster, cluster.workers(), hardware)
        self.assertEqual(placement.device_index_by_worker, {0: 2, 1: 3, 2: 0})
        self.assertEqual([placement.node_id(rank) for rank in range(3)], [1, 1, 0])

        cluster.worker_groups[-1].device_indices = [2]
        with self.assertRaisesRegex(ConfigurationError, "multiple networks"):
            build_topology_placement(cluster, cluster.workers(), hardware)
