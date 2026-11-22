"""Global Ring collective schedules and MegaMoE physical EP locality checks."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest

from TokenSim.comm.collective_types import CollectiveQuery
from TokenSim.comm.collectives import CollectiveModel
from TokenSim.comm.global_ring import route_loads
from TokenSim.config.parallel_config import ParallelConfig, ParallelRankInfo
from TokenSim.errors import ConfigurationError
from TokenSim.hardware import HardwareContext
from TokenSim.hardware.links import LinkCatalog, LinkClass
from TokenSim.hardware.topology import TopologyLevel, TopologyPlacement, TopologySpec
from TokenSim.latency.operator_table import OperatorTableLatencyBackend
from TokenSim.llm.forward import ForwardPass, RankShape
from TokenSim.parallel import ParallelCommunicator

ROOT = Path(__file__).resolve().parents[1]


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

    def test_contiguous_and_interleaved_rings_respect_nic_budgets(self):
        result = self.model.estimate(self.query(tuple(range(16)), 16_000_000))
        self.assertEqual(route_loads(self.topology, tuple(range(16))), (1, 1))
        self.assertAlmostEqual(result.latency_us, 8 + 30 * 2 + 30e6 / 50e9 * 1e6)
        self.assertEqual(result.bottleneck_level, "cluster")
        ranks = tuple(rank for pair in zip(range(8), range(8, 16)) for rank in pair)
        self.assertEqual(route_loads(self.topology, ranks), (0, 8))
        interleaved = self.model.estimate(self.query(ranks, 16_000_000))
        self.assertAlmostEqual(interleaved.latency_us, 8 + 60 + 30e6 * 8 / 50e9 * 1e6)

    def test_single_node_allgather_hierarchical_and_validation(self):
        single = self.model.estimate(self.query(tuple(range(8)), 8_000_000))
        self.assertAlmostEqual(single.latency_us, 8 + 14 + 14e6 / 300e9 * 1e6)
        ag = self.model.estimate(self.query(tuple(range(8)), 8_000_000, "all_gather"))
        self.assertAlmostEqual(ag.latency_us, 8 + 7 + 7e6 / 300e9 * 1e6)
        layout = self.topology.group_layout(tuple(range(16)))
        default_q = CollectiveQuery("all_reduce", 16_000_000, layout, "fp32", "ring")
        self.assertEqual(default_q.schedule, "hierarchical")
        with self.assertRaises(ConfigurationError):
            self.model.estimate(self.query(tuple(range(8)), 1024, algorithm="tree"))
        with self.assertRaises(ConfigurationError):
            self.model.estimate(self.query((0, 0), 1024))
        self.assertEqual(self.model.estimate(self.query((0,), 1024)).latency_us, 0)


class MegaMoETopologyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load(ROOT / "data")

    def estimator(self, label="nvlink_domain", *, remote=False, members=8):
        hardware = self.hardware
        topology = hardware.topologies.get("gb200_nvl72")
        topology = replace(
            topology,
            levels=(replace(topology.levels[0], name=label), *topology.levels[1:]),
        )
        placement = TopologyPlacement(
            topology,
            {i: (i + 68 if remote and i >= 4 else i) for i in range(8)},
        )
        parallel = ParallelConfig(
            data_parallel_size=8,
            data_parallel_size_local=4,
            enable_expert_parallel=True,
            expert_parallel_scope="global",
            expert_parallel_size=8,
        )
        communicator = ParallelCommunicator(
            placement=placement,
            collective_model=CollectiveModel(topology, hardware.links),
            workers=[
                SimpleNamespace(id=i, dp_rank=i, tp_rank=0, pp_rank=0)
                for i in range(members)
            ],
            worker_id=0,
            rank_info=ParallelRankInfo(),
            parallel_config=parallel,
        )
        backend = OperatorTableLatencyBackend(
            device=hardware.device("gb200"),
            model=hardware.models.resolve("deepseek-v4-pro"),
            parallel_config=parallel,
            communicator=communicator,
            package=hardware.operator_package("gb200", "vllm"),
            moe_execution={"backend": "deep_gemm_amxf4_mega_moe"},
        )
        estimator = backend.native_moe
        estimator.set_step(ForwardPass({i: RankShape(8, 1, False) for i in range(8)}))
        return estimator

    def test_nvlink_domain_label_independence_and_cross_domain_guard(self):
        conventional = self.estimator("node").estimate(8, layers=1)
        explicit = self.estimator("nvlink_domain").estimate(8, layers=1)
        self.assertEqual(conventional, explicit)
        for options in ({"remote": True}, {"members": 7}):
            with self.assertRaisesRegex(
                ConfigurationError, "complete single-node EP group"
            ):
                self.estimator(**options).estimate(8, layers=1)


if __name__ == "__main__":
    unittest.main()
