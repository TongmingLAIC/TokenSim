"""MegaMoE validates physical EP locality independently of topology labels."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest

from TokenSim.comm.collectives import CollectiveModel
from TokenSim.config.parallel_config import ParallelConfig, ParallelRankInfo
from TokenSim.errors import ConfigurationError
from TokenSim.hardware import HardwareContext
from TokenSim.hardware.topology import TopologyPlacement
from TokenSim.latency.operator_table import OperatorTableLatencyBackend
from TokenSim.llm.forward import ForwardPass, RankShape
from TokenSim.parallel import ParallelCommunicator

ROOT = Path(__file__).resolve().parents[1]


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

    def test_renaming_the_same_nvlink_domain_preserves_native_latency(self):
        conventional = self.estimator("node").estimate(8, layers=1)
        explicit = self.estimator("nvlink_domain").estimate(8, layers=1)
        self.assertEqual(conventional, explicit)
        self.assertGreater(explicit[0], 0)

    def test_cross_domain_and_incomplete_groups_keep_the_scope_guard(self):
        for options in ({"remote": True}, {"members": 7}):
            with (
                self.subTest(options=options),
                self.assertRaisesRegex(
                    ConfigurationError, "complete single-node EP group"
                ),
            ):
                self.estimator(**options).estimate(8, layers=1)


if __name__ == "__main__":
    unittest.main()
