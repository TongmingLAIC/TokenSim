"""Independent complete-model pools select destination DP coordinates locally."""

from collections import Counter
from types import SimpleNamespace
import unittest

from TokenSim.kv_transfer.pd_handoff import PDHandoff
from TokenSim.placement.policies import DataParallelWorkerPool, RoundRobinWorkerPool
from tests.test_grouped_pd_handoff import setup


class PDDomainRoutingTest(unittest.TestCase):
    def selector(self, independent=True, sticky=False):
        workers = [
            SimpleNamespace(
                id=i, dp_rank=i, execution_domain="D" if independent else None
            )
            for i in range(8)
        ]
        pool = DataParallelWorkerPool(
            workers, RoundRobinWorkerPool, sticky_sessions=sticky
        )
        handoff = PDHandoff.__new__(PDHandoff)
        handoff.engine = SimpleNamespace(
            af_runtime_plan=None, request_workers=workers, decode_workers=pool
        )
        return handoff

    def test_destination_round_robin_does_not_inherit_prefill_dp_coordinate(self):
        handoff = self.selector()
        requests = [SimpleNamespace(dp_rank=0, chat_id=str(i)) for i in range(16)]
        self.assertEqual(
            [handoff._select_decode_target(r).dp_rank for r in requests],
            list(range(8)) * 2,
        )
        self.assertEqual([r.dp_rank for r in requests], [0] * 16)

    def test_destination_session_affinity_uses_the_destination_namespace(self):
        handoff = self.selector(sticky=True)
        requests = [SimpleNamespace(dp_rank=i, chat_id="agent") for i in [1, 3, 0, 6]]
        self.assertEqual(
            [handoff._select_decode_target(r).dp_rank for r in requests], [0] * 4
        )

    def test_ordinary_shared_coordinates_preserve_their_existing_route(self):
        handoff = self.selector(independent=False)
        self.assertEqual(
            handoff._select_decode_target(SimpleNamespace(dp_rank=3)).dp_rank, 3
        )

    def test_grouped_delivery_changes_coordinate_only_after_dma(self):
        env, engine, handoff, req, failures = setup()
        source, target = engine.workers
        source.dp_rank, target.dp_rank, req.dp_rank = 2, 5, 2
        handoff.dispatch(source, [req])
        source.scheduler.running.clear()
        self.assertEqual(req.dp_rank, 2)
        env.run()
        self.assertEqual(failures, [])
        self.assertEqual(req.dp_rank, 5)
        rows = [e for e in handoff.events if e["kind"] == "transfer_start"]
        self.assertEqual(
            Counter((e["source_dp_rank"], e["target_dp_rank"]) for e in rows),
            {(2, 5): 1},
        )
        self.assertFalse(source.scheduler.block_manager.transfer_references)
        self.assertFalse(target.scheduler.block_manager.transfer_references)


if __name__ == "__main__":
    unittest.main()
