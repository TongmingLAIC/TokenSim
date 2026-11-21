"""Prefill DP coordinates cannot choose an independent AF decode replica."""

from collections import Counter
from types import SimpleNamespace
import unittest

from TokenSim.kv_transfer.pd_handoff import PDHandoff
from TokenSim.llm.llm_request import Request
from TokenSim.placement.policies import DataParallelWorkerPool, RoundRobinWorkerPool
from TokenSim.workload.source import LLMSource
from tests.af.support.config import _cluster
from tests.af.support.model import _engine


class AFDecodeRoutingTest(unittest.TestCase):
    def selector(self, *, af=True, sticky=False):
        workers = [SimpleNamespace(id=rank, dp_rank=rank) for rank in range(8)]
        pool = DataParallelWorkerPool(
            workers, RoundRobinWorkerPool, sticky_sessions=sticky
        )
        handoff = PDHandoff.__new__(PDHandoff)
        handoff.engine = SimpleNamespace(
            af_runtime_plan=object() if af else None, decode_workers=pool
        )
        return handoff

    def test_decode_round_robin_ignores_prefill_coordinate(self):
        handoff = self.selector()
        requests = [SimpleNamespace(dp_rank=1, chat_id=str(i)) for i in range(16)]
        targets = [handoff._select_decode_target(req).dp_rank for req in requests]
        self.assertEqual(targets, list(range(8)) * 2)
        self.assertEqual([req.dp_rank for req in requests], [1] * 16)

    def test_decode_session_affinity_belongs_to_destination_pool(self):
        handoff = self.selector(sticky=True)
        requests = [
            SimpleNamespace(dp_rank=rank, chat_id="same-agent") for rank in [3, 1, 7, 2]
        ]
        self.assertEqual(
            [handoff._select_decode_target(req).dp_rank for req in requests], [0] * 4
        )
        self.assertEqual([req.dp_rank for req in requests], [3, 1, 7, 2])

    def test_ordinary_shared_dp_namespace_keeps_its_mapping(self):
        handoff = self.selector(af=False)
        req = SimpleNamespace(dp_rank=3, chat_id="agent")
        self.assertEqual(handoff._select_decode_target(req).dp_rank, 3)
        self.assertEqual(req.dp_rank, 3)

    def test_native_handoff_delivers_to_every_af_replica(self):
        env, engine = _engine(_cluster(a_tp=1, a_dp=4, f_tp=1, f_dp=1, f_ep=1))
        engine.pd_handoff = PDHandoff(
            engine,
            dict(
                send_bytes_per_second=1e6,
                receive_bytes_per_second=1e6,
                fabric_bytes_per_second=4e6,
            ),
        )
        requests = [Request(i, 8, 3, block_size=4) for i in range(4)]
        for req in requests:
            req.dp_rank = 0
        env.process(LLMSource(env, engine, requests, 1, "burst"))
        env.run()
        engine.raise_if_failed()
        self.assertTrue(all(req.is_done for req in requests))
        starts = [
            row for row in engine.pd_handoff.events if row["kind"] == "transfer_start"
        ]
        self.assertEqual(
            Counter(row["target_dp_rank"] for row in starts), Counter(range(4))
        )
        self.assertEqual({row["source_dp_rank"] for row in starts}, {0})
        self.assertEqual({req.dp_rank for req in requests}, set(range(4)))
        report = engine.pd_handoff.report()
        self.assertEqual(report["pending_requests"], 0)
        self.assertEqual(report["receiver_reserved_pages"], 0)
        self.assertEqual(report["pinned_source_pages"], 0)


if __name__ == "__main__":
    unittest.main()
