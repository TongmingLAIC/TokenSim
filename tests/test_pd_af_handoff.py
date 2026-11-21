"""P/D ingress preserves capacity and clocks across explicit AF KV shards."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from TokenSim.kv_transfer.pd_handoff import PDHandoff
from TokenSim.llm.llm_request import Request
from TokenSim.workload.source import LLMSource
from tests.af.runtime import test_kv as kv_tests
from tests.af.support.config import _cluster
from tests.af.support.model import _engine, _request


class PDAFHandoffTest(unittest.TestCase):
    def test_all_rank_reservations_are_consumed_atomically(self):
        manager = kv_tests.ReplicaKVBlockManagerTest()._manager(gpu=(2, 2))
        request = _request(0, prefill_len=8)
        manager.transfer_reservations = {request.id: 2}
        for rank in manager.rank_managers.values():
            rank.transfer_reservations = manager.transfer_reservations
        manager.pd_handoff = SimpleNamespace(admitted=Mock())
        manager.reserve_gpu_blocks(2)
        self.assertTrue(manager.can_allocate(request))
        manager.allocate(request)
        manager.pd_handoff.admitted.assert_called_once_with(request)
        self.assertTrue(
            all(not m._reserved_gpu_blocks for m in manager.rank_managers.values())
        )
        manager.release_request_blocks(request)
        self.assertTrue(
            all(m.get_num_free_blocks() == 2 for m in manager.rank_managers.values())
        )

    def test_native_prefill_handoff_executes_af_decode_after_transfer(self):
        env, engine = _engine(_cluster(a_tp=2, a_dp=1, f_tp=1, f_dp=1, f_ep=1))
        source = engine.prefill_workers.workers[0]
        self.assertEqual(source.parallel_communicator.tp_group(), (source.id,))
        target = engine.decode_workers.workers[0]
        self.assertEqual(target.parallel_communicator.tp_group(), (0, 1))
        engine.pd_handoff = PDHandoff(
            engine,
            dict(
                send_bytes_per_second=1e6,
                receive_bytes_per_second=1e6,
                fabric_bytes_per_second=2e6,
            ),
        )
        req = Request(0, 8, 3, block_size=4)
        env.process(LLMSource(env, engine, [req], 1, "burst"))
        env.run()
        engine.raise_if_failed()
        self.assertTrue(req.is_done)
        self.assertEqual(len(engine.pd_handoff.fabric.completed), 2)
        self.assertEqual(
            len({flow["target"] for flow in engine.pd_handoff.fabric.completed}), 2
        )
        self.assertEqual(engine.pd_handoff.report()["receiver_reserved_pages"], 0)
        self.assertEqual(engine.pd_handoff.report()["pending_requests"], 0)
        for manager in engine.af_replica_block_managers.values():
            self.assertFalse(manager._request_views)
            self.assertTrue(
                all(not m._reserved_gpu_blocks for m in manager.rank_managers.values())
            )


if __name__ == "__main__":
    unittest.main()
