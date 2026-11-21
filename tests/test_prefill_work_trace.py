"""Ordinary prefill traces preserve shapes needed for complete-system FLOPs."""

import unittest

from TokenSim.llm.llm_request import Request
from TokenSim.operator_data.workload import work_for_table
from TokenSim.workload.source import LLMSource
from tests.af.support.config import _cluster
from tests.af.support.model import _engine, _moe


class PrefillWorkTraceTest(unittest.TestCase):
    def test_moe_prefill_exposes_exact_unpadded_work_shape(self):
        env, engine = _engine(
            _cluster(a_tp=1, a_dp=1, f_tp=1, f_dp=1, f_ep=1),
            moe_config=_moe(),
            trace=dict(mode="full"),
        )
        req = Request(0, 8, 2, block_size=4)
        env.process(LLMSource(env, engine, [req], 1, "burst"))
        env.run()
        engine.raise_if_failed()
        rows = [
            e
            for e in engine.trace_session.recorder.events
            if e.operation == "moe_module" and e.phase == "prefill"
        ]
        self.assertTrue(rows)
        source = engine.prefill_workers.workers[0]
        self.assertEqual({row.worker_id for row in rows}, {source.id})
        value = rows[0]
        self.assertEqual(value.details["table"], "moe")
        self.assertEqual(
            work_for_table("moe", value.details["shape"]).flops, 8 * 2 * 16 * 32 * 6
        )
        self.assertIn("device_id", value.details)
        attention = [
            e
            for e in engine.trace_session.recorder.events
            if e.operation == "attention" and e.phase == "prefill"
        ]
        self.assertEqual(
            attention[0].details["work_components"][0]["key"]["input_seq_len"], 8
        )


if __name__ == "__main__":
    unittest.main()
