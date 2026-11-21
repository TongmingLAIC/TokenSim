"""Ordinary prefill groups retain their own ranks beside explicit AF domains."""

from dataclasses import replace
import unittest

from TokenSim.config.config import ParallelConfig, WorkerGroupConfig
from tests.af.support.config import _cluster


class AFPrefillDomainsTest(unittest.TestCase):
    def test_four_prefill_replicas_have_distinct_dp_ranks(self):
        base = _cluster(a_tp=1, a_dp=1, f_tp=1, f_dp=1, f_ep=1)
        cluster = replace(
            base,
            num_workers=6,
            parallel_config=ParallelConfig(data_parallel_size=4),
            worker_groups=[
                *base.worker_groups,
                WorkerGroupConfig(
                    name="p",
                    role="prefill",
                    hardware="a-device",
                    num_workers=4,
                    network="af-net",
                ),
            ],
        )
        workers = cluster.workers()
        self.assertEqual(
            [w.rank_info.dp_rank for w in workers if w.name == "p"], [0, 1, 2, 3]
        )
        self.assertEqual(
            [w.rank_info.dp_rank for w in workers if w.name == "a-pool"], [0]
        )


if __name__ == "__main__":
    unittest.main()
