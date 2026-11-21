"""Complete-model P/D groups retain distinct TP, DP and expert ownership."""

import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import simpy

from TokenSim.config.config import (
    ClusterConfig,
    KVTransferConfig,
    ParallelConfig,
    WorkerGroupConfig,
)
from TokenSim.config.simulation import SimulationConfig
from TokenSim.errors import ConfigurationError
from TokenSim.llm.llm_engine import LLMEngine
from TokenSim.llm.llm_request import Request
from TokenSim.workload.source import LLMSource
from TokenSim.reporting.stats import (
    get_parallel_stats,
    get_moe_stats,
    get_cache_capacity_stats,
)
from TokenSim.reporting import RunReport, export_result
from tests.support.hardware import make_hardware, moe_test_model
from tests.test_kv_transfer_integration import _DeterministicBackend


def cluster():
    return ClusterConfig(
        num_workers=6,
        networks={"network": "ethernet-test"},
        worker_groups=[
            WorkerGroupConfig(
                name="P",
                role="prefill",
                hardware="TestGPU",
                num_workers=2,
                network="network",
                parallel_config=ParallelConfig(
                    tensor_parallel_size=2,
                    enable_expert_parallel=True,
                    expert_parallel_size=2,
                ),
            ),
            WorkerGroupConfig(
                name="D",
                role="decode",
                hardware="TestGPU",
                num_workers=4,
                network="network",
                parallel_config=ParallelConfig(
                    data_parallel_size=4,
                    enable_expert_parallel=True,
                    expert_parallel_scope="global",
                    expert_parallel_size=4,
                ),
            ),
        ],
    )


class ServingDomainsTest(unittest.TestCase):
    def test_role_settings_are_resolved_and_validated_separately(self):
        candidate = cluster()
        candidate.worker_groups[0].simulation_overrides = dict(
            max_num_seqs=8, max_num_batched_tokens=32, gpu_memory_utilization=0.85
        )
        candidate.worker_groups[1].simulation_overrides = dict(
            max_num_seqs=4, max_num_batched_tokens=16, gpu_memory_utilization=0.9
        )
        engine = LLMEngine(
            simpy.Environment(),
            KVTransferConfig.default(),
            moe_test_model(),
            candidate,
            make_hardware(),
            config=SimulationConfig(
                batching="paged-attn",
                latency_backend_type="analytical",
                latency_fallback="analytical_only",
                block_size=16,
            ),
        )
        self.assertEqual(engine.workers[0].scheduler.max_num_seqs, 8)
        self.assertEqual(engine.workers[2].scheduler.max_num_seqs, 4)
        self.assertEqual(engine.workers[0].scheduler.max_num_batched_tokens, 32)
        self.assertEqual(engine.workers[2].scheduler.max_num_batched_tokens, 16)
        self.assertEqual(
            get_parallel_stats(engine)["parallel_serving_domains"]["P"]["settings"][
                "gpu_memory_utilization"
            ],
            0.85,
        )
        from TokenSim.config.serving_domains import resolve_worker_config

        with self.assertRaises(ConfigurationError):
            resolve_worker_config(engine.config, {"block_size": 32})
        with self.assertRaises(ConfigurationError):
            resolve_worker_config(engine.config, {"max_num_seqs": 0})

    def test_prefill_and_decode_dp_waves_execute_independently_and_complete(self):
        candidate = cluster()
        candidate.worker_groups[0].parallel_config = ParallelConfig(
            data_parallel_size=2,
            enable_expert_parallel=True,
            expert_parallel_scope="global",
            expert_parallel_size=2,
        )
        candidate.kv_cache_capacity_tokens_per_dp_rank = 1024
        env = simpy.Environment()
        transfer = KVTransferConfig(
            kv_connector="P2PConnector",
            kv_connector_extra_config={
                "pd_endpoint_limits": dict(
                    send_bytes_per_second=1e9,
                    receive_bytes_per_second=1e9,
                    fabric_bytes_per_second=2e9,
                )
            },
        )
        with patch(
            "TokenSim.llm.engine.worker.build_latency_backend",
            side_effect=lambda *args, **kwargs: _DeterministicBackend(),
        ):
            engine = LLMEngine(
                env,
                transfer,
                moe_test_model(),
                candidate,
                make_hardware(),
                config=SimulationConfig(
                    batching="paged-attn", block_size=16, dp_lockstep=True
                ),
            )
        prefill_done = {}
        save_duration = 0.001

        def with_blocking_save(complete_step):
            def complete(batch, duration):
                result = complete_step(batch, duration)
                prefill_done.update((request.id, env.now) for request in batch)
                return max(result, save_duration)

            return complete

        for worker in engine.prefill_workers.workers:
            worker.complete_step = with_blocking_save(worker.complete_step)
        requests = [Request(i, 16, 4, 16) for i in range(4)]
        env.process(LLMSource(env, engine, requests, 1, "burst"))
        env.run(until=1)
        engine.raise_if_failed()
        self.assertTrue(all(req.is_done for req in requests))
        self.assertTrue(all(req.prefill_chunks == 1 for req in requests))
        self.assertEqual(len(engine.timing_recorder.timing.time), 4)
        self.assertEqual(engine.pd_handoff.jobs, {})
        starts = [
            row for row in engine.pd_handoff.events if row["kind"] == "transfer_start"
        ]
        self.assertEqual({row["target_dp_rank"] for row in starts}, {0, 1, 2, 3})
        self.assertEqual({row["source_dp_rank"] for row in starts}, {0, 1})
        self.assertTrue(
            all(
                row["time"] >= prefill_done[row["request_id"]] + save_duration
                for row in starts
            )
        )
        self.assertEqual(set(engine.dp_coordinator.coordinators), {"P", "D"})
        self.assertTrue(
            all(c.stats.steps > 0 for c in engine.dp_coordinator.coordinators.values())
        )
        self.assertEqual(get_parallel_stats(engine)["parallel_expected_rank_count"], 6)
        self.assertEqual(
            get_cache_capacity_stats(engine)["kv_cache_capacity_tokens_total"], 6144
        )
        moe = get_moe_stats(engine)
        self.assertEqual(moe["moe_ep_rank_count"], 6)
        self.assertEqual(set(moe["moe_serving_domains"]), {"P", "D"})
        self.assertTrue(all(":" in key for key in moe["moe_per_rank_expert_load"]))
        with tempfile.TemporaryDirectory() as temporary:
            export_result(
                RunReport(Path(temporary), 1, "paged-attn"),
                engine.timing_recorder.timing,
                engine,
                candidate,
                4,
                [16] * 4,
                [4] * 4,
                requests,
                [],
                env.now,
            )
            result = json.loads((Path(temporary) / "result_1.json").read_text())
            self.assertEqual(result["parallel_expected_rank_count"], 6)
            self.assertEqual(set(result["moe_serving_domains"]), {"P", "D"})
            self.assertEqual(result["pd_transport"]["pending_requests"], 0)

    def test_physical_collectives_and_experts_stay_in_their_own_domains(self):
        engine = LLMEngine(
            simpy.Environment(),
            KVTransferConfig.default(),
            moe_test_model(),
            cluster(),
            make_hardware(),
            config=SimulationConfig(
                batching="paged-attn",
                latency_backend_type="analytical",
                latency_fallback="analytical_only",
                block_size=16,
                dp_lockstep=True,
            ),
        )
        self.assertEqual([w.id for w in engine.prefill_workers.workers], [0])
        self.assertEqual([w.id for w in engine.decode_workers.workers], [2, 3, 4, 5])
        for worker in engine.workers[:2]:
            self.assertEqual(worker.parallel_communicator.tp_group(), (0, 1))
            self.assertEqual(worker.parallel_communicator.ep_group(), (0, 1))
            self.assertEqual(
                len(worker.expert_placement.experts_for_rank(worker.rank_info)), 2
            )
        for worker in engine.workers[2:]:
            self.assertEqual(worker.parallel_communicator.tp_group(), (worker.id,))
            self.assertEqual(worker.parallel_communicator.ep_group(), (2, 3, 4, 5))
            self.assertEqual(
                len(worker.expert_placement.experts_for_rank(worker.rank_info)), 1
            )
        self.assertEqual(engine.dp_coordinator.worker_ids, {2, 3, 4, 5})

    def test_incomplete_and_ambiguous_parallel_layouts_fail(self):
        for modify in [
            lambda c: setattr(c.worker_groups[0], "parallel_config", None),
            lambda c: setattr(
                c, "parallel_config", ParallelConfig(data_parallel_size=6)
            ),
            lambda c: setattr(c, "kv_cache_capacity_tokens_total", 4096),
        ]:
            candidate = cluster()
            modify(candidate)
            with (
                self.subTest(candidate=candidate),
                self.assertRaises(ConfigurationError),
            ):
                candidate.workers()


if __name__ == "__main__":
    unittest.main()
