"""Serving tests for rank capacity, request progression and placement."""

from dataclasses import replace
from types import SimpleNamespace
import unittest

import simpy

from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.config.simulation import SimulationConfig
from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.llm.engine.core import LLMEngine
from TokenSim.llm.llm_request import Request, RequestStatus
from TokenSim.llm.llm_scheduler import LLMPagedAttnScheduler
from TokenSim.placement.consistent_hash import FixedWorkerRouter
from TokenSim.placement.policies import DataParallelWorkerPool, RoundRobinWorkerPool
from TokenSim.reporting.serving_domains import parallel_domain_stats
from tests.support.hardware import make_device, make_hardware, make_model
from tests.test_kv_cache import layout


def group(n, *, name="hybrid", role="hybrid", hardware="TestGPU", **fields):
    return dict(
        name=name, role=role, hardware=hardware, num_workers=n, network="n", **fields
    )


def serving_cluster(*groups, parallel=None, kv_tokens=None):
    return ClusterConfig(
        num_workers=sum(g["num_workers"] for g in groups),
        networks={"n": "ethernet-test"},
        worker_groups=list(groups),
        parallel_config=parallel,
        kv_cache_capacity_tokens_per_dp_rank=kv_tokens,
    )


def start(cluster, model=None, hardware=None, **settings):
    options = {
        "batching": "paged-attn",
        "latency_fallback": "analytical_only",
        **settings,
    }
    return LLMEngine(
        simpy.Environment(),
        cluster.effective_kv_transfer(),
        model or make_model(),
        cluster,
        hardware or make_hardware(),
        config=SimulationConfig(**options),
    )


def submit(engine, requests):
    for request in requests:
        request.arrive(engine.env)
    engine.add_requests_burst(requests)


class ServingCoreTest(unittest.TestCase):
    def test_pipeline_executes_both_stages_and_charges_one_transfer_per_step(self):
        engine = start(
            serving_cluster(group(2), parallel={"pipeline_parallel_size": 2})
        )
        request = Request(0, 16, 2, 16)
        submit(engine, [request])
        engine.env.run()
        engine.raise_if_failed()
        self.assertTrue(request.is_done)
        first, last = engine.request_workers
        self.assertEqual(first.latency_backend.stats.step_count, 2)
        self.assertEqual(last.latency_backend.stats.step_count, 2)
        self.assertTrue(last.latency_backend.is_last_stage)
        self.assertGreater(first.parallel_communicator.stats.pp_transfer_latency, 0)
        self.assertEqual(last.parallel_communicator.stats.pp_transfer_latency, 0)

    def test_tp_replica_uses_narrowest_physical_rank_capacity(self):
        model = replace(make_model(), max_position_embeddings=65536)
        hardware = make_hardware(
            make_device(), make_device("SmallGPU", memory_gib=0.01)
        )
        cluster = serving_cluster(
            group(2, hardware=["TestGPU", "SmallGPU"]),
            parallel={"tensor_parallel_size": 2},
        )
        engine = start(cluster, model=model, hardware=hardware)
        leader, follower = engine.request_workers
        available = follower.scheduler.block_manager.num_total_gpu_blocks
        self.assertLess(available, leader.cache_config.num_gpu_blocks)
        self.assertEqual(leader.scheduler.block_manager.num_total_gpu_blocks, available)
        request = Request(0, (available + 1) * 16, 1, 16)
        with self.assertRaisesRegex(ConfigurationError, "cannot fit"):
            engine.validate_request_capacity([request])
        short = Request(1, 16, 2, 16)
        engine.validate_request_capacity([short])
        submit(engine, [short])
        engine.env.run()
        engine.raise_if_failed()
        self.assertTrue(short.is_done)

    def test_grouped_null_page_does_not_block_first_admission(self):
        scheduler = LLMPagedAttnScheduler(
            0,
            SimpleNamespace(
                hybrid_layout=layout(),
                block_size=16,
                num_gpu_blocks=40,
                model="fixture",
            ),
            max_occupy_ratio=1 / 40,
        )
        request = Request(0, 8, 2, 16)
        scheduler.add_requests([request])
        self.assertEqual(scheduler.schedule()[0], [request])

    def test_impossible_request_fails_instead_of_retrying_at_same_time(self):
        engine = start(serving_cluster(group(1), kv_tokens=16))
        submit(engine, [Request(0, 16, 4, 16)])
        for _ in range(50):
            if engine.failure_event.triggered or engine.env.peek() == float("inf"):
                break
            engine.env.step()
        with self.assertRaisesRegex(SimulationStateError, "cannot schedule"):
            engine.raise_if_failed()

    def test_model_context_limit_applies_to_every_batching_mode(self):
        model = replace(make_model(), max_position_embeddings=32)
        cluster = serving_cluster(group(1))
        for batching in ("dynamic", "static", "paged-attn"):
            with self.subTest(batching=batching):
                engine = start(cluster, model=model, batching=batching)
                with self.assertRaisesRegex(ConfigurationError, "max_model_len=32"):
                    engine.validate_request_capacity([Request(0, 32, 1, 16)])
                request = Request(1, 31, 10, 16)
                submit(engine, [request])
                engine.env.run()
                engine.raise_if_failed()
                self.assertEqual(request.generation_idx, 1)
                self.assertEqual(request.status, RequestStatus.FINISHED_LENGTH_CAPPED)

    def test_layout_validation_uses_effective_per_domain_settings(self):
        cluster = serving_cluster(
            group(
                2,
                name="P",
                role="prefill",
                parallel_config={
                    "data_parallel_size": 2,
                    "enable_expert_parallel": True,
                },
            ),
            group(1, name="D", role="decode", parallel_config={}),
        )
        with self.assertRaisesRegex(
            ConfigurationError, "expert parallelism on a dense model"
        ):
            start(cluster)
        cluster = serving_cluster(
            group(2, name="P", role="prefill", parallel_config={}),
            group(1, name="D", role="decode", parallel_config={}),
        )
        with self.assertRaisesRegex(ConfigurationError, "one worker per parallel rank"):
            start(cluster)
        cluster = serving_cluster(group(2), parallel={"data_parallel_size": 2})
        with self.assertRaisesRegex(ConfigurationError, "synchronized DP"):
            start(cluster, prefill_schedule_interval=2)
        engine = start(cluster, prefill_schedule_interval=2, dp_lockstep=True)
        self.assertEqual(engine.request_workers[0].prefill_schedule_interval, 2)

    def test_pending_burst_arrivals_count_toward_balanced_placement(self):
        engine = start(
            serving_cluster(group(2)), prefill_worker_pool_type="balanced_load"
        )
        requests = [Request(i, 16, 1, 16) for i in range(4)]
        submit(engine, requests)
        self.assertEqual(
            [next(iter(r.owner_worker_ids)) for r in requests], [0, 1, 0, 1]
        )
        engine.env.run()
        engine.raise_if_failed()
        self.assertTrue(all(w.pending_admissions == 0 for w in engine.request_workers))

    def test_replica_selection_and_counts_are_scoped_by_domain(self):
        cluster = serving_cluster(
            group(1, name="A", role="prefill", parallel_config={}),
            group(
                2, name="B", role="prefill", parallel_config={"data_parallel_size": 2}
            ),
            group(1, name="D", role="decode", parallel_config={}),
        )
        engine = start(cluster)
        requests = [Request(i, 16, 1, 16) for i in range(6)]
        submit(engine, requests)
        engine.env.run()
        engine.raise_if_failed()
        self.assertEqual(
            [(r.dp_domain, r.dp_rank) for r in requests],
            [("A", 0), ("B", 0), ("B", 1)] * 2,
        )
        domains = parallel_domain_stats(engine)["parallel_serving_domains"]
        self.assertEqual(domains["A"]["placements"]["prefill"], {0: 2})
        self.assertEqual(domains["B"]["placements"]["prefill"], {0: 2, 1: 2})

        engine = start(cluster, router_policy="consistent_hash")
        requests = [Request(i, 16, 1, 16, chat_id="same") for i in range(2)]
        submit(engine, requests)
        self.assertEqual(requests[0].owner_worker_ids, requests[1].owner_worker_ids)
        self.assertEqual(set(engine.router.workers), {"A:0", "B:0", "B:1"})

    def test_sticky_affinity_remembers_the_domain_and_stays_bounded(self):
        workers = [
            SimpleNamespace(id=i, execution_domain=name, dp_rank=0)
            for i, name in enumerate(("A", "B"))
        ]
        pool = DataParallelWorkerPool(
            workers, RoundRobinWorkerPool, sticky_sessions=True
        )
        pool.affinity_to_rank.max_entries = 2
        select = lambda chat: pool.select_prefill_worker(
            Request(0, 16, 1, 16, chat_id=chat)
        )
        self.assertIs(select("one"), workers[0])
        self.assertIs(select("two"), workers[1])
        self.assertIs(select("two"), workers[1])
        select("three")
        self.assertLessEqual(len(pool.affinity_to_rank), 2)

    def test_fixed_worker_router_validates_and_selects_expected_worker(self):
        workers = [SimpleNamespace(dp_rank=rank, id=rank) for rank in (0, 1)]
        router = FixedWorkerRouter(workers, {"chat": "1"})
        request = SimpleNamespace(source=None, chat_id="chat")
        self.assertIs(router.select(request), workers[1])
        self.assertEqual(request.dp_rank, 1)
        with self.assertRaisesRegex(ConfigurationError, "unknown DP rank"):
            FixedWorkerRouter(workers[:1], {"chat": 1})
