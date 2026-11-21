"""Grouped handoff with a live source request and an empty receiver."""

from dataclasses import replace
from types import SimpleNamespace

import simpy

from TokenSim.config.config import ParallelConfig
from TokenSim.kv_transfer.grouped_pd_handoff import GroupedPDHandoff
from TokenSim.llm.engine.messages import WorkerStatus
from TokenSim.llm.llm_request import Request
from TokenSim.llm.llm_scheduler import LLMPagedAttnScheduler
from tests.support.kv import layout


def setup(prefill=33, slots=40, target_inflight=None):
    env = simpy.Environment()
    cache = SimpleNamespace(
        hybrid_layout=layout(), num_gpu_blocks=slots, model="test", block_size=16
    )
    workers = []
    for i, role in enumerate(["prefill", "decode"]):
        current_cache = cache
        if i == 1 and target_inflight is not None:
            current_cache = SimpleNamespace(
                **{
                    **vars(cache),
                    "hybrid_layout": replace(
                        cache.hybrid_layout, max_in_flight_tokens=target_inflight
                    ),
                }
            )
        scheduler = LLMPagedAttnScheduler(
            i, current_cache, max_num_seqs=4, max_num_batched_tokens=16
        )
        workers.append(
            SimpleNamespace(
                id=i,
                role=role,
                execution_domain=role,
                parallel_config=ParallelConfig(),
                scheduler=scheduler,
                connector=scheduler.connector,
                status=WorkerStatus.WAITING,
            )
        )
    engine = SimpleNamespace(
        env=env,
        config=SimpleNamespace(batching="paged-attn"),
        request_workers=workers,
        workers=workers,
        dp_coordinator=None,
        decode_workers=SimpleNamespace(
            select_transfer_target=lambda requests: workers[1]
        ),
        send_task=lambda *args: None,
    )
    failures = []
    engine.record_worker_failure = lambda worker, error: failures.append(error)
    handoff = GroupedPDHandoff(
        engine,
        dict(
            send_bytes_per_second=1280,
            receive_bytes_per_second=2560,
            fabric_bytes_per_second=5120,
        ),
    )
    req = Request(1, prefill, 4, 16, hash_ids=list(range(prefill // 16)))
    req.arrive(env)
    scheduler = workers[0].scheduler
    scheduler.add_requests([req])
    while req.is_prefill:
        batch, _ = scheduler.schedule()
        env.run(until=env.now + 0.01)
        req.advance(env, 0.01, 1)
        scheduler.on_tokens_computed(req)
        scheduler.update(batch)
    return env, engine, handoff, req, failures
