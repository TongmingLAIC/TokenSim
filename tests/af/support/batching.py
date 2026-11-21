"""Deterministic multi-owner AF batches for scheduling tests."""

from dataclasses import replace
from types import SimpleNamespace

import simpy

from TokenSim.af.execution.executor import AFExecutor
from tests.af.support.config import _cluster
from tests.af.support.execution import (
    FixedComputeProvider,
    FixedTransferProvider,
    af_test_resources,
    execution_plan,
)
from tests.af.support.model import _moe


def batch_runtime(
    *,
    owners=4,
    max_tokens=0,
    wait_us=5,
    enabled=True,
    provider=None,
    f_tp=1,
    f_ep=1,
    submit_us=0,
    max_inflight=0,
    transport_effects=None,
    layers=1,
    inflight_scope="iteration",
):
    cluster = _cluster(a_tp=1, a_dp=owners, f_tp=f_tp, f_dp=1, f_ep=f_ep)
    cluster = replace(
        cluster,
        af=replace(
            cluster.af,
            af_block_size=2,
            ffn_batching=enabled,
            ffn_batch_wait_us=wait_us,
            ffn_batch_max_tokens=max_tokens,
            max_inflight_blocks=max_inflight,
            max_inflight_scope=inflight_scope,
            transport_effects=transport_effects or cluster.af.transport_effects,
            communication_timing=replace(
                cluster.af.communication_timing,
                enabled=bool(submit_us),
                compute_submit_us=submit_us,
            ),
        ),
    )
    env = simpy.Environment()
    compute = provider or FixedComputeProvider(1e-6, 20e-6, router=0, expert=20e-6)
    moe = replace(_moe(), num_moe_layers=layers)
    executor = AFExecutor(
        env,
        cluster,
        cluster.workers(),
        compute,
        FixedTransferProvider(0, 0, timing_scope="transport_service"),
        "TestModel",
        "bf16",
        layers,
        16,
        runtime_plan=execution_plan(cluster, layers=layers, moe=moe),
        moe_config=moe,
        resources=af_test_resources(env, cluster),
    )
    return env, executor, compute


def submit_owner(executor, owner, *, count=2, offset=0):
    requests = [
        SimpleNamespace(
            id=owner * 100 + offset + i,
            is_prefill=False,
            prefill_len=8,
            context_len=8,
            generation_idx=0,
        )
        for i in range(count)
    ]
    return executor.submit(owner, requests), requests
