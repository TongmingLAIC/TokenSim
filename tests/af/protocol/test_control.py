"""Native control serialization, preposted call order, and multi-owner rounds."""

from dataclasses import replace
import json
from types import SimpleNamespace
import unittest

import simpy

from TokenSim.af.caching.estimates import cached_estimate, initialize_cache
from TokenSim.af.execution.control import encode_eager_control
from TokenSim.af.execution.executor import AFExecutor
from TokenSim.af.factory import simulate_decode
from TokenSim.af.providers.queries import AFServiceEstimate, AFTransferQuery
from TokenSim.af.runtime.compiler import compile_af_runtime_plan
from TokenSim.af.trace.metrics import summarize_timeline
from TokenSim.errors import ConfigurationError, SimulationStateError
from tests.af.support.batching import submit_owner
from tests.af.support.config import _cluster
from tests.af.support.execution import (
    FixedComputeProvider,
    FixedTransferProvider,
    af_test_hardware,
    af_test_resources,
)
from tests.af.support.model import _moe
from tests.af.support.profiles import FixedHostFixture


class NativeControlAndOrderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FixedHostFixture.setUpClass()
        cls.fixture = FixedHostFixture()

    def run_case(self, **costs):
        env, ex = self.fixture.executor(native_control_enabled=True, **costs)
        self.fixture.run_step(env, ex)
        return env, ex

    def test_wire_schema_control_messages_and_readback_order(self):
        for count in (1, 8, 16):
            encoded = encode_eager_control(count)
            self.assertEqual(
                json.loads(encoded)["dp_metadata_list"]["0"][
                    "num_tokens_across_dp_cpu"
                ],
                [count],
            )
        env, ex = self.run_case(model=True)
        control = [e for e in ex.trace.events if e.event_type == "control_transfer"]
        data = [e for e in ex.trace.events if e.event_type == "transfer"]
        self.assertEqual(
            [e.payload_bytes for e in control], [8, len(encode_eager_control(1))]
        )
        self.assertTrue(
            all(
                e.details["transport_backend"] == "torch_distributed_nccl"
                for e in control
            )
        )
        self.assertTrue(
            all(e.details["transport_backend"] == "pynccl" for e in data)
            and len(data) == 4
        )

        _, ordered = self.run_case(
            control_length_readback_us=7,
            control_payload_readback_us=11,
            control_decode_us=200,
        )
        events = ordered.trace.events
        by_id = {e.node_id: e for e in events}
        self.assertAlmostEqual(by_id["i0:control:decode"].end_at, 228e-6)
        first_data = next(e for e in events if e.event_type == "transfer")
        self.assertAlmostEqual(first_data.start_at, 228e-6)
        summary = summarize_timeline(
            events,
            0,
            max(e.end_at for e in events),
            ordered.resources.resource_capacities(),
        )
        self.assertAlmostEqual(summary["unexplained_duration"], 0)

    def test_split_block_interleave_cancellation_and_cache_separation(self):
        env, ex = self.fixture.executor(native_control_enabled=True)
        with self.assertRaisesRegex(ConfigurationError, "unsplit"):
            self.fixture.run_step(env, ex, count=17)

        env2, ex2 = self.fixture.executor(
            native_control_enabled=True, control_decode_us=200
        )
        req = SimpleNamespace(
            id=0, is_prefill=False, prefill_len=128, context_len=129, generation_idx=0
        )
        process = env2.process(ex2.run_iteration(ex2.submit(0, [req])))
        env2.run(until=1e-6)
        with self.assertRaisesRegex(SimulationStateError, "in-flight"):
            self.fixture.run_step(env2, ex2)
        process.interrupt("cancel control")
        with self.assertRaises(simpy.Interrupt):
            env2.run(until=process)
        env2.run()
        self.assertFalse(ex2._native_controls)
        self.fixture.run_step(env2, ex2)

        for value in (-1, float("nan"), float("inf"), True):
            with self.assertRaises((ValueError, ConfigurationError)):
                self.fixture.executor(
                    native_control_enabled=True, control_encode_us=value
                )

        class Provider:
            fallback = "analytical_only"

            def __init__(self):
                initialize_cache(self)

            @cached_estimate
            def estimate(self, query):
                return AFServiceEstimate(
                    5 if query.transport_backend == "pynccl" else 7, "oracle"
                )

        q = AFTransferQuery(
            0, 0, 0, "decode", "a_to_f", 1, 256, "bf16", "a", "f", "link", 512
        )
        p = Provider()
        self.assertEqual(
            p.estimate(replace(q, transport_backend="pynccl")).service_duration, 5
        )
        self.assertEqual(
            p.estimate(
                replace(q, transport_backend="torch_distributed_nccl")
            ).service_duration,
            7,
        )

    def test_preposted_call_order_and_failure_cleanup(self):
        with self.assertRaises(ConfigurationError):
            self.fixture.executor(native_call_order_enabled=True)
        env, ex = self.fixture.executor(
            model=True,
            native_control_enabled=True,
            native_call_order_enabled=True,
            a_step_prepare_us=10,
            f_step_prepare_us=100,
            host_receive_submit_us=3,
            control_length_readback_us=50,
            compute_submit_us=3,
        )
        self.fixture.run_step(env, ex)
        by_id = {e.node_id: e for e in ex.trace.events_for(0)}
        for wire in [e for e in by_id.values() if e.event_type == "transfer"]:
            root = wire.node_id.removesuffix(":transport")
            self.assertLess(
                by_id[root + ":receive_submit"].start_at,
                by_id[root + ":host_submit"].start_at,
            )
            self.assertEqual(
                wire.details["readiness_mode"], "native_preposted_call_order"
            )
        ex.trace.validate_invariants(0)

        original = ex.transfer_provider.estimate

        def fail(query):
            if query.operation == "af_control_length":
                raise ValueError("injected transfer failure")
            return original(query)

        ex.transfer_provider.estimate = fail
        with self.assertRaisesRegex(SimulationStateError, "injected transfer failure"):
            self.fixture.run_step(env, ex)
        env.run()
        self.assertFalse(ex._native_controls or ex.active_iterations)
        ex.transfer_provider.estimate = original
        self.fixture.run_step(env, ex)


class NativeMultiOwnerTest(unittest.TestCase):
    def runtime(self, *, native_control=False, batching=True, max_tokens=0, layers=1):
        cluster = _cluster(a_tp=1, a_dp=2, f_tp=1, f_dp=1, f_ep=1)
        cluster = replace(
            cluster,
            af=replace(
                cluster.af,
                af_block_size=2,
                ffn_batching=batching,
                ffn_batch_max_tokens=max_tokens,
                execution_protocol="vllm_afd",
                communication_timing=replace(
                    cluster.af.communication_timing,
                    enabled=True,
                    native_control_enabled=native_control,
                ),
            ),
        )
        hardware = af_test_hardware(
            layers=layers, capacity_gib=0.1, moe=replace(_moe(), num_moe_layers=layers)
        )
        model = replace(
            hardware.model("TestModel"),
            model_id="moe-tiny",
            dtype="bf16",
            kv_cache_dtype="bf16",
            qk_norm=True,
        )
        plan = compile_af_runtime_plan(
            cluster_config=cluster,
            worker_configs=cluster.workers(),
            hardware=hardware,
            model=model,
            block_size=4,
            dtype="bf16",
        )
        env = simpy.Environment()
        provider = FixedComputeProvider(1e-6, 20e-6, router=1e-6, expert=20e-6)
        transfer = FixedTransferProvider(0, 0, timing_scope="transport_service")
        executor = AFExecutor(
            env,
            cluster,
            cluster.workers(),
            provider,
            transfer,
            model.model_id,
            "bf16",
            layers,
            16,
            runtime_plan=plan,
            moe_config=model.moe,
            resources=af_test_resources(env, cluster),
        )
        return env, executor, provider, transfer

    def test_cross_owner_batching_late_peer_and_round_ordering(self):
        with self.assertRaisesRegex(ConfigurationError, "native control requires 1A1F"):
            self.runtime(native_control=True)

        env, executor, provider, transfer = self.runtime(layers=2)
        for owner in range(2):
            env.process(
                executor.run_iteration(submit_owner(executor, owner, count=4)[0])
            )
        env.run()
        routers = [q for q in provider.queries if q.stage == "router"]
        self.assertEqual(
            [(q.layer_id, q.block_id) for q in routers],
            [(0, 0), (0, 1), (1, 0), (1, 1)],
        )
        self.assertEqual(
            len([q for q in transfer.queries if q.operation == "af_f_to_a"]), 8
        )

        env2, ex2, prov2, _ = self.runtime(batching=False)
        left, _ = submit_owner(ex2, 0)
        task = env2.process(ex2.run_iteration(left))
        env2.run(until=100e-6)
        self.assertFalse(task.triggered)
        right, _ = submit_owner(ex2, 1, count=1)
        env2.process(ex2.run_iteration(right))
        env2.run()
        self.assertTrue(task.triggered)
        self.assertEqual(
            [q for q in prov2.queries if q.stage == "router"][0].original_token_count, 3
        )

    def test_inconsistent_microbatches_workspace_limit_and_peer_abort(self):
        env, executor, _, _ = self.runtime()
        errors = []

        def guarded(ex, iteration):
            try:
                yield from ex.run_iteration(iteration)
            except (SimulationStateError, simpy.Interrupt) as error:
                errors.append(str(error))

        env.process(guarded(executor, submit_owner(executor, 0)[0]))
        env.run(until=100e-6)
        with self.assertRaisesRegex(ConfigurationError, "matching microbatch counts"):
            submit_owner(executor, 1, count=4)
        env.run()
        self.assertEqual(len(errors), 1)
        self.assertFalse(executor.active_iterations)

        env2, ex2, prov2, _ = self.runtime(max_tokens=2)
        errors.clear()
        for owner in range(2):
            env2.process(guarded(ex2, submit_owner(ex2, owner)[0]))
        env2.run()
        self.assertEqual(len(errors), 2)
        self.assertFalse(any(q.stage == "router" for q in prov2.queries))

        env3, ex3, prov3, _ = self.runtime()
        original = prov3.estimate
        prov3.estimate = lambda q: (
            (_ for _ in ()).throw(ValueError("peer attention failed"))
            if q.stage == "attention" and q.dp_rank == 1
            else original(q)
        )
        errors.clear()
        for owner in range(2):
            env3.process(guarded(ex3, submit_owner(ex3, owner)[0]))
        env3.run()
        self.assertEqual(len(errors), 2)
        with self.assertRaisesRegex(ConfigurationError, "submit every peer"):
            simulate_decode(SimpleNamespace(cluster=ex3.cluster_config), 2, 8)
