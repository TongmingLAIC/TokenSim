"""Pipeline stages own their model/cache shards and execute concurrently."""

from dataclasses import replace
from types import SimpleNamespace
import unittest

import simpy

from TokenSim.af.execution.model import ModelAFExecutor
from TokenSim.af.factory import build_model_runtime
from TokenSim.af.planning.batching import AFBlockBuilder
from TokenSim.af.providers.operator_table import build_operator_table_af_providers
from TokenSim.af.runtime.admission import ResidentKV, rank_kv_bytes
from TokenSim.af.runtime.compiler import compile_af_runtime_plan
from TokenSim.af.serving.service import AFService
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.context import HardwareContext
from TokenSim.hardware.placement import build_topology_placement
from TokenSim.trace.recorder import TraceRecorder

from tests.af.support.config import _cluster, _cluster_dict
from tests.af.support.execution import (
    FixedComputeProvider,
    FixedTransferProvider,
    af_test_hardware,
)
from tests.af.support.model import _moe, _runtime_hardware, _runtime_plan
from tests.af.support.pipeline import decode_request, pipeline_cluster
from tests.support.hardware import make_device


class PipelineRuntimeTest(unittest.TestCase):
    def runtime(self, a_pp=2, f_pp=2):
        cluster = pipeline_cluster(a_pp, f_pp)
        hardware = _runtime_hardware()
        model = replace(hardware.model("TestModel"), moe=_moe(top_k=1, shared=1))
        plan = compile_af_runtime_plan(
            cluster_config=cluster,
            worker_configs=cluster.workers(),
            hardware=hardware,
            model=model,
            block_size=4,
        )
        return cluster, model, plan

    def test_stage_weights_cache_and_collectives_use_physical_ranks(self):
        cluster, model, plan = self.runtime()
        _, _, single = self.runtime(1, 1)
        self.assertEqual(
            sum(state.model_bytes for state in plan.physical_states),
            sum(state.model_bytes for state in single.physical_states),
        )
        self.assertEqual(
            sum(
                b.kv_partition.bytes_per_token
                for b in plan.rank_bindings
                if b.kv_partition
            ),
            sum(
                b.kv_partition.bytes_per_token
                for b in single.rank_bindings
                if b.kv_partition
            ),
        )
        for binding in plan.rank_bindings:
            state = plan.physical_state(binding.physical_identity)
            self.assertTrue(
                all(
                    component.layer_id in (None, binding.pp_rank)
                    for component in state.components
                )
            )
            self.assertEqual(
                binding.tp_group, (2 * binding.pp_rank, 2 * binding.pp_rank + 1)
            )
        self.assertEqual(plan.replica("a-domain", 0).leader_rank_key, ("a-domain", 0))
        self.assertEqual(
            plan.collective_plan("a-domain", 0, "residual", 1).participant_ranks, (2, 3)
        )
        with self.assertRaises(ConfigurationError):
            self.runtime(3, 2)

    def test_pipeline_handoff_uses_the_production_transport_provider(self):
        cluster, model, plan = self.runtime()
        hardware = _runtime_hardware()
        workers = cluster.workers()
        placement = build_topology_placement(cluster, workers, hardware)
        compute, transfer = build_operator_table_af_providers(
            hardware=hardware,
            model=model,
            cluster_config=cluster,
            worker_configs=workers,
            placement=placement,
            fallback="analytical_only",
        )
        env = simpy.Environment()
        trace = TraceRecorder(True)
        executor = ModelAFExecutor(
            env,
            cluster,
            workers,
            compute,
            transfer,
            model.model_id,
            model.dtype,
            model.num_layers,
            model.hidden_size,
            runtime_plan=plan,
            model=model,
            moe_config=model.moe,
            trace_recorder=trace,
        )
        iteration = executor.submit(0, [decode_request(1)])
        env.process(executor.run_iteration(iteration))
        env.run()
        handoffs = [
            event
            for event in trace.events
            if event.operation == "af_pipeline_handoff"
            and event.event_type == "transfer"
        ]
        self.assertTrue(handoffs)
        self.assertTrue(all(event.match_type == "analytical" for event in handoffs))

    def test_uniform_moe_template_does_not_reuse_another_pipeline_stage(self):
        cluster = pipeline_cluster()
        hardware = _runtime_hardware()
        moe = replace(_moe(top_k=1), num_experts=1, num_moe_layers=2)
        model = replace(hardware.model("TestModel"), moe=moe)
        plan = compile_af_runtime_plan(
            cluster_config=cluster,
            worker_configs=cluster.workers(),
            hardware=hardware,
            model=model,
            block_size=4,
        )
        env = simpy.Environment()
        transfer = FixedTransferProvider(0, 0)
        transfer.operation_durations["af_pipeline_handoff"] = 0
        transfer.placement = build_topology_placement(
            cluster, cluster.workers(), hardware
        )
        trace = TraceRecorder(True)
        executor = ModelAFExecutor(
            env,
            cluster,
            cluster.workers(),
            FixedComputeProvider(1, 1),
            transfer,
            model.model_id,
            model.dtype,
            2,
            model.hidden_size,
            runtime_plan=plan,
            model=model,
            moe_config=moe,
            trace_recorder=trace,
        )
        iteration = executor.submit(0, [decode_request(1)])
        env.process(executor.run_iteration(iteration))
        env.run()
        stage_zero = {
            event.worker_id
            for event in trace.events
            if event.stage == "attention" and event.layer_id == 0
        }
        stage_one = {
            event.worker_id
            for event in trace.events
            if event.stage == "attention" and event.layer_id == 1
        }
        self.assertTrue(stage_zero.isdisjoint(stage_one))

    def simulate(self, parallel):
        cluster, model, plan = self.runtime()
        env = simpy.Environment()
        trace = TraceRecorder(True)
        transfer = FixedTransferProvider(0, 0)
        transfer.operation_durations["af_pipeline_handoff"] = 0.1
        transfer.placement = build_topology_placement(
            cluster, cluster.workers(), _runtime_hardware()
        )
        executor = ModelAFExecutor(
            env,
            cluster,
            cluster.workers(),
            FixedComputeProvider(1, 1),
            transfer,
            model.model_id,
            "bf16",
            2,
            model.hidden_size,
            runtime_plan=plan,
            model=model,
            moe_config=model.moe,
            trace_recorder=trace,
        )

        def run():
            processes = []
            for request_id in (1, 2, 3, 4):
                iteration = executor.submit(0, [decode_request(request_id)])
                process = env.process(executor.run_iteration(iteration))
                if parallel:
                    processes.append(process)
                else:
                    yield process
            if processes:
                yield env.all_of(processes)

        env.process(run())
        env.run()
        return env.now, trace

    def test_microbatches_overlap_stages_and_conserve_handoff_bytes(self):
        serial, _ = self.simulate(False)
        parallel, trace = self.simulate(True)
        self.assertLess(parallel, serial)
        transfers = [
            event
            for event in trace.events
            if event.operation == "af_pipeline_handoff"
            and event.event_type == "transfer"
        ]
        self.assertEqual(
            sum(event.payload_bytes for event in transfers), 4 * 2 * 16 * 2
        )
        attention = [
            event
            for event in trace.events
            if event.event_type == "compute" and event.service_duration > 0
        ]
        overlapping = [
            (left, right)
            for left in attention
            for right in attention
            if left.iteration_id != right.iteration_id
            and left.layer_id != right.layer_id
            and left.start_at < right.end_at
            and right.start_at < left.end_at
        ]
        self.assertTrue(overlapping)


class ExpertShardTest(unittest.TestCase):
    def test_expert_owners_memory_and_local_tensor_axes(self):
        cluster = _cluster(
            a_tp=1, a_dp=1, f_tp=4, f_dp=2, f_ep=4, placement_strategy="dedicated"
        )
        plan = _runtime_plan(cluster)
        self.assertEqual(plan.domain_runtime("f-domain").expert_tensor_parallel_size, 2)
        for expert in range(4):
            bindings = plan.expert_execution("f-domain", expert, 0)
            self.assertEqual(
                tuple(b.domain_rank for b in bindings), (2 * expert, 2 * expert + 1)
            )
            for offset, binding in enumerate(bindings):
                shape = plan.local_shape(
                    "f-domain", binding.domain_rank, 0, "routed_expert"
                )
                self.assertEqual(shape["expert_tp_rank"], offset)
                self.assertEqual(shape["intermediate_size"], 16)
                self.assertEqual(
                    plan.expert_tp_group(binding), (2 * expert, 2 * expert + 1)
                )
        routed = [
            component
            for state in plan.physical_states
            for component in state.components
            if component.category == "routed_expert"
        ]
        self.assertEqual(
            sum(component.bytes for component in routed), 4 * 3 * 16 * 32 * 2
        )

    def test_native_dag_executes_expert_collectives_on_the_owned_shards(self):
        cluster = _cluster(
            a_tp=1, a_dp=1, f_tp=4, f_dp=2, f_ep=4, placement_strategy="dedicated"
        )
        hardware = _runtime_hardware()
        plan = _runtime_plan(cluster, hardware=hardware)
        model = hardware.model("TestModel")
        # The shared runtime fixture gives this model a MoE layer.
        from dataclasses import replace
        from tests.af.support.model import _moe

        model = replace(model, moe=_moe(top_k=1, shared=1))
        workers = cluster.workers()
        placement = build_topology_placement(cluster, workers, hardware)
        compute, transfer = build_operator_table_af_providers(
            hardware=hardware,
            model=model,
            cluster_config=cluster,
            worker_configs=workers,
            placement=placement,
            fallback="analytical_only",
        )
        env = simpy.Environment()
        executor = ModelAFExecutor(
            env,
            cluster,
            workers,
            compute,
            transfer,
            model.model_id,
            model.dtype,
            model.num_layers,
            model.hidden_size,
            runtime_plan=plan,
            model=model,
            moe_config=model.moe,
            trace_recorder=TraceRecorder(True),
        )
        request = SimpleNamespace(
            id=1, is_prefill=False, prefill_len=8, context_len=8, generation_idx=0
        )
        iteration = executor.submit(0, [request])
        env.process(executor.run_iteration(iteration))
        env.run()
        routed = [e for e in executor.trace.events if e.stage == "routed_expert"]
        self.assertEqual(len(routed), 2)
        self.assertEqual(len({e.resource_id for e in routed}), 2)
        self.assertGreater(iteration.makespan, 0)


class ResidentKVTest(unittest.TestCase):
    def test_local_windows_are_bounded_and_full_layers_grow(self):
        hardware = _runtime_hardware()
        model = replace(hardware.model("TestModel"), attention_windows=(4, 0))
        cluster = pipeline_cluster()
        plan = compile_af_runtime_plan(
            cluster_config=cluster,
            worker_configs=cluster.workers(),
            hardware=hardware,
            model=model,
            block_size=4,
        )
        local, full = plan.binding("a-domain", 0), plan.binding("a-domain", 2)
        self.assertEqual(
            rank_kv_bytes(plan, local, 100), rank_kv_bytes(plan, local, 1000)
        )
        self.assertEqual(
            rank_kv_bytes(plan, full, 1000), 10 * rank_kv_bytes(plan, full, 100)
        )
        ledger = ResidentKV(plan)
        ledger.admit(1, "a-domain", "a0", maximum_context=1000, current_context=100)
        initial = ledger.summary()
        ledger.commit(1, 200)
        current = ledger.summary()
        self.assertEqual(
            initial["reserved_peak_kv_bytes"], current["reserved_peak_kv_bytes"]
        )
        self.assertGreater(
            sum(current["resident_current_kv_bytes"].values()),
            sum(initial["resident_current_kv_bytes"].values()),
        )
        with self.assertRaises(ConfigurationError):
            ledger.admit(2, "a-domain", "a0", maximum_context=10**12, current_context=0)
        self.assertEqual(len(ledger.leases), 1)
        ledger.release(1)
        self.assertEqual(sum(ledger.reserved.values()), 0)


class ServingLimitsTest(unittest.TestCase):
    @staticmethod
    def _limits_runtime(*, block_tokens=1, f_capacity=0.1, **fields):
        base = af_test_hardware(capacity_gib=0.1)
        model = replace(base.model("TestModel"), vocab_size=32)
        hardware = HardwareContext.in_memory(
            [
                make_device("a-device", memory_gib=0.1),
                make_device("f-device", memory_gib=f_capacity),
            ],
            [model],
            list(base.links),
            list(base.topologies),
        )
        document = _cluster_dict(af_block_size=block_tokens)
        document.update(fields)
        return build_model_runtime(
            document, hardware=hardware, model=model, block_size=4
        )

    def test_explicit_kv_limits_and_multi_owner_partition(self):
        for field in (
            "kv_cache_capacity_tokens_per_dp_rank",
            "kv_cache_capacity_tokens_total",
        ):
            env = simpy.Environment()
            service = AFService(env, self._limits_runtime(**{field: 32}))
            first = service.admit(0, 0, maximum_context=32, current_context=31)
            with self.assertRaisesRegex(ConfigurationError, "compiled replica budget"):
                service.admit(1, 0, maximum_context=32, current_context=31)
            env.process(service.forward([first]))
            env.run()
            self.assertEqual(first.cached_context, 32)
            service.release(first)
            service.admit(1, 0, maximum_context=32, current_context=31)

        base = af_test_hardware(capacity_gib=0.1)
        document = _cluster_dict(attention_workers=2)
        document["kv_cache_capacity_tokens_total"] = 64
        service = AFService(
            simpy.Environment(),
            build_model_runtime(
                document, hardware=base, model=base.model("TestModel"), block_size=4
            ),
        )
        for owner in (0, 1):
            service.admit(owner, owner, maximum_context=32, current_context=0)
            with self.assertRaisesRegex(ConfigurationError, "compiled replica budget"):
                service.admit(owner + 10, owner, maximum_context=1, current_context=0)

    def test_workspace_token_budget_and_block_packing(self):
        tight = self._limits_runtime(f_capacity=6400 / (1 << 30))
        env = simpy.Environment()
        service = AFService(env, tight)
        session = service.admit(0, 0, maximum_context=64, current_context=0)
        with self.assertRaisesRegex(ConfigurationError, "workspace token budget"):
            env.run(until=env.process(service.forward([session], query_tokens=64)))

        wide = self._limits_runtime(block_tokens=64)
        env2 = simpy.Environment()
        svc2 = AFService(env2, wide)
        s2 = svc2.admit(0, 0, maximum_context=64, current_context=0)
        env2.run(until=env2.process(svc2.forward([s2], query_tokens=64)))
        self.assertEqual(s2.cached_context, 64)

        packed = self._limits_runtime(block_tokens=8)
        requests = [
            SimpleNamespace(
                id=i, is_prefill=False, context_len=16, scheduled_query_tokens=count
            )
            for i, count in enumerate((3, 5, 2))
        ]
        iteration = AFBlockBuilder().build(
            0, requests, 8, 0, packed.cluster.af.links[0]
        )
        self.assertEqual([b.token_count for b in iteration.blocks], [8, 2])
