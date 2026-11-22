"""Compressed AF tensor ownership, native module scope and admission contracts."""

from dataclasses import replace
from pathlib import Path
import unittest

import simpy
from pydantic import TypeAdapter

from TokenSim.af.factory import build_model_runtime
from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.hardware import HardwareContext
from TokenSim.af.providers.queries import AFComputeQuery
from TokenSim.af.providers.operator_table import (
    OperatorTableAFComputeProvider,
    build_operator_table_af_providers,
)
from TokenSim.af.runtime.admission import ResidentKV, rank_kv_bytes
from TokenSim.af.runtime.compressed import compressed_layout
from TokenSim.af.serving.service import AFService
from TokenSim.config.weight_memory import packed_weight_memory
from TokenSim.errors import ConfigurationError
from TokenSim.operator_data.package import OperatorDataPackage, SourceRecord
from tests.af.support.config import _cluster


def compressed_runtime(a_tp=1, *, device="b200_sxm", topology="hgx_b200_8x_xdr"):
    hardware = HardwareContext.load(Path(__file__).resolve().parents[1] / "data")
    model = hardware.models.resolve("deepseek-v4-pro")
    cluster = _cluster(a_tp=a_tp, a_dp=4 // a_tp, f_tp=1, f_dp=12, f_ep=12)
    doc = TypeAdapter(ClusterConfig).dump_python(cluster, exclude_none=True)
    doc["topology"] = topology
    doc["networks"] = {"af-net": "ethernet400Gb"}
    for group in doc["worker_groups"]:
        group["hardware"] = device
    doc["af"].update(
        af_block_size=4, max_inflight_blocks=2, routed_expert_execution="grouped"
    )
    return build_model_runtime(doc, hardware=hardware, model=model)


class CompressedAFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = compressed_runtime()

    def provider(self):
        r = self.runtime
        return build_operator_table_af_providers(
            hardware=r.hardware,
            model=r.model,
            cluster_config=r.cluster,
            worker_configs=r.workers,
            placement=r.placement,
            dtype="bf16",
            fallback="analytical_only",
        )[0]

    def query(self, layer=0, stage="attention", operation="attention"):
        return AFComputeQuery(
            iteration_id=0,
            block_id=0,
            layer_id=layer,
            phase="decode",
            stage=stage,
            token_count=2,
            context_lengths=(4096, 4096),
            query_lengths=(1, 1),
            model_id=self.runtime.model.model_id,
            dtype="bf16",
            device_type="b200_sxm",
            worker_id=0,
            operation=operation,
            local_token_count=2,
            local_shape=dict(
                tokens=2, hidden_size=7168, global_hidden_size=7168, tp_size=1
            ),
        )

    def test_native_packed_attention_components_are_owned_by_a(self):
        r = self.runtime
        binding = next(
            b for b in r.plan.rank_bindings if b.capability == "af_attention"
        )
        state = r.plan.physical_state(binding.physical_identity)
        report = packed_weight_memory(r.model, include_mtp=False)
        actual = state.component_bytes()
        for category in (
            "attention",
            "indexer",
            "compressor",
            "mhc",
            "norm",
            "embedding",
            "lm_head",
        ):
            native = report["components"][category]
            self.assertEqual(
                actual[category],
                native["weight_bytes"]
                + native["metadata_bytes"]
                + native["scale_bytes"],
            )
        self.assertNotIn("routed_expert", actual)

    def test_engine_defaults_to_activation_precision_for_fp8_weights(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from TokenSim.llm.engine.af import AFEngineMixin

        class CompilationReached(Exception):
            pass

        engine = SimpleNamespace(
            env=simpy.Environment(),
            hardware=self.runtime.hardware,
            cluster_config=self.runtime.cluster,
            model_spec=self.runtime.model,
            config=SimpleNamespace(
                block_size=256,
                operator_backend=None,
                latency_fallback="analytical_only",
                latency_backend_type="analytical",
                random_seed=42,
                gpu_memory_utilization=1,
            ),
        )
        with patch(
            "TokenSim.llm.engine.af.compile_af_runtime_plan",
            side_effect=CompilationReached,
        ) as compile_plan:
            with self.assertRaises(CompilationReached):
                AFEngineMixin._initialize_workers(
                    engine,
                    None,
                    [SimpleNamespace(capabilities=["prefill"])],
                    None,
                    None,
                    None,
                )
        self.assertEqual(self.runtime.model.dtype, "fp8_block")
        self.assertEqual(compile_plan.call_args.kwargs["dtype"], "bf16")

    def test_exact_expert_partition_and_scales(self):
        r = self.runtime
        report = packed_weight_memory(r.model, include_mtp=False)
        all_experts = [
            c
            for s in r.plan.physical_states
            for c in s.components
            if c.category == "routed_expert"
        ]
        self.assertEqual(len(all_experts), 61 * 384)
        native = report["components"]["routed_expert"]
        self.assertEqual(
            sum(c.bytes for c in all_experts),
            native["weight_bytes"] + native["scale_bytes"],
        )
        self.assertEqual(sum(c.scale_bytes for c in all_experts), native["scale_bytes"])
        for state in r.plan.physical_states:
            self.assertLess(
                state.model_bytes + state.protocol_buffer_bytes, state.capacity_bytes
            )

    def test_native_group_slots_and_boundaries(self):
        plan = self.runtime.plan
        binding = next(b for b in plan.rank_bindings if b.kv_partition)
        layout = plan.compressed_kv_layout
        for context in (1, 127, 128, 255, 256, 4096, 131072):
            self.assertEqual(
                rank_kv_bytes(plan, binding, context),
                layout.max_request_slots(context) * layout.physical_slot_bytes,
            )
        owner = plan.request_owners[0]
        memory = ResidentKV(plan)
        memory.admit(
            1, owner.domain, owner.instance, maximum_context=4096, current_context=128
        )
        memory.release(1)
        self.assertEqual(sum(memory.reserved.values()), 0)

    def test_compressed_pool_does_not_advertise_unconfigured_host_swap(self):
        from TokenSim.block.hybrid.af_replica import (
            CompressedReplicaKV,
            cache_configuration,
        )
        from TokenSim.llm.llm_request import Request

        plan = self.runtime.plan
        original = next(r for r in plan.replicas if r.is_attention)
        self.assertEqual(original.logical_cpu_blocks, 0)
        replica = replace(
            original,
            logical_gpu_blocks=256,
            local_gpu_blocks=tuple((key, 256) for key, _ in original.local_gpu_blocks),
        )
        cache = cache_configuration(plan, replica)
        manager = CompressedReplicaKV(plan, replica)
        request = Request(0, 128, 2, 256)
        self.assertTrue(manager.can_allocate(request))
        manager.allocate(request)
        self.assertEqual(cache.num_cpu_blocks, 0)
        for rank, _ in replica.local_gpu_blocks:
            self.assertEqual(
                manager.rank_status(rank)["gpu_used"], manager.get_gpu_status()[1]
            )
        manager.release_request_blocks(request)

    def test_resident_admission_retains_the_shared_pool_null_slot(self):
        plan = self.runtime.plan
        owner = plan.request_owners[0]
        original = plan.replica_for_instance(owner.domain, owner.instance)
        needed = plan.compressed_kv_layout.max_request_slots(128)

        def limited(slots):
            replica = replace(
                original,
                logical_gpu_blocks=slots,
                local_gpu_blocks=tuple(
                    (rank, slots) for rank, _ in original.local_gpu_blocks
                ),
            )
            return replace(
                plan,
                replicas=tuple(
                    replica if r.replica_key == replica.replica_key else r
                    for r in plan.replicas
                ),
            )

        with self.assertRaisesRegex(ConfigurationError, "compiled replica budget"):
            ResidentKV(limited(needed)).admit(
                0, owner.domain, owner.instance, maximum_context=128, current_context=0
            )
        memory = ResidentKV(limited(needed + 1))
        memory.admit(
            0, owner.domain, owner.instance, maximum_context=128, current_context=0
        )
        self.assertEqual(len(memory.leases), 1)

    def test_attention_module_excludes_dense_qkv_and_retains_indexer(self):
        result = self.provider().estimate(self.query(layer=2))
        components = result.details["components"]
        self.assertEqual(
            [c["name"] for c in components],
            ["mhc_pre_site", "attention_norm", "compressed_attention"],
        )
        module = components[-1]
        self.assertEqual(module["key"]["compress_ratio"], 4)
        self.assertGreater(module["work_flops_by_dtype"]["fp8"], 0)
        self.assertGreater(module["work_flops_by_dtype"]["bf16"], 0)

    def test_cache_separates_csa_hca_and_context(self):
        p = self.provider()
        hca = p.estimate(self.query(0))
        csa = p.estimate(self.query(2))
        longer = p.estimate(replace(self.query(2), context_lengths=(131072, 131072)))
        self.assertNotEqual(hca.service_duration, csa.service_duration)
        self.assertGreater(longer.service_duration, csa.service_duration)
        self.assertEqual(csa, p.estimate(self.query(2)))

    def test_ragged_prefill_owns_projection_traffic_once(self):
        provider = self.provider()
        base = replace(
            self.query(layer=2),
            phase="prefill",
            token_count=4,
            local_token_count=4,
            query_lengths=(2, 2),
            context_lengths=(2048, 4096),
            local_shape=dict(
                tokens=4, hidden_size=7168, global_hidden_size=7168, tp_size=1
            ),
        )
        mixed = provider.estimate(base).details["components"][-1]
        separate = []
        for context in (2048, 4096):
            q = replace(
                base,
                token_count=2,
                local_token_count=2,
                query_lengths=(2,),
                context_lengths=(context,),
                local_shape=dict(
                    tokens=2, hidden_size=7168, global_hidden_size=7168, tp_size=1
                ),
            )
            separate.append(provider.estimate(q).details["components"][-1])
        self.assertLess(mixed["latency_us"], sum(c["latency_us"] for c in separate))
        self.assertGreaterEqual(
            mixed["latency_us"], max(mixed["analytical_bounds"].values())
        )
        for precision, value in mixed["work_flops_by_dtype"].items():
            self.assertEqual(
                value, sum(c["work_flops_by_dtype"].get(precision, 0) for c in separate)
            )
        with self.assertRaises(ConfigurationError):
            provider.estimate(replace(base, query_lengths=(2.0, 2.0)))

    def test_mhc_stays_on_a_and_replaces_generic_residual(self):
        p = self.provider()
        attention = p.estimate(self.query())
        middle = p.estimate(self.query(stage="layout", operation="post_attention_norm"))
        tail = p.estimate(self.query(stage="residual", operation="residual"))
        names = [
            c["name"]
            for e in (attention, middle, tail)
            for c in e.details["components"]
        ]
        self.assertEqual(names.count("mhc_pre_site"), 2)
        self.assertEqual(names.count("mhc_post_site"), 2)
        self.assertNotIn("residual_add", names)

    def test_invalid_queries_and_draft_are_rejected(self):
        with self.assertRaises(ConfigurationError):
            self.provider().estimate(replace(self.query(), query_lengths=(4, 4)))
        draft = self.runtime.hardware.models.resolve("deepseek-v4-pro-0813-dspark")
        with self.assertRaisesRegex(ConfigurationError, "DSpark"):
            compressed_layout(draft, 256, 4)
        with self.assertRaises(ConfigurationError):
            compressed_layout(self.runtime.model, 16, 4)

    def test_split_prefill_decode_and_lease_release(self):
        env = simpy.Environment()
        service = AFService(env, self.runtime, trace=True)
        session = service.admit(0, 0, maximum_context=8, current_context=0)

        def run():
            yield from service.forward(
                [session], phase="prefill", query_tokens=4, context=4
            )
            yield from service.forward([session], phase="decode", context=5)
            service.release(session)

        env.process(run())
        env.run()
        self.assertEqual(len(service.completed_passes), 2)
        self.assertFalse(service.memory.leases)
        modules = [
            e
            for e in service.trace.events
            if e.event_type == "compute" and e.stage == "attention"
        ]
        self.assertEqual(len(modules), 122)
        self.assertGreater(env.now, 0)

    def test_tp_tail_mhc_owns_only_its_hidden_shard(self):
        provider = self.provider()
        full = provider.estimate(self.query(stage="residual", operation="residual"))
        query = self.query(stage="residual", operation="residual")
        shard = replace(
            query,
            local_shape=dict(
                tokens=2, hidden_size=3584, global_hidden_size=7168, tp_size=2
            ),
        )
        half = provider.estimate(shard)
        self.assertEqual(
            half.details["components"][0]["work_flops_by_dtype"]["fp32"] * 2,
            full.details["components"][0]["work_flops_by_dtype"]["fp32"],
        )

    def test_tp_residual_collective_preserves_all_hyper_channels(self):
        from TokenSim.af.execution.model import ModelAFExecutor
        from TokenSim.af.providers.operator_table import (
            build_operator_table_af_providers,
        )
        from types import SimpleNamespace

        runtime = compressed_runtime(a_tp=2)
        compute, transfer = build_operator_table_af_providers(
            hardware=runtime.hardware,
            model=runtime.model,
            cluster_config=runtime.cluster,
            worker_configs=runtime.workers,
            placement=runtime.placement,
            fallback="analytical_only",
        )
        env = simpy.Environment()
        executor = ModelAFExecutor(
            env,
            runtime.cluster,
            runtime.workers,
            compute,
            transfer,
            runtime.model.model_id,
            "bf16",
            61,
            7168,
            moe_config=runtime.model.moe,
            runtime_plan=runtime.plan,
            model=runtime.model,
        )
        iteration = executor.submit(
            0,
            [
                SimpleNamespace(
                    id=0,
                    is_prefill=False,
                    prefill_len=128,
                    context_len=128,
                    generation_idx=0,
                )
            ],
        )
        env.process(executor.run_iteration(iteration))
        env.run()
        events = [
            e
            for e in executor.trace.events
            if e.event_type == "collective" and e.node_id.endswith(":residual:layout")
        ]
        self.assertEqual(len(events), 61)
        for event in events:
            self.assertEqual(event.payload_bytes, 7168 // 2 * 2 * 4)

    def test_measured_provenance_clamped_attention_and_dynamic_ffn(self):
        provider = OperatorTableAFComputeProvider(
            self.runtime.hardware,
            self.runtime.model,
            operator_backend="vllm",
            fallback="table_first",
        )
        module = provider.estimate(
            replace(self.query(layer=2), device_type="gb200")
        ).details["components"][-1]
        self.assertNotEqual(module["match_type"], "analytical")
        self.assertIn("compose:", module["source_id"])

        routed_q = replace(
            self.query(layer=2, stage="routed_expert", operation="routed_expert"),
            device_type="gb200",
            expert_ids=(0, 1),
            expert_workload=((0, 1), (1, 1)),
            weight_dtype="mxfp4",
            ep_group=tuple(range(8)),
        )
        comp = provider.estimate(routed_q).details["components"][0]
        self.assertEqual(
            (comp["table"], comp["match_type"]), ("routed_experts", "analytical")
        )

        rt = compressed_runtime(device="b300_sxm", topology="gb300_nvl72")
        hw, model = rt.hardware, rt.model
        pkg = hw.operator_package("b300_sxm", "vllm")
        source = SourceRecord("test:rmsnorm", "A", "measured")
        meta = replace(pkg.meta, sources={**pkg.meta.sources, source.source_id: source})
        norm = OperatorDataPackage.from_rows(
            meta,
            {
                "elementwise": [
                    dict(
                        op_name="rmsnorm",
                        dtype=model.operator_dtype("norm"),
                        num_tokens=1,
                        hidden_size=model.hidden_size,
                        latency_us=1.0,
                        source_id=source.source_id,
                    )
                ]
            },
        )
        hw.register_package(
            replace(pkg, meta=meta, tables={**pkg.tables, **norm.tables})
        )
        providers = build_operator_table_af_providers(
            hardware=hw,
            model=model,
            cluster_config=rt.cluster,
            worker_configs=rt.workers,
            placement=rt.placement,
            dtype=model.activation_dtype,
            operator_backend="vllm",
        )
        env = simpy.Environment()
        service = AFService(env, rt, providers=providers)
        session = service.admit(0, 0, maximum_context=1, current_context=0)
        env.process(service.forward([session], phase="prefill"))
        env.run()
        csa = next(l.layer_id for l in model.layer_specs() if l.attention_kind == "csa")
        event = next(
            e
            for e in service.trace.events
            if e.stage == "attention" and e.layer_id == csa
        )
        self.assertEqual(event.match_type, "clamped")
