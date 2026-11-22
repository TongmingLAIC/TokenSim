"""Operator ownership, precision, ragged attention, and wire codec contracts."""

from __future__ import annotations

from dataclasses import replace
import unittest

import simpy

from TokenSim.af.providers.operator_table import OperatorTableAFComputeProvider
from TokenSim.af.providers.queries import AF_MATCH_TYPES
from TokenSim.config.config import KVTransferConfig
from TokenSim.config.simulation import SimulationConfig
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.device import DeviceCatalog
from TokenSim.llm.llm_engine import LLMEngine
from TokenSim.llm.llm_request import Request
from TokenSim.operator_data.lookup import MissingOperatorDataError
from TokenSim.operator_data.package import (
    OperatorDataPackage,
    PackageMeta,
    SourceRecord,
)
from TokenSim.operator_data.workload import elementwise_work, work_for_table
from tests.af.support.config import _cluster, _with_prefill_pool
from tests.af.support.execution import af_test_hardware
from tests.af.support.model import HIDDEN, _component_names, _compute_query, _moe


def ragged_key(lengths="[[2,2],[16,1]]", batch=3):
    return dict(
        attn_dtype="fp16",
        kv_cache_dtype="fp16",
        num_heads=8,
        num_kv_heads=8,
        head_dim=2,
        window_size=4,
        batch_size=batch,
        context_lengths=lengths,
    )


class OperatorTableComputeProviderTest(unittest.TestCase):
    def setUp(self):
        self.hardware = af_test_hardware(moe=_moe())
        self.model = self.hardware.model("TestModel")
        self.provider = OperatorTableAFComputeProvider(self.hardware, self.model)

    def test_stage_decomposition_tp_sharding_and_dense_ffn(self):
        a = self.provider.estimate(_compute_query("a"))
        router = self.provider.estimate(_compute_query("router"))
        expert = self.provider.estimate(_compute_query("expert"))
        self.assertEqual(
            _component_names(a),
            [
                "a_rmsnorm",
                "qkv_projection",
                "rope",
                "attention_core",
                "output_projection",
                "a_residual_add",
            ],
        )
        self.assertEqual(_component_names(router), ["f_rmsnorm", "router"])
        self.assertEqual(_component_names(expert), ["moe_expert", "f_residual_add"])
        for estimate in (a, router, expert):
            total = sum(item["latency_us"] for item in estimate.details["components"])
            self.assertAlmostEqual(estimate.service_duration, total / 1e6)
            self.assertIn(estimate.match_type, AF_MATCH_TYPES)

        tp_a = self.provider.estimate(_compute_query("a", tp_size=2))
        keys = {item["name"]: item["key"] for item in tp_a.details["components"]}
        self.assertEqual(
            (keys["qkv_projection"]["n"], keys["output_projection"]["k"]), (24, 8)
        )
        self.assertEqual(keys["attention_core"]["num_heads"], 4)

        dense_hw = af_test_hardware()
        dense_provider = OperatorTableAFComputeProvider(
            dense_hw, dense_hw.model("TestModel")
        )
        dense_exp = dense_provider.estimate(
            _compute_query("expert", tokens=2, tp_size=2, expert_workload=())
        )
        self.assertEqual(
            _component_names(dense_exp),
            ["dense_gate_up", "dense_swiglu", "dense_down", "f_residual_add"],
        )

    def test_attention_grouping_moe_measured_distribution_and_table_only(self):
        prefill = self.provider.estimate(
            _compute_query("a", phase="prefill", tokens=5, context=(2, 3))
        )
        attn = [
            c for c in prefill.details["components"] if c["name"] == "attention_core"
        ]
        self.assertEqual([c["key"]["input_seq_len"] for c in attn], [2, 3])
        decode = self.provider.estimate(
            _compute_query("a", tokens=3, context=(4, 8, 4))
        )
        d_attn = [
            c for c in decode.details["components"] if c["name"] == "attention_core"
        ]
        self.assertEqual(d_attn[0]["table"], "ragged_generation_attention")

        source = SourceRecord("measured:balanced", "A", "measured")
        package = OperatorDataPackage.from_rows(
            PackageMeta(
                "test-moe", "f-device", "vllm", sources={source.source_id: source}
            ),
            {
                "moe": [
                    dict(
                        dtype="fp16",
                        distribution="balanced",
                        num_tokens=6,
                        hidden_size=HIDDEN,
                        inter_size=32,
                        top_k=2,
                        num_experts=4,
                        tp_size=1,
                        ep_size=2,
                        latency_us=12.0,
                        source_id=source.source_id,
                    )
                ]
            },
        )
        self.hardware.register_package(package)
        moe_res = OperatorTableAFComputeProvider(self.hardware, self.model).estimate(
            _compute_query(
                "expert", tokens=6, ep_size=2, expert_workload=((0, 4), (1, 2))
            )
        )
        self.assertEqual(
            moe_res.details["components"][0]["source_id"], source.source_id
        )

        strict = OperatorTableAFComputeProvider(
            self.hardware, self.model, fallback="table_only"
        )
        with self.assertRaises(ConfigurationError):
            strict.resolver_for("a-device")
        with self.assertRaisesRegex(ValueError, "unsupported AF compute stage"):
            self.provider.estimate(replace(_compute_query("a"), stage="other"))

    def test_ragged_attention_work_and_model_qk_norm(self):
        key = ragged_key()
        actual = work_for_table("ragged_generation_attention", key)
        parts = [
            work_for_table(
                "generation_attention",
                {**key, "context_len": length, "batch_size": count},
            )
            for length, count in ((2, 2), (16, 1))
        ]
        self.assertEqual(actual.flops, sum(p.flops for p in parts))
        self.assertEqual(actual.details["kernel_invocations"], 1)
        for bad in ("[[2,2]]", "[[2,1.5],[16,1.5]]", "[]"):
            with self.assertRaises(ValueError):
                work_for_table("ragged_generation_attention", ragged_key(bad))

        hardware = af_test_hardware()
        model = replace(hardware.model("TestModel"), qk_norm=True)
        names = [
            [
                c["name"]
                for c in OperatorTableAFComputeProvider(
                    hardware, model, fallback="analytical_only", execution_protocol=p
                )
                .estimate(_compute_query())
                .details["components"]
            ]
            for p in ("partial_return", "vllm_afd")
        ]
        self.assertEqual(names[0], names[1])
        self.assertTrue({"q_norm", "k_norm"}.issubset(names[0]))
        with self.assertRaises(ConfigurationError):
            replace(model, qk_norm=1)

        source = SourceRecord("measured:test", "A", "measured")
        hardware.register_package(
            OperatorDataPackage.from_rows(
                PackageMeta(
                    "test", "a-device", "measured", sources={source.source_id: source}
                ),
                {
                    "generation_attention": [
                        {
                            **ragged_key(),
                            "context_len": 8,
                            "latency_us": 10,
                            "source_id": source.source_id,
                        }
                    ]
                },
            )
        )
        resolver = OperatorTableAFComputeProvider(
            hardware, hardware.model("TestModel"), fallback="table_only"
        ).resolver_for("a-device")
        with self.assertRaises(MissingOperatorDataError):
            resolver.query("ragged_generation_attention", ragged_key())


class AFProviderCorrectnessAndConversionTest(unittest.TestCase):
    def setUp(self):
        hardware = af_test_hardware(moe=_moe())
        hardware = replace(
            hardware,
            devices=DeviceCatalog(
                [
                    replace(
                        device,
                        peak_compute={
                            **device.peak_compute,
                            "fp8": device.peak_compute["fp16"],
                        },
                    )
                    for device in hardware.devices
                ]
            ),
        )
        self.provider = OperatorTableAFComputeProvider(
            hardware, hardware.model("TestModel"), fallback="analytical_only"
        )

    def test_moe_stage_ownership_dtype_independence_and_prefill_kv(self):
        router = self.provider.estimate(
            replace(_compute_query("router"), layer_type="moe")
        )
        self.assertEqual(
            [c["name"] for c in router.details["components"]], ["router", "router_topk"]
        )
        routed = self.provider.estimate(
            replace(_compute_query("expert"), stage="routed_expert", layer_type="moe")
        )
        self.assertEqual(
            [c["name"] for c in routed.details["components"]], ["moe_expert"]
        )

        query = replace(
            _compute_query("a", dtype="bf16", context=(8192,)),
            weight_dtype="fp8",
            activation_dtype="bf16",
            kv_cache_dtype="fp32",
        )
        estimate = self.provider.estimate(query)
        smaller_kv = self.provider.estimate(replace(query, kv_cache_dtype="fp16"))
        self.assertGreater(estimate.service_duration, smaller_kv.service_duration)

        pf = replace(
            _compute_query("a", phase="prefill", tokens=8, context=(8,)),
            query_lengths=(8,),
        )
        self.assertGreater(
            self.provider.estimate(
                replace(pf, context_lengths=(8192,))
            ).service_duration,
            self.provider.estimate(pf).service_duration,
        )
        with self.assertRaisesRegex(ValueError, "KV contexts"):
            self.provider.estimate(replace(pf, context_lengths=(4,)))

    def test_wire_cast_work_and_tp_codec_execution(self):
        for source, target, byte_width in (("bf16", "fp8", 3), ("fp8", "fp32", 5)):
            work = elementwise_work(f"cast_{source}_to_{target}", 2, 16, target)
            self.assertEqual(work.activation_bytes, 32 * byte_width)
        with self.assertRaisesRegex(ValueError, "target dtype"):
            elementwise_work("cast_bf16_to_fp8", 1, 16, "bf16")

        cluster = _cluster(a_tp=2, a_dp=1, f_tp=2, f_dp=1, f_ep=1)
        cluster = replace(
            cluster,
            af=replace(
                cluster.af,
                latency_provider="operator_table",
                activation_dtype="bf16",
                links=[
                    replace(
                        cluster.af.links[0], a_to_f_dtype="fp8", f_to_a_dtype="fp32"
                    )
                ],
            ),
        )
        hardware = af_test_hardware(capacity_gib=0.1)
        hardware = replace(
            hardware,
            devices=DeviceCatalog(
                [
                    replace(
                        d,
                        peak_compute={**d.peak_compute, "fp32": d.peak_compute["fp16"]},
                    )
                    for d in hardware.devices
                ]
            ),
        )
        env = simpy.Environment()
        engine = LLMEngine(
            env=env,
            kv_transfer_config=KVTransferConfig.default(),
            model_spec=hardware.model("TestModel"),
            cluster_config=_with_prefill_pool(cluster),
            hardware=hardware,
            config=SimulationConfig(
                trace={"mode": "full"}, block_size=4, latency_fallback="analytical_only"
            ),
        )
        req = Request(1, 8, 4, 4)
        req.generation_idx = 1
        iteration = engine.af_executor.submit(0, [req])
        env.run(until=env.process(engine.af_executor.run_iteration(iteration)))
        codecs = [
            e
            for e in engine.af_executor.trace.events
            if e.stage in {"encode", "decode"}
        ]
        self.assertEqual(
            {e.details["components"][0]["key"]["op_name"] for e in codecs},
            {
                "cast_bf16_to_fp8",
                "cast_fp8_to_bf16",
                "cast_bf16_to_fp32",
                "cast_fp32_to_bf16",
            },
        )
