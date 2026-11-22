"""LP30 catalog, unit/workspace accounting, native AFService, and CLI behavior."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from io import StringIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import simpy
import yaml

from TokenSim.af.factory import simulate_decode
from TokenSim.af.runtime.lpx_memory import (
    LP30Memory,
    validate_queries,
    workspace_components,
)
from TokenSim.af.serving.service import AFService
from TokenSim.config.cache_config import stage_layer_range
from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.errors import ConfigurationError
from TokenSim.hardware import DeviceCatalog, HardwareContext
from TokenSim.hardware.device import DeviceSpec
from TokenSim.hardware.placement import build_topology_placement
from TokenSim.operator_data.analytical import analytical_model_for
from TokenSim.operator_data.analytical.lp30 import LP30Pipeline
from scripts.lpx_runtime import Layout, configuration
from scripts.run_lpx import main
from tests.support.lpx import tiny_moe

ROOT = Path(__file__).resolve().parents[1]


class LPXHardwareAndUnitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load(ROOT / "data")

    def estimate(self, table, key, **policy):
        base = self.hardware.device("groq_lp30")
        device = replace(base, analytical={**base.analytical, "lpx_pipeline": policy})
        return analytical_model_for(device.family).estimate(table, key, device)

    def test_lp30_catalog_and_topology_injection_budget(self):
        lpu = self.hardware.device("LP30")
        self.assertEqual(lpu.device_id, "groq_lp30")
        self.assertEqual(self.hardware.device("groq").device_id, "groqchip_v1")
        self.assertEqual(lpu.memory_capacity_bytes.value, 500_000_000)
        self.assertEqual(lpu.peak_compute_for("fp8").value * 256, 315e15)
        aggregate = self.hardware.link("groq_lp30_injection")
        self.assertEqual(aggregate.bandwidth_per_direction_bytes_per_s, 1.25e12)
        cluster = ClusterConfig(
            num_workers=2,
            networks={"net": "ethernet1600Gb"},
            worker_groups=[
                dict(
                    hardware="groq_lp30",
                    num_workers=2,
                    network="net",
                    capabilities=["prefill", "decode_full"],
                )
            ],
        )
        placement = build_topology_placement(cluster, cluster.workers(), self.hardware)
        self.assertEqual(placement.topology.levels[0].links_per_endpoint, 1)

    def test_unit_accounting_mixed_precision_softmax_and_validation(self):
        raw = yaml.safe_load((ROOT / "data/devices/groq_lp30.yaml").read_text())
        raw["device_id"] = "custom-lpx"
        raw["analytical"].update(lpx_pipeline={}, lpx_memory={"prefill_chunk": 4})
        custom = DeviceSpec.from_mapping(raw)
        self.assertEqual(custom.lpx_memory.prefill_chunk, 4)

        key = dict(dtype="fp8", m=3, n=64, k=128)
        estimate = self.estimate("gemm", key)
        self.assertEqual(estimate.work.weight_bytes, 64 * 128)
        self.assertGreater(
            self.estimate("gemm", key, vector_ops_per_s=1e8).latency_us,
            estimate.latency_us,
        )
        pipelined = self.estimate("gemm", dict(dtype="fp8", m=1, n=128, k=256))
        serialized = self.estimate(
            "gemm", dict(dtype="fp8", m=1, n=128, k=256), schedule="serialized"
        )
        self.assertLessEqual(pipelined.latency_us, serialized.latency_us)

        attn_key = dict(
            attn_dtype="bf16",
            kv_cache_dtype="fp8",
            batch_size=1,
            context_len=1000,
            num_heads=2,
            num_kv_heads=1,
            head_dim=64,
            window_size=0,
        )
        short = self.estimate("generation_attention", attn_key)
        capped = self.estimate(
            "generation_attention",
            {**attn_key, "context_len": 2000, "window_size": 1000},
        )
        self.assertEqual(capped.latency_us, short.latency_us)
        softcap = self.estimate(
            "elementwise",
            dict(op_name="logit_softcap", dtype="bf16", num_tokens=2, hidden_size=128),
        )
        self.assertEqual(softcap.details["exponential_operations"], 256)
        for values in (
            {"exp_ops_per_s": 0},
            {"vector_efficiency": 2},
            {"schedule": "unknown"},
        ):
            with self.assertRaises(ConfigurationError):
                LP30Pipeline(**values)
        with self.assertRaises(ConfigurationError):
            LP30Memory(prefill_chunk=17, maximum_context=16)


class NativeLPXRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load(ROOT / "data")
        cls.model = tiny_moe(cls.hardware)
        cls.base = Layout(a_tp=2, f_ep=2)
        cls.lpu = Layout(
            a_device="groq_lp30",
            f_device="groq_lp30",
            a_tp=2,
            a_pp=4,
            f_ep=4,
            f_pp=4,
            colocated=False,
        )

    def test_workspace_reservations_and_bounded_window_admission(self):
        c = SimpleNamespace(
            category="dense_ffn",
            shape=(("hidden_size", 128), ("local_intermediate_size", 256)),
        )
        one = dict(
            workspace_components(
                (c,), self.model, dict(prefill_chunk=1, reserve_bytes=0)
            )
        )
        self.assertEqual(one["lp30_live_scratch"], (3 * 256 + 2 * 128) * 2)
        with self.assertRaises(ConfigurationError):
            validate_queries((LP30Memory(prefill_chunk=4),), 5, 1000)

        device = self.hardware.device("groq_lp30")
        bounded_dev = replace(
            device,
            device_id="bounded-lpx",
            display_name="bounded-lpx",
            aliases=(),
            memory_capacity_bytes=replace(
                device.memory_capacity_bytes, value=2_000_000
            ),
            on_chip_memory_bytes=replace(device.memory_capacity_bytes, value=2_000_000),
        )
        hw = replace(
            self.hardware,
            devices=DeviceCatalog([*self.hardware.devices, bounded_dev]),
        )
        window_model = replace(
            self.model,
            max_position_embeddings=131072,
            attention_windows=(16, 16, 16, 0),
            kv_cache_dtype="fp8",
        )
        window_rt = configuration(
            hw,
            window_model,
            Layout(
                a_device="bounded-lpx",
                f_device="bounded-lpx",
                a_tp=2,
                f_tp=2,
                f_ep=1,
            ),
            namespace="window-admission",
        )
        owner = next(iter(window_rt.plan.request_owners))
        replica = window_rt.plan.replica_for_instance(owner.domain, owner.instance)
        self.assertGreater((100001 + 15) // 16, replica.logical_gpu_blocks)
        executor, rows, _, _ = simulate_decode(
            window_rt, 1, 100000, steps=1, trace=False, fallback="analytical_only"
        )
        self.assertEqual(len(rows), 1)
        executor.trace.close()
        with self.assertRaises(ConfigurationError):
            simulate_decode(
                window_rt, 64, 100000, steps=1, trace=False, fallback="analytical_only"
            )

    def test_colocated_validation_return_routes_and_af_service_prefill(self):
        with self.assertRaisesRegex(ConfigurationError, "hybrid expert TP/EP"):
            configuration(
                self.hardware, self.model, Layout(a_tp=4, f_tp=2, f_ep=2), namespace="x"
            )
        four_layer = replace(
            self.model, num_layers=4, moe=replace(self.model.moe, num_moe_layers=4)
        )
        sparse = configuration(
            self.hardware,
            four_layer,
            Layout(
                a_tp=2, a_pp=4, f_device="groq_lp30", f_ep=4, f_pp=2, colocated=False
            ),
            namespace="sparse",
        )
        for route in sparse.plan.partial_destinations:
            src = sparse.plan.binding(route.source_domain, route.source_rank)
            dst = sparse.plan.binding(route.target_domain, route.target_rank)
            self.assertTrue(
                set(stage_layer_range(4, 2, src.pp_rank)).intersection(
                    stage_layer_range(4, 4, dst.pp_rank)
                )
            )

        runtime = configuration(self.hardware, self.model, self.base, namespace="chunk")
        env = simpy.Environment()
        service = AFService(env, runtime)
        session = service.admit(0, 0, maximum_context=24, current_context=16)
        env.run(
            until=env.process(
                service.forward(
                    [session], phase="prefill", query_tokens=2, emit_logits=False
                )
            )
        )
        self.assertEqual(session.cached_context, 18)
        env.run(
            until=env.process(
                service.forward([session], phase="prefill", query_tokens=2)
            )
        )
        self.assertEqual(session.cached_context, 20)
        split = configuration(self.hardware, self.model, self.lpu, namespace="split")
        other = AFService(env, split)
        split_session = other.admit(1, 0, maximum_context=24, current_context=16)
        with self.assertRaisesRegex(ConfigurationError, "complete-model pool"):
            env.run(
                until=env.process(
                    other.forward([split_session], phase="prefill", query_tokens=2)
                )
            )

    def test_graph_kernel_assumption_and_colocated_expert_tp(self):
        original = self.hardware.device("rubin").analytical["kernel_launch_us"]
        for device, expect_diff in (("rubin", True), ("groq_lp30", False)):
            times = []
            for mode in ("eager", "cuda_graph"):
                rt = configuration(
                    self.hardware,
                    self.model,
                    Layout(a_device=device, f_device=device, a_tp=2, f_ep=2),
                    namespace=f"{device}-{mode}",
                    execution_mode=mode,
                    graph_kernel_us=0.1,
                )
                env = simpy.Environment()
                svc = AFService(env, rt)
                s = svc.admit(0, 0, maximum_context=24, current_context=16)
                env.run(
                    until=env.process(svc.forward([s], phase="prefill", query_tokens=2))
                )
                times.append(env.now)
            if expect_diff:
                self.assertLess(times[1], times[0])
            else:
                self.assertEqual(times[0], times[1])
        self.assertEqual(
            self.hardware.device("rubin").analytical["kernel_launch_us"], original
        )

        pure = configuration(
            self.hardware, self.model, Layout(a_tp=2, f_tp=2, f_ep=1), namespace="pure"
        )
        env = simpy.Environment()
        service = AFService(env, pure)
        session = service.admit(0, 0, maximum_context=24, current_context=16)
        for phase, queries in (("prefill", 2), ("decode", 1)):
            env.run(
                until=env.process(
                    service.forward([session], phase=phase, query_tokens=queries)
                )
            )
        self.assertEqual(session.cached_context, 19)
        service.release(session)
        self.assertFalse(service.memory.leases)

    def test_native_cli_checkpoints_and_invalid_partition_rejection(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            arguments = [
                "--model",
                "moe-toy",
                "--tp",
                "2",
                "--pp",
                "1",
                "--batches",
                "1",
                "2",
                "--context",
                "8",
                "--steps",
                "2",
                "--trace",
                "off",
                "--output",
                str(output),
            ]
            with redirect_stdout(StringIO()):
                self.assertEqual(main(arguments), 0)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(len(summary["results"]), 2)
            with (
                redirect_stdout(StringIO()),
                patch(
                    "TokenSim.af.factory.simulate_decode",
                    side_effect=AssertionError("reran"),
                ),
            ):
                for location in (str(output), os.path.relpath(output)):
                    arguments[-1] = location
                    self.assertEqual(main(arguments), 0)
            absent = output / "absent"
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                main(
                    [
                        "--model",
                        "moe-toy",
                        "--tp",
                        "2",
                        "--pp",
                        "2",
                        "--pipeline-layer-counts",
                        "1",
                        "2",
                        "--output",
                        str(absent),
                    ]
                )
            self.assertFalse(absent.exists())
