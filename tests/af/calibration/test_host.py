"""Host overhead profiles, native call groups, and deterministic routing replay."""

import copy
from dataclasses import asdict, replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import simpy
from pydantic import TypeAdapter

from TokenSim.af.calibration.host import (
    COMPUTE_CALL_FIELDS,
    EXTENDED_HOST_COVERAGE,
    HOST_COVERAGE,
    MESSAGE_CALL_FIELDS,
)
from TokenSim.af.execution.model import ModelAFExecutor
from TokenSim.af.factory import build_model_runtime, simulate_decode
from TokenSim.af.providers.operator_table import OperatorTableAFComputeProvider
from TokenSim.af.providers.queries import AFServiceEstimate
from TokenSim.config.config import ClusterConfig, KVTransferConfig
from TokenSim.config.simulation import SimulationConfig
from TokenSim.errors import ConfigurationError, WorkloadValidationError
from TokenSim.hardware.device.sourced_value import SourcedValue
from TokenSim.llm.llm_engine import LLMEngine
from tests.af.support.config import _with_prefill_pool
from tests.af.support.profiles import HostProfileFixture, OperationProfileFixture


class HostProfileTest(HostProfileFixture, unittest.TestCase):
    def test_domain_placement_and_explicit_conflicts_invalidate_profile(self):
        original = self.runtime()
        doc = copy.deepcopy(original.doc)
        doc["af"]["transfer_topology_mode"] = "physical"
        doc["af"]["communication_domains"] = dict(
            node_size=8,
            node_count=2,
            placement_strategy="manual",
            device_indices={"a-pool:0": 2, "f-pool:0": 3},
        )
        base = build_model_runtime(
            doc, hardware=original.hardware, model=original.model
        )
        profile = self.document(base)
        for changes in (
            {"local_bandwidth_bytes_per_s": 150e9},
            {"local_latency_us": 2},
            {"device_indices": {"a-pool:0": 2, "f-pool:0": 8}},
        ):
            altered = copy.deepcopy(doc)
            altered["af"]["communication_domains"].update(changes)
            runtime = build_model_runtime(
                altered, hardware=original.hardware, model=original.model
            )
            with self.assertRaisesRegex(ConfigurationError, "fingerprint mismatch"):
                self.load(profile, runtime)
        conflict = self.runtime({"host_submit_us": 5})
        with self.assertRaisesRegex(ConfigurationError, "duplicate"):
            self.load(
                self.document(conflict),
                conflict,
                conflict.cluster.af.communication_timing._explicit_cost_fields,
            )

    def test_measured_cost_origin_scope_validation_and_bounds(self):
        runtime = self.runtime()
        doc = copy.deepcopy(runtime.doc)
        profile = self.document(runtime)
        profile["evidence"]["kind"] = "measured"
        profile["costs"] = {
            name: dict(fixed_us=value, per_token_us=0, coverage=[HOST_COVERAGE[name]])
            for name, value in [("control_encode_us", 7), ("control_decode_us", 11)]
        }
        profile["evidence"]["quantiles_us"] = {
            name: dict(p50=entry["fixed_us"], p95=entry["fixed_us"])
            for name, entry in profile["costs"].items()
        }
        doc["af"].update(
            engine_overhead_profile=profile,
            calibration_environment=profile["environment"],
        )
        configured = build_model_runtime(
            doc, hardware=runtime.hardware, model=runtime.model
        )
        ex, _, _, _ = simulate_decode(
            configured, 1, 129, steps=3, fallback="analytical_only"
        )
        for name, expected in [("control_encode_us", 7), ("control_decode_us", 11)]:
            charged = [
                e for e in ex.trace.events if e.details.get("cost_field") == name
            ]
            self.assertEqual(len(charged), 3)
            self.assertTrue(
                all(abs(e.service_duration * 1e6 - expected) < 1e-9 for e in charged)
            )
            self.assertEqual(
                ex.host_cost_provenance(name, tokens=1)["origin"], "measured_profile"
            )
        self.assertEqual(
            ex.host_cost_provenance("a_step_prepare_us", tokens=1)["origin"],
            "unmeasured_default",
        )
        original = self.document(runtime)
        for edit in (
            lambda d: d.update(contains_wait=True),
            lambda d: d.update(excludes_device_launch=False),
            lambda d: d.update(timing_scope="inclusive"),
            lambda d: d["environment"].update(software_sha256="changed"),
            lambda d: d.update(runtime_fingerprint="changed"),
            lambda d: d["costs"]["host_submit_us"].update(coverage=["a_input_prepare"]),
            lambda d: d["costs"]["host_submit_us"].update(fixed_us=-1),
            lambda d: d["costs"]["host_submit_us"].update(fixed_us=True),
            lambda d: d["evidence"].update(window_count=1),
            lambda d: d["evidence"].update(heldout_batches=[8]),
        ):
            bad = copy.deepcopy(original)
            edit(bad)
            with self.assertRaises(ConfigurationError):
                self.load(bad)
        loaded = self.load(original)
        for requests, phase in (
            ([SimpleNamespace(context_len=128)], "decode"),
            ([SimpleNamespace(context_len=129)] * 9, "decode"),
            ([SimpleNamespace(context_len=129)], "prefill"),
        ):
            with self.assertRaises(ConfigurationError):
                loaded.validate_workload(requests, phase)

    def test_standard_engine_boundaries_and_protocol_buffer_reservation(self):
        runtime = self.runtime()
        states = {s.physical_identity: s for s in runtime.plan.physical_states}
        a = states["a-pool:0"]
        self.assertEqual(dict(a.protocol_buffers)["a_return_hidden"], 8 * 256 * 2)
        original_device = runtime.hardware.device
        with patch.object(
            runtime.hardware,
            "device",
            side_effect=lambda name: replace(
                original_device(name),
                memory_capacity_bytes=SourcedValue(
                    a.model_bytes + a.protocol_buffer_bytes - 1, "test", "D"
                ),
                memory_reserved_bytes=SourcedValue(0, "test", "D"),
            ),
        ):
            with self.assertRaisesRegex(ConfigurationError, "required_bytes"):
                build_model_runtime(
                    runtime.doc, hardware=runtime.hardware, model=runtime.model
                )
        raw = TypeAdapter(ClusterConfig).dump_python(
            _with_prefill_pool(runtime.cluster), mode="json", exclude_none=True
        )
        runtime = build_model_runtime(
            raw, hardware=runtime.hardware, model=runtime.model
        )
        profile = self.document(runtime)
        doc = copy.deepcopy(runtime.doc)
        doc["af"].update(
            engine_overhead_profile=profile,
            calibration_environment=profile["environment"],
            communication_timing=dict(enabled=True, native_control_enabled=True),
        )
        env = simpy.Environment()
        with patch.object(runtime.hardware, "model", return_value=runtime.model):
            engine = LLMEngine(
                env=env,
                kv_transfer_config=KVTransferConfig.default(),
                model_spec=runtime.model,
                cluster_config=_with_prefill_pool(ClusterConfig(**doc)),
                hardware=runtime.hardware,
                config=SimulationConfig(
                    trace={"mode": "full"},
                    block_size=16,
                    batching="paged-attn",
                    max_num_seqs=8,
                    latency_fallback="analytical_only",
                ),
            )
        ex = engine.af_executor
        self.assertIsInstance(ex, ModelAFExecutor)
        iteration = ex.submit(
            0,
            [
                SimpleNamespace(
                    id=0,
                    is_prefill=False,
                    prefill_len=128,
                    context_len=129,
                    generation_idx=0,
                )
            ],
        )
        env.run(until=env.process(ex.run_iteration(iteration)))
        ex.trace.validate_invariants(iteration.iteration_id)

    def test_measured_compute_scope_and_fused_combine_coverage(self):
        provider = OperatorTableAFComputeProvider(
            self.runtime().hardware,
            self.fixture.model,
            fallback="analytical_only",
            require_device_scope=True,
        )
        estimate = SimpleNamespace(
            latency_us=1,
            match_type="exact",
            source_id="fixture",
            dataset_version="test",
        )
        metadata = {}
        resolver = SimpleNamespace(
            query=lambda *args, **kwargs: estimate,
            package=SimpleNamespace(meta=SimpleNamespace(extra=metadata)),
            dataset_version="test",
        )
        metadata["timing_scopes"] = {"routed_experts": "inclusive"}
        with self.assertRaises(ConfigurationError):
            provider._component(resolver, "experts", "routed_experts", {})
        metadata["timing_scopes"]["routed_experts"] = "device_service"
        metadata["operation_coverage"] = {
            "routed_experts": ("routed_experts", "weighted_combine")
        }
        self.assertEqual(
            provider._component(
                resolver, "experts", "routed_experts", {}
            ).covered_operations,
            ("routed_experts", "weighted_combine"),
        )
        runtime = self.runtime({"compute_submit_us": 7})
        ex, _, _, _ = simulate_decode(runtime, 1, 129, fallback="analytical_only")
        original = ex.compute_provider.estimate

        def fused(query):
            if query.stage == "routed_expert":
                return AFServiceEstimate(
                    10e-6,
                    "fused-oracle",
                    details={
                        "covered_operations": ("routed_experts", "weighted_combine"),
                        "timing_scope": "device_service",
                    },
                )
            if query.stage == "weighted_combine":
                raise AssertionError("combine must not be estimated again")
            return original(query)

        with patch.object(ex.compute_provider, "estimate", fused):
            iteration = ex.submit(
                0,
                [
                    SimpleNamespace(
                        id=0,
                        is_prefill=False,
                        context_len=129,
                        prefill_len=128,
                        generation_idx=0,
                    )
                ],
            )
            ex.env.run(until=ex.env.process(ex.run_iteration(iteration)))
        combined = [
            e
            for e in ex.trace.events_for(iteration.iteration_id)
            if e.stage == "weighted_combine"
        ]
        self.assertTrue(all(e.service_duration == 0 for e in combined))
        ex.trace.validate_invariants(iteration.iteration_id)


class NativeCallAndRoutingReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OperationProfileFixture.setUpClass()
        cls.fixture = OperationProfileFixture()

    def profile(self, fields):
        doc = self.fixture.host.document(self.fixture.host.runtime())
        doc["schema_version"] = 2
        doc["evidence"]["heldout_batches"] = [2]
        doc["costs"] = {
            name: dict(
                fixed_us=value, per_token_us=0, coverage=[EXTENDED_HOST_COVERAGE[name]]
            )
            for name, value in fields.items()
        }
        doc["evidence"]["quantiles_us"] = {
            name: dict(p50=value, p95=value) for name, value in fields.items()
        }
        return doc

    def run_profile(self, fields, steps=2, **options):
        runtime = self.fixture.configured(
            self.fixture.sampled_profile(),
            engine_overhead_profile=self.profile(fields),
            **options,
        )
        return simulate_decode(
            runtime,
            1,
            129,
            steps=steps,
            fallback="analytical_only",
            trace=options.get("trace_enabled", True),
        )

    def test_native_call_groups_ordering_and_duplicate_rejection(self):
        fields = {name: 7 for name in COMPUTE_CALL_FIELDS.values()}
        fields.update({name: 3 for name in MESSAGE_CALL_FIELDS.values()})
        ex, _, _, _ = self.run_profile(fields)
        counts = {}
        for e in ex.trace.events:
            field = e.details.get("cost_field")
            if field in fields:
                counts[field] = counts.get(field, 0) + 1
        for name in MESSAGE_CALL_FIELDS.values():
            self.assertEqual(counts.get(name, 0), 4)
        book_fields = {
            "compute_f_moe_block_us": 23,
            "engine_step_schedule_us": 13,
            "engine_runner_bookkeeping_us": 19,
            "engine_output_process_us": 17,
        }
        ex, rows, _, _ = self.run_profile(book_fields)
        _, quiet, _, _ = self.run_profile(
            book_fields,
            trace_enabled=False,
            service_estimate_cache=False,
            dag_template_cache=False,
        )
        self.assertEqual(rows, quiet)
        events = ex.trace.events_for(0)
        readback = next(e for e in events if e.operation == "sampled_token_readback")
        book = next(
            e
            for e in events
            if e.details.get("cost_field") == "engine_runner_bookkeeping_us"
        )
        output = next(
            e
            for e in events
            if e.details.get("cost_field") == "engine_output_process_us"
        )
        self.assertGreaterEqual(book.start_at, readback.end_at)
        self.assertGreaterEqual(output.start_at, book.end_at)
        with self.assertRaisesRegex(ConfigurationError, "duplicate"):
            self.fixture.host.load(
                self.profile({"compute_f_moe_block_us": 23, "compute_f_router_us": 7})
            )

    def test_routing_replay_assignments_and_validation(self):
        runtime = self.fixture.host.runtime()
        profile = self.fixture.host.document(runtime)
        replay = dict(
            schema_version=1,
            runtime_fingerprint=profile["runtime_fingerprint"],
            environment=profile["environment"],
            source="unit-test://routing",
            kind="synthetic_test",
            assignments=[
                dict(
                    phase="decode",
                    decode_step=step,
                    layer_id=layer,
                    request_id=0,
                    token_id=128 + step,
                    top_k_index=k,
                    expert_id=(layer + k) % 4,
                    routing_weight=0.25 if k == 0 else 0.75,
                    a_dp_rank=0,
                    a_tp_rank=0,
                )
                for step in range(2)
                for layer in range(2)
                for k in range(2)
            ],
        )
        doc = copy.deepcopy(runtime.doc)
        doc["af"].update(
            routing_replay=replay, calibration_environment=profile["environment"]
        )
        configured = build_model_runtime(
            doc, hardware=runtime.hardware, model=runtime.model
        )
        executor, rows, _, _ = simulate_decode(
            configured, 1, 129, steps=2, fallback="analytical_only"
        )
        iteration = executor.submit(
            0,
            [
                SimpleNamespace(
                    id=0, is_prefill=False, context_len=129, generation_idx=0
                )
            ],
        )
        records = [
            {k: v for k, v in asdict(a).items() if k != "iteration_id"}
            for snap in iteration.blocks[0].routing_snapshots
            for a in snap.assignments
        ]
        self.assertEqual(records, replay["assignments"][:4])
        for edit in (
            lambda r: r["assignments"].pop(),
            lambda r: r["assignments"][0].update(expert_id=99),
            lambda r: r.update(runtime_fingerprint="wrong"),
        ):
            bad = copy.deepcopy(doc)
            edit(bad["af"]["routing_replay"])
            with self.assertRaises((ConfigurationError, WorkloadValidationError)):
                simulate_decode(
                    build_model_runtime(
                        bad, hardware=runtime.hardware, model=runtime.model
                    ),
                    1,
                    129,
                    steps=2,
                    fallback="analytical_only",
                )
