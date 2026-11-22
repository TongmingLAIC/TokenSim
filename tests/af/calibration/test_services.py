"""Operation service and transport service calibration profile contracts."""

import copy
from dataclasses import asdict
from types import SimpleNamespace
import unittest

import simpy

from TokenSim.af.calibration.services import TransportServiceProfile
from TokenSim.af.factory import build_model_runtime, simulate_decode
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.domains import CommunicationDomainConfig, MODEL_VERSION
from tests.af.support.profiles import NativeFixture, OperationProfileFixture


class OperationProfileTest(OperationProfileFixture, unittest.TestCase):
    def test_rejects_invalid_scope_evidence_and_duplicate_costs(self):
        original = self.document(self.host.runtime())
        for edit in (
            lambda d: d.update(timing_scope="host_service"),
            lambda d: d.update(runtime_fingerprint="wrong"),
            lambda d: d.update(environment={}),
            lambda d: d["operations"]["h2d"].update(coverage=["host_submit"]),
            lambda d: d["operations"]["h2d"].update(fixed_us=-1),
            lambda d: d["operations"]["h2d"].update(fixed_us=True),
            lambda d: d["operations"]["h2d"].update(completion="api_enqueue"),
            lambda d: d["evidence"].update(external_wait_excluded=False),
            lambda d: d["evidence"].update(heldout_batches=None),
        ):
            doc = copy.deepcopy(original)
            edit(doc)
            with self.assertRaises(ConfigurationError):
                self.load(doc)
        for cost in ("control_h2d_us", "control_length_readback_us"):
            with self.assertRaisesRegex(ConfigurationError, "duplicate"):
                self.load(original, (cost,))
        v1 = self.sampled_profile()
        v1["schema_version"] = 1
        with self.assertRaises(ConfigurationError):
            self.load(v1)
        for count, context, phase in (
            (9, 129, "decode"),
            (1, 128, "decode"),
            (1, 129, "prefill"),
        ):
            with self.assertRaises(ConfigurationError):
                self.load(original).validate_workload(
                    [SimpleNamespace(context_len=context)] * count, phase
                )

    def test_sampled_readback_and_codec_coexistence(self):
        baseline = self.document(self.host.runtime())
        _, before, _, _ = simulate_decode(
            self.configured(baseline), 1, 129, steps=3, fallback="analytical_only"
        )
        ex, after, _, _ = simulate_decode(
            self.configured(self.sampled_profile()),
            1,
            129,
            steps=3,
            fallback="analytical_only",
        )
        for old, new in zip(before, after):
            self.assertAlmostEqual(new["makespan_us"] - old["makespan_us"], 11)
        outputs = [
            e for e in ex.trace.events if e.operation == "sampled_token_readback"
        ]
        self.assertEqual(len(outputs), 3)
        codec = self.host.document(self.host.runtime())
        ex_codec, _, _, _ = simulate_decode(
            self.configured(engine_overhead_profile=codec),
            1,
            129,
            steps=2,
            fallback="analytical_only",
        )
        self.assertFalse(
            any(e.event_type == "control_device" for e in ex_codec.trace.events)
        )
        self.assertEqual(
            len(
                [
                    e
                    for e in ex_codec.trace.events
                    if e.event_type == "operation_service"
                ]
            ),
            6,
        )

    def test_calling_thread_serialization_and_cancellation_recovery(self):
        env, ex, block, node = self.oracle()
        identity = node.physical_workers[1]
        for name in ("length_readback", "payload_readback"):
            env.process(
                ex._control_stage(block, node, name, (), identity, 0, device=True)
            )
        env.run(until=1e-6)
        slots = ex.resources.host_resources_for((identity,))
        self.assertEqual(sum(len(r.users) for _, r in slots), 1)
        self.assertEqual(sum(len(r.queue) for _, r in slots), 1)
        env.run()
        self.assertAlmostEqual(env.now, 22e-6)
        runtime = self.configured(self.input_profile())
        ex2, _, _, _ = simulate_decode(runtime, 1, 129, fallback="analytical_only")
        request = SimpleNamespace(
            id=0, is_prefill=False, prefill_len=128, context_len=129, generation_idx=0
        )
        process = ex2.env.process(ex2.run_iteration(ex2.submit(0, [request])))
        ex2.env.run(until=ex2.env.now + 1e-6)
        process.interrupt("cancel input prepare")
        with self.assertRaises(simpy.Interrupt):
            ex2.env.run(until=process)
        ex2.env.run()
        self.assertFalse(ex2._step_preparation)
        self.fixed.run_step(ex2.env, ex2)

    def test_input_and_metadata_preparation_ordering(self):
        profile = self.input_profile()
        for name, value in (("a_attention_metadata", 17), ("f_dp_metadata", 13)):
            profile["operations"][name] = dict(
                fixed_us=value,
                coverage=[name],
                input_ready=(
                    "prior_device_work_complete"
                    if name.startswith("a_")
                    else "control_decoded_cpu_metadata_ready"
                ),
                completion=(
                    "device_synchronize_after_operation"
                    if name.startswith("a_")
                    else "cpu_api_return_data_ready"
                ),
                resource_model="calling_thread_single_inflight",
            )
            profile["evidence"]["statistics"][name] = dict(
                p50_us=value, p95_us=value, samples=200
            )
        ex, _, _, _ = simulate_decode(
            self.configured(profile), 1, 129, fallback="analytical_only"
        )
        events = ex.trace.events
        by_op = {e.operation: e for e in events if e.event_type == "operation_service"}
        self.assertGreaterEqual(
            by_op["a_attention_metadata"].start_at, by_op["a_input_prepare"].end_at
        )
        decoded = next(
            e for e in events if e.details.get("cost_field") == "control_decode_us"
        )
        self.assertGreaterEqual(by_op["f_dp_metadata"].start_at, decoded.end_at)
        ex.trace.validate_invariants(0)


class TransportProfileTest(unittest.TestCase):
    def document(self):
        return dict(
            schema_version=1,
            unit="us",
            profile_id="synthetic-transport",
            timing_scope="transport_service",
            contains_wait=False,
            contains_cpu_submit=False,
            model_version=MODEL_VERSION,
            environment={"gpu_pair": "A,F"},
            domains=asdict(CommunicationDomainConfig()),
            device_indices={"a": 2, "f": 3},
            evidence=dict(
                kind="synthetic_test",
                source="unit test",
                window_count=3,
                sample_count=150,
                measurement_boundary="after_both_device_kernels_started_until_both_complete",
                peer_delay_audit_passed=True,
                limitations=["synthetic data"],
            ),
            records=[
                dict(
                    source_device=2,
                    target_device=3,
                    backend="pynccl",
                    dtype="bf16",
                    points=[
                        dict(bytes=512, service_us=8),
                        dict(bytes=4096, service_us=10),
                    ],
                    interpolation="linear_within_training_range",
                )
            ],
        )

    def load(self, d):
        return TransportServiceProfile(
            d,
            environment={"gpu_pair": "A,F"},
            domains=CommunicationDomainConfig(),
            device_indices={"a": 2, "f": 3},
        )

    def test_runtime_prices_each_message_once_and_cache_preserves_timeline(self):
        NativeFixture.setUpClass()
        fixture = NativeFixture()
        domains = CommunicationDomainConfig(
            node_count=1, device_indices={"a-pool:0": 2, "f-pool:0": 3}
        )
        profile = self.document()
        profile["domains"] = asdict(domains)
        profile["device_indices"] = {"a-pool:0": 2, "f-pool:0": 3}
        profile["records"].append(
            {
                **copy.deepcopy(profile["records"][0]),
                "source_device": 3,
                "target_device": 2,
            }
        )
        for dtype, size in [("int64", 8), ("uint8", 179)]:
            profile["records"].append(
                dict(
                    source_device=2,
                    target_device=3,
                    backend="torch_distributed_nccl",
                    dtype=dtype,
                    points=[dict(bytes=size, service_us=11)],
                    interpolation="linear_within_training_range",
                )
            )
        times = []
        for cached in (False, True):
            doc = fixture.document(
                communication_timing=dict(enabled=True, native_control_enabled=True),
                communication_domains=asdict(domains),
                transfer_topology_mode="physical",
                calibration_environment={"gpu_pair": "A,F"},
                transport_service_profile=profile,
                service_estimate_cache=cached,
            )
            runtime = build_model_runtime(
                doc, hardware=fixture.base.hardware, model=fixture.model
            )
            executor, rows, _, _ = simulate_decode(
                runtime, 1, 129, steps=2, fallback="analytical_only"
            )
            wires = [
                e
                for e in executor.trace.events
                if e.event_type in ("transfer", "control_transfer")
            ]
            self.assertEqual(len(wires), 12)
            times.append(rows)
        self.assertEqual(times[0], times[1])

    def test_bounded_interpolation_and_invalid_samples(self):
        p = self.load(self.document())
        query = dict(source=2, target=3, backend="pynccl", dtype="bf16", size=2304)
        self.assertEqual(p.estimate_us(**query), 9)
        for patch in (
            {"size": 4097},
            {"backend": "torch_distributed_nccl"},
            {"dtype": "fp32"},
        ):
            with self.assertRaises(ConfigurationError):
                p.estimate_us(**{**query, **patch})
        for patch in (
            {"contains_wait": True},
            {"contains_cpu_submit": True},
            {"device_indices": {"a": 2, "f": 4}},
        ):
            with self.assertRaises(ConfigurationError):
                self.load({**self.document(), **patch})
        for value in (True, -1, float("nan"), float("inf")):
            bad = self.document()
            bad["records"][0]["points"][0]["service_us"] = value
            with self.assertRaises(ConfigurationError):
                self.load(bad)
