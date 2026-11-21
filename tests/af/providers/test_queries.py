"""Compute/transfer query validation and precision contracts."""

from __future__ import annotations

from dataclasses import replace
import unittest

import simpy

from TokenSim.af.execution.executor import AFExecutor
from TokenSim.af.providers.queries import (
    AFComputeQuery,
    AFServiceEstimate,
    AFTensorSlice,
    AFTransferQuery,
)
from TokenSim.dtypes import DTYPE_REGISTRY, canonicalize_dtype, dtype_size_bytes
from TokenSim.errors import ConfigurationError
from tests.af.support.config import _cluster
from tests.af.support.execution import _requests, af_test_resources
from tests.af.support.model import _moe, _runtime_plan


class CanonicalDTypeRegistryTest(unittest.TestCase):
    def test_canonical_names_aliases_and_byte_widths(self):
        cases = {
            "float32": ("fp32", 4),
            "FLOAT16": ("fp16", 2),
            "bfloat16": ("bf16", 2),
            "float8_e4m3fn": ("fp8", 1),
            "float8_e5m2": ("fp8", 1),
            "int8": ("int8", 1),
        }
        for value, (canonical, size_bytes) in cases.items():
            with self.subTest(dtype=value):
                self.assertEqual(canonicalize_dtype(value), canonical)
                self.assertEqual(dtype_size_bytes(value), size_bytes)

        self.assertEqual(
            {name: spec.size_bytes for name, spec in DTYPE_REGISTRY.items()},
            {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1},
        )

    def test_unknown_empty_and_invalid_role_fail_without_a_default(self):
        for value in ("", "mystery"):
            with self.subTest(dtype=value):
                with self.assertRaisesRegex(ValueError, "activation_dtype"):
                    dtype_size_bytes(value, role="activation_dtype")
        with self.assertRaisesRegex(ValueError, "unknown dtype role"):
            canonicalize_dtype("bf16", role="optimizer_dtype")


class AFLatencyQueryContractTest(unittest.TestCase):
    def test_legacy_query_construction_keeps_compatible_defaults(self):
        compute = AFComputeQuery(
            iteration_id=1,
            block_id=2,
            layer_id=3,
            phase="decode",
            stage="a",
            token_count=4,
            context_lengths=(8,),
            model_id="model",
            dtype="bf16",
            device_type="a-device",
            worker_id=0,
        )
        transfer = AFTransferQuery(
            iteration_id=1,
            block_id=2,
            layer_id=3,
            phase="decode",
            direction="a_to_f",
            token_count=4,
            hidden_size=16,
            dtype="bf16",
            source_device="a-device",
            target_device="f-device",
            network="af-net",
        )

        self.assertIsNone(compute.layer_type)
        self.assertIsNone(compute.activation_dtype)
        self.assertEqual(compute.tensor_slices, ())
        self.assertIsNone(transfer.communication_kind)
        self.assertIsNone(transfer.logical_stage_id)
        self.assertEqual(transfer.logical_input_bytes_per_rank, ())

    def test_tensor_slice_rejects_an_invalid_global_interval(self):
        with self.assertRaisesRegex(ValueError, "half-open"):
            AFTensorSlice("tensor", (4, 4), "sharded")


class RecordingComputeProvider:
    def __init__(self, *, rejection: str | None = None) -> None:
        self.rejection = rejection
        self.validated = []
        self.queries = []

    def validate_effective_dtypes(self, model_id, effective_dtypes) -> None:
        self.validated.append((model_id, effective_dtypes))
        if self.rejection is not None:
            raise ValueError(self.rejection)

    def estimate(self, query) -> AFServiceEstimate:
        self.queries.append(query)
        return AFServiceEstimate(0.0, "dtype-contract:compute")


class RecordingTransferProvider:
    def __init__(self) -> None:
        self.queries = []

    def estimate(self, query) -> AFServiceEstimate:
        self.queries.append(query)
        return AFServiceEstimate(
            0.0,
            "dtype-contract:transfer",
            payload_bytes=query.payload_bytes,
        )


def _typed_runtime():
    cluster = _cluster(a_tp=1, a_dp=1, f_tp=1, f_dp=1, f_ep=1)
    cluster.af = replace(
        cluster.af,
        weight_dtype="fp8",
        activation_dtype="bf16",
        kv_cache_dtype="fp32",
    )
    cluster.af.links = [
        replace(
            cluster.af.links[0],
            a_to_f_dtype="fp8",
            f_to_a_dtype="fp32",
        )
    ]
    moe_config = _moe(top_k=1, shared=1)
    plan = _runtime_plan(cluster, moe_config=moe_config, dtype="fp16")
    return cluster, moe_config, plan


def _executor(env, cluster, moe_config, plan, compute, transfer):
    return AFExecutor(
        env=env,
        cluster_config=cluster,
        worker_configs=cluster.workers(),
        compute_provider=compute,
        transfer_provider=transfer,
        model_id="TestModel",
        # This compatibility input intentionally differs from the runtime plan.
        dtype="fp16",
        total_layers=plan.total_layers,
        hidden_size=plan.hidden_size,
        moe_config=moe_config,
        runtime_plan=plan,
        resources=af_test_resources(env, cluster, cluster.workers()),
    )


class AFExecutorDTypeContractTest(unittest.TestCase):
    def test_effective_roles_cross_executor_provider_boundary(self):
        env = simpy.Environment()
        cluster, moe_config, plan = _typed_runtime()
        compute = RecordingComputeProvider()
        transfer = RecordingTransferProvider()
        executor = _executor(env, cluster, moe_config, plan, compute, transfer)
        iteration = executor.submit(0, _requests(env, 1))
        env.run(until=env.process(executor.run_iteration(iteration)))

        model_id, validated = compute.validated[0]
        self.assertEqual(model_id, "TestModel")
        expected_roles = {
            "weight_dtype": "fp8",
            "activation_dtype": "bf16",
            "kv_cache_dtype": "fp32",
            "a_to_f_dtype": "fp8",
            "f_to_a_dtype": "fp32",
        }
        self.assertEqual(
            {role: getattr(validated, role) for role in expected_roles}, expected_roles
        )
        queries = [query for query in compute.queries if query.stage == "attention"]
        self.assertTrue(queries)
        for query in queries:
            self.assertEqual(query.dtype, "bf16")
            self.assertEqual(
                {role: getattr(query, role) for role in expected_roles}, expected_roles
            )

        for direction, dtype in (("a_to_f", "fp8"), ("f_to_a", "fp32")):
            wire = [
                query
                for query in transfer.queries
                if query.operation == f"af_{direction}"
            ]
            self.assertEqual({query.dtype for query in wire}, {dtype})
            self.assertEqual({query.activation_dtype for query in wire}, {"bf16"})

    def test_manifest_mismatch_fails_before_event_loop_or_query(self):
        env = simpy.Environment()
        cluster, moe_config, plan = _typed_runtime()
        compute = RecordingComputeProvider(
            rejection=(
                "operator manifest dtype mismatch: model='TestModel', "
                "role='activation_dtype', effective='bf16', manifest='fp16'"
            )
        )
        transfer = RecordingTransferProvider()

        with self.assertRaisesRegex(
            ConfigurationError,
            "provider dtype contract validation failed.*activation_dtype",
        ):
            _executor(env, cluster, moe_config, plan, compute, transfer)

        self.assertEqual(env.now, 0)
        self.assertEqual(compute.queries, [])
        self.assertEqual(transfer.queries, [])


if __name__ == "__main__":
    unittest.main()
