"""Preserve measured compressed-attention boundary rows in the AF trace."""

from dataclasses import replace
import unittest

import simpy

from TokenSim.af.providers.operator_table import build_operator_table_af_providers
from TokenSim.af.serving.service import AFService
from TokenSim.operator_data.package import OperatorDataPackage, SourceRecord
from TokenSim.trace.host import run_environment
from tests.test_compressed_af import compressed_runtime


class CompressedProvenanceTest(unittest.TestCase):
    def test_b300_one_token_prefill_records_clamped_attention(self):
        runtime = compressed_runtime(device="b300_sxm", topology="gb300_nvl72")
        hardware, model = runtime.hardware, runtime.model
        package = hardware.operator_package("b300_sxm", "vllm")
        # Supply the RMS norm row absent from the bundled package so attention
        # determines both its component match and its enclosing stage match.
        source = SourceRecord("test:rmsnorm", "A", "measured")
        meta = replace(
            package.meta, sources={**package.meta.sources, source.source_id: source}
        )
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
        hardware.register_package(
            replace(package, meta=meta, tables={**package.tables, **norm.tables})
        )
        providers = build_operator_table_af_providers(
            hardware=hardware,
            model=model,
            cluster_config=runtime.cluster,
            worker_configs=runtime.workers,
            placement=runtime.placement,
            dtype=model.activation_dtype,
            operator_backend="vllm",
        )
        env = simpy.Environment()
        service = AFService(env, runtime, providers=providers)
        session = service.admit(0, 0, maximum_context=1, current_context=0)
        env.process(service.forward([session], phase="prefill"))
        run_environment(env)

        csa = next(
            layer.layer_id
            for layer in model.layer_specs()
            if layer.attention_kind == "csa"
        )
        event = next(
            event
            for event in service.trace.events
            if event.stage == "attention" and event.layer_id == csa
        )
        component = next(
            row
            for row in event.details["components"]
            if row["name"] == "compressed_attention"
        )
        self.assertEqual(event.match_type, "clamped")
        self.assertEqual(component["match_type"], "clamped")
        self.assertEqual(component["dataset_version"], package.dataset_version)
        self.assertIn(
            component["source_id"].removeprefix("compose:"), package.meta.sources
        )
