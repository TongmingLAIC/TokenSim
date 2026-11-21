"""A/F measurements follow the physical endpoint's selected operator package."""

from dataclasses import replace
import unittest

from TokenSim.af.providers.operator_table import build_operator_table_af_providers
from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.hardware.placement import build_topology_placement
from TokenSim.operator_data.package import (
    OperatorDataPackage,
    PackageMeta,
    SourceRecord,
)
from tests.af.communication.test_contract import measured_record
from tests.af.support.config import _cluster_dict, _with_prefill_pool
from tests.af.support.execution import af_test_hardware
from tests.af.support.model import HIDDEN, _compute_query, _transfer_query


class AFEndpointBackendTest(unittest.TestCase):
    def test_compute_and_transport_respect_per_worker_backend(self):
        values = _cluster_dict()
        values["worker_groups"][0]["operator_backend"] = "vllm"
        values["worker_groups"][1].update(
            hardware="a-device", operator_backend="trtllm"
        )
        cluster = _with_prefill_pool(ClusterConfig(**values))
        cluster.worker_groups[-1].operator_backend = "sglang"
        hardware = af_test_hardware()
        workers = cluster.workers()
        arguments = dict(
            hardware=hardware,
            model=hardware.model("TestModel"),
            cluster_config=cluster,
            worker_configs=workers,
            placement=build_topology_placement(cluster, workers, hardware),
        )
        query = replace(_transfer_query(), target_device="a-device")
        _, analytical = build_operator_table_af_providers(
            **arguments, fallback="analytical_only"
        )
        transfer_source = "test:mixed-endpoints"
        measurement = measured_record(
            analytical,
            query,
            participant_hardware_backends=(
                ("a-device", "vllm"),
                ("a-device", "trtllm"),
            ),
            source_id=transfer_source,
        )
        for backend, cost_us in (("vllm", 101), ("trtllm", 123), ("sglang", 145)):
            source_id = f"test:{backend}"
            metadata = PackageMeta(
                f"test-{backend}",
                "a-device",
                backend,
                sources={source_id: SourceRecord(source_id, "A", source_id)},
                extra=(
                    {"communication_phase_services": [measurement]}
                    if backend == "trtllm"
                    else {}
                ),
            )
            package = OperatorDataPackage.from_rows(
                metadata,
                {
                    "gemm": [
                        dict(
                            dtype="fp16",
                            m=2,
                            n=3 * HIDDEN,
                            k=HIDDEN,
                            latency_us=cost_us,
                            source_id=source_id,
                        )
                    ]
                },
            )
            hardware.register_package(package)

        compute, transfer = build_operator_table_af_providers(**arguments)
        compute_query = _compute_query()
        for physical_worker, worker_id, backend, cost_us in (
            ("a-pool:0", 0, "vllm", 101),
            ("f-pool:0", 1, "trtllm", 123),
            ("a-pool:0", 0, "vllm", 101),
        ):
            with self.subTest(physical_worker=physical_worker, backend=backend):
                estimate = compute.estimate(
                    replace(
                        compute_query,
                        physical_worker=physical_worker,
                        worker_id=worker_id,
                    )
                )
                projection = next(
                    item
                    for item in estimate.details["components"]
                    if item["name"] == "qkv_projection"
                )
                self.assertEqual(projection["latency_us"], cost_us)
                self.assertEqual(projection["source_id"], f"test:{backend}")
        self.assertEqual(
            {
                device["operator_backend"]
                for device in compute.describe()["devices"].values()
            },
            {"vllm", "trtllm"},
        )
        self.assertEqual(transfer.estimate(query).source_id, transfer_source)
        phase = transfer.communication_plan(query).phases[0]
        self.assertEqual(
            transfer.phase_estimate(phase).details["participant_hardware_backends"],
            measurement["participant_hardware_backends"],
        )
