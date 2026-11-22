"""AF transfer provider, collective algorithms, endpoint backends, and cache."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import unittest

import simpy

from TokenSim.af.factory import build_model_runtime
from TokenSim.af.providers.operator_table import (
    OperatorTableAFComputeProvider,
    TopologyAFTransferProvider,
    build_operator_table_af_providers,
)
from TokenSim.af.providers.queries import AF_MATCH_TYPES
from TokenSim.af.serving.service import AFService
from TokenSim.comm.flat_collectives import flat_collective
from TokenSim.config.af.communication import AFCommunicationTimingConfig
from TokenSim.config.af.domains import DomainCollectiveConfig
from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.errors import ConfigurationError
from TokenSim.hardware import HardwareContext
from TokenSim.hardware.links import LinkClass
from TokenSim.hardware.placement import build_topology_placement
from TokenSim.hardware.topology import TopologyLevel, TopologyPlacement, TopologySpec
from TokenSim.operator_data.package import (
    OperatorDataPackage,
    PackageMeta,
    SourceRecord,
)
from scripts.lpx_runtime import Layout, configuration
from tests.af.support.config import _cluster_dict, _with_prefill_pool
from tests.af.support.execution import af_test_hardware
from tests.af.support.model import HIDDEN, _compute_query, _moe, _transfer_query
from tests.af.support.pipeline import cluster_document
from tests.support.lpx import tiny_moe


class TopologyTransferProviderTest(unittest.TestCase):
    def setUp(self):
        self.cluster = ClusterConfig(
            **_cluster_dict(attention_workers=2, ffn_workers=2)
        )
        self.hardware = af_test_hardware()
        self.workers = self.cluster.workers()
        self.placement = build_topology_placement(
            self.cluster, self.workers, self.hardware
        )
        self.provider = TopologyAFTransferProvider(
            self.hardware, self.cluster, self.workers, self.placement, dtype="fp16"
        )

    def test_cross_domain_colocated_flat_loopback_and_collectives(self):
        estimate = self.provider.estimate(
            _transfer_query("af_a_to_f", payload_bytes=4096)
        )
        self.assertAlmostEqual(estimate.service_duration, 4096 / 300e9)
        self.assertIn(estimate.match_type, AF_MATCH_TYPES)
        self.assertEqual(
            self.provider.estimate(_transfer_query(payload_bytes=0)).service_duration,
            0.0,
        )

        colocated = _transfer_query(
            "af_a_to_f", payload_bytes=8192, endpoints=("a-pool:0", "a-pool:0")
        )
        self.assertEqual(self.provider.estimate(colocated).service_duration, 0)
        for operation in ("tp_collective", "all_to_all"):
            with self.assertRaises(ConfigurationError):
                self.provider.estimate(replace(colocated, operation=operation))

        flat_doc = _cluster_dict(attention_workers=2, ffn_workers=2)
        flat_doc["af"]["transfer_topology_mode"] = "flat_link"
        flat = TopologyAFTransferProvider(
            self.hardware, ClusterConfig(**flat_doc), self.workers, None
        )
        self.assertEqual(flat.estimate(colocated).service_duration, 0)
        tp_query = _transfer_query(
            "tp_collective", payload_bytes=8192, endpoints=("a-pool:0", "a-pool:1")
        )
        self.assertEqual(flat.estimate(tp_query).details["algorithm"], "ring")

    def test_node_domains_separated_timing_and_invalid_endpoints(self):
        with self.assertRaisesRegex(
            ConfigurationError, "participants must be nonempty, distinct and mapped"
        ):
            self.provider.estimate(
                _transfer_query(payload_bytes=8192, endpoints=("a-pool:0", "missing:0"))
            )
        topology = TopologySpec(
            "three-level-af",
            (
                TopologyLevel("node", 8, "nvlink-test", links_per_endpoint=2),
                TopologyLevel("rack", 2, "ethernet-test"),
                TopologyLevel("cluster", None, "ethernet-test", oversubscription=2),
            ),
        )
        self.cluster.af.communication_domains = replace(
            self.cluster.af.communication_domains, node_count=4
        )
        times = [
            TopologyAFTransferProvider(
                self.hardware,
                self.cluster,
                self.workers,
                TopologyPlacement(topology, {0: 0, 1: 2, 2: index, 3: 24}),
            )
            .estimate(_transfer_query(payload_bytes=1 << 20))
            .service_duration
            for index in (1, 8, 16)
        ]
        self.assertLess(times[0], times[1])
        self.assertEqual(times[1], times[2])

        tp_query = _transfer_query(
            "tp_collective", payload_bytes=8192, endpoints=("a-pool:0", "a-pool:1")
        )
        legacy = self.provider.estimate(tp_query)
        self.cluster.af.communication_timing = AFCommunicationTimingConfig(
            enabled=True, host_submit_us=5
        )
        separated = TopologyAFTransferProvider(
            self.hardware, self.cluster, self.workers, self.placement, dtype="fp16"
        ).estimate(tp_query)
        self.assertEqual(legacy.service_duration, separated.service_duration)
        with self.assertRaisesRegex(ConfigurationError, "not declared"):
            self.provider.estimate(_transfer_query(network="ghost-net"))


class CollectiveAlgorithmAndFlatModelTest(unittest.TestCase):
    def test_collective_algorithm_cache_physical_guard_and_lpx_policy(self):
        doc = _cluster_dict(attention_workers=2, ffn_workers=2)
        doc["af"]["transfer_topology_mode"] = "flat_link"
        cluster = ClusterConfig(**doc)
        hardware = af_test_hardware()
        workers = cluster.workers()
        provider = TopologyAFTransferProvider(
            hardware,
            cluster,
            workers,
            build_topology_placement(cluster, workers, hardware),
        )
        query = _transfer_query(
            "all_reduce", payload_bytes=8192, endpoints=("a-pool:0", "a-pool:1")
        )
        ring = provider.estimate(query)
        tree = provider.estimate(replace(query, collective_algorithm="tree"))
        self.assertEqual(
            (ring.details["algorithm"], tree.details["algorithm"]), ("ring", "tree")
        )
        with self.assertRaises(ConfigurationError):
            provider.estimate(replace(query, collective_algorithm="invalid"))

        doc["af"]["transfer_topology_mode"] = "physical"
        physical = ClusterConfig(**doc)
        phys_provider = TopologyAFTransferProvider(
            hardware,
            physical,
            workers,
            build_topology_placement(physical, workers, hardware),
        )
        with self.assertRaisesRegex(ConfigurationError, "ring only"):
            phys_provider.estimate(replace(query, collective_algorithm="tree"))
        with self.assertRaises(ConfigurationError):
            DomainCollectiveConfig(
                "all_reduce", "a_compute", (0, 1), algorithm="invalid"
            )

        real_hw = HardwareContext.load(Path(__file__).resolve().parents[3] / "data")
        model = tiny_moe(real_hw)
        layout = Layout(
            a_tp=2,
            f_ep=2,
            a_collective_algorithm="direct",
            f_collective_algorithm="direct",
        )
        runtime = configuration(real_hw, model, layout, namespace="policy")
        env = simpy.Environment()
        service = AFService(env, runtime)
        session = service.admit(0, 0, maximum_context=24, current_context=16)
        env.run(until=env.process(service.forward([session])))
        events = [
            e
            for e in service.trace.events
            if e.operation in {"all_reduce", "all_gather"}
        ]
        self.assertEqual({e.details["algorithm"] for e in events}, {"direct"})
        bad_doc = cluster_document(runtime.cluster)
        bad_doc["af"].update(
            transfer_topology_mode="physical",
            communication_resource_scope="fabric",
            endpoint_transfer_occupancy="service",
        )
        with self.assertRaisesRegex(ConfigurationError, "wire schedule"):
            build_model_runtime(bad_doc, hardware=real_hw, model=model)

    def test_flat_collective_arithmetic_and_validation(self):
        link = LinkClass("flat", "test fabric", 100e9, 0.5)
        ring = flat_collective(
            "all_reduce", 8, 8192, link, algorithm="ring", dtype="bf16"
        )
        self.assertEqual(
            (ring.detail["rounds"], ring.detail["wire_bytes"]), (14, 14336)
        )
        small = flat_collective(
            "all_reduce", 8, 8192, link, algorithm="auto", dtype="bf16"
        )
        large = flat_collective(
            "all_reduce", 8, 8 << 20, link, algorithm="auto", dtype="bf16"
        )
        self.assertEqual(
            (small.detail["algorithm"], large.detail["algorithm"]), ("tree", "ring")
        )
        self.assertEqual(
            flat_collective(
                "all_reduce", 1, 8192, link, algorithm="ring", dtype="bf16"
            ).latency_us,
            0,
        )
        with self.assertRaisesRegex(ConfigurationError, "integer number"):
            flat_collective("all_reduce", 8, 8194, link, algorithm="ring", dtype="bf16")


class EndpointBackendAndEstimateCacheTest(unittest.TestCase):
    def test_per_worker_backend_selection_and_estimate_cache_immutability(self):
        values = _cluster_dict()
        values["worker_groups"][0]["operator_backend"] = "vllm"
        values["worker_groups"][1].update(
            hardware="a-device", operator_backend="trtllm"
        )
        cluster = _with_prefill_pool(ClusterConfig(**values))
        hardware = af_test_hardware()
        workers = cluster.workers()
        args = dict(
            hardware=hardware,
            model=hardware.model("TestModel"),
            cluster_config=cluster,
            worker_configs=workers,
            placement=build_topology_placement(cluster, workers, hardware),
        )
        query = replace(_transfer_query(), target_device="a-device")
        _, analytical = build_operator_table_af_providers(
            **args, fallback="analytical_only"
        )
        phase = analytical.communication_plan(query).phases[0]
        nodes = {
            n: i
            for i, n in enumerate(
                sorted({analytical.node_by_worker[r] for r in phase.participants})
            )
        }
        measurement = dict(
            model_version="node-domain-fifo-v1",
            algorithm="direct",
            phase_kind="transfer",
            operation="send_recv",
            dtype=phase.dtype,
            participants=len(phase.participants),
            nodes=len(nodes),
            logical_input_bytes=phase.logical_input_bytes,
            wire_bytes=phase.wire_bytes,
            wire_schedule=tuple(
                tuple(
                    (nodes[m.source_node], nodes[m.target_node], m.wire_bytes)
                    for m in row
                )
                for row in phase.rounds
            ),
            domain_parameters=tuple(
                (name, getattr(analytical.domain_config, name))
                for name in (
                    "local_bandwidth_bytes_per_s",
                    "inter_tx_bandwidth_bytes_per_s",
                    "inter_rx_bandwidth_bytes_per_s",
                    "local_latency_us",
                    "inter_latency_us",
                )
            ),
            timing_scope="transport_service",
            latency_us=7,
            source_id="test:mixed-endpoints",
            participant_hardware_backends=(
                ("a-device", "vllm"),
                ("a-device", "trtllm"),
            ),
        )
        for backend, cost_us in (("vllm", 101), ("trtllm", 123)):
            source_id = f"test:{backend}"
            meta = PackageMeta(
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
            hardware.register_package(
                OperatorDataPackage.from_rows(
                    meta,
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
            )
        compute, transfer = build_operator_table_af_providers(**args)
        cq = _compute_query()
        for worker, wid, cost_us in (("a-pool:0", 0, 101), ("f-pool:0", 1, 123)):
            est = compute.estimate(replace(cq, physical_worker=worker, worker_id=wid))
            proj = next(
                c for c in est.details["components"] if c["name"] == "qkv_projection"
            )
            self.assertEqual(proj["latency_us"], cost_us)
        self.assertEqual(transfer.estimate(query).source_id, "test:mixed-endpoints")

        moe_hw = af_test_hardware(moe=_moe())
        cached_provider = OperatorTableAFComputeProvider(
            moe_hw, moe_hw.model("TestModel"), fallback="analytical_only"
        )
        q = _compute_query(context=(8, 8))
        first = cached_provider.estimate(q)
        second = cached_provider.estimate(
            replace(q, iteration_id=12, block_id=4, layer_id=10)
        )
        self.assertIs(first, second)
        self.assertEqual(cached_provider.estimate_cache_hits, 1)
        with self.assertRaises(TypeError):
            second.details["components"][0]["key"]["m"] = 999
        self.assertIn("components", json.loads(json.dumps(second.details)))
        strict = OperatorTableAFComputeProvider(
            moe_hw, moe_hw.model("TestModel"), fallback="table_only"
        )
        with self.assertRaisesRegex(
            ConfigurationError, "table_only.*requires operator data"
        ):
            strict.estimate(q)
