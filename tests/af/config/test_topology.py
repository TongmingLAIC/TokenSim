"""AF settings, worker capabilities, links, communication fabrics, and domains."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import unittest

from TokenSim.config.af.communication import (
    AFCommunicationFabricConfig,
    AFCommunicationLaneConfig,
)
from TokenSim.config.af.domains import DomainCollectiveConfig
from TokenSim.config.af.links import AFEffectiveDTypes, AFLinkConfig
from TokenSim.config.af.settings import AFConfig
from TokenSim.config.cluster_config import ClusterConfig, WorkerGroupConfig
from TokenSim.config.constants import LEGACY_ROLE_CAPABILITIES
from TokenSim.errors import ConfigurationError
from tests.af.support.config_validation import (
    _af_group,
    _cluster,
    _explicit_decode_cluster_raw,
    _full_link,
    _raw_link,
)


class AFConfigAndWorkerCapabilityTest(unittest.TestCase):
    def test_defaults_and_scalar_validation(self):
        cluster = ClusterConfig(num_workers=0, networks={})
        self.assertFalse(cluster.af.enabled)
        self.assertEqual((cluster.af.af_block_size, cluster.af.links), (1, []))
        self.assertEqual(cluster.af_communication_resources, {})
        for field, values in (
            ("af_block_size", (0, -1, True, 1.5, "2")),
            ("enabled", (1,)),
            ("trace_enabled", ("true",)),
            ("ffn_batching", (1, "true")),
            ("ffn_batch_wait_us", (-1, True, float("nan"), float("inf"))),
            ("ffn_batch_max_tokens", (-1, True, 1.5, "8")),
            ("max_inflight_blocks", (-1, True, 1.5, "8")),
        ):
            for value in values:
                with self.assertRaises(ConfigurationError):
                    AFConfig(**{field: value})

    def test_dtype_resolution_precedence_and_legacy_mapping(self):
        default = AFConfig().resolve_dtypes(model_dtype="BFloat16")
        self.assertIsInstance(default, AFEffectiveDTypes)
        self.assertEqual(
            (
                default.weight_dtype,
                default.activation_dtype,
                default.kv_cache_dtype,
                default.a_to_f_dtype,
                default.f_to_a_dtype,
            ),
            ("bf16",) * 5,
        )
        link = _full_link(a_to_f_dtype="float8_e4m3fn", f_to_a_dtype="float32")
        independent = AFConfig(
            weight_dtype="FP8",
            activation_dtype="BF16",
            kv_cache_dtype="half",
            links=[link],
        ).resolve_dtypes(model_dtype="fp32", link=link)
        self.assertEqual(
            (
                independent.weight_dtype,
                independent.activation_dtype,
                independent.kv_cache_dtype,
                independent.a_to_f_dtype,
                independent.f_to_a_dtype,
            ),
            ("fp8", "bf16", "fp16", "fp8", "fp32"),
        )
        legacy = AFConfig(af_dtype="BFloat16").resolve_dtypes(model_dtype="fp32")
        self.assertTrue(legacy.legacy_mapped)
        self.assertEqual(legacy.deprecated_fields, ("af_dtype",))
        with self.assertRaisesRegex(ConfigurationError, "af_dtype.*weight_dtype"):
            AFConfig(af_dtype="bf16", weight_dtype="bf16")
        with self.assertRaisesRegex(ConfigurationError, "af_dtype.*a_to_f_dtype"):
            AFConfig(af_dtype="bf16", links=[_full_link(a_to_f_dtype="fp8")])
        model_kv = AFConfig(activation_dtype="bf16").resolve_dtypes(
            model_dtype="fp32", model_kv_cache_dtype="float16"
        )
        self.assertEqual(model_kv.kv_cache_dtype, "fp16")
        default_link = _full_link()
        self.assertEqual(
            replace(default_link, a_to_f_dtype="fp8").explicit_dtype_fields,
            frozenset({"a_to_f_dtype"}),
        )
        with self.assertRaisesRegex(ConfigurationError, "f_to_a_dtype"):
            AFLinkConfig(**{**_raw_link(default_link), "f_to_a_dtype": None})

    def test_worker_capabilities_and_repository_examples(self):
        for role, expected in LEGACY_ROLE_CAPABILITIES.items():
            group = WorkerGroupConfig(role=role, hardware="device", num_workers=1)
            self.assertEqual(set(group.capabilities), set(expected))
        explicit = WorkerGroupConfig(
            role="prefill",
            name="mixed",
            capabilities=("af_attention", "prefill"),
            hardware="device",
            num_workers=1,
        ).workers({"net1": "test-network"})[0]
        self.assertEqual(explicit.capabilities, ("prefill", "af_attention"))
        for values in (
            {"role": "bad", "hardware": "device", "num_workers": 1},
            {"hardware": "device", "num_workers": 1, "capabilities": ("bad",)},
            {"hardware": "device", "num_workers": 0, "capabilities": ("af_ffn",)},
            {
                "role": "prefill",
                "hardware": "device",
                "num_workers": 1,
                "capabilities": ("decode_full",),
            },
        ):
            with self.assertRaises(ConfigurationError):
                WorkerGroupConfig(**values)
        with self.assertRaisesRegex(ConfigurationError, "duplicate.*same"):
            ClusterConfig(
                num_workers=2,
                networks={"net1": "network"},
                worker_groups=[
                    _af_group("same", "af_attention"),
                    _af_group("same", "af_ffn"),
                ],
            )
        examples = Path(__file__).parents[3] / "data/clusters/af"
        for name, workers in (
            ("decode_half_duplex.json", 5),
            ("decode_tp2_dp2_ep4_rank_reuse.json", 9),
        ):
            cluster = ClusterConfig.from_file(examples / name)
            self.assertEqual(len(cluster.workers()), workers)
            self.assertEqual(
                ClusterConfig(**json.loads((examples / name).read_text())).af_link,
                cluster.af_link,
            )


class AFLinkAndFabricTopologyTest(unittest.TestCase):
    def test_link_duplex_endpoints_and_capabilities(self):
        full = _full_link()
        half = _full_link(
            duplex="half",
            a_to_f_capacity=None,
            f_to_a_capacity=None,
            half_duplex_capacity=2,
        )
        self.assertEqual(
            (full.effective_capacities, half.effective_capacities), ((1, 1), (2,))
        )
        for overrides in (
            {"a_to_f_capacity": None},
            {"duplex": "simplex"},
            {"a_to_f_capacity": 0},
            {"f_to_a_capacity": True},
            {"half_duplex_capacity": 1},
        ):
            with self.assertRaises(ConfigurationError):
                _full_link(**overrides)
        for links in (
            [],
            [_full_link(name="one"), _full_link(name="two", resource_group="two")],
        ):
            with self.assertRaisesRegex(ConfigurationError, "exactly one link"):
                _cluster(links=links)
        with self.assertRaisesRegex(ConfigurationError, "af-link.*missing"):
            _cluster(links=[_full_link(attention_group="missing")])
        with self.assertRaisesRegex(ConfigurationError, "must differ"):
            _cluster(links=[_full_link(ffn_group="attention")])
        with self.assertRaisesRegex(ConfigurationError, "af-link.*unknown network"):
            _cluster(links=[_full_link(network="missing")])

    def test_communication_resources_and_fabric_conflicts(self):
        resources = _cluster().af_communication_resources
        self.assertEqual(set(resources), {("fabric", "a_to_f"), ("fabric", "f_to_a")})
        half = _full_link(
            duplex="half",
            a_to_f_capacity=None,
            f_to_a_capacity=None,
            half_duplex_capacity=2,
        )
        self.assertEqual(
            set(_cluster(links=[half]).af_communication_resources),
            {("fabric", "shared")},
        )
        fabric = AFCommunicationFabricConfig(
            resource_group="fabric",
            network="net1",
            duplex="full",
            lanes=(
                AFCommunicationLaneConfig("a_to_f", 1),
                AFCommunicationLaneConfig("f_to_a", 1),
                AFCommunicationLaneConfig("collective", 2),
            ),
        )
        link = AFLinkConfig(
            name="af-link",
            attention_group="attention",
            ffn_group="ffn",
            resource_group="fabric",
        )
        cluster = _cluster(links=[link], communication_fabrics=[fabric])
        self.assertEqual(
            cluster.af_communication_resources[("fabric", "collective")].capacity, 2
        )
        with self.assertRaisesRegex(ConfigurationError, "capacity differs"):
            _cluster(
                links=[_full_link(a_to_f_capacity=2)], communication_fabrics=[fabric]
            )
        with self.assertRaisesRegex(ConfigurationError, "missing communication fabric"):
            _cluster(
                links=[
                    AFLinkConfig(
                        name="af-link",
                        attention_group="attention",
                        ffn_group="ffn",
                        resource_group="missing",
                    )
                ]
            )


class AFDomainAndCollectiveConfigTest(unittest.TestCase):
    @staticmethod
    def _with_collective_fabric(raw: dict) -> dict:
        raw["af"]["communication_fabrics"].append(
            {
                "resource_group": "a-collective-fabric",
                "network": "net1",
                "duplex": "half",
                "lanes": [{"name": "collective", "capacity": 2}],
            }
        )
        return raw

    @staticmethod
    def _collective(**overrides) -> dict:
        return {
            "operation": "all_gather",
            "after_stage": "a_compute",
            "participants": [0, 2],
            "resource_group": "a-collective-fabric",
            "resource_lane": "collective",
            "network": "net1",
            **overrides,
        }

    def test_f_world_size_and_ep_divisibility(self):
        for ep_size, expected_mode, expected_expert_tp, expected_ep_ranks in (
            (1, "expert_tp", 2, [0, 0, 0, 0]),
            (4, "rank_reuse", 1, [0, 1, 2, 3]),
            (2, "expert_sharded", 2, [0, 0, 1, 1]),
        ):
            cluster = ClusterConfig(**_explicit_decode_cluster_raw(f_ep=ep_size))
            domain = cluster.execution_domains["f-domain"]
            mappings = [
                m for inst in domain.logical_instances for m in inst.rank_mapping
            ]
            self.assertEqual(domain.parallel.world_size, 4)
            self.assertEqual(domain.expert_parallel.mode, expected_mode)
            self.assertEqual(
                domain.expert_parallel.expert_tensor_parallel_size, expected_expert_tp
            )
            self.assertEqual([m.ep_rank for m in mappings], expected_ep_ranks)
        with self.assertRaisesRegex(ConfigurationError, "expert_parallel_size=3"):
            ClusterConfig(**_explicit_decode_cluster_raw(f_ep=3))

    def test_collective_participants_and_fabric_resolution(self):
        raw = self._with_collective_fabric(
            _explicit_decode_cluster_raw(a_tp=4, a_dp=2, f_tp=1, f_dp=1)
        )
        raw["af"]["execution_domains"][0]["communication_plan"] = [self._collective()]
        cluster = ClusterConfig(**raw)
        domain = cluster.execution_domains["a-domain"]
        collective = domain.communication_plan[0]
        self.assertEqual(domain.resolve_collective_participants(collective, 0), (0, 2))
        self.assertEqual(domain.resolve_collective_participants(collective, 1), (4, 6))
        with self.assertRaisesRegex(
            ConfigurationError, "participants contains duplicates"
        ):
            DomainCollectiveConfig(
                operation="all_reduce", after_stage="a_compute", participants=(0, 0)
            )
        out_of_range = deepcopy(raw)
        out_of_range["af"]["execution_domains"][0]["communication_plan"] = [
            self._collective(participants=[0, 4])
        ]
        with self.assertRaisesRegex(ConfigurationError, "out-of-range"):
            ClusterConfig(**out_of_range)
        missing_lane = deepcopy(raw)
        del missing_lane["af"]["execution_domains"][0]["communication_plan"][0][
            "resource_lane"
        ]
        with self.assertRaisesRegex(
            ConfigurationError, "requires explicit resource_lane"
        ):
            ClusterConfig(**missing_lane)
