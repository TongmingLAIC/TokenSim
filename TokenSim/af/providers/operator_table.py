"""Operator provider."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence
from dataclasses import replace

from TokenSim.af.caching.estimates import (
    ConfigurationOwner,
    HardwareSnapshot,
    initialize_cache,
    snapshot,
    traffic_feature_fingerprint,
)
from TokenSim.af.providers.components import ComputeComponentsMixin
from TokenSim.af.providers.compute import ComputeEstimateMixin
from TokenSim.af.providers.traffic import TransferPlanMixin
from TokenSim.af.providers.transfer import TransferEstimateMixin
from TokenSim.config.model_config import ModelSpec
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.domains import MODEL_VERSION
from TokenSim.hardware.links import LinkClass
from TokenSim.latency.operator_table import (
    FALLBACK_POLICIES,
    OperatorResolver,
    OperatorStats,
    _MOE_DISTRIBUTION_CANDIDATES,
)
from TokenSim.operator_data.lookup import LookupPolicy


def analytical_execution_device(device, graph_device_kernel_us):
    """Apply the same graph kernel assumption to complete and AF execution."""
    if graph_device_kernel_us is None or device.family != "nvidia_gpu":
        return device
    return replace(
        device,
        analytical={**device.analytical, "kernel_launch_us": graph_device_kernel_us},
    )


class OperatorTableAFComputeProvider(
    ComputeEstimateMixin, ComputeComponentsMixin, ConfigurationOwner
):
    """Compose rank-local attention, FFN and model-boundary operator estimates.

    Compiled local work controls tensor widths, routing assignments and dtypes.
    Kernel/link service excludes queueing, which belongs to executor resources.
    """

    _configuration_fields = (
        "hardware",
        "model",
        "fallback",
        "lookup_policy",
        "operator_backend",
        "backend_by_device",
        "operator_data_by_endpoint",
        "cluster_config",
        "placement",
        "dtype",
        "execution_protocol",
        "require_device_scope",
        "graph_device_kernel_us",
    )

    def __init__(
        self,
        hardware: Any,
        model: ModelSpec,
        *,
        operator_backend: str | None = None,
        backend_by_device: Mapping[str, str | None] | None = None,
        operator_data_by_endpoint: Mapping[str, tuple[str, str | None]] | None = None,
        fallback: str = "table_first",
        lookup_policy: LookupPolicy | None = None,
        cache_enabled: bool = True,
        execution_protocol: str = "partial_return",
        require_device_scope: bool = False,
        graph_device_kernel_us: float | None = None,
    ) -> None:
        if fallback not in FALLBACK_POLICIES:
            raise ConfigurationError(
                f"AF compute provider fallback must be one of {FALLBACK_POLICIES}, got {fallback!r}"
            )
        self.hardware = HardwareSnapshot(hardware)
        self.require_device_scope = (
            require_device_scope or execution_protocol == "vllm_afd"
        )
        self.model = snapshot(model)
        self.operator_backend = operator_backend
        self.backend_by_device = snapshot(dict(backend_by_device or {}))
        self.operator_data_by_endpoint = snapshot(dict(operator_data_by_endpoint or {}))
        self.fallback = fallback
        self.lookup_policy = snapshot(lookup_policy)
        self.execution_protocol = execution_protocol
        self.graph_device_kernel_us = graph_device_kernel_us
        if graph_device_kernel_us is not None and fallback != "analytical_only":
            raise ConfigurationError(
                "analytical graph execution requires analytical_only; measured graph profiles must declare their scope"
            )
        initialize_cache(self, cache_enabled)
        self._estimate_namespace += (
            ("graph_device_kernel_us", graph_device_kernel_us),
        )
        self._resolvers: dict[tuple[str, str | None], OperatorResolver] = {}
        routing = model.moe.routing.distribution if model.moe.routing else "uniform"
        self.moe_distributions = _MOE_DISTRIBUTION_CANDIDATES.get(
            routing, ("uniform", "balanced")
        )

    def resolver_for(
        self, device_type: str, physical_worker: str | None = None
    ) -> OperatorResolver:
        """Share a resolver among endpoints using the same device and backend."""
        key = self.operator_data_by_endpoint.get(
            physical_worker,
            (
                device_type,
                self.backend_by_device.get(device_type, self.operator_backend),
            ),
        )
        resolver = self._resolvers.get(key)
        if resolver is not None:
            return resolver
        device_type, backend = key
        device = analytical_execution_device(
            self.hardware.device(device_type), self.graph_device_kernel_us
        )
        package = None
        if self.fallback != "analytical_only":
            package = self.hardware.operator_package(device.device_id, backend)
        resolver = OperatorResolver(
            device=device,
            package=package,
            fallback=self.fallback,
            lookup_policy=self.lookup_policy,
            gated=self.model.gated,
        )
        self._resolvers[key] = resolver
        return resolver

    def estimate_signature(self, query):
        if self.model.components is not None:
            layers = self.model.layer_specs(include_mtp=False)
            layer = (
                layers[query.layer_id] if 0 <= query.layer_id < len(layers) else None
            )
            return (
                "compressed",
                layer.attention_kind if layer else None,
                layer.compress_ratio if layer else None,
                layer.moe_routing if layer else None,
            )
        if (
            self.model.attention_windows or self.model.attention_geometries
        ) and query.stage in {"a", "attention"}:
            return (
                self.model.attention_window(query.layer_id),
                self.model.attention_geometry(query.layer_id),
            )
        return ()

    @property
    def stats(self) -> OperatorStats:
        aggregate = OperatorStats()
        for resolver in self._resolvers.values():
            aggregate = aggregate.aggregate(resolver.stats)
        return aggregate

    def reset_profile_stats(self) -> None:
        for resolver in self._resolvers.values():
            resolver.stats = OperatorStats()
            # A cached analytical fallback has no record of the failed lookup.
            resolver._cache.clear()
        self._estimate_cache.clear()
        getattr(self, "_compressed_states", {}).clear()
        self.estimate_requests = 0
        self.estimate_cache_hits = 0

    def stats_dict(self) -> dict[str, Any]:
        stats = self.stats
        return {
            **stats.as_dict(),
            "operator_missing_shapes": stats.missing_shape_records()[:200],
            "estimate_requests": self.estimate_requests,
            "estimate_cache_hits": self.estimate_cache_hits,
        }

    def describe(self) -> dict[str, Any]:
        device_counts = Counter(device for device, _ in self._resolvers)
        return {
            "provider": "operator_table",
            "model_id": self.model.model_id,
            "execution_protocol": self.execution_protocol,
            "fallback": self.fallback,
            "graph_device_kernel_us_assumption": self.graph_device_kernel_us,
            "devices": {
                (
                    f"{device_type}:{backend}"
                    if device_counts[device_type] > 1 and backend is not None
                    else device_type
                ): {
                    "device_id": resolver.device.device_id,
                    "operator_backend": resolver.operator_backend,
                    "dataset_version": resolver.dataset_version,
                    "lookup_semantics": (
                        resolver.lookup.semantics if resolver.lookup else None
                    ),
                }
                for (device_type, backend), resolver in sorted(
                    self._resolvers.items(),
                    key=lambda item: (item[0][0], item[0][1] or ""),
                )
            },
        }


class TopologyAFTransferProvider(
    TransferPlanMixin, TransferEstimateMixin, ConfigurationOwner
):
    """Price static AF traffic with node domains or flat alpha-beta links.

    Physical collectives expose ring subgroup phases and matching measured
    transport services. Runtime reservations and queues belong to the executor.
    """

    _configuration_fields = OperatorTableAFComputeProvider._configuration_fields + (
        "domain_config",
        "node_by_worker",
        "phase_measurements",
        "_scope_by_worker",
        "transport_profile",
    )

    def link_class(self, network: str) -> LinkClass:
        link_id = self.cluster_config.networks.get(network)
        if link_id is None:
            raise ConfigurationError(
                f"AF network {network!r} is not declared in the cluster's networks"
            )
        try:
            return self.hardware.link(link_id)
        except ConfigurationError as exc:
            raise ConfigurationError(
                f"AF network {network!r} maps to {link_id!r}, which is not a link class "
                f"in the hardware catalog (data/topologies/links.yaml)"
            ) from exc

    def describe(self) -> dict[str, Any]:
        return {
            "provider": "topology_links",
            "transport_service_profile": (
                self.transport_profile.fingerprint if self.transport_profile else None
            ),
            "dtype": self.dtype,
            "links": {
                network: link.link_id
                for network, link in sorted(self.link_classes.items())
            },
            "placement": (
                self.placement.topology.topology_id
                if self.placement is not None
                else None
            ),
            "model_version": MODEL_VERSION,
            "domain_config": dict(vars(self.domain_config)),
            "worker_nodes": dict(self.node_by_worker),
            "worker_device_indices": (
                {
                    worker: self.placement.device_index(worker)
                    for worker in self.node_by_worker
                }
                if self.placement is not None
                else {}
            ),
            "placement_source": (
                "explicit:af.communication_domains"
                if self.cluster_config.af.enabled
                and self.cluster_config.af.transfer_topology_mode == "physical"
                else "legacy:topology"
            ),
            "configuration_fingerprint": traffic_feature_fingerprint(
                (
                    {
                        "model_version": MODEL_VERSION,
                        "domains": dict(vars(self.domain_config)),
                        "placement": self.node_by_worker,
                        "mode": self.cluster_config.af.transfer_topology_mode,
                        "measurements": self.phase_measurements,
                        "hardware_backends": self._scope_by_worker,
                    },
                )
            ),
        }


def build_operator_table_af_providers(
    *,
    hardware: Any,
    model: ModelSpec,
    cluster_config: Any,
    worker_configs: Sequence[Any],
    placement: Any | None,
    dtype: str = "fp16",
    operator_backend: str | None = None,
    fallback: str = "table_first",
    latency_backend_type: str = "operator_table",
) -> tuple[OperatorTableAFComputeProvider, TopologyAFTransferProvider]:
    """Build the pair of providers the engine uses for ``af.latency_provider = operator_table``.

    Every AF device is resolved eagerly so that a missing device, an unknown
    operator backend or an unmapped network fails before the event loop.
    """
    if latency_backend_type == "analytical":
        fallback = "analytical_only"
    transfer = TopologyAFTransferProvider(
        hardware,
        cluster_config,
        worker_configs,
        placement,
        dtype=dtype,
        operator_backend=operator_backend,
        fallback=fallback,
    )
    compute = OperatorTableAFComputeProvider(
        hardware,
        model,
        operator_backend=operator_backend,
        operator_data_by_endpoint=transfer.operator_data_by_endpoint,
        fallback=fallback,
        cache_enabled=cluster_config.af.service_estimate_cache,
        execution_protocol=cluster_config.af.execution_protocol,
        require_device_scope=cluster_config.af.engine_overhead_profile is not None,
        graph_device_kernel_us=(
            cluster_config.af.graph_device_kernel_us
            if cluster_config.af.compute_execution_mode == "cuda_graph"
            else None
        ),
    )
    for endpoint, (hardware_name, _) in transfer.operator_data_by_endpoint.items():
        compute.resolver_for(hardware_name, endpoint)
    if cluster_config.af.engine_overhead_profile is not None:
        from TokenSim.af.calibration.host import content_fingerprint

        fingerprint = content_fingerprint(cluster_config.af.engine_overhead_profile)
        for provider in (compute, transfer):
            provider._estimate_namespace += (fingerprint,)
    if cluster_config.af.operation_service_profile is not None:
        from TokenSim.af.calibration.host import content_fingerprint

        fingerprint = content_fingerprint(cluster_config.af.operation_service_profile)
        for provider in (compute, transfer):
            provider._estimate_namespace += (("operation_service", fingerprint),)
    return compute, transfer
