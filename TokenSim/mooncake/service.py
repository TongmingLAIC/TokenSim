from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from TokenSim.config.kv_transfer_config import KVTransferConfig
from TokenSim.mooncake.config import MooncakeConfig, parse_mooncake_config
from TokenSim.mooncake.store import MooncakeStore
from TokenSim.errors import ConfigurationError
from TokenSim.storage import StorageBackendFactory


@dataclass(frozen=True)
class _Contribution:
    worker: object
    pool_bytes: int
    staging_bytes: int


class CapacityLedger:
    """Count the capacity and staging of the ranks attached to one master."""

    def __init__(self, config: MooncakeConfig, *, fixed_capacity=None):
        self.config = config
        self.fixed_capacity = (
            config.global_segment_size
            if fixed_capacity is None and config.mode == "standalone-store"
            else fixed_capacity
        )
        self.contributors: dict[int, _Contribution] = {}
        self.capacity_bytes = 0
        self.staging_bytes = 0

    def register(self, worker, config, *, ranks=1):
        contribution = _Contribution(
            worker, ranks * config.global_segment_size, ranks * config.local_buffer_size
        )
        previous = self.contributors.get(id(worker))
        if previous is not None:
            if (
                contribution.staging_bytes != previous.staging_bytes
                or self.fixed_capacity is None
                and contribution.pool_bytes != previous.pool_bytes
            ):
                raise ConfigurationError(
                    "one rank cannot contribute different capacities to a Mooncake master"
                )
            return
        capacity = (
            self.capacity_bytes + contribution.pool_bytes
            if self.fixed_capacity is None
            else self.fixed_capacity
        )
        staging = self.staging_bytes + contribution.staging_bytes
        if (
            self.config.host_available_bytes is not None
            and capacity + staging > self.config.host_available_bytes
        ):
            raise ConfigurationError(
                "shared Mooncake pool and staging exceed available host DRAM"
            )
        self.contributors[id(worker)] = contribution
        self.capacity_bytes, self.staging_bytes = capacity, staging


class MooncakeService:
    def __init__(self, config: MooncakeConfig, block_bytes: int):
        self.store = MooncakeStore(config, block_bytes)
        fixed_capacity = (
            config.memory_capacity_blocks * block_bytes
            if config.memory_capacity_blocks is not None
            else None
        )
        self.capacity = CapacityLedger(config, fixed_capacity=fixed_capacity)

    def register(self, contributor, ranks, config):
        self.capacity.register(contributor, config, ranks=ranks)
        self.store.memory_capacity_bytes = self.capacity.capacity_bytes


def shared_store(owner, config, payload_format, create):
    """One simulated master per environment, independent of engine identity."""
    if owner is None:
        return create()
    stores = getattr(owner, "mooncake_stores", None)
    if stores is None:
        stores = owner.mooncake_stores = {}
    identity = config.master_server_address
    signature = _config_fingerprint(config)
    if identity not in stores:
        stores[identity] = (signature, payload_format, create())
    expected, payload, service = stores[identity]
    if signature != expected or payload != payload_format:
        raise ConfigurationError(
            "engines sharing a Mooncake master need compatible store settings and payload formats"
        )
    return service


def get_mooncake_service(
    kv_config: KVTransferConfig | MooncakeConfig,
    *,
    block_bytes: int,
    extra_config: dict[str, Any] | None = None,
    owner=None,
    contributor=None,
    ranks=1,
) -> MooncakeService:
    mooncake_config = (
        kv_config
        if isinstance(kv_config, MooncakeConfig)
        else parse_mooncake_config(
            extra_config
            if extra_config is not None
            else kv_config.kv_connector_extra_config
        )
    )

    def create():
        service = MooncakeService(mooncake_config, block_bytes)
        service.register(
            contributor if contributor is not None else service, ranks, mooncake_config
        )
        return service

    service = shared_store(
        owner,
        mooncake_config,
        ("uniform", block_bytes),
        create,
    )
    service.register(
        contributor if contributor is not None else service, ranks, mooncake_config
    )
    return service


def _config_fingerprint(config: MooncakeConfig) -> str:
    payload = asdict(config)
    if config.offload is not None:
        payload["offload"]["config"] = StorageBackendFactory.create(
            config.offload
        ).describe()
    client_fields = [
        "tenant_id",
        "cache_prefix",
        "group_id",
        "pcp_rank",
        "dcp_rank",
        "admission_policy",
        "save_policy",
        "load_async",
        "enable_lookup",
        "transfer_overlap",
        "local_buffer_size",
    ]
    if config.mode == "embedded":
        client_fields.append("global_segment_size")
    if config.cache_layout == "uniform" or config.mode == "embedded":
        client_fields.extend(
            [
                "protocol",
                "num_nics",
                "parallel_paths",
                "fixed_latency_us",
                "bandwidth_bytes_per_s",
                "transfer_utilization",
                "protocol_bandwidth_bytes_per_s",
                "protocol_latency_us",
            ]
        )
    for name in client_fields:
        payload.pop(name)
    return json.dumps(payload, sort_keys=True, default=str)
