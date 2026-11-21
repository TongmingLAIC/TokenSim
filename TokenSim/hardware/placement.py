"""Map simulator workers onto topology device indices."""

from __future__ import annotations

from itertools import count
from typing import Any, Sequence

from TokenSim.errors import ConfigurationError
from TokenSim.hardware.context import HardwareContext
from TokenSim.hardware.topology import TopologyPlacement, TopologySpec
from TokenSim.hardware.domains import CommunicationDomainConfig, physical_domain_config


def build_topology_placement(
    cluster_config: Any,
    worker_configs: Sequence[Any],
    hardware: HardwareContext,
) -> TopologyPlacement:
    """Resolve the cluster's topology and each worker's device index.

    AF physical mode uses explicit communication-domain node inventory.
    With a named topology, explicit slots are reserved before filling the
    remaining slots in device order. A synthesized topology places each
    network on one node. Explicit slots select that node and local slots.
    """
    af = getattr(cluster_config, "af", None)
    domains = physical_domain_config(af) if af is not None else None
    if domains is not None:
        return build_domain_placement(cluster_config, worker_configs, hardware, domains)
    for worker in worker_configs:
        hardware.link(worker.nettype)

    topology_id = getattr(cluster_config, "topology", None)
    topology = (
        hardware.topology(topology_id)
        if topology_id
        else synthesize_topology(cluster_config, worker_configs, hardware)
    )
    explicit = {
        rank: topology.validate_device_index(slot)
        for rank, worker in enumerate(worker_configs)
        if (slot := getattr(worker, "device_index", None)) is not None
    }
    if len(set(explicit.values())) != len(explicit):
        raise ConfigurationError("two workers are placed on the same device index")

    if topology_id:
        slots = count() if topology.capacity is None else range(topology.capacity)
        occupied = set(explicit.values())
        available = (slot for slot in slots if slot not in occupied)
        placement = _fill_device_slots(worker_configs, explicit, available)
    else:
        placement = _place_synthesized_networks(
            worker_configs, explicit, topology.levels[0].size
        )
    return TopologyPlacement(topology=topology, device_index_by_worker=placement)


def _fill_device_slots(worker_configs, explicit, available):
    placement = dict(explicit)
    for rank in range(len(worker_configs)):
        if rank not in placement:
            slot = next(available, None)
            if slot is None:
                raise ConfigurationError("worker placement exceeds topology inventory")
            placement[rank] = slot
    return placement


def _place_synthesized_networks(worker_configs, explicit, node_size):
    networks = [worker.network for worker in worker_configs]
    network_nodes = {}
    node_networks = {}
    occupied = set(explicit.values())
    for rank, slot in explicit.items():
        network, node = networks[rank], slot // node_size
        if network_nodes.setdefault(network, node) != node:
            raise ConfigurationError(
                f"network {network!r} spans device slots on multiple synthesized nodes; "
                "declare a topology for this placement"
            )
        if node_networks.setdefault(node, network) != network:
            raise ConfigurationError(
                f"device indices place multiple networks on synthesized node {node}; "
                "declare a topology for this placement"
            )

    nodes = (node for node in count() if node not in node_networks)
    free_slots = {}
    placement = dict(explicit)
    for rank, network in enumerate(networks):
        if network not in network_nodes:
            network_nodes[network] = next(nodes)
        if rank in placement:
            continue
        if network not in free_slots:
            start = network_nodes[network] * node_size
            free_slots[network] = (
                slot for slot in range(start, start + node_size) if slot not in occupied
            )
        placement[rank] = next(free_slots[network])
    return placement


def build_domain_placement(cluster_config, worker_configs, hardware, config):
    """Reserve explicit slots first; never expand a declared inventory."""
    config.validate_physical_inventory()
    node_count = config.node_count
    refs = [getattr(worker, "physical_worker_ref", None) for worker in worker_configs]
    if any(ref is None for ref in refs) or len({ref.identity for ref in refs}) != len(
        refs
    ):
        raise ConfigurationError(
            "AF physical placement needs unique physical worker identities"
        )
    identities = {ref.identity: index for index, ref in enumerate(refs)}
    overrides = dict(config.device_indices)
    for group in cluster_config.worker_groups:
        if group.device_indices is not None:
            for index, slot in enumerate(group.device_indices):
                identity = f"{group.name}:{index}"
                if identity in overrides and overrides[identity] != slot:
                    raise ConfigurationError(
                        f"conflicting device index aliases for {identity}"
                    )
                overrides[identity] = slot
    if set(overrides) - set(identities):
        raise ConfigurationError(
            "domain placement override references an unknown worker"
        )
    if len(set(overrides.values())) != len(overrides):
        raise ConfigurationError("domain placement duplicates a GPU slot")
    if any(type(slot) is not int or slot < 0 for slot in overrides.values()):
        raise ConfigurationError(
            "domain placement requires nonnegative integer GPU slots"
        )
    attention = {
        index
        for index, worker in enumerate(worker_configs)
        if "af_attention" in worker.capabilities
    }
    roles = {
        index: ("a" if index in attention else "f")
        for index in range(len(worker_configs))
    }
    dedicated_nodes = {}
    if config.placement_strategy == "dedicated":
        for identity, slot in overrides.items():
            node = slot // config.node_size
            role = roles[identities[identity]]
            if node in dedicated_nodes and dedicated_nodes[node] != role:
                raise ConfigurationError(
                    "dedicated placement override mixes A/F in one node"
                )
            dedicated_nodes[node] = role

    def permitted(worker_id, slot):
        node, local = divmod(slot, config.node_size)
        if node >= node_count or slot < 0:
            return False
        role = roles[worker_id]
        if config.placement_strategy == "mixed":
            return (local < config.mixed_a_slots) == (role == "a")
        if config.placement_strategy == "dedicated":
            return dedicated_nodes.get(node, role) == role
        return True

    placed = {}
    occupied = set(overrides.values())
    for identity, slot in overrides.items():
        worker_id = identities[identity]
        if not permitted(worker_id, slot):
            raise ConfigurationError(
                f"placement override violates inventory or role policy: {identity}"
            )
        placed[worker_id] = slot
    for worker_id in range(len(worker_configs)):
        if worker_id in placed:
            continue
        slot = next(
            (
                slot
                for slot in range(node_count * config.node_size)
                if slot not in occupied and permitted(worker_id, slot)
            ),
            None,
        )
        if slot is None:
            raise ConfigurationError(
                "domain placement exceeds declared inventory or role slots"
            )
        placed[worker_id] = slot
        occupied.add(slot)
        if config.placement_strategy == "dedicated":
            dedicated_nodes[slot // config.node_size] = roles[worker_id]
    topology_id = getattr(cluster_config, "topology", None)
    legacy = (
        hardware.topology(topology_id)
        if topology_id
        else synthesize_topology(cluster_config, worker_configs, hardware)
    )
    topology = TopologySpec.two_level(
        "domain_" + legacy.topology_id,
        node_size=config.node_size,
        node_link=legacy.levels[0].link,
        cluster_link=legacy.levels[-1].link,
    )
    result = TopologyPlacement(topology, placed)
    validate_domain_placement(result, worker_configs, config)
    return result


def validate_domain_placement(placement, worker_configs, config):
    """Validate supplied and generated AF mappings before creating resources."""
    config.validate_physical_inventory()
    if placement is None:
        raise ConfigurationError(
            "AF physical requires placement from explicit communication_domains"
        )
    if placement.topology.levels[0].size != config.node_size:
        raise ConfigurationError(
            "AF physical placement node size must equal communication_domains.node_size; migrate legacy placement"
        )
    mapping = placement.device_index_by_worker
    if set(mapping) != set(range(len(worker_configs))) or any(
        type(rank) is not int for rank in mapping
    ):
        raise ConfigurationError(
            "AF physical placement must map every actual worker exactly once"
        )
    slots = tuple(mapping.values())
    if any(type(slot) is not int or slot < 0 for slot in slots):
        raise ConfigurationError(
            "AF physical placement requires nonnegative integer GPU slots"
        )
    if len(set(slots)) != len(slots):
        raise ConfigurationError("AF physical placement duplicates a GPU slot")
    if any(slot >= config.node_count * config.node_size for slot in slots):
        raise ConfigurationError(
            "AF physical placement exceeds declared node inventory"
        )
    identities = set()
    node_roles = {}
    for worker_id, worker in enumerate(worker_configs):
        ref = worker.physical_worker_ref
        if ref is None or ref.identity in identities:
            raise ConfigurationError(
                "AF physical placement needs unique physical worker identities"
            )
        identities.add(ref.identity)
        slot = mapping[worker_id]
        for explicit in (config.device_indices.get(ref.identity), worker.device_index):
            if explicit is not None and (type(explicit) is not int or explicit < 0):
                raise ConfigurationError(
                    f"AF placement needs a nonnegative integer GPU slot for {ref.identity}"
                )
            if explicit is not None and explicit != slot:
                raise ConfigurationError(
                    f"AF placement conflicts with explicit GPU slot for {ref.identity}"
                )
        is_a = "af_attention" in worker.capabilities
        node, local = placement.node_id(worker_id), slot % config.node_size
        if (
            config.placement_strategy == "mixed"
            and (local < config.mixed_a_slots) != is_a
        ):
            raise ConfigurationError("AF mixed placement violates declared role slots")
        if config.placement_strategy == "dedicated":
            if node_roles.setdefault(node, is_a) != is_a:
                raise ConfigurationError("AF dedicated placement mixes A/F in one node")
    if set(config.device_indices) - identities:
        raise ConfigurationError(
            "AF placement override references an unknown physical worker"
        )


def synthesize_topology(
    cluster_config: Any,
    worker_configs: Sequence[Any],
    hardware: HardwareContext,
) -> TopologySpec:
    counts: dict[str, int] = {}
    for worker in worker_configs:
        network = getattr(worker, "network", "net")
        counts[network] = counts.get(network, 0) + 1
    node_size = max(counts.values()) if counts else 1

    first_device = (
        hardware.device(worker_configs[0].hardware) if worker_configs else None
    )
    node_link = None
    if first_device is not None:
        node_link = first_device.scale_up_link or first_device.host_link
    if node_link is None or node_link not in hardware.links:
        raise ConfigurationError(
            "synthesized topology requires a known device scale-up or host link"
        )
    device_link_ids = {
        hardware.device(worker.hardware).scale_up_link
        or hardware.device(worker.hardware).host_link
        for worker in worker_configs
    }
    if len(device_link_ids) != 1:
        raise ConfigurationError(
            "mixed device links require an explicit cluster topology"
        )
    node_fabric = (
        "all_to_all"
        if first_device is not None and first_device.family == "groq_tsp"
        else "switched"
    )
    node_links = (
        first_device.scale_up_ports
        if (first_device and first_device.family == "groq_tsp")
        else 1
    )
    if hardware.link(node_link).bandwidth_scope == "endpoint":
        node_links = 1

    nettypes = {getattr(worker, "nettype", None) for worker in worker_configs}
    nettypes.discard(None)
    if len(nettypes) != 1:
        raise ConfigurationError(
            "mixed network links require an explicit cluster topology"
        )
    cluster_link = next(iter(nettypes))
    hardware.link(cluster_link)
    name = "synth_" + "_".join(sorted(counts)) if counts else "synth"
    return TopologySpec.two_level(
        name,
        node_size=node_size,
        node_link=node_link,
        cluster_link=cluster_link,
        node_fabric=node_fabric,
        node_links_per_endpoint=node_links,
    )
