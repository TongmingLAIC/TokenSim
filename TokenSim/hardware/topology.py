from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import cached_property
from math import prod
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from TokenSim.config.validation import integer
from TokenSim.errors import ConfigurationError
from TokenSim.hardware._yaml import load_yaml_mapping, require
from TokenSim.hardware.links import LinkCatalog, LinkClass

FABRICS = ("switched", "all_to_all", "ring", "fat_tree", "bus")


@dataclass(frozen=True)
class TopologyLevel:
    """One level of a hierarchical interconnect.

    Level 0 groups devices into the first enclosure (a node or an NVLink
    domain); level ``i`` groups level ``i-1`` groups. ``size`` is the number
    of children per group (``None`` means unbounded, used for the outermost
    level). ``link`` names the :class:`LinkClass` used by an endpoint to talk
    to peers inside this level; ``links_per_endpoint`` is how many such links
    (ports / NICs) each child has, and ``fabric`` describes how they connect.
    """

    name: str
    size: int | None
    link: str
    fabric: str = "switched"
    links_per_endpoint: int = 1
    oversubscription: float = 1.0
    collective_algorithm: str = "auto"

    def __post_init__(self) -> None:
        if self.size is not None:
            integer(self.size, f"topology level {self.name!r} size", minimum=1)
        if self.fabric not in FABRICS:
            raise ConfigurationError(
                f"topology level {self.name!r}: fabric must be one of {FABRICS}"
            )
        if self.links_per_endpoint < 1:
            raise ConfigurationError(
                f"topology level {self.name!r}: links_per_endpoint must be >= 1"
            )
        if self.oversubscription < 1.0:
            raise ConfigurationError(
                f"topology level {self.name!r}: oversubscription must be >= 1"
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], context: str) -> "TopologyLevel":
        name = str(require(raw, "name", context))
        size = raw.get("size")
        return cls(
            name=name,
            size=size,
            link=str(require(raw, "link", f"{context} level {name!r}")),
            fabric=str(raw.get("fabric", "switched")),
            links_per_endpoint=int(raw.get("links_per_endpoint", 1)),
            oversubscription=float(raw.get("oversubscription", 1.0)),
            collective_algorithm=str(raw.get("collective_algorithm", "auto")),
        )


@dataclass(frozen=True)
class GroupLayout:
    """How a set of ranks is spread across the topology levels.

    ``fan[i]`` is the largest number of level ``i-1`` groups (devices for
    ``i == 0``) that participate inside one level ``i`` group. A TP group of
    16 GPUs over two 8-GPU nodes has ``fan == (8, 2)``; ``levels`` are the
    topology levels actually crossed (``fan[i] > 1``) plus the innermost.
    ``node_count`` is the exact number of occupied level-zero groups.
    """

    ranks: tuple[int, ...]
    fan: tuple[int, ...]
    lowest_common_level: int  # -1: single device, 0: same level-0 group, ...
    node_count: int | None = None

    @property
    def size(self) -> int:
        return len(self.ranks)

    @property
    def spans_levels(self) -> int:
        return self.lowest_common_level + 1


@dataclass(frozen=True)
class TopologySpec:
    topology_id: str
    levels: tuple[TopologyLevel, ...]
    description: str = ""
    source_id: str = "unspecified"
    # Reserved for measured collective tables keyed by topology.
    collective_table_scope: str | None = None

    def __post_init__(self) -> None:
        if not self.levels:
            raise ConfigurationError(
                f"topology {self.topology_id!r} needs at least one level"
            )
        for level in self.levels[:-1]:
            if level.size is None:
                raise ConfigurationError(
                    f"topology {self.topology_id!r}: only the outermost level may be unbounded"
                )
        names = [level.name for level in self.levels]
        if len(set(names)) != len(names):
            raise ConfigurationError(
                f"topology {self.topology_id!r}: level names must be unique"
            )

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], context: str = "topology"
    ) -> "TopologySpec":
        topology_id = str(require(raw, "topology_id", context))
        if raw.get("schema_version", 1) != 1:
            raise ConfigurationError(
                f"topology {topology_id!r}: unsupported schema_version"
            )
        levels_raw = require(raw, "levels", f"topology {topology_id!r}")
        if not isinstance(levels_raw, Sequence) or not levels_raw:
            raise ConfigurationError(
                f"topology {topology_id!r}: levels must be a non-empty list"
            )
        return cls(
            topology_id=topology_id,
            levels=tuple(
                TopologyLevel.from_mapping(item, f"topology {topology_id!r}")
                for item in levels_raw
            ),
            description=str(raw.get("description", "")),
            source_id=str(raw.get("source_id", "unspecified")),
            collective_table_scope=raw.get("collective_table_scope"),
        )

    @classmethod
    def two_level(
        cls,
        topology_id: str,
        *,
        node_size: int,
        node_link: str,
        cluster_link: str,
        node_fabric: str = "switched",
        node_links_per_endpoint: int = 1,
        cluster_links_per_endpoint: int = 1,
    ) -> "TopologySpec":
        return cls(
            topology_id=topology_id,
            levels=(
                TopologyLevel(
                    name="node",
                    size=max(1, node_size),
                    link=node_link,
                    fabric=node_fabric,
                    links_per_endpoint=node_links_per_endpoint,
                ),
                TopologyLevel(
                    name="cluster",
                    size=None,
                    link=cluster_link,
                    fabric="fat_tree",
                    links_per_endpoint=cluster_links_per_endpoint,
                ),
            ),
            description="synthesized from cluster networks",
            source_id="derived:cluster-config",
        )

    def validate_links(self, links: LinkCatalog) -> None:
        for level in self.levels:
            if level.link not in links:
                raise ConfigurationError(
                    f"topology {self.topology_id!r} level {level.name!r} references unknown link {level.link!r}"
                )

    # -- coordinates ------------------------------------------------------

    @cached_property
    def capacity(self) -> int | None:
        """Number of devices in a finite topology."""
        if self.levels[-1].size is None:
            return None
        return prod(level.size for level in self.levels)

    def validate_device_index(self, device_index: int) -> int:
        slot = integer(device_index, "device index")
        if self.capacity is not None and slot >= self.capacity:
            raise ConfigurationError(
                f"topology {self.topology_id!r} has {self.capacity} device slots; "
                f"device index {slot} is outside the inventory"
            )
        return slot

    def coordinates(self, device_index: int) -> tuple[int, ...]:
        """Group index of ``device_index`` at every level (innermost first)."""
        coords: list[int] = []
        cursor = self.validate_device_index(device_index)
        for level in self.levels:
            cursor = 0 if level.size is None else cursor // level.size
            coords.append(cursor)
        return tuple(coords)

    def _occupied_groups(self, device_indices: Sequence[int]):
        coordinates = {
            index: self.coordinates(index) for index in dict.fromkeys(device_indices)
        }
        groups = tuple(set(level) for level in zip(*coordinates.values()))
        return coordinates, groups

    @staticmethod
    def _common_level(coordinates, groups) -> int:
        if len(coordinates) <= 1:
            return -1
        return next(
            level for level, group_ids in enumerate(groups) if len(group_ids) == 1
        )

    def lowest_common_level(self, device_indices: Sequence[int]) -> int:
        return self._common_level(*self._occupied_groups(device_indices))

    def group_layout(self, device_indices: Sequence[int]) -> GroupLayout:
        ranks = tuple(device_indices)
        coords, groups = self._occupied_groups(ranks)
        if not ranks:
            raise ConfigurationError("group layout requires at least one rank")
        if len(coords) < len(ranks):
            raise ConfigurationError("group layout requires unique device indices")
        fan = [max(Counter(c[0] for c in coords.values()).values())]
        for level_index, child_groups in enumerate(groups[:-1], 1):
            # An unbounded level owns all child groups.
            size = self.levels[level_index].size
            parents = Counter(
                0 if size is None else child // size for child in child_groups
            )
            fan.append(max(parents.values()))
        return GroupLayout(
            ranks=ranks,
            fan=tuple(fan),
            lowest_common_level=self._common_level(coords, groups),
            node_count=len(groups[0]),
        )

    def level_link(self, level_index: int, links: LinkCatalog) -> LinkClass:
        return links.get(self.levels[level_index].link)

    def to_dict(self) -> dict[str, Any]:
        return {
            "topology_id": self.topology_id,
            "levels": [
                {
                    "name": level.name,
                    "size": level.size,
                    "link": level.link,
                    "fabric": level.fabric,
                    "links_per_endpoint": level.links_per_endpoint,
                    "oversubscription": level.oversubscription,
                }
                for level in self.levels
            ],
        }


@dataclass(frozen=True)
class TopologyPlacement:
    """Maps simulator worker ids onto device indices of a topology."""

    topology: TopologySpec
    device_index_by_worker: Mapping[int, int] = field(default_factory=dict)

    def device_index(self, worker_id: int) -> int:
        return self.device_index_by_worker.get(worker_id, worker_id)

    def node_id(self, worker_id: int) -> int:
        """Read a mapped worker's node without guessing a missing GPU index."""
        if worker_id not in self.device_index_by_worker:
            raise ConfigurationError(f"placement has no mapping for worker {worker_id}")
        slot = self.device_index_by_worker[worker_id]
        if type(slot) is not int or slot < 0:
            raise ConfigurationError(
                f"placement worker {worker_id} needs a nonnegative integer GPU slot"
            )
        size = self.topology.levels[0].size
        if type(size) is not int or size < 1:
            raise ConfigurationError(
                "placement node level requires a positive integer size"
            )
        return self.topology.coordinates(slot)[0]

    def layout(self, worker_ids: Sequence[int]) -> GroupLayout:
        return self.topology.group_layout([self.device_index(w) for w in worker_ids])

    def lowest_common_level(self, worker_ids: Sequence[int]) -> int:
        return self.topology.lowest_common_level(
            [self.device_index(w) for w in worker_ids]
        )

    def level_name(self, level_index: int) -> str:
        if level_index < 0:
            return "local"
        return self.topology.levels[level_index].name


class TopologyCatalog:
    def __init__(self, topologies: Iterable[TopologySpec] = ()) -> None:
        self._topologies: dict[str, TopologySpec] = {}
        for topology in topologies:
            self.add(topology)

    def add(self, topology: TopologySpec) -> None:
        if topology.topology_id in self._topologies:
            raise ConfigurationError(f"duplicate topology_id {topology.topology_id!r}")
        self._topologies[topology.topology_id] = topology

    @classmethod
    def load(cls, directory: str | Path) -> "TopologyCatalog":
        directory = Path(directory)
        if not directory.is_dir():
            raise ConfigurationError(
                f"topology catalog directory not found: {directory}"
            )
        catalog = cls()
        for path in sorted(directory.glob("*.yaml")):
            document = load_yaml_mapping(path)
            if "topology_id" not in document:
                # links.yaml and other auxiliary files live in the same directory
                continue
            catalog.add(TopologySpec.from_mapping(document, context=str(path)))
        return catalog

    def get(self, topology_id: str) -> TopologySpec:
        try:
            return self._topologies[topology_id]
        except KeyError as exc:
            raise ConfigurationError(
                f"unknown topology {topology_id!r}; known: {sorted(self._topologies)}"
            ) from exc

    def __contains__(self, topology_id: object) -> bool:
        return topology_id in self._topologies

    def __iter__(self):
        return iter(self._topologies.values())
