"""Select collective anchors by their measured schedule and placement.

A package can opt a source in to Global Ring matching with this metadata:

    collective_measurement_scopes:
      my-nccl-run:
        schedule: global_ring
        algorithm: ring
        topology_id: my-topology
        ranks: [0, 1, 8, 9]

Ranks are physical device indices in ring order. A source id describes one
communicator placement. Unscoped sources remain eligible for hierarchical
queries. Filter the curve before lookup so interpolation and extrapolation
cannot use anchors from a different schedule or placement.
"""

from collections.abc import Mapping
from dataclasses import replace

from TokenSim.comm.collective_types import CollectiveQuery
from TokenSim.config.validation import integer
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.topology import TopologySpec
from TokenSim.operator_data.lookup import OperatorLookup
from TokenSim.utils import BoundedCache


class ScopedCollectiveMeasurements:
    def __init__(self, topology: TopologySpec, lookup: OperatorLookup):
        self.topology_id = topology.topology_id
        self.lookup = lookup
        self.scopes: dict[str, tuple[str, tuple[int, ...]]] = {}
        self.curves = BoundedCache(32)
        metadata = lookup.package.meta
        scopes = metadata.extra.get("collective_measurement_scopes", {})
        if not isinstance(scopes, Mapping):
            raise ConfigurationError(
                "collective_measurement_scopes must map source IDs to scopes"
            )
        for source, scope in scopes.items():
            if (
                source not in metadata.sources
                or metadata.sources[source].method
                not in {"measured", "imported", "public_benchmark"}
                or not isinstance(scope, Mapping)
                or scope.get("schedule") != "global_ring"
                or scope.get("algorithm") != "ring"
                or not isinstance(scope.get("topology_id"), str)
                or not scope["topology_id"]
                or not isinstance(scope.get("ranks"), (list, tuple))
            ):
                raise ConfigurationError(
                    f"invalid collective measurement scope for {source!r}"
                )
            ranks = tuple(
                integer(rank, "collective measurement rank") for rank in scope["ranks"]
            )
            if len(ranks) < 2 or len(set(ranks)) != len(ranks):
                raise ConfigurationError(
                    f"collective measurement scope for {source!r} requires distinct ranks"
                )
            self.scopes[source] = (scope["topology_id"], ranks)

    def for_query(self, query: CollectiveQuery) -> OperatorLookup | None:
        if not self.scopes:
            return self.lookup if query.schedule == "hierarchical" else None
        scope = (
            (self.topology_id, query.layout.ranks)
            if query.schedule == "global_ring"
            else None
        )
        if scope not in self.curves:
            rows = tuple(
                row
                for row in self.lookup.package.tables.get("collective", ())
                if self.scopes.get(row["source_id"]) == scope
            )
            self.curves[scope] = (
                OperatorLookup(
                    replace(
                        self.lookup.package, tables={"collective": rows}, analysis={}
                    ),
                    self.lookup.policy,
                    self.lookup.analytical_scaler,
                    bind_analytical_axes=self.lookup.bind_analytical_axes,
                )
                if rows
                else None
            )
        return self.curves[scope]
