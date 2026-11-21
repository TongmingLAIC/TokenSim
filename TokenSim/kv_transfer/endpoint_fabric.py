"""Compatibility imports for endpoint contention modeling."""

from TokenSim.kv_transfer.transport.fabric import (
    EndpointFabric,
    Flow,
    fair_rates,
    positive,
)

__all__ = ["EndpointFabric", "Flow", "fair_rates", "positive"]
