"""Physical assumptions for the experimental grouped L2/L3 connector."""

from dataclasses import dataclass

from TokenSim.config.validation import integer, number, boolean
from TokenSim.errors import ConfigurationError


@dataclass(frozen=True)
class TieredConfig:
    l2_bytes_per_rank: int = 64 << 30
    l3_bytes: int = 2 << 40
    staging_bytes_per_rank: int = 64 << 20
    write_policy: str = "write_through"
    promote_reads: bool = True
    read_priority: bool = False
    high_watermark: float = 0.8
    low_watermark: float = 0.6
    network_bytes_per_second: float = 100e9
    network_latency_us: float = 10
    backend: str = "dram"
    backend_bytes_per_second: float = 100e9
    backend_latency_us: float = 1
    transfer_utilization: float = 0.85
    drain_tail: bool = True
    watermark_eviction: bool = True

    def __post_init__(self):
        for name in ("l2_bytes_per_rank", "l3_bytes", "staging_bytes_per_rank"):
            integer(getattr(self, name), name, minimum=1)
        if self.staging_bytes_per_rank >= self.l2_bytes_per_rank:
            raise ConfigurationError("L2 budget must exceed its staging reservation")
        for name in (
            "promote_reads",
            "read_priority",
            "drain_tail",
            "watermark_eviction",
        ):
            boolean(getattr(self, name), name)
        for name in ("network_bytes_per_second", "backend_bytes_per_second"):
            number(getattr(self, name), name, positive=True)
        for name in ("network_latency_us", "backend_latency_us"):
            number(getattr(self, name), name)
        number(
            self.transfer_utilization, "transfer_utilization", positive=True, maximum=1
        )
        number(self.low_watermark, "low_watermark", positive=True, maximum=1)
        number(self.high_watermark, "high_watermark", positive=True, maximum=1)
        if self.low_watermark >= self.high_watermark:
            raise ConfigurationError("low_watermark must be below high_watermark")
        if self.write_policy not in (
            "write_through",
            "write_back",
            "write_around",
            "selective_write_back",
        ):
            raise ConfigurationError("Unsupported tiered write policy")
        if self.backend not in ("dram", "ssd"):
            raise ConfigurationError("L3 backend must be dram or ssd")

    @classmethod
    def from_mapping(cls, raw):
        try:
            return cls(**dict(raw))
        except TypeError as error:
            raise ConfigurationError(f"Invalid tiered connector: {error}") from error
