"""Validated online policy inputs for queued uniform KV storage."""

from dataclasses import dataclass

from TokenSim.config.validation import integer, number, boolean
from TokenSim.errors import ConfigurationError


@dataclass(frozen=True)
class OnlineStoragePolicy:
    reuse_threshold: int = 1
    read_priority: bool = False
    write_bytes_per_second: float | None = None
    write_burst_bytes: int = 8 << 30
    max_backlog_bytes: int = 64 << 30
    save_batch_bytes: int = 256 << 20
    prefetch: bool = False
    prefetch_budget_bytes: int = 16 << 30
    prefetch_lead_seconds: float = 0.1
    history_entries: int = 1000000
    history_seconds: float = 300.0
    backend_includes_transport: bool = True

    def __post_init__(self):
        for name in (
            "reuse_threshold",
            "write_burst_bytes",
            "max_backlog_bytes",
            "save_batch_bytes",
            "prefetch_budget_bytes",
            "history_entries",
        ):
            integer(getattr(self, name), name, minimum=1)
        for name in ("read_priority", "prefetch", "backend_includes_transport"):
            boolean(getattr(self, name), name)
        if self.write_bytes_per_second is not None:
            number(self.write_bytes_per_second, "write_bytes_per_second", positive=True)
        number(self.prefetch_lead_seconds, "prefetch_lead_seconds")
        number(self.history_seconds, "history_seconds", positive=True)
        if (
            self.save_batch_bytes > self.write_burst_bytes
            or self.save_batch_bytes > self.max_backlog_bytes
        ):
            raise ConfigurationError(
                "Save batches must fit the write burst and backlog budgets"
            )

    @classmethod
    def from_mapping(cls, raw):
        try:
            return cls(**dict(raw or {}))
        except (TypeError, ValueError) as error:
            raise ConfigurationError(
                f"Invalid online storage policy: {error}"
            ) from error
