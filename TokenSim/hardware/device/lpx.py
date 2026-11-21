"""Optional execution and workspace capabilities for LPX-class devices."""

from collections.abc import Mapping
from dataclasses import dataclass

from TokenSim.config.validation import integer, number
from TokenSim.dtypes import canonicalize_dtype
from TokenSim.errors import ConfigurationError


@dataclass(frozen=True)
class LP30Pipeline:
    activation_dtype: str = "bf16"
    vector_ops_per_s: float = 20e12
    exp_ops_per_s: float = 1e12
    shuffle_bytes_per_s: float = 10e12
    vector_efficiency: float = 0.5
    shuffle_efficiency: float = 0.8
    vector_bytes: int = 320
    attention_tile_keys: int = 256
    fill_drain_us: float = 0.05
    schedule: str = "tiled"

    def __post_init__(self):
        object.__setattr__(
            self,
            "activation_dtype",
            canonicalize_dtype(self.activation_dtype, role="activation_dtype"),
        )
        for name in ("vector_ops_per_s", "exp_ops_per_s", "shuffle_bytes_per_s"):
            number(getattr(self, name), f"LP30 {name}", positive=True)
        for name in ("vector_efficiency", "shuffle_efficiency"):
            number(getattr(self, name), f"LP30 {name}", positive=True, maximum=1)
        for name in ("vector_bytes", "attention_tile_keys"):
            integer(getattr(self, name), f"LP30 {name}", minimum=1)
        number(self.fill_drain_us, "LP30 fill/drain", minimum=0)
        if self.schedule not in {"tiled", "serialized"}:
            raise ConfigurationError("LP30 schedule must be tiled or serialized")


@dataclass(frozen=True)
class LP30Memory:
    prefill_chunk: int = 16
    maximum_context: int = 102000
    attention_tile_keys: int = 256
    tile_buffers: int = 2
    reserve_bytes: int = 5_000_000

    def __post_init__(self):
        for name in (
            "prefill_chunk",
            "maximum_context",
            "attention_tile_keys",
            "tile_buffers",
        ):
            integer(getattr(self, name), f"LP30 memory {name}", minimum=1)
        integer(self.reserve_bytes, "LP30 memory reserve_bytes")
        if self.prefill_chunk > self.maximum_context:
            raise ConfigurationError("LP30 prefill chunk exceeds maximum context")


def read_lpx_policy(device, key, policy_type):
    values = device.analytical.get(key)
    if values is None:
        return None
    if device.family != "groq_tsp":
        raise ConfigurationError(
            f"{device.device_id}: {key} requires the groq_tsp family"
        )
    if not isinstance(values, Mapping):
        raise ConfigurationError(f"{device.device_id}: {key} must be a mapping")
    try:
        return policy_type(**values)
    except (TypeError, ValueError) as error:
        raise ConfigurationError(
            f"{device.device_id}: invalid {key}: {error}"
        ) from error
