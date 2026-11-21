"""Synthetic packet-tail and finite receiver assumptions for AF experiments."""

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass
from TokenSim.config.validation import boolean, integer, number
from TokenSim.errors import ConfigurationError


@dataclass(config=ConfigDict(extra="forbid"))
class AFTransportEffectsConfig:
    enabled: bool = False
    packet_bytes: int = 4096
    packet_tail_probability: float = 0.0
    tail_scale_us: float = 0.0
    tail_shape: float = 2.0
    switch_buffer_bytes: int = 0
    retry_us: float = 0.0
    receive_cohorts: int = 8
    ordered_receive: bool = False
    seed: int = 0

    @model_validator(mode="before")
    @classmethod
    def validate_inputs(cls, value):
        data = value.kwargs if hasattr(value, "kwargs") else value
        if isinstance(data, dict):
            for name in ("enabled", "ordered_receive"):
                boolean(data.get(name, False), name)
            for name, default in (("packet_bytes", 4096), ("receive_cohorts", 8)):
                integer(data.get(name, default), name, minimum=1)
            for name in ("seed", "switch_buffer_bytes"):
                integer(data.get(name, 0), name)
            number(
                data.get("packet_tail_probability", 0),
                "packet tail probability",
                maximum=1,
            )
            number(data.get("tail_scale_us", 0), "tail scale")
            number(data.get("retry_us", 0), "retry duration")
            if number(data.get("tail_shape", 2), "tail shape") <= 1:
                raise ConfigurationError(
                    "tail_shape must exceed one for a finite packet-tail mean"
                )
        return value
