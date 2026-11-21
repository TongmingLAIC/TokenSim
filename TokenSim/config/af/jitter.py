"""Explicit synthetic service variability for AF sensitivity experiments."""

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass
from TokenSim.config.validation import boolean, integer, number


@dataclass(config=ConfigDict(extra="forbid"))
class AFJitterConfig:
    enabled: bool = False
    compute_cv: float = 0.0
    network_cv: float = 0.0
    rank_compute_cv: float = 0.0
    tp_cv: float = 0.0
    seed: int = 0
    tp_rank_cv: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def validate_inputs(cls, value):
        data = value.kwargs if hasattr(value, "kwargs") else value
        if isinstance(data, dict):
            boolean(data.get("enabled", False), "AF jitter enabled")
            integer(data.get("seed", 0), "AF jitter seed")
            for name in (
                "compute_cv",
                "network_cv",
                "rank_compute_cv",
                "tp_cv",
                "tp_rank_cv",
            ):
                number(data.get(name, 0), name, maximum=2)
        return value
