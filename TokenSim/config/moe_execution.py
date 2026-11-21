"""Serving execution choices, separate from the model's architecture."""

from dataclasses import dataclass

from TokenSim.errors import ConfigurationError


@dataclass(frozen=True)
class MoEExecutionConfig:
    backend: str = "generic"
    shared_expert_fusion: bool = True
    table: str | None = None

    def __post_init__(self):
        if self.backend not in ("generic", "deep_gemm_amxf4_mega_moe"):
            raise ConfigurationError(
                f"unsupported MoE execution backend: {self.backend}"
            )

    @classmethod
    def from_value(cls, value=None):
        if value is None:
            return cls()
        return value if isinstance(value, cls) else cls(**value)

    @property
    def native(self):
        return self.backend == "deep_gemm_amxf4_mega_moe"

    @property
    def weight_layout(self):
        return "mega_moe" if self.native else "fused"

    @classmethod
    def for_compressed_memory(cls, parallel, value):
        if value is None and parallel.enable_expert_parallel:
            return cls(backend="deep_gemm_amxf4_mega_moe")
        return cls.from_value(value)
