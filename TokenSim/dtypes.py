"""Canonical dtype names, byte sizes, role rules, and compute-pipeline mappings."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from TokenSim.errors import ConfigurationError

CANONICAL_DTYPE_ROLES = frozenset(
    {
        "weight_dtype",
        "activation_dtype",
        "kv_cache_dtype",
        "a_to_f_dtype",
        "f_to_a_dtype",
    }
)


@dataclass(frozen=True)
class DTypeSpec:
    canonical_name: str
    size_bytes: int | float
    aliases: tuple[str, ...]
    compute_pipe: str = "fp16"


_UNPACKED_SPECS = (
    DTypeSpec("fp32", 4, ("fp32", "float32", "float"), "fp32"),
    DTypeSpec("fp16", 2, ("fp16", "float16", "half"), "fp16"),
    DTypeSpec("bf16", 2, ("bf16", "bfloat16"), "bf16"),
    DTypeSpec(
        "fp8",
        1,
        (
            "fp8",
            "float8",
            "fp8_e4m3",
            "fp8_static",
            "fp8e4m3",
            "fp8e5m2",
            "float8_e4m3fn",
            "float8_e4m3fnuz",
            "float8_e5m2",
            "float8_e5m2fnuz",
        ),
        "fp8",
    ),
    DTypeSpec("int8", 1, ("int8",), "int8"),
)

_EXTENDED_SPECS = (
    DTypeSpec("tf32", 4.0, ("tf32",), "tf32"),
    DTypeSpec("fp8_block", 1, ("fp8_block",), "fp8"),
    DTypeSpec("int8_wo", 1.0, ("int8_wo",), "fp16"),
    DTypeSpec("int8_sq", 1.0, ("int8_sq", "sq"), "int8"),
    DTypeSpec("int4", 0.5, ("int4",), "int4"),
    DTypeSpec("int4_wo", 0.5, ("int4_wo", "w4a16_awq"), "fp16"),
    DTypeSpec("nvfp4", 0.5, ("nvfp4", "fp4"), "nvfp4"),
    DTypeSpec("int4_a8", 0.5, ("int4_a8", "w4a8", "w4afp8", "w4a8_awq"), "fp8"),
    DTypeSpec(
        "mxfp4",
        0.5,
        ("mxfp4", "w4a8_mxfp4_mxfp8", "w4a8_mxfp4_fp8"),
        "nvfp4",
    ),
    DTypeSpec("mxfp4_wo", 0.5, ("mxfp4_wo", "w4a16_mxfp4"), "fp16"),
)

DTYPE_REGISTRY: Mapping[str, DTypeSpec] = MappingProxyType(
    {spec.canonical_name: spec for spec in _UNPACKED_SPECS}
)
CAST_OPERATIONS = tuple(
    f"cast_{source}_to_{target}"
    for source in DTYPE_REGISTRY
    for target in DTYPE_REGISTRY
    if source != target
)

_ALL_SPECS = {
    spec.canonical_name: spec for spec in (*_UNPACKED_SPECS, *_EXTENDED_SPECS)
}
_ROLE_ALIAS_TO_CANONICAL: Mapping[str, str] = MappingProxyType(
    {
        alias: spec.canonical_name
        for spec in _UNPACKED_SPECS
        for alias in spec.aliases
        if alias not in {"float", "fp8_e4m3", "fp8_static", "fp8e4m3", "fp8e5m2"}
    }
)
_PACKED_WEIGHT_ALIASES: Mapping[str, DTypeSpec] = MappingProxyType(
    {
        alias: _ALL_SPECS[name]
        for name, aliases in (
            ("fp8_block", ("fp8_block",)),
            ("mxfp4", ("mxfp4",)),
            ("mxfp4_wo", ("mxfp4_wo", "w4a16_mxfp4")),
        )
        for alias in aliases
    }
)
_TABLE_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        alias: spec.canonical_name
        for spec in _ALL_SPECS.values()
        for alias in spec.aliases
    }
)

COMPUTE_PIPE_FALLBACKS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "bf16": ("fp16",),
        "fp16": ("bf16",),
        "tf32": ("fp32",),
        "nvfp4": ("fp8", "fp16", "bf16"),
        "int4": ("int8", "fp16", "bf16"),
    }
)


def normalize_dtype(dtype: str) -> str:
    key = str(dtype).strip().lower()
    try:
        return _TABLE_ALIASES[key]
    except KeyError as exc:
        raise ConfigurationError(f"unsupported dtype {dtype!r}") from exc


def dtype_bytes(dtype: str) -> float:
    return float(_ALL_SPECS[normalize_dtype(dtype)].size_bytes)


def compute_dtype(dtype: str) -> str:
    return _ALL_SPECS[normalize_dtype(dtype)].compute_pipe


def resolve_dtype(dtype: str, *, role: str | None = None) -> DTypeSpec:
    role_name = _validate_role(role)
    if not isinstance(dtype, str) or not dtype.strip():
        raise ValueError(f"{role_name} must be a non-empty dtype string")
    value = dtype.strip().lower()
    if role == "weight_dtype" and value in _PACKED_WEIGHT_ALIASES:
        return _PACKED_WEIGHT_ALIASES[value]
    canonical = _ROLE_ALIAS_TO_CANONICAL.get(value)
    if canonical is None:
        supported = ", ".join(DTYPE_REGISTRY)
        raise ValueError(
            f"{role_name}={dtype!r} has no explicit byte size; "
            f"supported canonical dtypes are [{supported}]"
        )
    return DTYPE_REGISTRY[canonical]


def canonicalize_dtype(dtype: str, *, role: str | None = None) -> str:
    return resolve_dtype(dtype, role=role).canonical_name


def dtype_size_bytes(dtype: str, *, role: str | None = None) -> int | float:
    return resolve_dtype(dtype, role=role).size_bytes


def _validate_role(role: str | None) -> str:
    if role is None:
        return "dtype"
    if not isinstance(role, str) or role not in CANONICAL_DTYPE_ROLES:
        raise ValueError(
            f"unknown dtype role {role!r}; expected one of "
            f"{sorted(CANONICAL_DTYPE_ROLES)}"
        )
    return role
