from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from TokenSim.errors import ConfigurationError


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ConfigurationError(
            "YAML catalogs require PyYAML; install project requirements"
        ) from exc
    path = Path(path)
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"catalog file not found: {path}") from exc
    except Exception as exc:
        raise ConfigurationError(f"failed to parse {path}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise ConfigurationError(f"{path}: top level must be a mapping")
    return dict(document)


def require(mapping: Mapping[str, Any], field: str, context: str) -> Any:
    if field not in mapping:
        raise ConfigurationError(f"{context}: missing required field {field!r}")
    return mapping[field]


def finite_number(value: Any, field: str, context: str, *, allow_zero=False) -> float:
    minimum = "non-negative" if allow_zero else "positive"
    error = f"{context}: {field} must be finite and {minimum}, got {value!r}"
    if isinstance(value, bool):
        raise ConfigurationError(error)
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError(error) from exc
    if not math.isfinite(number) or number < 0 or (not allow_zero and number == 0):
        raise ConfigurationError(error)
    return number
