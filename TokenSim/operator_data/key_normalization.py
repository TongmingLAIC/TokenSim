"""Canonical numeric and composite shape fields for operator table keys."""

from numbers import Integral, Real
from typing import Any
import json
import math


def integral_shape(value: Any) -> int:
    """Accept exact integer numerals from CSV and integral numeric values."""
    if type(value) is int:
        return value
    if not isinstance(value, bool):
        if isinstance(value, (Integral, str)):
            return int(value)
        if isinstance(value, Real) and math.isfinite(value):
            result = int(value)
            if value == result:
                return result
    raise ValueError("must be an integer")


def canonical_histogram(value: Any, *, field: str) -> str:
    """Return a stable JSON representation for length/token histograms.

    Tables are often written by parquet/CSV tooling with ``1`` versus ``1.0``
    or with a different pair order.  Canonicalising at both package ingest and
    query time keeps those representations equivalent and hashable.
    """
    raw = value
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{field} must be a JSON list of [value,count] pairs"
            ) from exc
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(f"{field} must be a non-empty list of [value,count] pairs")
    pairs: list[tuple[int, float]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"{field}[{index}] must contain exactly [value,count]")
        try:
            value_i = integral_shape(item[0])
            if isinstance(item[1], bool):
                raise ValueError("count cannot be boolean")
            count_f = float(item[1])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{field}[{index}] values must be numeric") from exc
        if value_i <= 0 or not math.isfinite(count_f) or count_f <= 0:
            raise ValueError(f"{field}[{index}] requires positive value and count")
        pairs.append((value_i, count_f))
    pairs.sort(key=lambda pair: pair[0])
    return json.dumps(pairs, separators=(",", ":"), ensure_ascii=False)


def normalize_table_key_value(table: str, field: str, value: Any) -> Any:
    if (
        table in {"mla_attention", "ragged_generation_attention"}
        and field == "context_lengths"
    ):
        return canonical_histogram(value, field=field)
    if table == "routed_experts" and field == "expert_tokens":
        raw = value
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{field} must be a JSON list of positive counts"
                ) from exc
        if not isinstance(raw, (list, tuple)) or not raw:
            raise ValueError(f"{field} must be a non-empty list of positive counts")
        counts: list[int] = []
        for index, item in enumerate(raw):
            try:
                count = integral_shape(item)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field}[{index}] must be an integer") from exc
            if count <= 0:
                raise ValueError(f"{field}[{index}] must be positive")
            counts.append(count)
        return json.dumps(counts, separators=(",", ":"))
    return value
