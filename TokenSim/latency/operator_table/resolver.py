"""Primitive operator resolution for ordinary and AF serving."""

from typing import Any, Iterable, Mapping
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.device import DeviceSpec
from TokenSim.operator_data.analytical import Calibration, analytical_model_for
from TokenSim.operator_data.coverage import MissingShapeReport
from TokenSim.operator_data.lookup import (
    FALLBACK_POLICIES,
    LookupPolicy,
    MissingOperatorDataError,
    OperatorLookup,
    normalize_query_key,
)
from TokenSim.operator_data.package import OperatorDataPackage
from TokenSim.utils import BoundedCache
from .stats import OperatorStats, OperatorEstimate


class OperatorResolver:
    """Resolve an operator shape and record its estimate and provenance."""

    _CACHE_LIMIT = 262144

    def __init__(
        self,
        *,
        device: DeviceSpec,
        package: OperatorDataPackage | None = None,
        fallback: str = "table_first",
        lookup_policy: LookupPolicy | None = None,
        gated: bool = True,
        missing_report: MissingShapeReport | None = None,
    ) -> None:
        if fallback not in FALLBACK_POLICIES:
            raise ConfigurationError(
                f"fallback must be one of {FALLBACK_POLICIES}, got {fallback!r}"
            )
        if fallback == "table_only" and package is None:
            raise ConfigurationError(
                f"latency backend 'table_only' requires operator data for device {device.device_id!r}"
            )
        self.device = device
        self.fallback = fallback
        self.gated = gated
        self.package = package if fallback != "analytical_only" else None
        self.analytical = analytical_model_for(device.family)
        # Out-of-range table queries keep the boundary row's measured efficiency
        # and follow the analytical model's growth (see LookupPolicy).
        self.lookup = (
            OperatorLookup(
                self.package,
                lookup_policy,
                analytical_scaler=self.analytical_reference_us,
            )
            if self.package is not None
            else None
        )
        self.stats = OperatorStats(missing=missing_report or MissingShapeReport())
        self._cache = BoundedCache(self._CACHE_LIMIT)
        self._calibrations = {
            table: Calibration.from_mapping(values)
            for table, values in (
                self.package.meta.calibration.items() if self.package else []
            )
        }

    @property
    def dataset_version(self) -> str | None:
        return self.package.dataset_version if self.package else None

    @property
    def operator_backend(self) -> str:
        return self.package.backend if self.package else "analytical"

    def query(
        self, table: str, key: Mapping[str, Any], *, record_stats: bool = True
    ) -> OperatorEstimate:
        normalized_key = {**key, **normalize_query_key(table, key)}
        identity = (table, tuple(sorted(normalized_key.items())))
        return self._resolve(
            table, (normalized_key,), identity, record_stats=record_stats
        )

    def query_moe(
        self, base_key: Mapping[str, Any], distributions: Iterable[str]
    ) -> OperatorEstimate:
        """Try the requested MoE routing distributions in order."""
        labels = tuple(distributions) or ("uniform",)
        keys = []
        for label in labels:
            key = {**base_key, "distribution": label}
            keys.append({**key, **normalize_query_key("moe", key)})
        identity = (
            "moe-distributions",
            tuple(tuple(sorted(key.items())) for key in keys),
        )
        return self._resolve("moe", keys, identity)

    def _resolve(self, table, keys, identity, *, record_stats: bool = True):
        estimate = self._cache.get(identity)
        if estimate is None:
            if self.fallback != "analytical_only":
                estimate = self._measured(table, keys)
            if estimate is None:
                estimate = self.analytical_estimate(table, keys[0], gated=self.gated)
            self._cache[identity] = estimate
        if record_stats:
            self.stats.record(table, estimate.match_type)
        return estimate

    def _measured(self, table, keys):
        if self.lookup is None:
            error = MissingOperatorDataError(
                table, keys[0], "no operator package for this device", "table_absent"
            )
        else:
            for key in keys:
                try:
                    result = self.lookup.lookup(table, key)
                except MissingOperatorDataError as exc:
                    error = exc
                    continue
                return OperatorEstimate(
                    result.latency_us,
                    result.match_type,
                    result.source_id,
                    workload_distribution=(
                        key.get("distribution") if table == "moe" else None
                    ),
                )
        if self.fallback == "table_only":
            raise error
        self.stats.record_missing(error)
        return None

    def analytical_estimate(
        self, table: str, key: Mapping[str, Any], **context: Any
    ) -> OperatorEstimate:
        calibration = self._calibrations.get(table) or self._calibrations.get("default")
        result = self.analytical.estimate(
            table, key, self.device, calibration, **context
        )
        return OperatorEstimate(result.latency_us, "analytical", result.source_id)

    def analytical_reference_us(
        self, table: str, key: Mapping[str, Any]
    ) -> float | None:
        """Uncalibrated analytical latency used as the growth reference for extrapolation."""
        try:
            return self.analytical.estimate(
                table, key, self.device, Calibration(), gated=self.gated
            ).latency_us
        except Exception:
            return None
