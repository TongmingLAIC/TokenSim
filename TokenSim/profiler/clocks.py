"""Timing alignment shared by comparison artifacts."""

import math

from TokenSim.errors import ConfigurationError


def profile_clock_shifts(starts, anchors, *, shared):
    """Align each clock, retaining one physical time base when proven shared."""
    origins = {}
    for clock, start in starts:
        origins[clock] = min(start, origins.get(clock, start))
    shifts = {}
    for clock in origins.keys() & anchors.keys():
        anchor = anchors[clock]
        shift = anchor.get("translation_us") if isinstance(anchor, dict) else None
        if type(shift) not in (int, float) or not math.isfinite(shift):
            raise ConfigurationError("comparison clock translation must be finite")
        shifts[clock] = shift
    if shared and origins:
        known = set(shifts.values())
        if len(known) > 1:
            raise ConfigurationError(
                "shared profile clocks have inconsistent translations"
            )
        shift = next(iter(known)) if known else -min(origins.values())
        return dict.fromkeys(origins, shift)
    return {clock: shifts.get(clock, -start) for clock, start in origins.items()}
