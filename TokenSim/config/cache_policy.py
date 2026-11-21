"""Named combinations of retention, admission and movement policy."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CachePolicyPreset:
    retain: bool = False
    quota: bool = False
    movement: bool = False
    cooperative: bool = False
    forced_wait: float | None = None
    minimum_wait: float = 0.5
    minimum_occupancy: float = 0
    age_seconds: float = 30
    age_tokens: int = 2048
    root_bonus_tokens: int = 1024

    def waiting_score(self, *, hit_tokens, age, root):
        return (
            hit_tokens
            + age / self.age_seconds * self.age_tokens
            + int(root) * self.root_bonus_tokens
        )


PRESETS = {
    "apc": CachePolicyPreset(),
    "noop": CachePolicyPreset(),
    "retain": CachePolicyPreset(retain=True),
    "quota": CachePolicyPreset(quota=True),
    "movement": CachePolicyPreset(movement=True),
    "retain_movement": CachePolicyPreset(retain=True, movement=True),
    "all": CachePolicyPreset(
        retain=True, quota=True, movement=True, minimum_occupancy=0.8
    ),
    "forced_movement": CachePolicyPreset(
        retain=True, quota=True, movement=True, forced_wait=30.0
    ),
    "enhanced_coop": CachePolicyPreset(
        retain=True, quota=True, movement=True, cooperative=True, minimum_occupancy=0.8
    ),
}
