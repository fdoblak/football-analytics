"""Deterministic self-contained acceptance scenario (no network, no real video)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ScenarioConfig:
    seed: int = 16_040
    run_id: str = "run_self_contained_r4"
    video_id: str = "vid_self_contained_r4"
    target_player_id: str = "T24"
    team_side: str = "left"
    jersey_number: int = 24
    pitch_length_m: float = 105.0
    pitch_width_m: float = 68.0
    fps: float = 25.0
    duration_s: float = 120.0
    namespace: str = "self_contained_deterministic_acceptance"

    def fingerprint_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScenarioEvent:
    t_ms: int
    label: str
    player_id: str
    team: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioPoint:
    t_ms: int
    x_m: float
    y_m: float
    visible: bool = True


@dataclass
class ScenarioBundle:
    config: ScenarioConfig
    events: list[ScenarioEvent]
    trajectory: list[ScenarioPoint]
    expected_metrics: dict[str, Any]
    camera_intervals: list[dict[str, Any]]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "self_contained_scenario_v1",
            "config": self.config.fingerprint_payload(),
            "events": [asdict(e) for e in self.events],
            "trajectory": [asdict(p) for p in self.trajectory],
            "expected_metrics": self.expected_metrics,
            "camera_intervals": self.camera_intervals,
            "notes": self.notes,
            "synthetic_disclaimer": (
                "Synthetic/self-contained fixture; not real football footage; "
                "not real-match accuracy evidence."
            ),
        }


__all__ = [
    "ScenarioBundle",
    "ScenarioConfig",
    "ScenarioEvent",
    "ScenarioPoint",
]
