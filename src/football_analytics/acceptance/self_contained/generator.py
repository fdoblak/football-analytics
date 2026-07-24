"""Generate a bounded deterministic scenario with expected metric vectors."""

from __future__ import annotations

import math
import random
from typing import Any

from football_analytics.acceptance.self_contained.scenario import (
    ScenarioBundle,
    ScenarioConfig,
    ScenarioEvent,
    ScenarioPoint,
)


def _dist(a: ScenarioPoint, b: ScenarioPoint) -> float:
    return math.hypot(b.x_m - a.x_m, b.y_m - a.y_m)


def generate_scenario(config: ScenarioConfig | None = None) -> ScenarioBundle:
    cfg = config or ScenarioConfig()
    rng = random.Random(cfg.seed)
    n_frames = int(cfg.duration_s * cfg.fps)
    traj: list[ScenarioPoint] = []
    x, y = -20.0, 0.0
    for i in range(n_frames):
        t_ms = int(round(1000.0 * i / cfg.fps))
        # smooth progression toward opponent box with small noise
        x = min(45.0, x + 0.045 + rng.uniform(-0.01, 0.01))
        y = max(-30.0, min(30.0, y + rng.uniform(-0.08, 0.08)))
        visible = not (40 <= i % 200 <= 48)  # short occlusion gaps
        traj.append(ScenarioPoint(t_ms=t_ms, x_m=x, y_m=y, visible=visible))

    events = [
        ScenarioEvent(8_000, "Pass", cfg.target_player_id, cfg.team_side, {"outcome": "completed"}),
        ScenarioEvent(12_000, "Pass", cfg.target_player_id, cfg.team_side, {"outcome": "failed"}),
        ScenarioEvent(18_000, "High Pass", cfg.target_player_id, cfg.team_side, {}),
        ScenarioEvent(28_000, "Drive", cfg.target_player_id, cfg.team_side, {}),
        ScenarioEvent(
            36_000, "Take-On", cfg.target_player_id, cfg.team_side, {"outcome": "success"}
        ),
        ScenarioEvent(44_000, "Take-On", cfg.target_player_id, cfg.team_side, {"outcome": "fail"}),
        ScenarioEvent(52_000, "Duel", cfg.target_player_id, cfg.team_side, {"outcome": "won"}),
        ScenarioEvent(60_000, "Player Successful Tackle", cfg.target_player_id, cfg.team_side, {}),
        ScenarioEvent(68_000, "Recovery", cfg.target_player_id, cfg.team_side, {}),
        ScenarioEvent(76_000, "Turnover", cfg.target_player_id, cfg.team_side, {}),
        ScenarioEvent(84_000, "Header", cfg.target_player_id, cfg.team_side, {}),
        ScenarioEvent(92_000, "Clearance", cfg.target_player_id, cfg.team_side, {}),
        ScenarioEvent(100_000, "Box Contact", cfg.target_player_id, cfg.team_side, {}),
        ScenarioEvent(
            108_000, "Pass", cfg.target_player_id, cfg.team_side, {"outcome": "completed"}
        ),
    ]

    camera_intervals = [
        {"t0_ms": 0, "t1_ms": 30_000, "view": "main", "playable": True},
        {"t0_ms": 30_000, "t1_ms": 34_000, "view": "replay", "playable": False},
        {"t0_ms": 34_000, "t1_ms": 90_000, "view": "main", "playable": True},
        {"t0_ms": 90_000, "t1_ms": 95_000, "view": "graphics", "playable": False},
        {"t0_ms": 95_000, "t1_ms": 120_000, "view": "main", "playable": True},
    ]

    visible_pts = [p for p in traj if p.visible]
    distance = 0.0
    speeds: list[float] = []
    for a, b in zip(visible_pts, visible_pts[1:], strict=False):
        d = _dist(a, b)
        distance += d
        dt = max(1e-6, (b.t_ms - a.t_ms) / 1000.0)
        speeds.append(d / dt)
    sprint_count = sum(1 for s in speeds if s >= 7.0)
    box_presence = sum(1 for p in visible_pts if p.x_m >= 36.0)
    coverage = len(visible_pts) / max(1, len(traj))

    passes = [e for e in events if e.label == "Pass"]
    completed = sum(1 for e in passes if e.meta.get("outcome") == "completed")
    expected: dict[str, Any] = {
        "analyzed_duration_s": cfg.duration_s,
        "visibility_coverage": coverage,
        "measured_distance_m": distance,
        "mean_speed_m_s": (sum(speeds) / len(speeds)) if speeds else 0.0,
        "peak_speed_m_s": max(speeds) if speeds else 0.0,
        "sprint_count": sprint_count,
        "pass_attempts": len(passes),
        "pass_completed": completed,
        "pass_accuracy": completed / len(passes) if passes else 0.0,
        "long_pass_attempts": sum(1 for e in events if e.label == "High Pass"),
        "successful_take_ons": sum(
            1 for e in events if e.label == "Take-On" and e.meta.get("outcome") == "success"
        ),
        "failed_take_ons": sum(
            1 for e in events if e.label == "Take-On" and e.meta.get("outcome") == "fail"
        ),
        "duels_won": sum(1 for e in events if e.label == "Duel" and e.meta.get("outcome") == "won"),
        "tackles": sum(1 for e in events if e.label == "Player Successful Tackle"),
        "recoveries": sum(1 for e in events if e.label == "Recovery"),
        "turnovers": sum(1 for e in events if e.label == "Turnover"),
        "aerials": sum(1 for e in events if e.label == "Header"),
        "clearances": sum(1 for e in events if e.label == "Clearance"),
        "penalty_area_contacts": sum(1 for e in events if e.label == "Box Contact"),
        "box_presence_frames": box_presence,
        "activity_index": len(events) / cfg.duration_s,
        "n_events": len(events),
        "n_trajectory_points": len(traj),
    }

    notes = [
        "Deterministic synthetic scenario for contract/lineage/metric arithmetic tests.",
        "Not real football video; not SoccerTrack/TeamTrack accuracy evidence.",
    ]
    return ScenarioBundle(
        config=cfg,
        events=events,
        trajectory=traj,
        expected_metrics=expected,
        camera_intervals=camera_intervals,
        notes=notes,
    )


__all__ = ["generate_scenario"]
