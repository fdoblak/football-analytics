"""Ground-truth ledger helpers for self-contained scenarios."""

from __future__ import annotations

from typing import Any

from football_analytics.acceptance.self_contained.scenario import ScenarioBundle


def build_gt_event_ledger(bundle: ScenarioBundle) -> list[dict[str, Any]]:
    return [
        {
            "t_ms": e.t_ms,
            "label": e.label,
            "player_id": e.player_id,
            "team": e.team,
            "meta": dict(e.meta),
            "provenance": "self_contained_synthetic_gt",
        }
        for e in bundle.events
    ]


def build_gt_identity(bundle: ScenarioBundle) -> dict[str, Any]:
    cfg = bundle.config
    return {
        "target_player_id": cfg.target_player_id,
        "team_side": cfg.team_side,
        "jersey_number": cfg.jersey_number,
        "namespace": cfg.namespace,
        "provenance": "self_contained_synthetic_gt",
    }


def build_gt_calibration(bundle: ScenarioBundle) -> dict[str, Any]:
    cfg = bundle.config
    return {
        "pitch_length_m": cfg.pitch_length_m,
        "pitch_width_m": cfg.pitch_width_m,
        "origin": "pitch_center",
        "units": "metres",
        "provenance": "self_contained_synthetic_gt",
    }


__all__ = [
    "build_gt_calibration",
    "build_gt_event_ledger",
    "build_gt_identity",
]
