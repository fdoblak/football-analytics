"""Physical metric definition contracts (Stage 9A — definitions only, no computation)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from football_analytics.physical.types import MetricResultStatus, PhysicalContractError

DISTANCE_DEFINITION: dict[str, Any] = {
    "metric_name": "distance",
    "metric_version": 1,
    "unit": "m",
    "method": "euclidean_2d_pitch_same_segment",
    "gap_bridge": False,
    "not_meanings": [
        "3d_body_motion",
        "energy_effort",
        "invisible_time_estimate",
        "outside_calibration_distance",
    ],
}

SPEED_DEFINITION: dict[str, Any] = {
    "metric_name": "speed",
    "metric_version": 1,
    "canonical_unit": "m_s",
    "display_unit_optional": "km_h",
    "time_source": "video_time_us",
    "forbid_fps_time": True,
}

SPRINT_DEFINITION: dict[str, Any] = {
    "metric_name": "sprint",
    "metric_version": 1,
    "requires_threshold_profile": True,
    "single_frame_not_sprint": True,
    "not_universal_standard": True,
}

HEATMAP_DEFINITION: dict[str, Any] = {
    "metric_name": "heatmap",
    "metric_version": 1,
    "png_not_canonical": True,
    "unseen_not_zero_activity": True,
    "attack_flip_requires_known_direction": True,
}

ACTIVITY_DEFINITION: dict[str, Any] = {
    "metric_name": "activity",
    "metric_version": 1,
    "composite_disabled_by_default": True,
    "low_coverage_not_low_activity": True,
}


def contract_stub_result(
    *,
    metric_name: str,
    unit: str,
    sample_layer: str = "none",
    status: str = MetricResultStatus.CONTRACT_STUB.value,
    reason: str = "STAGE_9A_CONTRACTS_ONLY",
) -> dict[str, Any]:
    if status == MetricResultStatus.COMPUTED.value:
        raise PhysicalContractError("Stage 9A must not emit computed physical metrics")
    return {
        "metric_name": metric_name,
        "unit": unit,
        "value": None,
        "status": status,
        "sample_layer": sample_layer,
        "reason_codes": [reason],
    }


def metric_definition(name: str) -> Mapping[str, Any]:
    table = {
        "distance": DISTANCE_DEFINITION,
        "speed": SPEED_DEFINITION,
        "sprint": SPRINT_DEFINITION,
        "heatmap": HEATMAP_DEFINITION,
        "activity": ACTIVITY_DEFINITION,
    }
    if name not in table:
        raise PhysicalContractError(f"unknown metric definition: {name}")
    return table[name]


__all__ = [
    "DISTANCE_DEFINITION",
    "SPEED_DEFINITION",
    "SPRINT_DEFINITION",
    "HEATMAP_DEFINITION",
    "ACTIVITY_DEFINITION",
    "contract_stub_result",
    "metric_definition",
]
