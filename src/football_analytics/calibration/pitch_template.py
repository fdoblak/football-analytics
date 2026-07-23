"""Canonical pitch template (configurable; FIFA-range validate; fingerprint)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from football_analytics.calibration.types import CalibrationContractError
from football_analytics.core.hashing import hash_canonical_json

DEFAULT_LENGTH_M = 105.0
DEFAULT_WIDTH_M = 68.0

# FIFA/IFAB permissive validation ranges (not an official-size claim).
FIFA_LENGTH_MIN = 90.0
FIFA_LENGTH_MAX = 120.0
FIFA_WIDTH_MIN = 45.0
FIFA_WIDTH_MAX = 90.0


@dataclass(frozen=True)
class PitchPoint:
    feature_id: str
    x_m: float
    y_m: float
    kind: str = "keypoint"


@dataclass(frozen=True)
class PitchLine:
    feature_id: str
    x1_m: float
    y1_m: float
    x2_m: float
    y2_m: float


@dataclass(frozen=True)
class PitchTemplate:
    length_m: float
    width_m: float
    keypoints: tuple[PitchPoint, ...]
    lines: tuple[PitchLine, ...]
    centre_circle_radius_m: float
    penalty_area_depth_m: float
    penalty_area_width_m: float
    goal_area_depth_m: float
    goal_area_width_m: float
    penalty_spot_distance_m: float
    corner_arc_radius_m: float
    goal_width_m: float
    origin: str
    x_axis: str
    y_axis: str
    real_size_known: bool
    frame_id: str = "canonical_pitch"

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "length_m": self.length_m,
            "width_m": self.width_m,
            "origin": self.origin,
            "x_axis": self.x_axis,
            "y_axis": self.y_axis,
            "real_size_known": self.real_size_known,
            "centre_circle_radius_m": self.centre_circle_radius_m,
            "penalty_area_depth_m": self.penalty_area_depth_m,
            "penalty_area_width_m": self.penalty_area_width_m,
            "goal_area_depth_m": self.goal_area_depth_m,
            "goal_area_width_m": self.goal_area_width_m,
            "penalty_spot_distance_m": self.penalty_spot_distance_m,
            "corner_arc_radius_m": self.corner_arc_radius_m,
            "goal_width_m": self.goal_width_m,
            "keypoints": [
                {"feature_id": p.feature_id, "x_m": p.x_m, "y_m": p.y_m, "kind": p.kind}
                for p in self.keypoints
            ],
            "lines": [
                {
                    "feature_id": ln.feature_id,
                    "x1_m": ln.x1_m,
                    "y1_m": ln.y1_m,
                    "x2_m": ln.x2_m,
                    "y2_m": ln.y2_m,
                }
                for ln in self.lines
            ],
        }


def validate_fifa_range(length_m: float, width_m: float) -> None:
    if not (FIFA_LENGTH_MIN <= length_m <= FIFA_LENGTH_MAX):
        raise CalibrationContractError(
            f"pitch length_m {length_m} outside FIFA validation range "
            f"[{FIFA_LENGTH_MIN}, {FIFA_LENGTH_MAX}]"
        )
    if not (FIFA_WIDTH_MIN <= width_m <= FIFA_WIDTH_MAX):
        raise CalibrationContractError(
            f"pitch width_m {width_m} outside FIFA validation range "
            f"[{FIFA_WIDTH_MIN}, {FIFA_WIDTH_MAX}]"
        )


def _build_keypoints(
    L: float, W: float, *, pa_d: float, pa_w: float, ga_d: float, ga_w: float, ps: float, gw: float
) -> tuple[PitchPoint, ...]:
    pa_y0 = (W - pa_w) / 2.0
    pa_y1 = pa_y0 + pa_w
    ga_y0 = (W - ga_w) / 2.0
    ga_y1 = ga_y0 + ga_w
    goal_y0 = (W - gw) / 2.0
    goal_y1 = goal_y0 + gw
    mid_y = W / 2.0
    return (
        PitchPoint("corner_a_left", 0.0, 0.0),
        PitchPoint("corner_a_right", 0.0, W),
        PitchPoint("corner_b_left", L, 0.0),
        PitchPoint("corner_b_right", L, W),
        PitchPoint("halfway_left", L / 2.0, 0.0),
        PitchPoint("halfway_right", L / 2.0, W),
        PitchPoint("centre_spot", L / 2.0, mid_y),
        PitchPoint("penalty_spot_a", ps, mid_y),
        PitchPoint("penalty_spot_b", L - ps, mid_y),
        PitchPoint("pa_a_near_left", 0.0, pa_y0),
        PitchPoint("pa_a_near_right", 0.0, pa_y1),
        PitchPoint("pa_a_far_left", pa_d, pa_y0),
        PitchPoint("pa_a_far_right", pa_d, pa_y1),
        PitchPoint("pa_b_near_left", L, pa_y0),
        PitchPoint("pa_b_near_right", L, pa_y1),
        PitchPoint("pa_b_far_left", L - pa_d, pa_y0),
        PitchPoint("pa_b_far_right", L - pa_d, pa_y1),
        PitchPoint("ga_a_far_left", ga_d, ga_y0),
        PitchPoint("ga_a_far_right", ga_d, ga_y1),
        PitchPoint("ga_b_far_left", L - ga_d, ga_y0),
        PitchPoint("ga_b_far_right", L - ga_d, ga_y1),
        PitchPoint("goal_a_left_post", 0.0, goal_y0),
        PitchPoint("goal_a_right_post", 0.0, goal_y1),
        PitchPoint("goal_b_left_post", L, goal_y0),
        PitchPoint("goal_b_right_post", L, goal_y1),
    )


def _build_lines(
    L: float, W: float, *, pa_d: float, pa_w: float, ga_d: float, ga_w: float
) -> tuple[PitchLine, ...]:
    pa_y0 = (W - pa_w) / 2.0
    pa_y1 = pa_y0 + pa_w
    ga_y0 = (W - ga_w) / 2.0
    ga_y1 = ga_y0 + ga_w
    return (
        PitchLine("touchline_left", 0.0, 0.0, L, 0.0),
        PitchLine("touchline_right", 0.0, W, L, W),
        PitchLine("goalline_a", 0.0, 0.0, 0.0, W),
        PitchLine("goalline_b", L, 0.0, L, W),
        PitchLine("halfway_line", L / 2.0, 0.0, L / 2.0, W),
        PitchLine("pa_a_far", pa_d, pa_y0, pa_d, pa_y1),
        PitchLine("pa_b_far", L - pa_d, pa_y0, L - pa_d, pa_y1),
        PitchLine("ga_a_far", ga_d, ga_y0, ga_d, ga_y1),
        PitchLine("ga_b_far", L - ga_d, ga_y0, L - ga_d, ga_y1),
    )


def build_pitch_template(
    *,
    length_m: float = DEFAULT_LENGTH_M,
    width_m: float = DEFAULT_WIDTH_M,
    centre_circle_radius_m: float = 9.15,
    penalty_area_depth_m: float = 16.5,
    penalty_area_width_m: float = 40.32,
    goal_area_depth_m: float = 5.5,
    goal_area_width_m: float = 18.32,
    penalty_spot_distance_m: float = 11.0,
    corner_arc_radius_m: float = 1.0,
    goal_width_m: float = 7.32,
    origin: str = "corner_goal_a_touchline_left",
    x_axis: str = "length_toward_goal_b",
    y_axis: str = "width_toward_right_touchline",
    real_size_known: bool = False,
    validate_fifa: bool = True,
) -> PitchTemplate:
    if length_m <= 0 or width_m <= 0:
        raise CalibrationContractError("pitch dimensions must be positive")
    if validate_fifa:
        validate_fifa_range(length_m, width_m)
    kps = _build_keypoints(
        length_m,
        width_m,
        pa_d=penalty_area_depth_m,
        pa_w=penalty_area_width_m,
        ga_d=goal_area_depth_m,
        ga_w=goal_area_width_m,
        ps=penalty_spot_distance_m,
        gw=goal_width_m,
    )
    lines = _build_lines(
        length_m,
        width_m,
        pa_d=penalty_area_depth_m,
        pa_w=penalty_area_width_m,
        ga_d=goal_area_depth_m,
        ga_w=goal_area_width_m,
    )
    return PitchTemplate(
        length_m=float(length_m),
        width_m=float(width_m),
        keypoints=kps,
        lines=lines,
        centre_circle_radius_m=float(centre_circle_radius_m),
        penalty_area_depth_m=float(penalty_area_depth_m),
        penalty_area_width_m=float(penalty_area_width_m),
        goal_area_depth_m=float(goal_area_depth_m),
        goal_area_width_m=float(goal_area_width_m),
        penalty_spot_distance_m=float(penalty_spot_distance_m),
        corner_arc_radius_m=float(corner_arc_radius_m),
        goal_width_m=float(goal_width_m),
        origin=origin,
        x_axis=x_axis,
        y_axis=y_axis,
        real_size_known=bool(real_size_known),
    )


def pitch_template_from_coord_config(cfg: Mapping[str, Any]) -> PitchTemplate:
    pitch = cfg.get("pitch")
    if not isinstance(pitch, Mapping):
        raise CalibrationContractError("coordinate config missing pitch mapping")
    return build_pitch_template(
        length_m=float(pitch.get("default_length_m", DEFAULT_LENGTH_M)),
        width_m=float(pitch.get("default_width_m", DEFAULT_WIDTH_M)),
        centre_circle_radius_m=float(pitch.get("centre_circle_radius_m", 9.15)),
        penalty_area_depth_m=float(pitch.get("penalty_area_depth_m", 16.5)),
        penalty_area_width_m=float(pitch.get("penalty_area_width_m", 40.32)),
        goal_area_depth_m=float(pitch.get("goal_area_depth_m", 5.5)),
        goal_area_width_m=float(pitch.get("goal_area_width_m", 18.32)),
        penalty_spot_distance_m=float(pitch.get("penalty_spot_distance_m", 11.0)),
        corner_arc_radius_m=float(pitch.get("corner_arc_radius_m", 1.0)),
        goal_width_m=float(pitch.get("goal_width_m", 7.32)),
        origin=str(pitch.get("origin", "corner_goal_a_touchline_left")),
        x_axis=str(pitch.get("x_axis", "length_toward_goal_b")),
        y_axis=str(pitch.get("y_axis", "width_toward_right_touchline")),
        real_size_known=False,
        validate_fifa=True,
    )


def pitch_template_fingerprint(template: PitchTemplate) -> str:
    return hash_canonical_json(template.to_dict())


def freeze_template_dict(template: PitchTemplate) -> Mapping[str, Any]:
    return MappingProxyType(template.to_dict())


__all__ = [
    "DEFAULT_LENGTH_M",
    "DEFAULT_WIDTH_M",
    "FIFA_LENGTH_MIN",
    "FIFA_LENGTH_MAX",
    "FIFA_WIDTH_MIN",
    "FIFA_WIDTH_MAX",
    "PitchPoint",
    "PitchLine",
    "PitchTemplate",
    "validate_fifa_range",
    "build_pitch_template",
    "pitch_template_from_coord_config",
    "pitch_template_fingerprint",
    "freeze_template_dict",
]
