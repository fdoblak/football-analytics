"""Image↔pitch correspondence construction from calibration_features (Stage 8C)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from football_analytics.calibration.pitch_template import PitchTemplate
from football_analytics.calibration.types import FeatureType, Suitability

# Known planar line-pair → pitch keypoint (template ids).
LINE_INTERSECTION_PITCH: dict[frozenset[str], str] = {
    frozenset({"touchline_left", "goalline_a"}): "corner_a_left",
    frozenset({"touchline_right", "goalline_a"}): "corner_a_right",
    frozenset({"touchline_left", "goalline_b"}): "corner_b_left",
    frozenset({"touchline_right", "goalline_b"}): "corner_b_right",
    frozenset({"touchline_left", "halfway_line"}): "halfway_left",
    frozenset({"touchline_right", "halfway_line"}): "halfway_right",
}


@dataclass(frozen=True)
class Correspondence:
    correspondence_id: str
    source_type: str  # keypoint | line_intersection | hybrid
    feature_ids: tuple[str, ...]
    canonical_pitch_feature_id: str
    image_x: float
    image_y: float
    pitch_x_m: float
    pitch_y_m: float
    score: float | None
    quality: float | None
    reason_codes: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "correspondence_id": self.correspondence_id,
            "source_type": self.source_type,
            "feature_ids": list(self.feature_ids),
            "canonical_pitch_feature_id": self.canonical_pitch_feature_id,
            "image_x": self.image_x,
            "image_y": self.image_y,
            "pitch_x_m": self.pitch_x_m,
            "pitch_y_m": self.pitch_y_m,
            "score": self.score,
            "quality": self.quality,
            "reason_codes": list(self.reason_codes),
            "provenance": dict(self.provenance),
        }


@dataclass
class CorrespondenceBuildResult:
    accepted: list[Correspondence]
    rejected: list[dict[str, Any]]
    stats: dict[str, int]


def template_keypoint_index(template: PitchTemplate) -> dict[str, tuple[float, float]]:
    return {p.feature_id: (float(p.x_m), float(p.y_m)) for p in template.keypoints}


def template_line_index(
    template: PitchTemplate,
) -> dict[str, tuple[float, float, float, float]]:
    return {
        ln.feature_id: (float(ln.x1_m), float(ln.y1_m), float(ln.x2_m), float(ln.y2_m))
        for ln in template.lines
    }


def _finite_xy(x: Any, y: Any) -> bool:
    try:
        fx, fy = float(x), float(y)
    except (TypeError, ValueError):
        return False
    return math.isfinite(fx) and math.isfinite(fy)


def _in_image_bounds(x: float, y: float, *, width: float | None, height: float | None) -> bool:
    if width is None or height is None:
        return True
    return 0.0 <= x < width and 0.0 <= y < height


def _line_length(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(math.hypot(x2 - x1, y2 - y1))


def _line_intersection(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float] | None:
    x1, y1, x2, y2 = a
    x3, y3, x4, y4 = b
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    if not (math.isfinite(px) and math.isfinite(py)):
        return None
    return float(px), float(py)


def _abs_sin_angle(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    ax, ay = a[2] - a[0], a[3] - a[1]
    bx, by = b[2] - b[0], b[3] - b[1]
    na = math.hypot(ax, ay)
    nb = math.hypot(bx, by)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    cross = abs(ax * by - ay * bx) / (na * nb)
    return float(min(1.0, max(0.0, cross)))


def _rank_key(score: float | None, feature_id: str) -> tuple[float, str]:
    return (-(float(score) if score is not None else -1.0), feature_id)


def build_correspondences_from_features(
    features: Sequence[Mapping[str, Any]],
    *,
    template: PitchTemplate,
    config: Mapping[str, Any],
    image_width: float | None = None,
    image_height: float | None = None,
    mode: str = "hybrid",  # keypoint_only | line_intersection_only | hybrid
) -> CorrespondenceBuildResult:
    """Build unique image↔pitch correspondences from mapped canonical features."""
    corr_cfg = config["correspondence"]
    kp_index = template_keypoint_index(template)
    accepted_raw: list[Correspondence] = []
    rejected: list[dict[str, Any]] = []
    stats = {
        "input": len(features),
        "unknown": 0,
        "low_score": 0,
        "unsuitable": 0,
        "oob": 0,
        "nan": 0,
        "duplicate_canonical": 0,
        "duplicate_image": 0,
        "keypoints_accepted": 0,
        "intersections_accepted": 0,
        "intersections_unstable": 0,
        "accepted": 0,
        "rejected": 0,
    }

    keypoints: list[Mapping[str, Any]] = []
    lines: list[Mapping[str, Any]] = []
    for feat in features:
        ft = str(feat.get("feature_type"))
        if ft == FeatureType.KEYPOINT.value:
            keypoints.append(feat)
        elif ft == FeatureType.LINE.value:
            lines.append(feat)

    # --- keypoints ---
    if mode in {"keypoint_only", "hybrid"}:
        ranked_kp = sorted(
            keypoints,
            key=lambda f: _rank_key(
                float(f["score"]) if f.get("score") is not None else None,
                str(f.get("feature_id", "")),
            ),
        )
        for feat in ranked_kp:
            fid = str(feat.get("feature_id", ""))
            canon = feat.get("canonical_pitch_feature_id")
            if canon is None or canon == "":
                stats["unknown"] += 1
                rejected.append(
                    {"feature_id": fid, "reason": "UNKNOWN_OR_UNMAPPED", "source": "keypoint"}
                )
                continue
            canon_s = str(canon)
            if canon_s not in kp_index:
                stats["unknown"] += 1
                rejected.append(
                    {"feature_id": fid, "reason": "UNKNOWN_CANONICAL_ID", "source": "keypoint"}
                )
                continue
            suit = str(feat.get("suitability", Suitability.UNKNOWN.value))
            if corr_cfg["reject_unsuitable"] and suit == Suitability.UNSUITABLE.value:
                stats["unsuitable"] += 1
                rejected.append({"feature_id": fid, "reason": "UNSUITABLE", "source": "keypoint"})
                continue
            if not corr_cfg["allow_marginal"] and suit == Suitability.MARGINAL.value:
                stats["unsuitable"] += 1
                rejected.append({"feature_id": fid, "reason": "MARGINAL", "source": "keypoint"})
                continue
            score = feat.get("score")
            if score is not None and float(score) < float(corr_cfg["min_keypoint_score"]):
                stats["low_score"] += 1
                rejected.append({"feature_id": fid, "reason": "LOW_SCORE", "source": "keypoint"})
                continue
            if not _finite_xy(feat.get("image_x"), feat.get("image_y")):
                stats["nan"] += 1
                rejected.append({"feature_id": fid, "reason": "NON_FINITE", "source": "keypoint"})
                continue
            ix, iy = float(feat["image_x"]), float(feat["image_y"])
            if not _in_image_bounds(ix, iy, width=image_width, height=image_height):
                stats["oob"] += 1
                rejected.append(
                    {"feature_id": fid, "reason": "OUT_OF_BOUNDS", "source": "keypoint"}
                )
                continue
            px, py = kp_index[canon_s]
            accepted_raw.append(
                Correspondence(
                    correspondence_id=f"kp:{fid}",
                    source_type="keypoint",
                    feature_ids=(fid,),
                    canonical_pitch_feature_id=canon_s,
                    image_x=ix,
                    image_y=iy,
                    pitch_x_m=px,
                    pitch_y_m=py,
                    score=float(score) if score is not None else None,
                    quality=None,
                    reason_codes=(),
                    provenance={
                        "feature_type": "keypoint",
                        "status": feat.get("status"),
                        "suitability": suit,
                        "raw_score_not_calibrated_probability": True,
                    },
                )
            )
            stats["keypoints_accepted"] += 1

    # --- line intersections ---
    if mode in {"line_intersection_only", "hybrid"} and corr_cfg["line_intersection"]["enabled"]:
        li_cfg = corr_cfg["line_intersection"]
        mapped_lines: list[tuple[Mapping[str, Any], str, tuple[float, float, float, float]]] = []
        for feat in lines:
            fid = str(feat.get("feature_id", ""))
            canon = feat.get("canonical_pitch_feature_id")
            if canon is None or canon == "":
                stats["unknown"] += 1
                rejected.append(
                    {"feature_id": fid, "reason": "UNKNOWN_OR_UNMAPPED", "source": "line"}
                )
                continue
            if not all(
                _finite_xy(feat.get(a), feat.get(b))
                for a, b in (("line_x1", "line_y1"), ("line_x2", "line_y2"))
            ):
                stats["nan"] += 1
                rejected.append({"feature_id": fid, "reason": "NON_FINITE", "source": "line"})
                continue
            score = feat.get("score")
            if score is not None and float(score) < float(corr_cfg["min_line_score"]):
                stats["low_score"] += 1
                rejected.append({"feature_id": fid, "reason": "LOW_SCORE", "source": "line"})
                continue
            seg = (
                float(feat["line_x1"]),
                float(feat["line_y1"]),
                float(feat["line_x2"]),
                float(feat["line_y2"]),
            )
            if _line_length(*seg) < float(li_cfg["min_segment_length_px"]):
                stats["intersections_unstable"] += 1
                rejected.append({"feature_id": fid, "reason": "SHORT_LINE", "source": "line"})
                continue
            mapped_lines.append((feat, str(canon), seg))

        mapped_lines.sort(key=lambda t: _rank_key(t[0].get("score"), str(t[0].get("feature_id"))))
        for i in range(len(mapped_lines)):
            for j in range(i + 1, len(mapped_lines)):
                fa, ca, sa = mapped_lines[i]
                fb, cb, sb = mapped_lines[j]
                key = frozenset({ca, cb})
                pitch_id = LINE_INTERSECTION_PITCH.get(key)
                if pitch_id is None or pitch_id not in kp_index:
                    continue
                sin_a = _abs_sin_angle(sa, sb)
                if sin_a < float(li_cfg["min_abs_sin_angle"]):
                    stats["intersections_unstable"] += 1
                    rejected.append(
                        {
                            "feature_ids": [fa.get("feature_id"), fb.get("feature_id")],
                            "reason": "NEAR_PARALLEL",
                            "source": "line_intersection",
                        }
                    )
                    continue
                inter = _line_intersection(sa, sb)
                if inter is None:
                    stats["intersections_unstable"] += 1
                    rejected.append(
                        {
                            "feature_ids": [fa.get("feature_id"), fb.get("feature_id")],
                            "reason": "NO_INTERSECTION",
                            "source": "line_intersection",
                        }
                    )
                    continue
                ix, iy = inter
                if not _in_image_bounds(ix, iy, width=image_width, height=image_height):
                    stats["oob"] += 1
                    rejected.append(
                        {
                            "feature_ids": [fa.get("feature_id"), fb.get("feature_id")],
                            "reason": "OUT_OF_BOUNDS",
                            "source": "line_intersection",
                        }
                    )
                    continue
                scores = [
                    float(s)
                    for s in (fa.get("score"), fb.get("score"))
                    if s is not None and math.isfinite(float(s))
                ]
                score_v = float(min(scores)) if scores else None
                if score_v is not None and score_v < float(corr_cfg["min_intersection_score"]):
                    stats["low_score"] += 1
                    rejected.append(
                        {
                            "feature_ids": [fa.get("feature_id"), fb.get("feature_id")],
                            "reason": "LOW_SCORE",
                            "source": "line_intersection",
                        }
                    )
                    continue
                px, py = kp_index[pitch_id]
                accepted_raw.append(
                    Correspondence(
                        correspondence_id=f"ix:{fa.get('feature_id')}+{fb.get('feature_id')}",
                        source_type="line_intersection",
                        feature_ids=(str(fa.get("feature_id")), str(fb.get("feature_id"))),
                        canonical_pitch_feature_id=pitch_id,
                        image_x=ix,
                        image_y=iy,
                        pitch_x_m=px,
                        pitch_y_m=py,
                        score=score_v,
                        quality=sin_a,
                        reason_codes=(),
                        provenance={
                            "line_canonical_ids": [ca, cb],
                            "abs_sin_angle": sin_a,
                            "raw_score_not_calibrated_probability": True,
                        },
                    )
                )
                stats["intersections_accepted"] += 1

    # Deduplicate: keep best score per canonical pitch id and per image location.
    accepted_raw.sort(
        key=lambda c: _rank_key(c.score, c.correspondence_id),
    )
    kept: list[Correspondence] = []
    used_canon: set[str] = set()
    used_img: list[tuple[float, float]] = []
    dup_img = float(corr_cfg["duplicate_image_distance_px"])
    for c in accepted_raw:
        if c.canonical_pitch_feature_id in used_canon:
            stats["duplicate_canonical"] += 1
            rejected.append(
                {
                    "correspondence_id": c.correspondence_id,
                    "reason": "DUPLICATE_CANONICAL",
                    "source": c.source_type,
                }
            )
            continue
        if any(math.hypot(c.image_x - ux, c.image_y - uy) <= dup_img for ux, uy in used_img):
            stats["duplicate_image"] += 1
            rejected.append(
                {
                    "correspondence_id": c.correspondence_id,
                    "reason": "DUPLICATE_IMAGE_POINT",
                    "source": c.source_type,
                }
            )
            continue
        kept.append(c)
        used_canon.add(c.canonical_pitch_feature_id)
        used_img.append((c.image_x, c.image_y))
        if len(kept) >= int(corr_cfg["max_correspondences_per_frame"]):
            break

    # Deterministic order for solver.
    kept.sort(key=lambda c: (c.canonical_pitch_feature_id, c.correspondence_id))
    stats["accepted"] = len(kept)
    stats["rejected"] = len(rejected)
    return CorrespondenceBuildResult(accepted=kept, rejected=rejected, stats=stats)


def correspondences_to_arrays(
    items: Sequence[Correspondence],
) -> tuple[np.ndarray, np.ndarray]:
    if not items:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)
    img = np.array([[c.image_x, c.image_y] for c in items], dtype=np.float64)
    pitch = np.array([[c.pitch_x_m, c.pitch_y_m] for c in items], dtype=np.float64)
    return img, pitch


__all__ = [
    "LINE_INTERSECTION_PITCH",
    "Correspondence",
    "CorrespondenceBuildResult",
    "template_keypoint_index",
    "template_line_index",
    "build_correspondences_from_features",
    "correspondences_to_arrays",
]
