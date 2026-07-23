"""calibration_features validation helpers (Stage 8A)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from football_analytics.calibration.coordinates import IMAGE_FRAME, validate_coordinate_frame_id
from football_analytics.calibration.types import (
    CONTRACT_VERSION,
    CalibrationContractError,
    FeatureStatus,
    FeatureType,
    Suitability,
)


def feature_row(
    *,
    run_id: str,
    video_id: str,
    frame_index: int,
    video_time_us: int,
    feature_id: str,
    feature_type: str,
    coordinate_frame_id: str = IMAGE_FRAME,
    image_x: float | None = None,
    image_y: float | None = None,
    line_x1: float | None = None,
    line_y1: float | None = None,
    line_x2: float | None = None,
    line_y2: float | None = None,
    canonical_pitch_feature_id: str | None = None,
    score: float | None = None,
    confidence: float | None = None,
    source: str = "synthetic",
    model_ref: str | None = None,
    suitability: str = "suitable",
    status: str = "matched",
    manual_review_required: bool = False,
    reason_codes: Sequence[str] | None = None,
    quality_flags: Sequence[str] | None = None,
    provenance_json: str | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "frame_index": int(frame_index),
        "video_time_us": int(video_time_us),
        "feature_id": feature_id,
        "feature_type": feature_type,
        "canonical_pitch_feature_id": canonical_pitch_feature_id,
        "image_x": image_x,
        "image_y": image_y,
        "line_x1": line_x1,
        "line_y1": line_y1,
        "line_x2": line_x2,
        "line_y2": line_y2,
        "coordinate_frame_id": coordinate_frame_id,
        "score": score,
        "confidence": confidence,
        "source": source,
        "model_ref": model_ref,
        "suitability": suitability,
        "status": status,
        "manual_review_required": bool(manual_review_required),
        "reason_codes": list(reason_codes or []),
        "quality_flags": list(quality_flags or []),
        "provenance_json": provenance_json,
        "contract_version": CONTRACT_VERSION,
    }


def _finite_or_none(v: Any, *, label: str) -> None:
    if v is None:
        return
    if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise CalibrationContractError(f"non-finite {label}")


def validate_feature_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    ft = str(out.get("feature_type"))
    if ft not in {t.value for t in FeatureType}:
        raise CalibrationContractError(f"invalid feature_type: {ft}")
    st = str(out.get("status"))
    if st not in {s.value for s in FeatureStatus}:
        raise CalibrationContractError(f"invalid status: {st}")
    su = str(out.get("suitability"))
    if su not in {s.value for s in Suitability}:
        raise CalibrationContractError(f"invalid suitability: {su}")
    validate_coordinate_frame_id(str(out.get("coordinate_frame_id")))

    for key in (
        "image_x",
        "image_y",
        "line_x1",
        "line_y1",
        "line_x2",
        "line_y2",
        "score",
        "confidence",
    ):
        _finite_or_none(out.get(key), label=key)

    # Do not invent canonical pitch matches.
    if out.get("canonical_pitch_feature_id") in ("",):
        raise CalibrationContractError("empty canonical_pitch_feature_id not allowed; use null")

    if ft == FeatureType.KEYPOINT.value:
        if out.get("image_x") is None or out.get("image_y") is None:
            raise CalibrationContractError("keypoint requires image_x/image_y")
    elif ft == FeatureType.LINE.value:
        needed = ("line_x1", "line_y1", "line_x2", "line_y2")
        if any(out.get(k) is None for k in needed):
            raise CalibrationContractError("line requires line endpoints")

    # Keypoint vs line semantics stay separate — reject mixed filled geometry.
    if ft == FeatureType.KEYPOINT.value and any(
        out.get(k) is not None for k in ("line_x1", "line_y1", "line_x2", "line_y2")
    ):
        raise CalibrationContractError("keypoint must not carry line parameters")
    if ft == FeatureType.LINE.value and (
        out.get("image_x") is not None or out.get("image_y") is not None
    ):
        raise CalibrationContractError("line must not carry keypoint image_x/y")

    return out


def validate_feature_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, int, str]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        validated = validate_feature_row(row)
        key = (
            str(validated["run_id"]),
            str(validated["video_id"]),
            int(validated["frame_index"]),
            str(validated["feature_id"]),
        )
        if key in seen:
            raise CalibrationContractError(f"duplicate feature PK: {key}")
        seen.add(key)
        out.append(validated)
    return out


__all__ = [
    "feature_row",
    "validate_feature_row",
    "validate_feature_rows",
]
