"""Pure Stage 5B human-detection post-processing (no Ultralytics import)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from football_analytics.perception.adapters.base import RawPersonBox
from football_analytics.perception.taxonomy import ClassMappingResult, map_model_class
from football_analytics.perception.transforms import (
    TransformError,
    build_preprocessing_transform,
    clip_bbox_xyxy,
    inverse_bbox,
    validate_bbox_xyxy,
)
from football_analytics.perception.types import (
    ColorSpace,
    EntityType,
    ReviewStatus,
    RoleLabel,
    RoleSource,
)


@dataclass(frozen=True)
class MappedHumanDetection:
    detection_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    quality_flags: tuple[str, ...]
    entity_type: EntityType
    role_label: RoleLabel
    role_source: RoleSource
    mapping_note: str | None


def aspect_ratio(bbox: Sequence[float]) -> float:
    x1, y1, x2, y2 = (float(v) for v in bbox)
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    if w <= 0 or h <= 0:
        return float("inf")
    return max(w / h, h / w)


def bbox_area(bbox: Sequence[float]) -> float:
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def resolve_source_bbox(
    raw: RawPersonBox,
    *,
    frame_width: int,
    frame_height: int,
    boxes_in_source_space: bool,
    model_input_size: int,
    apply_inverse: bool = False,
) -> tuple[tuple[float, float, float, float], list[str], str | None]:
    """Map raw adapter box → source-frame xyxy with quality flags.

    When Ultralytics returns source-space boxes (default Stage 5B path),
    ``boxes_in_source_space=True`` and inverse is skipped. Inverse is available
    for adapters that emit model-input-space boxes.
    """
    flags: list[str] = []
    try:
        box = validate_bbox_xyxy((raw.x1, raw.y1, raw.x2, raw.y2), allow_clip_check=False)
    except TransformError:
        return (0.0, 0.0, 0.0, 0.0), flags, "INVALID_BBOX"

    if boxes_in_source_space and not apply_inverse:
        source = box
    else:
        transform = build_preprocessing_transform(
            source_width=frame_width,
            source_height=frame_height,
            model_input_width=model_input_size,
            model_input_height=model_input_size,
            color_space=ColorSpace.BGR,
        )
        try:
            source = inverse_bbox(box, transform)
        except TransformError:
            return (0.0, 0.0, 0.0, 0.0), flags, "INVERSE_TRANSFORM_FAILED"

    clipped, was_clipped = clip_bbox_xyxy(
        source, frame_width=frame_width, frame_height=frame_height
    )
    if was_clipped:
        flags.append("clipped_to_frame")
    try:
        validate_bbox_xyxy(clipped, allow_clip_check=False)
    except TransformError:
        return clipped, flags, "INVALID_BBOX"
    return clipped, flags, None


def filter_raw_person_boxes(
    raw_boxes: Sequence[RawPersonBox],
    *,
    confidence_threshold: float,
    minimum_bbox_area: float,
    maximum_aspect_ratio: float,
    frame_width: int,
    frame_height: int,
    model_input_size: int,
    boxes_in_source_space: bool = True,
    taxonomy: Mapping[str, Any] | None = None,
    max_detections: int = 256,
) -> list[MappedHumanDetection]:
    """Threshold, geometry filter, taxonomy map → human/unknown only."""
    accepted: list[MappedHumanDetection] = []
    next_id = 0
    for raw in raw_boxes:
        if float(raw.score) < float(confidence_threshold):
            continue
        source, flags, err = resolve_source_bbox(
            raw,
            frame_width=frame_width,
            frame_height=frame_height,
            boxes_in_source_space=boxes_in_source_space,
            model_input_size=model_input_size,
            apply_inverse=not boxes_in_source_space,
        )
        if err is not None:
            continue
        area = bbox_area(source)
        if area < float(minimum_bbox_area):
            flags = list(flags) + ["below_min_area"]
            continue
        ar = aspect_ratio(source)
        if ar > float(maximum_aspect_ratio):
            flags = list(flags) + ["aspect_ratio_rejected"]
            continue

        mapped: ClassMappingResult = map_model_class(
            int(raw.class_id), str(raw.class_name), taxonomy=taxonomy
        )
        if mapped.rejected or not mapped.mapped:
            continue
        if mapped.entity_type != EntityType.HUMAN:
            # Ball / other → Stage 5C; never emit here.
            continue
        # Hard rule: generic person never upgrades to player/referee/gk.
        role = mapped.role_label
        if str(raw.class_name).strip().lower() in {"person", "human", "pedestrian"}:
            role = RoleLabel.UNKNOWN
        if role == RoleLabel.PLAYER and str(raw.class_name).strip().lower() == "person":
            role = RoleLabel.UNKNOWN

        # Role source from taxonomy mapping (contracts have no model_taxonomy enum).
        role_source = mapped.role_source

        accepted.append(
            MappedHumanDetection(
                detection_id=next_id,
                class_id=int(raw.class_id),
                class_name="person",
                confidence=float(raw.score),
                bbox_x1=float(source[0]),
                bbox_y1=float(source[1]),
                bbox_x2=float(source[2]),
                bbox_y2=float(source[3]),
                quality_flags=tuple(flags),
                entity_type=EntityType.HUMAN,
                role_label=RoleLabel.UNKNOWN,
                role_source=role_source,
                mapping_note=mapped.rule_note,
            )
        )
        next_id += 1
        if len(accepted) >= int(max_detections):
            break
    return accepted


def build_detection_rows(
    mapped: Sequence[MappedHumanDetection],
    *,
    run_id: str,
    video_id: str,
    frame_index: int,
    model_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for det in mapped:
        rows.append(
            {
                "run_id": run_id,
                "video_id": video_id,
                "frame_index": int(frame_index),
                "detection_id": int(det.detection_id),
                "class_id": int(det.class_id),
                "class_name": det.class_name,
                "confidence": float(det.confidence),
                "bbox_x1": float(det.bbox_x1),
                "bbox_y1": float(det.bbox_y1),
                "bbox_x2": float(det.bbox_x2),
                "bbox_y2": float(det.bbox_y2),
                "model_id": model_id,
                "is_interpolated": False,
                "quality_flags": list(det.quality_flags),
            }
        )
    return rows


def build_attribute_rows(
    mapped: Sequence[MappedHumanDetection],
    *,
    run_id: str,
    video_id: str,
    frame_index: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for det in mapped:
        prov = {
            "mapping": "detection_taxonomy",
            "role_policy": "person_never_player",
            "note": det.mapping_note,
        }
        rows.append(
            {
                "run_id": run_id,
                "video_id": video_id,
                "frame_index": int(frame_index),
                "detection_id": int(det.detection_id),
                "entity_type": EntityType.HUMAN.value,
                "role_label": RoleLabel.UNKNOWN.value,
                "role_source": det.role_source.value,
                "role_score": None,
                "occlusion": None,
                "truncation": None,
                "visibility": None,
                "review_status": ReviewStatus.UNREVIEWED.value,
                "attribute_source_ref": "human_detection_baseline",
                "provenance_json": json.dumps(prov, sort_keys=True, separators=(",", ":")),
                "contract_version": 1,
            }
        )
    return rows


def coverage_from_boxes(
    boxes: Sequence[Sequence[float]], *, frame_width: int, frame_height: int
) -> float:
    frame_area = float(frame_width) * float(frame_height)
    if frame_area <= 0:
        return 0.0
    total = sum(bbox_area(b) for b in boxes)
    return max(0.0, min(1.0, total / frame_area))


__all__ = [
    "MappedHumanDetection",
    "aspect_ratio",
    "bbox_area",
    "resolve_source_bbox",
    "filter_raw_person_boxes",
    "build_detection_rows",
    "build_attribute_rows",
    "coverage_from_boxes",
]
