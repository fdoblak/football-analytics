"""Pure Stage 5C ball-detection post-processing (no Ultralytics import)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from football_analytics.perception.adapters.base import RawDetectionBox
from football_analytics.perception.adapters.ultralytics_common import normalize_class_name
from football_analytics.perception.candidate_merge import BallCandidate
from football_analytics.perception.human_detection import (
    aspect_ratio,
    bbox_area,
    resolve_source_bbox,
)
from football_analytics.perception.taxonomy import ClassMappingResult, map_model_class
from football_analytics.perception.types import (
    EntityType,
    ReviewStatus,
    RoleLabel,
    RoleSource,
)


@dataclass(frozen=True)
class MappedBallDetection:
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
    candidate_source: str


def filter_raw_ball_boxes(
    raw_boxes: Sequence[RawDetectionBox],
    *,
    confidence_threshold: float,
    filters: Mapping[str, Any],
    frame_width: int,
    frame_height: int,
    model_input_size: int,
    boxes_in_source_space: bool = True,
    taxonomy: Mapping[str, Any] | None = None,
    max_detections: int = 64,
    candidate_source: str = "full_frame",
) -> list[MappedBallDetection]:
    """Threshold + ball-specific geometry filters + taxonomy → ball/unknown only."""
    frame_area = float(frame_width) * float(frame_height)
    if frame_area <= 0:
        return []
    min_w = float(filters["min_width"])
    max_w = float(filters["max_width"])
    min_h = float(filters["min_height"])
    max_h = float(filters["max_height"])
    min_af = float(filters["min_area_fraction"])
    max_af = float(filters["max_area_fraction"])
    max_ar = float(filters["max_aspect_ratio"])

    accepted: list[MappedBallDetection] = []
    next_id = 0
    for raw in raw_boxes:
        if float(raw.score) < float(confidence_threshold):
            continue
        # Reject person explicitly even if mis-routed.
        cnorm = normalize_class_name(str(raw.class_name))
        if cnorm in {"person", "human", "pedestrian"}:
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
        w = max(0.0, source[2] - source[0])
        h = max(0.0, source[3] - source[1])
        area = bbox_area(source)
        area_frac = area / frame_area
        if w < min_w or w > max_w or h < min_h or h > max_h:
            continue
        if area_frac < min_af or area_frac > max_af:
            continue
        ar = aspect_ratio(source)
        if ar > max_ar:
            continue

        # Taxonomy names use underscores; COCO uses "sports ball".
        tax_name = cnorm if cnorm else "sports_ball"
        mapped: ClassMappingResult = map_model_class(int(raw.class_id), tax_name, taxonomy=taxonomy)
        if mapped.rejected or not mapped.mapped:
            continue
        if mapped.entity_type != EntityType.BALL:
            continue
        # Never invent player ownership / human roles on ball.
        role = RoleLabel.UNKNOWN
        role_source = mapped.role_source if mapped.role_source else RoleSource.UNKNOWN

        accepted.append(
            MappedBallDetection(
                detection_id=next_id,
                class_id=int(raw.class_id),
                class_name="sports ball",
                confidence=float(raw.score),
                bbox_x1=float(source[0]),
                bbox_y1=float(source[1]),
                bbox_x2=float(source[2]),
                bbox_y2=float(source[3]),
                quality_flags=tuple(flags),
                entity_type=EntityType.BALL,
                role_label=role,
                role_source=role_source,
                mapping_note=mapped.rule_note,
                candidate_source=candidate_source,
            )
        )
        next_id += 1
        if len(accepted) >= int(max_detections):
            break
    return accepted


def mapped_to_candidates(mapped: Sequence[MappedBallDetection]) -> list[BallCandidate]:
    return [
        BallCandidate(
            x1=m.bbox_x1,
            y1=m.bbox_y1,
            x2=m.bbox_x2,
            y2=m.bbox_y2,
            score=m.confidence,
            class_id=m.class_id,
            class_name=m.class_name,
            candidate_source=m.candidate_source,
        )
        for m in mapped
    ]


def candidates_to_mapped(
    candidates: Sequence[BallCandidate],
    *,
    taxonomy: Mapping[str, Any] | None = None,
) -> list[MappedBallDetection]:
    out: list[MappedBallDetection] = []
    for i, c in enumerate(candidates):
        tax_name = normalize_class_name(c.class_name) or "sports_ball"
        mapped = map_model_class(int(c.class_id), tax_name, taxonomy=taxonomy)
        out.append(
            MappedBallDetection(
                detection_id=i,
                class_id=int(c.class_id),
                class_name="sports ball",
                confidence=float(c.score),
                bbox_x1=float(c.x1),
                bbox_y1=float(c.y1),
                bbox_x2=float(c.x2),
                bbox_y2=float(c.y2),
                quality_flags=(),
                entity_type=EntityType.BALL,
                role_label=RoleLabel.UNKNOWN,
                role_source=(mapped.role_source if mapped.mapped else RoleSource.UNKNOWN),
                mapping_note=mapped.rule_note,
                candidate_source=c.candidate_source,
            )
        )
    return out


def build_detection_rows(
    mapped: Sequence[MappedBallDetection],
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
                "quality_flags": list(det.quality_flags)
                + ([f"src:{det.candidate_source}"] if det.candidate_source else []),
            }
        )
    return rows


def build_attribute_rows(
    mapped: Sequence[MappedBallDetection],
    *,
    run_id: str,
    video_id: str,
    frame_index: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for det in mapped:
        prov = {
            "mapping": "detection_taxonomy",
            "role_policy": "ball_role_unknown",
            "candidate_source": det.candidate_source,
            "note": det.mapping_note,
            "no_player_ownership": True,
        }
        rows.append(
            {
                "run_id": run_id,
                "video_id": video_id,
                "frame_index": int(frame_index),
                "detection_id": int(det.detection_id),
                "entity_type": EntityType.BALL.value,
                "role_label": RoleLabel.UNKNOWN.value,
                "role_source": det.role_source.value,
                "role_score": None,
                "occlusion": None,
                "truncation": None,
                "visibility": None,
                "review_status": ReviewStatus.UNREVIEWED.value,
                "attribute_source_ref": "ball_detection_baseline",
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
    "MappedBallDetection",
    "filter_raw_ball_boxes",
    "mapped_to_candidates",
    "candidates_to_mapped",
    "build_detection_rows",
    "build_attribute_rows",
    "coverage_from_boxes",
]
