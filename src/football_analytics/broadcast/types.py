"""Stage 4A broadcast shot/camera typed contracts (immutable, no inference)."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from football_analytics.core.run_id import validate_run_id
from football_analytics.data.types import SAFE_ID_RE, assert_safe_identifier
from football_analytics.video.types import MappingQuality

CONTRACT_VERSION = 1


class BroadcastError(ValueError):
    """Base error for Stage 4A broadcast contracts."""


class BroadcastContractError(BroadcastError):
    """Contract construction or validation failure."""


class TransitionType(str, Enum):
    HARD_CUT = "hard_cut"
    DISSOLVE = "dissolve"
    FADE = "fade"
    WIPE = "wipe"
    FLASH = "flash"
    UNKNOWN = "unknown"


class DetectionSource(str, Enum):
    MODEL = "model"
    RULE = "rule"
    MANUAL = "manual"
    IMPORTED = "imported"


class ReviewStatus(str, Enum):
    UNREVIEWED = "unreviewed"
    REVIEWED = "reviewed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"


class SegmentStatus(str, Enum):
    ACTIVE = "active"
    GAP_COVERAGE = "gap_coverage"
    INCOMPLETE = "incomplete"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


class ViewFamily(str, Enum):
    MAIN_BROADCAST = "main_broadcast"
    TACTICAL = "tactical"
    GOAL_VIEW = "goal_view"
    REVERSE_ANGLE = "reverse_angle"
    AERIAL = "aerial"
    PLAYER_ISOLATION = "player_isolation"
    CROWD = "crowd"
    BENCH = "bench"
    STUDIO = "studio"
    GRAPHICS = "graphics"
    UNKNOWN = "unknown"


class FramingScale(str, Enum):
    WIDE = "wide"
    MEDIUM = "medium"
    CLOSE_UP = "close_up"
    EXTREME_CLOSE_UP = "extreme_close_up"
    UNKNOWN = "unknown"


class CameraPosition(str, Enum):
    SIDELINE = "sideline"
    BEHIND_GOAL = "behind_goal"
    CORNER = "corner"
    OVERHEAD = "overhead"
    FIELD_LEVEL = "field_level"
    UNKNOWN = "unknown"


class CameraMotion(str, Enum):
    STATIC = "static"
    PAN = "pan"
    TILT = "tilt"
    ZOOM = "zoom"
    COMPOUND = "compound"
    UNSTABLE = "unstable"
    UNKNOWN = "unknown"


class ReplayStatus(str, Enum):
    LIVE = "live"
    REPLAY = "replay"
    REPLAY_TRANSITION = "replay_transition"
    UNKNOWN = "unknown"


class GraphicsStatus(str, Enum):
    NONE = "none"
    PARTIAL_OVERLAY = "partial_overlay"
    DOMINANT_OVERLAY = "dominant_overlay"
    FULL_SCREEN = "full_screen"
    UNKNOWN = "unknown"


class Playability(str, Enum):
    PLAYABLE = "playable"
    PARTIALLY_PLAYABLE = "partially_playable"
    NON_PLAYABLE = "non_playable"
    UNCERTAIN = "uncertain"


class Suitability(str, Enum):
    SUITABLE = "suitable"
    CONDITIONALLY_SUITABLE = "conditionally_suitable"
    UNSUITABLE = "unsuitable"
    UNKNOWN = "unknown"


class ClassificationSource(str, Enum):
    MODEL = "model"
    RULE = "rule"
    MANUAL = "manual"
    IMPORTED = "imported"


NON_PLAYABLE_VIEW_FAMILIES = frozenset(
    {
        ViewFamily.CROWD,
        ViewFamily.BENCH,
        ViewFamily.STUDIO,
        ViewFamily.GRAPHICS,
    }
)


def _require_str(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise BroadcastContractError(f"{label} must be a non-empty string")
    return value


def _require_safe_id(value: Any, *, label: str) -> str:
    text = _require_str(value, label=label)
    try:
        assert_safe_identifier(text)
    except Exception as exc:  # noqa: BLE001
        raise BroadcastContractError(f"{label} must be a safe identifier") from exc
    return text


def _require_safe_id_or_none(value: Any, *, label: str) -> str | None:
    if value is None:
        return None
    return _require_safe_id(value, label=label)


def _require_int(
    value: Any, *, label: str, minimum: int | None = None, allow_none: bool = False
) -> int | None:
    if value is None:
        if allow_none:
            return None
        raise BroadcastContractError(f"{label} must be an int")
    if not isinstance(value, int) or isinstance(value, bool):
        raise BroadcastContractError(f"{label} must be an int")
    if minimum is not None and value < minimum:
        raise BroadcastContractError(f"{label} must be >= {minimum}")
    return value


def _require_float_unit(value: Any, *, label: str, allow_none: bool = False) -> float | None:
    if value is None:
        if allow_none:
            return None
        raise BroadcastContractError(f"{label} must be a float in [0,1]")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BroadcastContractError(f"{label} must be a number")
    f = float(value)
    if not math.isfinite(f) or not (0.0 <= f <= 1.0):
        raise BroadcastContractError(f"{label} must be finite and in [0,1]")
    return f


def _require_provenance(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise BroadcastContractError("provenance_json must be a string or null")
    try:
        obj = json.loads(value)
    except Exception as exc:  # noqa: BLE001
        raise BroadcastContractError("provenance_json must be canonical JSON object") from exc
    if not isinstance(obj, dict):
        raise BroadcastContractError("provenance_json must be a JSON object")
    return value


def _require_enum(enum_cls: type[Enum], value: Any, *, label: str) -> Any:
    if isinstance(value, enum_cls):
        return value
    if not isinstance(value, str):
        raise BroadcastContractError(f"{label} must be a string enum")
    try:
        return enum_cls(value)
    except ValueError as exc:
        raise BroadcastContractError(f"invalid {label}: {value}") from exc


def _require_contract_version(value: Any) -> int:
    v = _require_int(value, label="contract_version", minimum=1)
    assert v is not None
    if v != CONTRACT_VERSION:
        raise BroadcastContractError(f"contract_version must be {CONTRACT_VERSION}")
    return v


def _require_evidence_refs(value: Any) -> tuple[str, ...]:
    if value is None:
        raise BroadcastContractError("evidence_refs must be a list (may be empty)")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise BroadcastContractError("evidence_refs must be a list of strings")
    out: list[str] = []
    for i, item in enumerate(value):
        if item is None:
            raise BroadcastContractError(f"evidence_refs[{i}] is null")
        if not isinstance(item, str):
            raise BroadcastContractError(f"evidence_refs[{i}] must be a string")
        if not item:
            raise BroadcastContractError(f"evidence_refs[{i}] is empty")
        # Safe identifier only — never invent fake URIs
        if not SAFE_ID_RE.fullmatch(item):
            raise BroadcastContractError(f"evidence_refs[{i}] must be a safe identifier")
        out.append(item)
    return tuple(out)


@dataclass(frozen=True)
class ShotBoundary:
    run_id: str
    video_id: str
    boundary_id: str
    boundary_time_us: int
    left_frame_index: int | None
    right_frame_index: int | None
    transition_type: TransitionType
    transition_duration_us: int | None
    confidence: float | None
    detection_source: DetectionSource
    evidence_ref: str | None
    review_status: ReviewStatus
    provenance_json: str | None
    contract_version: int = CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "video_id": self.video_id,
            "boundary_id": self.boundary_id,
            "boundary_time_us": self.boundary_time_us,
            "left_frame_index": self.left_frame_index,
            "right_frame_index": self.right_frame_index,
            "transition_type": self.transition_type.value,
            "transition_duration_us": self.transition_duration_us,
            "confidence": self.confidence,
            "detection_source": self.detection_source.value,
            "evidence_ref": self.evidence_ref,
            "review_status": self.review_status.value,
            "provenance_json": self.provenance_json,
            "contract_version": self.contract_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ShotBoundary:
        validate_run_id(data["run_id"])
        return cls(
            run_id=str(data["run_id"]),
            video_id=_require_safe_id(data["video_id"], label="video_id"),
            boundary_id=_require_safe_id(data["boundary_id"], label="boundary_id"),
            boundary_time_us=_require_int(
                data["boundary_time_us"], label="boundary_time_us", minimum=0
            )
            or 0,
            left_frame_index=_require_int(
                data.get("left_frame_index"), label="left_frame_index", minimum=0, allow_none=True
            ),
            right_frame_index=_require_int(
                data.get("right_frame_index"), label="right_frame_index", minimum=0, allow_none=True
            ),
            transition_type=_require_enum(
                TransitionType, data["transition_type"], label="transition_type"
            ),
            transition_duration_us=_require_int(
                data.get("transition_duration_us"),
                label="transition_duration_us",
                minimum=0,
                allow_none=True,
            ),
            confidence=_require_float_unit(
                data.get("confidence"), label="confidence", allow_none=True
            ),
            detection_source=_require_enum(
                DetectionSource, data["detection_source"], label="detection_source"
            ),
            evidence_ref=_require_safe_id_or_none(data.get("evidence_ref"), label="evidence_ref"),
            review_status=_require_enum(ReviewStatus, data["review_status"], label="review_status"),
            provenance_json=_require_provenance(data.get("provenance_json")),
            contract_version=_require_contract_version(
                data.get("contract_version", CONTRACT_VERSION)
            ),
        )


@dataclass(frozen=True)
class ShotSegment:
    run_id: str
    video_id: str
    shot_id: str
    start_time_us: int
    end_time_us: int
    start_frame_index: int | None
    end_frame_index_exclusive: int | None
    start_boundary_id: str | None
    end_boundary_id: str | None
    duration_us: int
    frame_count: int | None
    timeline_mapping_quality: MappingQuality
    segment_status: SegmentStatus
    provenance_json: str | None
    contract_version: int = CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "video_id": self.video_id,
            "shot_id": self.shot_id,
            "start_time_us": self.start_time_us,
            "end_time_us": self.end_time_us,
            "start_frame_index": self.start_frame_index,
            "end_frame_index_exclusive": self.end_frame_index_exclusive,
            "start_boundary_id": self.start_boundary_id,
            "end_boundary_id": self.end_boundary_id,
            "duration_us": self.duration_us,
            "frame_count": self.frame_count,
            "timeline_mapping_quality": self.timeline_mapping_quality.value,
            "segment_status": self.segment_status.value,
            "provenance_json": self.provenance_json,
            "contract_version": self.contract_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ShotSegment:
        validate_run_id(data["run_id"])
        start = _require_int(data["start_time_us"], label="start_time_us", minimum=0) or 0
        end = _require_int(data["end_time_us"], label="end_time_us", minimum=0) or 0
        duration = _require_int(data["duration_us"], label="duration_us", minimum=1) or 0
        if end <= start:
            raise BroadcastContractError("end_time_us must be > start_time_us")
        if duration != end - start:
            raise BroadcastContractError("duration_us must equal end_time_us - start_time_us")
        return cls(
            run_id=str(data["run_id"]),
            video_id=_require_safe_id(data["video_id"], label="video_id"),
            shot_id=_require_safe_id(data["shot_id"], label="shot_id"),
            start_time_us=start,
            end_time_us=end,
            start_frame_index=_require_int(
                data.get("start_frame_index"), label="start_frame_index", minimum=0, allow_none=True
            ),
            end_frame_index_exclusive=_require_int(
                data.get("end_frame_index_exclusive"),
                label="end_frame_index_exclusive",
                minimum=0,
                allow_none=True,
            ),
            start_boundary_id=_require_safe_id_or_none(
                data.get("start_boundary_id"), label="start_boundary_id"
            ),
            end_boundary_id=_require_safe_id_or_none(
                data.get("end_boundary_id"), label="end_boundary_id"
            ),
            duration_us=duration,
            frame_count=_require_int(
                data.get("frame_count"), label="frame_count", minimum=1, allow_none=True
            ),
            timeline_mapping_quality=_require_enum(
                MappingQuality, data["timeline_mapping_quality"], label="timeline_mapping_quality"
            ),
            segment_status=_require_enum(
                SegmentStatus, data["segment_status"], label="segment_status"
            ),
            provenance_json=_require_provenance(data.get("provenance_json")),
            contract_version=_require_contract_version(
                data.get("contract_version", CONTRACT_VERSION)
            ),
        )


@dataclass(frozen=True)
class CameraViewSegment:
    run_id: str
    video_id: str
    camera_segment_id: str
    shot_id: str | None
    start_time_us: int
    end_time_us: int
    start_frame_index: int | None
    end_frame_index_exclusive: int | None
    view_family: ViewFamily
    framing_scale: FramingScale
    camera_position: CameraPosition
    camera_motion: CameraMotion
    replay_status: ReplayStatus
    graphics_status: GraphicsStatus
    playability: Playability
    calibration_suitability: Suitability
    tracking_suitability: Suitability
    target_identity_suitability: Suitability
    classification_source: ClassificationSource
    confidence: float | None
    coverage: float
    review_status: ReviewStatus
    evidence_refs: tuple[str, ...]
    provenance_json: str | None
    contract_version: int = CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "video_id": self.video_id,
            "camera_segment_id": self.camera_segment_id,
            "shot_id": self.shot_id,
            "start_time_us": self.start_time_us,
            "end_time_us": self.end_time_us,
            "start_frame_index": self.start_frame_index,
            "end_frame_index_exclusive": self.end_frame_index_exclusive,
            "view_family": self.view_family.value,
            "framing_scale": self.framing_scale.value,
            "camera_position": self.camera_position.value,
            "camera_motion": self.camera_motion.value,
            "replay_status": self.replay_status.value,
            "graphics_status": self.graphics_status.value,
            "playability": self.playability.value,
            "calibration_suitability": self.calibration_suitability.value,
            "tracking_suitability": self.tracking_suitability.value,
            "target_identity_suitability": self.target_identity_suitability.value,
            "classification_source": self.classification_source.value,
            "confidence": self.confidence,
            "coverage": self.coverage,
            "review_status": self.review_status.value,
            "evidence_refs": list(self.evidence_refs),
            "provenance_json": self.provenance_json,
            "contract_version": self.contract_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CameraViewSegment:
        validate_run_id(data["run_id"])
        start = _require_int(data["start_time_us"], label="start_time_us", minimum=0) or 0
        end = _require_int(data["end_time_us"], label="end_time_us", minimum=0) or 0
        if end <= start:
            raise BroadcastContractError("end_time_us must be > start_time_us")
        coverage = _require_float_unit(data["coverage"], label="coverage")
        assert coverage is not None
        return cls(
            run_id=str(data["run_id"]),
            video_id=_require_safe_id(data["video_id"], label="video_id"),
            camera_segment_id=_require_safe_id(
                data["camera_segment_id"], label="camera_segment_id"
            ),
            shot_id=_require_safe_id_or_none(data.get("shot_id"), label="shot_id"),
            start_time_us=start,
            end_time_us=end,
            start_frame_index=_require_int(
                data.get("start_frame_index"), label="start_frame_index", minimum=0, allow_none=True
            ),
            end_frame_index_exclusive=_require_int(
                data.get("end_frame_index_exclusive"),
                label="end_frame_index_exclusive",
                minimum=0,
                allow_none=True,
            ),
            view_family=_require_enum(ViewFamily, data["view_family"], label="view_family"),
            framing_scale=_require_enum(FramingScale, data["framing_scale"], label="framing_scale"),
            camera_position=_require_enum(
                CameraPosition, data["camera_position"], label="camera_position"
            ),
            camera_motion=_require_enum(CameraMotion, data["camera_motion"], label="camera_motion"),
            replay_status=_require_enum(ReplayStatus, data["replay_status"], label="replay_status"),
            graphics_status=_require_enum(
                GraphicsStatus, data["graphics_status"], label="graphics_status"
            ),
            playability=_require_enum(Playability, data["playability"], label="playability"),
            calibration_suitability=_require_enum(
                Suitability, data["calibration_suitability"], label="calibration_suitability"
            ),
            tracking_suitability=_require_enum(
                Suitability, data["tracking_suitability"], label="tracking_suitability"
            ),
            target_identity_suitability=_require_enum(
                Suitability,
                data["target_identity_suitability"],
                label="target_identity_suitability",
            ),
            classification_source=_require_enum(
                ClassificationSource, data["classification_source"], label="classification_source"
            ),
            confidence=_require_float_unit(
                data.get("confidence"), label="confidence", allow_none=True
            ),
            coverage=coverage,
            review_status=_require_enum(ReviewStatus, data["review_status"], label="review_status"),
            evidence_refs=_require_evidence_refs(data.get("evidence_refs", [])),
            provenance_json=_require_provenance(data.get("provenance_json")),
            contract_version=_require_contract_version(
                data.get("contract_version", CONTRACT_VERSION)
            ),
        )


__all__ = [
    "CONTRACT_VERSION",
    "BroadcastError",
    "BroadcastContractError",
    "TransitionType",
    "DetectionSource",
    "ReviewStatus",
    "SegmentStatus",
    "ViewFamily",
    "FramingScale",
    "CameraPosition",
    "CameraMotion",
    "ReplayStatus",
    "GraphicsStatus",
    "Playability",
    "Suitability",
    "ClassificationSource",
    "NON_PLAYABLE_VIEW_FAMILIES",
    "MappingQuality",
    "ShotBoundary",
    "ShotSegment",
    "CameraViewSegment",
]
