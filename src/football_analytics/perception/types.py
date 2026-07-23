"""Stage 5A perception detection typed contracts (immutable, no inference)."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from football_analytics.core.run_id import validate_run_id
from football_analytics.data.types import assert_safe_identifier

CONTRACT_VERSION = 1
RECEIPT_SCHEMA_VERSION = 1
TRANSFORM_SCHEMA_VERSION = 1

ERROR_CODES = frozenset(
    {
        "FRAME_NOT_ELIGIBLE",
        "BALL_ANALYSIS_NOT_ELIGIBLE",
        "UNKNOWN_PLAYABILITY",
        "MISSING_FRAME",
        "FRAME_DECODE_FAILED",
        "MODEL_NOT_AVAILABLE",
        "MODEL_CLASS_UNMAPPED",
        "INVALID_MODEL_OUTPUT",
        "INVALID_BBOX",
        "INVERSE_TRANSFORM_FAILED",
        "INFERENCE_FAILED",
        "OUTPUT_INTEGRITY_FAILED",
        "MANUAL_REVIEW_REQUIRED",
    }
)


class PerceptionError(ValueError):
    """Base error for Stage 5A perception contracts."""


class PerceptionContractError(PerceptionError):
    """Contract construction or validation failure."""


class EntityType(str, Enum):
    HUMAN = "human"
    BALL = "ball"
    UNKNOWN = "unknown"


class RoleLabel(str, Enum):
    PLAYER = "player"
    GOALKEEPER = "goalkeeper"
    REFEREE = "referee"
    ASSISTANT_REFEREE = "assistant_referee"
    STAFF = "staff"
    UNKNOWN = "unknown"


class RoleSource(str, Enum):
    DETECTOR_NATIVE = "detector_native"
    DOWNSTREAM_CLASSIFIER = "downstream_classifier"
    MANUAL_REVIEW = "manual_review"
    IMPORTED = "imported"
    UNKNOWN = "unknown"


class ProcessingStatus(str, Enum):
    PROCESSED = "processed"
    PROCESSED_NO_DETECTIONS = "processed_no_detections"
    SKIPPED = "skipped"
    FAILED = "failed"
    NOT_ELIGIBLE = "not_eligible"


class Eligibility(str, Enum):
    ELIGIBLE = "eligible"
    CONDITIONALLY_ELIGIBLE = "conditionally_eligible"
    INELIGIBLE = "ineligible"
    UNKNOWN = "unknown"


class ReviewStatus(str, Enum):
    UNREVIEWED = "unreviewed"
    REVIEWED = "reviewed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"


class ResizeMode(str, Enum):
    LETTERBOX = "letterbox"
    STRETCH = "stretch"


class ReceiptStatus(str, Enum):
    SUCCEEDED = "succeeded"
    REJECTED = "rejected"
    FAILED = "failed"
    PARTIAL = "partial"


class ColorSpace(str, Enum):
    RGB = "rgb"
    BGR = "bgr"
    GRAY = "gray"
    YUV = "yuv"
    UNKNOWN = "unknown"


class ChannelOrder(str, Enum):
    CHANNELS_LAST = "channels_last"
    CHANNELS_FIRST = "channels_first"
    UNKNOWN = "unknown"


class Orientation(str, Enum):
    IDENTITY = "identity"
    ROTATED_90 = "rotated_90"
    ROTATED_180 = "rotated_180"
    ROTATED_270 = "rotated_270"
    UNKNOWN = "unknown"


PROCESSED_STATUSES = frozenset(
    {ProcessingStatus.PROCESSED, ProcessingStatus.PROCESSED_NO_DETECTIONS}
)
UNPROCESSED_STATUSES = frozenset(
    {
        ProcessingStatus.SKIPPED,
        ProcessingStatus.FAILED,
        ProcessingStatus.NOT_ELIGIBLE,
    }
)


def _require_str(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PerceptionContractError(f"{label} must be a non-empty string")
    return value


def _require_safe_id(value: Any, *, label: str) -> str:
    text = _require_str(value, label=label)
    try:
        assert_safe_identifier(text)
    except Exception as exc:  # noqa: BLE001
        raise PerceptionContractError(f"{label} must be a safe identifier") from exc
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
        raise PerceptionContractError(f"{label} must be an int")
    if not isinstance(value, int) or isinstance(value, bool):
        raise PerceptionContractError(f"{label} must be an int")
    if minimum is not None and value < minimum:
        raise PerceptionContractError(f"{label} must be >= {minimum}")
    return value


def _require_float_unit(value: Any, *, label: str, allow_none: bool = False) -> float | None:
    if value is None:
        if allow_none:
            return None
        raise PerceptionContractError(f"{label} must be a float in [0,1]")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PerceptionContractError(f"{label} must be a number")
    f = float(value)
    if not math.isfinite(f) or not (0.0 <= f <= 1.0):
        raise PerceptionContractError(f"{label} must be finite and in [0,1]")
    return f


def _require_provenance(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise PerceptionContractError("provenance_json must be a string or null")
    try:
        obj = json.loads(value)
    except Exception as exc:  # noqa: BLE001
        raise PerceptionContractError("provenance_json must be canonical JSON object") from exc
    if not isinstance(obj, dict):
        raise PerceptionContractError("provenance_json must be a JSON object")
    return value


def _require_enum(enum_cls: type[Enum], value: Any, *, label: str) -> Any:
    if isinstance(value, enum_cls):
        return value
    if not isinstance(value, str):
        raise PerceptionContractError(f"{label} must be a string enum")
    try:
        return enum_cls(value)
    except ValueError as exc:
        raise PerceptionContractError(f"invalid {label}: {value}") from exc


def _require_contract_version(value: Any) -> int:
    v = _require_int(value, label="contract_version", minimum=1)
    assert v is not None
    if v != CONTRACT_VERSION:
        raise PerceptionContractError(f"contract_version must be {CONTRACT_VERSION}")
    return v


def _require_sha256_or_none(value: Any, *, label: str) -> str | None:
    if value is None:
        return None
    text = _require_str(value, label=label)
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise PerceptionContractError(f"{label} must be lowercase hex sha256 or null")
    return text


def _require_error_code_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = _require_str(value, label="error_code")
    if text not in ERROR_CODES:
        raise PerceptionContractError(f"unknown error_code: {text}")
    return text


def _require_message_list(value: Any, *, label: str) -> tuple[dict[str, str], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PerceptionContractError(f"{label} must be a list")
    out: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise PerceptionContractError(f"{label} items must be objects")
        code = _require_str(item.get("code"), label=f"{label}.code")
        message = _require_str(item.get("message"), label=f"{label}.message")
        out.append({"code": code, "message": message})
    return tuple(out)


@dataclass(frozen=True)
class DetectionFrameStatus:
    run_id: str
    video_id: str
    frame_index: int
    video_time_us: int
    analysis_window_id: str | None
    processing_status: ProcessingStatus
    eligibility: Eligibility
    detector_id: str
    input_artifact_ref: str | None
    detection_count: int
    human_count: int
    ball_count: int
    skip_reason: str | None
    error_code: str | None
    coverage: float
    provenance_json: str | None
    contract_version: int = CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "video_id": self.video_id,
            "frame_index": self.frame_index,
            "video_time_us": self.video_time_us,
            "analysis_window_id": self.analysis_window_id,
            "processing_status": self.processing_status.value,
            "eligibility": self.eligibility.value,
            "detector_id": self.detector_id,
            "input_artifact_ref": self.input_artifact_ref,
            "detection_count": self.detection_count,
            "human_count": self.human_count,
            "ball_count": self.ball_count,
            "skip_reason": self.skip_reason,
            "error_code": self.error_code,
            "coverage": self.coverage,
            "provenance_json": self.provenance_json,
            "contract_version": self.contract_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DetectionFrameStatus:
        run_id = _require_str(data["run_id"], label="run_id")
        try:
            validate_run_id(run_id)
        except Exception as exc:  # noqa: BLE001
            raise PerceptionContractError("invalid run_id") from exc
        status = _require_enum(
            ProcessingStatus, data["processing_status"], label="processing_status"
        )
        det_count = _require_int(data["detection_count"], label="detection_count", minimum=0)
        human = _require_int(data["human_count"], label="human_count", minimum=0)
        ball = _require_int(data["ball_count"], label="ball_count", minimum=0)
        assert det_count is not None and human is not None and ball is not None
        coverage = _require_float_unit(data["coverage"], label="coverage")
        assert coverage is not None

        if status == ProcessingStatus.PROCESSED_NO_DETECTIONS and (
            det_count != 0 or human != 0 or ball != 0
        ):
            raise PerceptionContractError(
                "processed_no_detections requires zero detection/human/ball counts"
            )
        if status == ProcessingStatus.PROCESSED and det_count <= 0:
            raise PerceptionContractError("processed requires detection_count > 0")
        # Counts must be zero — never invent fake empty detections for skipped/failed.
        if status in UNPROCESSED_STATUSES and (det_count != 0 or human != 0 or ball != 0):
            raise PerceptionContractError(
                f"{status.value} must not carry non-zero detection counts"
            )
        if human + ball > det_count:
            raise PerceptionContractError("human_count + ball_count exceeds detection_count")

        skip_reason = data.get("skip_reason")
        if skip_reason is not None:
            skip_reason = _require_str(skip_reason, label="skip_reason")
        error_code = _require_error_code_or_none(data.get("error_code"))
        if status == ProcessingStatus.FAILED and error_code is None:
            raise PerceptionContractError("failed status requires error_code")
        if (
            status in {ProcessingStatus.SKIPPED, ProcessingStatus.NOT_ELIGIBLE}
            and skip_reason is None
            and error_code is None
        ):
            raise PerceptionContractError(f"{status.value} requires skip_reason or error_code")

        return cls(
            run_id=run_id,
            video_id=_require_safe_id(data["video_id"], label="video_id"),
            frame_index=_require_int(data["frame_index"], label="frame_index", minimum=0) or 0,
            video_time_us=_require_int(data["video_time_us"], label="video_time_us", minimum=0)
            or 0,
            analysis_window_id=_require_safe_id_or_none(
                data.get("analysis_window_id"), label="analysis_window_id"
            ),
            processing_status=status,
            eligibility=_require_enum(Eligibility, data["eligibility"], label="eligibility"),
            detector_id=_require_safe_id(data["detector_id"], label="detector_id"),
            input_artifact_ref=(
                None
                if data.get("input_artifact_ref") is None
                else _require_str(data["input_artifact_ref"], label="input_artifact_ref")
            ),
            detection_count=det_count,
            human_count=human,
            ball_count=ball,
            skip_reason=skip_reason,
            error_code=error_code,
            coverage=coverage,
            provenance_json=_require_provenance(data.get("provenance_json")),
            contract_version=_require_contract_version(
                data.get("contract_version", CONTRACT_VERSION)
            ),
        )


@dataclass(frozen=True)
class DetectionAttributes:
    run_id: str
    video_id: str
    frame_index: int
    detection_id: int
    entity_type: EntityType
    role_label: RoleLabel
    role_source: RoleSource
    role_score: float | None
    occlusion: float | None
    truncation: float | None
    visibility: float | None
    review_status: ReviewStatus
    attribute_source_ref: str | None
    provenance_json: str | None
    contract_version: int = CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "video_id": self.video_id,
            "frame_index": self.frame_index,
            "detection_id": self.detection_id,
            "entity_type": self.entity_type.value,
            "role_label": self.role_label.value,
            "role_source": self.role_source.value,
            "role_score": self.role_score,
            "occlusion": self.occlusion,
            "truncation": self.truncation,
            "visibility": self.visibility,
            "review_status": self.review_status.value,
            "attribute_source_ref": self.attribute_source_ref,
            "provenance_json": self.provenance_json,
            "contract_version": self.contract_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DetectionAttributes:
        run_id = _require_str(data["run_id"], label="run_id")
        try:
            validate_run_id(run_id)
        except Exception as exc:  # noqa: BLE001
            raise PerceptionContractError("invalid run_id") from exc
        entity = _require_enum(EntityType, data["entity_type"], label="entity_type")
        role = _require_enum(RoleLabel, data["role_label"], label="role_label")
        role_source = _require_enum(RoleSource, data["role_source"], label="role_source")
        if entity == EntityType.BALL and role != RoleLabel.UNKNOWN:
            raise PerceptionContractError("ball entity forbids non-unknown role_label")
        return cls(
            run_id=run_id,
            video_id=_require_safe_id(data["video_id"], label="video_id"),
            frame_index=_require_int(data["frame_index"], label="frame_index", minimum=0) or 0,
            detection_id=_require_int(data["detection_id"], label="detection_id", minimum=0) or 0,
            entity_type=entity,
            role_label=role,
            role_source=role_source,
            role_score=_require_float_unit(
                data.get("role_score"), label="role_score", allow_none=True
            ),
            occlusion=_require_float_unit(
                data.get("occlusion"), label="occlusion", allow_none=True
            ),
            truncation=_require_float_unit(
                data.get("truncation"), label="truncation", allow_none=True
            ),
            visibility=_require_float_unit(
                data.get("visibility"), label="visibility", allow_none=True
            ),
            review_status=_require_enum(ReviewStatus, data["review_status"], label="review_status"),
            attribute_source_ref=(
                None
                if data.get("attribute_source_ref") is None
                else _require_str(data["attribute_source_ref"], label="attribute_source_ref")
            ),
            provenance_json=_require_provenance(data.get("provenance_json")),
            contract_version=_require_contract_version(
                data.get("contract_version", CONTRACT_VERSION)
            ),
        )


@dataclass(frozen=True)
class PreprocessingTransform:
    source_width: int
    source_height: int
    model_input_width: int
    model_input_height: int
    resize_mode: ResizeMode
    scale_x: float
    scale_y: float
    pad_left: float
    pad_top: float
    pad_right: float
    pad_bottom: float
    color_space: ColorSpace
    channel_order: ChannelOrder
    normalization: Mapping[str, Any]
    orientation: Orientation
    transform_fingerprint: str
    schema_version: int = TRANSFORM_SCHEMA_VERSION
    roundtrip_tolerance_px: float = 0.5
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "model_input_width": self.model_input_width,
            "model_input_height": self.model_input_height,
            "resize_mode": self.resize_mode.value,
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "pad_left": self.pad_left,
            "pad_top": self.pad_top,
            "pad_right": self.pad_right,
            "pad_bottom": self.pad_bottom,
            "color_space": self.color_space.value,
            "channel_order": self.channel_order.value,
            "normalization": dict(self.normalization),
            "orientation": self.orientation.value,
            "transform_fingerprint": self.transform_fingerprint,
            "roundtrip_tolerance_px": self.roundtrip_tolerance_px,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PreprocessingTransform:
        from football_analytics.perception.transforms import compute_transform_fingerprint

        schema_version = _require_int(
            data.get("schema_version", 1), label="schema_version", minimum=1
        )
        if schema_version != TRANSFORM_SCHEMA_VERSION:
            raise PerceptionContractError(f"schema_version must be {TRANSFORM_SCHEMA_VERSION}")
        sw = _require_int(data["source_width"], label="source_width", minimum=1)
        sh = _require_int(data["source_height"], label="source_height", minimum=1)
        mw = _require_int(data["model_input_width"], label="model_input_width", minimum=1)
        mh = _require_int(data["model_input_height"], label="model_input_height", minimum=1)
        assert sw and sh and mw and mh
        for key in ("scale_x", "scale_y"):
            v = data[key]
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(float(v))
            ):
                raise PerceptionContractError(f"{key} must be finite positive number")
            if float(v) <= 0:
                raise PerceptionContractError(f"{key} must be > 0")
        for key in ("pad_left", "pad_top", "pad_right", "pad_bottom"):
            v = data[key]
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(float(v))
            ):
                raise PerceptionContractError(f"{key} must be finite number")
            if float(v) < 0:
                raise PerceptionContractError(f"{key} must be >= 0")
        norm = data.get("normalization")
        if not isinstance(norm, Mapping) or "kind" not in norm:
            raise PerceptionContractError("normalization.kind required")
        fp = _require_str(data["transform_fingerprint"], label="transform_fingerprint")
        if len(fp) != 64:
            raise PerceptionContractError("transform_fingerprint must be sha256 hex")
        obj = cls(
            source_width=sw,
            source_height=sh,
            model_input_width=mw,
            model_input_height=mh,
            resize_mode=_require_enum(ResizeMode, data["resize_mode"], label="resize_mode"),
            scale_x=float(data["scale_x"]),
            scale_y=float(data["scale_y"]),
            pad_left=float(data["pad_left"]),
            pad_top=float(data["pad_top"]),
            pad_right=float(data["pad_right"]),
            pad_bottom=float(data["pad_bottom"]),
            color_space=_require_enum(ColorSpace, data["color_space"], label="color_space"),
            channel_order=_require_enum(ChannelOrder, data["channel_order"], label="channel_order"),
            normalization=dict(norm),
            orientation=_require_enum(Orientation, data["orientation"], label="orientation"),
            transform_fingerprint=fp,
            schema_version=schema_version or TRANSFORM_SCHEMA_VERSION,
            roundtrip_tolerance_px=float(data.get("roundtrip_tolerance_px", 0.5)),
            notes=None if data.get("notes") is None else _require_str(data["notes"], label="notes"),
        )
        expected = compute_transform_fingerprint(obj)
        if obj.transform_fingerprint != expected:
            raise PerceptionContractError("transform_fingerprint mismatch")
        return obj


@dataclass(frozen=True)
class DetectionRunReceipt:
    receipt_id: str
    run_id: str
    detector_id: str
    model_registry_id: str | None
    model_sha256: str | None
    adapter_id: str
    adapter_version: str
    config_fingerprint: str
    taxonomy_version: str
    source_video_ref: str
    frames_ref: str
    analysis_windows_ref: str | None
    eligible_frame_count: int
    processed_frame_count: int
    skipped_frame_count: int
    failed_frame_count: int
    processed_no_detection_count: int
    total_detection_count: int
    human_detection_count: int
    ball_detection_count: int
    pre_nms_count: int | None
    post_nms_count: int | None
    started_at_utc: str
    completed_at_utc: str
    status: ReceiptStatus
    warnings: tuple[dict[str, str], ...]
    errors: tuple[dict[str, str], ...]
    artifacts: Mapping[str, Any]
    environment_ref: str | None
    provenance: Mapping[str, Any]
    schema_version: int = RECEIPT_SCHEMA_VERSION
    execution_provider: str | None = None
    precision: str | None = None
    software_versions: Mapping[str, str] | None = None
    transform_fingerprint: str | None = None
    threshold_config_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "receipt_id": self.receipt_id,
            "run_id": self.run_id,
            "detector_id": self.detector_id,
            "model_registry_id": self.model_registry_id,
            "model_sha256": self.model_sha256,
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "config_fingerprint": self.config_fingerprint,
            "taxonomy_version": self.taxonomy_version,
            "source_video_ref": self.source_video_ref,
            "frames_ref": self.frames_ref,
            "analysis_windows_ref": self.analysis_windows_ref,
            "eligible_frame_count": self.eligible_frame_count,
            "processed_frame_count": self.processed_frame_count,
            "skipped_frame_count": self.skipped_frame_count,
            "failed_frame_count": self.failed_frame_count,
            "processed_no_detection_count": self.processed_no_detection_count,
            "total_detection_count": self.total_detection_count,
            "human_detection_count": self.human_detection_count,
            "ball_detection_count": self.ball_detection_count,
            "pre_nms_count": self.pre_nms_count,
            "post_nms_count": self.post_nms_count,
            "started_at_utc": self.started_at_utc,
            "completed_at_utc": self.completed_at_utc,
            "status": self.status.value,
            "warnings": [dict(w) for w in self.warnings],
            "errors": [dict(e) for e in self.errors],
            "artifacts": dict(self.artifacts),
            "environment_ref": self.environment_ref,
            "execution_provider": self.execution_provider,
            "precision": self.precision,
            "software_versions": (
                None if self.software_versions is None else dict(self.software_versions)
            ),
            "transform_fingerprint": self.transform_fingerprint,
            "threshold_config_fingerprint": self.threshold_config_fingerprint,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DetectionRunReceipt:
        run_id = _require_str(data["run_id"], label="run_id")
        try:
            validate_run_id(run_id)
        except Exception as exc:  # noqa: BLE001
            raise PerceptionContractError("invalid run_id") from exc
        schema_version = _require_int(
            data.get("schema_version", 1), label="schema_version", minimum=1
        )
        if schema_version != RECEIPT_SCHEMA_VERSION:
            raise PerceptionContractError(f"schema_version must be {RECEIPT_SCHEMA_VERSION}")
        total = _require_int(
            data["total_detection_count"], label="total_detection_count", minimum=0
        )
        human = _require_int(
            data["human_detection_count"], label="human_detection_count", minimum=0
        )
        ball = _require_int(data["ball_detection_count"], label="ball_detection_count", minimum=0)
        assert total is not None and human is not None and ball is not None
        if human + ball > total:
            raise PerceptionContractError("human+ball detection counts exceed total")
        cfg_fp = _require_str(data["config_fingerprint"], label="config_fingerprint")
        if len(cfg_fp) != 64:
            raise PerceptionContractError("config_fingerprint must be sha256 hex")
        provenance = data.get("provenance")
        if not isinstance(provenance, Mapping):
            raise PerceptionContractError("provenance must be an object")
        if provenance.get("stage") != "5A":
            raise PerceptionContractError("provenance.stage must be 5A")
        _require_str(provenance.get("label"), label="provenance.label")
        artifacts = data.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise PerceptionContractError("artifacts must be an object")
        return cls(
            receipt_id=_require_safe_id(data["receipt_id"], label="receipt_id"),
            run_id=run_id,
            detector_id=_require_safe_id(data["detector_id"], label="detector_id"),
            model_registry_id=(
                None
                if data.get("model_registry_id") is None
                else _require_str(data["model_registry_id"], label="model_registry_id")
            ),
            model_sha256=_require_sha256_or_none(data.get("model_sha256"), label="model_sha256"),
            adapter_id=_require_str(data["adapter_id"], label="adapter_id"),
            adapter_version=_require_str(data["adapter_version"], label="adapter_version"),
            config_fingerprint=cfg_fp,
            taxonomy_version=_require_str(data["taxonomy_version"], label="taxonomy_version"),
            source_video_ref=_require_str(data["source_video_ref"], label="source_video_ref"),
            frames_ref=_require_str(data["frames_ref"], label="frames_ref"),
            analysis_windows_ref=(
                None
                if data.get("analysis_windows_ref") is None
                else _require_str(data["analysis_windows_ref"], label="analysis_windows_ref")
            ),
            eligible_frame_count=_require_int(
                data["eligible_frame_count"], label="eligible_frame_count", minimum=0
            )
            or 0,
            processed_frame_count=_require_int(
                data["processed_frame_count"], label="processed_frame_count", minimum=0
            )
            or 0,
            skipped_frame_count=_require_int(
                data["skipped_frame_count"], label="skipped_frame_count", minimum=0
            )
            or 0,
            failed_frame_count=_require_int(
                data["failed_frame_count"], label="failed_frame_count", minimum=0
            )
            or 0,
            processed_no_detection_count=_require_int(
                data["processed_no_detection_count"],
                label="processed_no_detection_count",
                minimum=0,
            )
            or 0,
            total_detection_count=total,
            human_detection_count=human,
            ball_detection_count=ball,
            pre_nms_count=_require_int(
                data.get("pre_nms_count"), label="pre_nms_count", minimum=0, allow_none=True
            ),
            post_nms_count=_require_int(
                data.get("post_nms_count"), label="post_nms_count", minimum=0, allow_none=True
            ),
            started_at_utc=_require_str(data["started_at_utc"], label="started_at_utc"),
            completed_at_utc=_require_str(data["completed_at_utc"], label="completed_at_utc"),
            status=_require_enum(ReceiptStatus, data["status"], label="status"),
            warnings=_require_message_list(data.get("warnings", []), label="warnings"),
            errors=_require_message_list(data.get("errors", []), label="errors"),
            artifacts=dict(artifacts),
            environment_ref=(
                None
                if data.get("environment_ref") is None
                else _require_str(data["environment_ref"], label="environment_ref")
            ),
            provenance=dict(provenance),
            schema_version=schema_version or RECEIPT_SCHEMA_VERSION,
            execution_provider=(
                None
                if data.get("execution_provider") is None
                else _require_str(data["execution_provider"], label="execution_provider")
            ),
            precision=data.get("precision"),
            software_versions=(
                None
                if data.get("software_versions") is None
                else {str(k): str(v) for k, v in dict(data["software_versions"]).items()}
            ),
            transform_fingerprint=_require_sha256_or_none(
                data.get("transform_fingerprint"), label="transform_fingerprint"
            ),
            threshold_config_fingerprint=_require_sha256_or_none(
                data.get("threshold_config_fingerprint"), label="threshold_config_fingerprint"
            ),
        )


__all__ = [
    "CONTRACT_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "TRANSFORM_SCHEMA_VERSION",
    "ERROR_CODES",
    "PROCESSED_STATUSES",
    "UNPROCESSED_STATUSES",
    "PerceptionError",
    "PerceptionContractError",
    "EntityType",
    "RoleLabel",
    "RoleSource",
    "ProcessingStatus",
    "Eligibility",
    "ReviewStatus",
    "ResizeMode",
    "ReceiptStatus",
    "ColorSpace",
    "ChannelOrder",
    "Orientation",
    "DetectionFrameStatus",
    "DetectionAttributes",
    "PreprocessingTransform",
    "DetectionRunReceipt",
]
