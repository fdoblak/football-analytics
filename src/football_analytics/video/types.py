"""Stage 3A video ingest typed contracts (immutable, side-effect free on import)."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from football_analytics.core.hashing import hash_canonical_json
from football_analytics.core.run_id import validate_run_id

SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
PATH_SAFE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
UTC_Z_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]{1,6})?Z$")
FILENAME_RE = re.compile(r"^[^/\\]+$")
ROTATION_ALLOWED = frozenset({0, 90, 180, 270})


class VideoError(ValueError):
    """Base error for Stage 3A video contracts."""


class VideoContractError(VideoError):
    """Contract construction or validation failure."""


class VideoSourceError(VideoError):
    """Path safety or source provenance failure."""


class VideoPolicyError(VideoError):
    """Ingest policy load or validation failure."""


class SourceKind(str, Enum):
    USER_LOCAL_VIDEO = "user_local_video"
    SYNTHETIC_FIXTURE = "synthetic_fixture"
    PROJECT_FIXTURE = "project_fixture"


class IngestMode(str, Enum):
    VALIDATE_ONLY = "validate_only"
    PLAN_ONLY = "plan_only"
    FIXTURE_DESIGN = "fixture_design"


class ReceiptStatus(str, Enum):
    PLANNED = "planned"
    VALIDATED = "validated"
    REJECTED = "rejected"
    FAILED = "failed"


class FrameRateMode(str, Enum):
    CFR = "cfr"
    VFR = "vfr"
    UNKNOWN = "unknown"


class FrameCountSource(str, Enum):
    NB_FRAMES = "nb_frames"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"


class WarningSeverity(str, Enum):
    WARNING = "warning"
    UNSUPPORTED = "unsupported"
    HARD_FAILURE = "hard_failure"


def require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise VideoContractError(f"{label} must be 64 lowercase hex")
    return value


def require_path_safe_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not PATH_SAFE_ID_RE.fullmatch(value):
        raise VideoContractError(f"{label} must be path-safe id [a-z][a-z0-9_]{{1,63}}")
    return value


def require_non_bool_int(
    value: Any, *, label: str, minimum: int | None = None, allow_none: bool = False
) -> int | None:
    if value is None:
        if allow_none:
            return None
        raise VideoContractError(f"{label} must be an int")
    if not isinstance(value, int) or isinstance(value, bool):
        raise VideoContractError(f"{label} must be an int")
    if minimum is not None and value < minimum:
        raise VideoContractError(f"{label} must be >= {minimum}")
    return value


def require_utc_z(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not UTC_Z_RE.fullmatch(value):
        raise VideoContractError(f"{label} must be UTC timestamp ending with Z")
    # Parse to ensure timezone-aware UTC semantics
    text = value
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise VideoContractError(f"{label} is not a valid timestamp") from exc
    if parsed.tzinfo is None:
        raise VideoContractError(f"{label} must be timezone-aware UTC")
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise VideoContractError(f"{label} must be UTC")
    return value


def require_finite_number_absent(value: Any, *, label: str) -> None:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise VideoContractError(f"{label}: NaN/Infinity forbidden")


def normalize_rotation_degrees(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise VideoContractError("rotation_degrees must be int")
    # Map signed equivalents onto allowlist
    mapping = {
        0: 0,
        90: 90,
        180: 180,
        270: 270,
        -90: 270,
        -180: 180,
        -270: 90,
    }
    if value not in mapping:
        raise VideoContractError("rotation_degrees not in allowlist")
    return mapping[value]


@dataclass(frozen=True)
class Rational:
    """Exact rational; denominator must be > 0. Never store float frame-rate strings."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if not isinstance(self.numerator, int) or isinstance(self.numerator, bool):
            raise VideoContractError("rational numerator must be int")
        if not isinstance(self.denominator, int) or isinstance(self.denominator, bool):
            raise VideoContractError("rational denominator must be int")
        if self.denominator <= 0:
            raise VideoContractError("rational denominator must be >= 1")

    def to_dict(self) -> dict[str, int]:
        return {"numerator": self.numerator, "denominator": self.denominator}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, label: str = "rational") -> Rational:
        if not isinstance(data, Mapping):
            raise VideoContractError(f"{label} must be object")
        unknown = set(data) - {"numerator", "denominator"}
        if unknown:
            raise VideoContractError(f"{label} unknown keys: {sorted(unknown)}")
        return cls(numerator=data["numerator"], denominator=data["denominator"])


@dataclass(frozen=True)
class StreamDisposition:
    default: bool
    attached_pic: bool
    forced: bool

    def __post_init__(self) -> None:
        for name in ("default", "attached_pic", "forced"):
            if not isinstance(getattr(self, name), bool):
                raise VideoContractError(f"disposition.{name} must be bool")

    def to_dict(self) -> dict[str, bool]:
        return {
            "default": self.default,
            "attached_pic": self.attached_pic,
            "forced": self.forced,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StreamDisposition:
        if not isinstance(data, Mapping):
            raise VideoContractError("disposition must be object")
        unknown = set(data) - {"default", "attached_pic", "forced"}
        if unknown:
            raise VideoContractError(f"disposition unknown keys: {sorted(unknown)}")
        return cls(
            default=bool(data["default"]),
            attached_pic=bool(data["attached_pic"]),
            forced=bool(data["forced"]),
        )


@dataclass(frozen=True)
class ProvenanceInfo:
    origin: str
    label: str
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.origin not in {"local_file", "synthetic_generated", "project_fixture"}:
            raise VideoContractError("provenance.origin invalid")
        if not isinstance(self.label, str) or not self.label:
            raise VideoContractError("provenance.label empty")
        if self.notes is not None and not isinstance(self.notes, str):
            raise VideoContractError("provenance.notes must be str or null")

    def to_dict(self) -> dict[str, Any]:
        return {"origin": self.origin, "label": self.label, "notes": self.notes}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProvenanceInfo:
        unknown = set(data) - {"origin", "label", "notes"}
        if unknown:
            raise VideoContractError(f"provenance unknown keys: {sorted(unknown)}")
        return cls(
            origin=str(data["origin"]),
            label=str(data["label"]),
            notes=data.get("notes"),
        )


@dataclass(frozen=True)
class Issue:
    code: str
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise VideoContractError("issue.code empty")
        if not isinstance(self.message, str):
            raise VideoContractError("issue.message must be str")

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Issue:
        unknown = set(data) - {"code", "message"}
        if unknown:
            raise VideoContractError(f"issue unknown keys: {sorted(unknown)}")
        return cls(code=str(data["code"]), message=str(data["message"]))


@dataclass(frozen=True)
class ProbeWarning:
    code: str
    message: str
    severity: WarningSeverity

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise VideoContractError("warning.code empty")
        if not isinstance(self.message, str):
            raise VideoContractError("warning.message must be str")
        if not isinstance(self.severity, WarningSeverity):
            raise VideoContractError("warning.severity invalid")

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProbeWarning:
        unknown = set(data) - {"code", "message", "severity"}
        if unknown:
            raise VideoContractError(f"probe warning unknown keys: {sorted(unknown)}")
        return cls(
            code=str(data["code"]),
            message=str(data["message"]),
            severity=WarningSeverity(str(data["severity"])),
        )


@dataclass(frozen=True)
class VideoSource:
    source_id: str
    source_kind: SourceKind
    original_filename: str
    source_path: str
    source_size_bytes: int
    source_sha256: str
    media_type: str
    container_hint: str | None
    created_at_utc: str
    registered_at_utc: str
    immutability_policy: str
    provenance: ProvenanceInfo
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise VideoContractError("unsupported VideoSource schema_version")
        object.__setattr__(
            self, "source_id", require_path_safe_id(self.source_id, label="source_id")
        )
        if not isinstance(self.source_kind, SourceKind):
            raise VideoContractError("source_kind invalid")
        if not isinstance(self.original_filename, str) or not FILENAME_RE.fullmatch(
            self.original_filename
        ):
            raise VideoContractError("original_filename invalid")
        if not isinstance(self.source_path, str) or not self.source_path:
            raise VideoContractError("source_path empty")
        if "\x00" in self.source_path:
            raise VideoContractError("source_path contains null byte")
        object.__setattr__(
            self,
            "source_size_bytes",
            require_non_bool_int(self.source_size_bytes, label="source_size_bytes", minimum=0),
        )
        object.__setattr__(
            self, "source_sha256", require_sha256(self.source_sha256, label="source_sha256")
        )
        if not isinstance(self.media_type, str) or not self.media_type:
            raise VideoContractError("media_type empty")
        if self.container_hint is not None and (
            not isinstance(self.container_hint, str) or not self.container_hint
        ):
            raise VideoContractError("container_hint must be non-empty string or null")
        object.__setattr__(
            self, "created_at_utc", require_utc_z(self.created_at_utc, label="created_at_utc")
        )
        object.__setattr__(
            self,
            "registered_at_utc",
            require_utc_z(self.registered_at_utc, label="registered_at_utc"),
        )
        if self.immutability_policy not in {"immutable_source", "detect_mutation"}:
            raise VideoContractError("immutability_policy invalid")
        if not isinstance(self.provenance, ProvenanceInfo):
            raise VideoContractError("provenance must be ProvenanceInfo")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "source_kind": self.source_kind.value,
            "original_filename": self.original_filename,
            "source_path": self.source_path,
            "source_size_bytes": self.source_size_bytes,
            "source_sha256": self.source_sha256,
            "media_type": self.media_type,
            "container_hint": self.container_hint,
            "created_at_utc": self.created_at_utc,
            "registered_at_utc": self.registered_at_utc,
            "immutability_policy": self.immutability_policy,
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VideoSource:
        allowed = {
            "schema_version",
            "source_id",
            "source_kind",
            "original_filename",
            "source_path",
            "source_size_bytes",
            "source_sha256",
            "media_type",
            "container_hint",
            "created_at_utc",
            "registered_at_utc",
            "immutability_policy",
            "provenance",
        }
        unknown = set(data) - allowed
        if unknown:
            raise VideoContractError(f"VideoSource unknown keys: {sorted(unknown)}")
        return cls(
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
            source_id=str(data["source_id"]),
            source_kind=SourceKind(str(data["source_kind"])),
            original_filename=str(data["original_filename"]),
            source_path=str(data["source_path"]),
            source_size_bytes=int(data["source_size_bytes"]),
            source_sha256=str(data["source_sha256"]),
            media_type=str(data["media_type"]),
            container_hint=data.get("container_hint"),
            created_at_utc=str(data["created_at_utc"]),
            registered_at_utc=str(data["registered_at_utc"]),
            immutability_policy=str(data["immutability_policy"]),
            provenance=ProvenanceInfo.from_dict(data["provenance"]),
        )

    def fingerprint(self) -> str:
        return hash_canonical_json(self.to_dict())


@dataclass(frozen=True)
class IngestRequest:
    request_id: str
    run_id: str
    source_id: str
    source_path: str
    requested_at_utc: str
    ingest_mode: IngestMode
    policy_version: str
    probe_requested: bool
    normalization_requested: bool
    expected_source_sha256: str
    expected_source_size_bytes: int
    output_root: str
    fixture_mode: bool
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise VideoContractError("unsupported IngestRequest schema_version")
        object.__setattr__(
            self, "request_id", require_path_safe_id(self.request_id, label="request_id")
        )
        object.__setattr__(self, "run_id", validate_run_id(self.run_id))
        object.__setattr__(
            self, "source_id", require_path_safe_id(self.source_id, label="source_id")
        )
        if not isinstance(self.source_path, str) or not self.source_path:
            raise VideoContractError("source_path empty")
        if "\x00" in self.source_path or "\x00" in self.output_root:
            raise VideoContractError("path contains null byte")
        object.__setattr__(
            self,
            "requested_at_utc",
            require_utc_z(self.requested_at_utc, label="requested_at_utc"),
        )
        if not isinstance(self.ingest_mode, IngestMode):
            raise VideoContractError("ingest_mode invalid")
        if not isinstance(self.policy_version, str) or not self.policy_version:
            raise VideoContractError("policy_version empty")
        for name in ("probe_requested", "normalization_requested", "fixture_mode"):
            if not isinstance(getattr(self, name), bool):
                raise VideoContractError(f"{name} must be bool")
        object.__setattr__(
            self,
            "expected_source_sha256",
            require_sha256(self.expected_source_sha256, label="expected_source_sha256"),
        )
        object.__setattr__(
            self,
            "expected_source_size_bytes",
            require_non_bool_int(
                self.expected_source_size_bytes,
                label="expected_source_size_bytes",
                minimum=0,
            ),
        )
        if not isinstance(self.output_root, str) or not self.output_root:
            raise VideoContractError("output_root empty")
        if self.fixture_mode and self.ingest_mode not in {
            IngestMode.FIXTURE_DESIGN,
            IngestMode.VALIDATE_ONLY,
            IngestMode.PLAN_ONLY,
        }:
            raise VideoContractError("fixture_mode incompatible with ingest_mode")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "run_id": self.run_id,
            "source_id": self.source_id,
            "source_path": self.source_path,
            "requested_at_utc": self.requested_at_utc,
            "ingest_mode": self.ingest_mode.value,
            "policy_version": self.policy_version,
            "probe_requested": self.probe_requested,
            "normalization_requested": self.normalization_requested,
            "expected_source_sha256": self.expected_source_sha256,
            "expected_source_size_bytes": self.expected_source_size_bytes,
            "output_root": self.output_root,
            "fixture_mode": self.fixture_mode,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> IngestRequest:
        allowed = {
            "schema_version",
            "request_id",
            "run_id",
            "source_id",
            "source_path",
            "requested_at_utc",
            "ingest_mode",
            "policy_version",
            "probe_requested",
            "normalization_requested",
            "expected_source_sha256",
            "expected_source_size_bytes",
            "output_root",
            "fixture_mode",
        }
        unknown = set(data) - allowed
        if unknown:
            raise VideoContractError(f"IngestRequest unknown keys: {sorted(unknown)}")
        return cls(
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
            request_id=str(data["request_id"]),
            run_id=str(data["run_id"]),
            source_id=str(data["source_id"]),
            source_path=str(data["source_path"]),
            requested_at_utc=str(data["requested_at_utc"]),
            ingest_mode=IngestMode(str(data["ingest_mode"])),
            policy_version=str(data["policy_version"]),
            probe_requested=bool(data["probe_requested"]),
            normalization_requested=bool(data["normalization_requested"]),
            expected_source_sha256=str(data["expected_source_sha256"]),
            expected_source_size_bytes=int(data["expected_source_size_bytes"]),
            output_root=str(data["output_root"]),
            fixture_mode=bool(data["fixture_mode"]),
        )

    def fingerprint(self) -> str:
        return hash_canonical_json(self.to_dict())


@dataclass(frozen=True)
class VideoStreamInfo:
    stream_index: int
    codec_name: str
    codec_long_name: str | None
    profile: str | None
    pixel_format: str | None
    width: int
    height: int
    coded_width: int | None
    coded_height: int | None
    sample_aspect_ratio: Rational
    display_aspect_ratio: Rational
    rotation_degrees: int
    time_base: Rational
    codec_time_base: Rational | None
    r_frame_rate: Rational
    avg_frame_rate: Rational
    nominal_frame_rate: Rational | None
    frame_rate_mode: FrameRateMode
    start_pts: int | None
    duration_ts: int | None
    duration_us: int | None
    frame_count: int | None
    frame_count_source: FrameCountSource
    bit_rate_bps: int | None
    color_range: str | None
    color_space: str | None
    color_transfer: str | None
    color_primaries: str | None
    field_order: str | None
    disposition: StreamDisposition
    codec_type: str = "video"

    def __post_init__(self) -> None:
        if self.codec_type != "video":
            raise VideoContractError("VideoStreamInfo.codec_type must be video")
        require_non_bool_int(self.stream_index, label="stream_index", minimum=0)
        require_non_bool_int(self.width, label="width", minimum=1)
        require_non_bool_int(self.height, label="height", minimum=1)
        object.__setattr__(
            self, "rotation_degrees", normalize_rotation_degrees(self.rotation_degrees)
        )
        if not isinstance(self.frame_rate_mode, FrameRateMode):
            raise VideoContractError("frame_rate_mode invalid")
        if not isinstance(self.frame_count_source, FrameCountSource):
            raise VideoContractError("frame_count_source invalid")
        if self.frame_count is None and self.frame_count_source != FrameCountSource.UNKNOWN:
            raise VideoContractError("null frame_count requires frame_count_source=unknown")
        if self.frame_count is not None:
            require_non_bool_int(self.frame_count, label="frame_count", minimum=0)
        if self.duration_us is not None:
            require_non_bool_int(self.duration_us, label="duration_us", minimum=0)
        if self.duration_us == 0 and self.frame_rate_mode == FrameRateMode.CFR:
            # zero duration is allowed only if explicitly measured; do not coerce unknown→0
            pass

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream_index": self.stream_index,
            "codec_type": self.codec_type,
            "codec_name": self.codec_name,
            "codec_long_name": self.codec_long_name,
            "profile": self.profile,
            "pixel_format": self.pixel_format,
            "width": self.width,
            "height": self.height,
            "coded_width": self.coded_width,
            "coded_height": self.coded_height,
            "sample_aspect_ratio": self.sample_aspect_ratio.to_dict(),
            "display_aspect_ratio": self.display_aspect_ratio.to_dict(),
            "rotation_degrees": self.rotation_degrees,
            "time_base": self.time_base.to_dict(),
            "codec_time_base": (
                None if self.codec_time_base is None else self.codec_time_base.to_dict()
            ),
            "r_frame_rate": self.r_frame_rate.to_dict(),
            "avg_frame_rate": self.avg_frame_rate.to_dict(),
            "nominal_frame_rate": (
                None if self.nominal_frame_rate is None else self.nominal_frame_rate.to_dict()
            ),
            "frame_rate_mode": self.frame_rate_mode.value,
            "start_pts": self.start_pts,
            "duration_ts": self.duration_ts,
            "duration_us": self.duration_us,
            "frame_count": self.frame_count,
            "frame_count_source": self.frame_count_source.value,
            "bit_rate_bps": self.bit_rate_bps,
            "color_range": self.color_range,
            "color_space": self.color_space,
            "color_transfer": self.color_transfer,
            "color_primaries": self.color_primaries,
            "field_order": self.field_order,
            "disposition": self.disposition.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VideoStreamInfo:
        def _rat(key: str, *, allow_none: bool = False) -> Rational | None:
            raw = data.get(key)
            if raw is None:
                if allow_none:
                    return None
                raise VideoContractError(f"{key} required")
            return Rational.from_dict(raw, label=key)

        return cls(
            stream_index=int(data["stream_index"]),
            codec_name=str(data["codec_name"]),
            codec_long_name=data.get("codec_long_name"),
            profile=data.get("profile"),
            pixel_format=data.get("pixel_format"),
            width=int(data["width"]),
            height=int(data["height"]),
            coded_width=data.get("coded_width"),
            coded_height=data.get("coded_height"),
            sample_aspect_ratio=_rat("sample_aspect_ratio"),  # type: ignore[arg-type]
            display_aspect_ratio=_rat("display_aspect_ratio"),  # type: ignore[arg-type]
            rotation_degrees=int(data["rotation_degrees"]),
            time_base=_rat("time_base"),  # type: ignore[arg-type]
            codec_time_base=_rat("codec_time_base", allow_none=True),
            r_frame_rate=_rat("r_frame_rate"),  # type: ignore[arg-type]
            avg_frame_rate=_rat("avg_frame_rate"),  # type: ignore[arg-type]
            nominal_frame_rate=_rat("nominal_frame_rate", allow_none=True),
            frame_rate_mode=FrameRateMode(str(data["frame_rate_mode"])),
            start_pts=data.get("start_pts"),
            duration_ts=data.get("duration_ts"),
            duration_us=data.get("duration_us"),
            frame_count=data.get("frame_count"),
            frame_count_source=FrameCountSource(str(data["frame_count_source"])),
            bit_rate_bps=data.get("bit_rate_bps"),
            color_range=data.get("color_range"),
            color_space=data.get("color_space"),
            color_transfer=data.get("color_transfer"),
            color_primaries=data.get("color_primaries"),
            field_order=data.get("field_order"),
            disposition=StreamDisposition.from_dict(data["disposition"]),
            codec_type=str(data.get("codec_type", "video")),
        )


@dataclass(frozen=True)
class AudioStreamInfo:
    stream_index: int
    codec_name: str
    sample_rate_hz: int | None
    channels: int | None
    channel_layout: str | None
    time_base: Rational
    duration_us: int | None
    bit_rate_bps: int | None
    disposition: StreamDisposition
    codec_type: str = "audio"

    def __post_init__(self) -> None:
        if self.codec_type != "audio":
            raise VideoContractError("AudioStreamInfo.codec_type must be audio")
        require_non_bool_int(self.stream_index, label="stream_index", minimum=0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream_index": self.stream_index,
            "codec_type": self.codec_type,
            "codec_name": self.codec_name,
            "sample_rate_hz": self.sample_rate_hz,
            "channels": self.channels,
            "channel_layout": self.channel_layout,
            "time_base": self.time_base.to_dict(),
            "duration_us": self.duration_us,
            "bit_rate_bps": self.bit_rate_bps,
            "disposition": self.disposition.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AudioStreamInfo:
        return cls(
            stream_index=int(data["stream_index"]),
            codec_name=str(data["codec_name"]),
            sample_rate_hz=data.get("sample_rate_hz"),
            channels=data.get("channels"),
            channel_layout=data.get("channel_layout"),
            time_base=Rational.from_dict(data["time_base"], label="time_base"),
            duration_us=data.get("duration_us"),
            bit_rate_bps=data.get("bit_rate_bps"),
            disposition=StreamDisposition.from_dict(data["disposition"]),
            codec_type=str(data.get("codec_type", "audio")),
        )


def select_primary_video_stream(
    streams: tuple[VideoStreamInfo | AudioStreamInfo, ...],
) -> int:
    """Deterministic selection: ignore attached pictures; prefer largest area; lowest index."""
    candidates: list[VideoStreamInfo] = []
    for stream in streams:
        if isinstance(stream, VideoStreamInfo) and not stream.disposition.attached_pic:
            candidates.append(stream)
    if not candidates:
        raise VideoContractError("no selectable video stream (attached_pic excluded)")
    candidates.sort(key=lambda s: (-(s.width * s.height), s.stream_index))
    return candidates[0].stream_index


@dataclass(frozen=True)
class VideoProbe:
    source_id: str
    source_sha256: str
    probe_tool: str
    probe_tool_version: str
    probed_at_utc: str
    container: str | None
    format_name: str | None
    duration_us: int | None
    start_time_us: int | None
    bit_rate_bps: int | None
    file_size_bytes: int
    streams: tuple[VideoStreamInfo | AudioStreamInfo, ...]
    selected_video_stream_index: int
    selected_audio_stream_index: int | None
    warnings: tuple[ProbeWarning, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise VideoContractError("unsupported VideoProbe schema_version")
        object.__setattr__(
            self, "source_id", require_path_safe_id(self.source_id, label="source_id")
        )
        object.__setattr__(
            self, "source_sha256", require_sha256(self.source_sha256, label="source_sha256")
        )
        if self.probe_tool not in {"ffprobe", "synthetic_metadata"}:
            raise VideoContractError("probe_tool invalid")
        object.__setattr__(
            self, "probed_at_utc", require_utc_z(self.probed_at_utc, label="probed_at_utc")
        )
        require_non_bool_int(self.file_size_bytes, label="file_size_bytes", minimum=0)
        if self.duration_us is not None:
            require_non_bool_int(self.duration_us, label="duration_us", minimum=0)
        if not self.streams:
            raise VideoContractError("streams empty")
        object.__setattr__(self, "streams", tuple(self.streams))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        expected = select_primary_video_stream(self.streams)
        if self.selected_video_stream_index != expected:
            raise VideoContractError(
                "selected_video_stream_index does not match deterministic policy"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "source_sha256": self.source_sha256,
            "probe_tool": self.probe_tool,
            "probe_tool_version": self.probe_tool_version,
            "probed_at_utc": self.probed_at_utc,
            "container": self.container,
            "format_name": self.format_name,
            "duration_us": self.duration_us,
            "start_time_us": self.start_time_us,
            "bit_rate_bps": self.bit_rate_bps,
            "file_size_bytes": self.file_size_bytes,
            "streams": [s.to_dict() for s in self.streams],
            "selected_video_stream_index": self.selected_video_stream_index,
            "selected_audio_stream_index": self.selected_audio_stream_index,
            "warnings": [w.to_dict() for w in self.warnings],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VideoProbe:
        streams: list[VideoStreamInfo | AudioStreamInfo] = []
        for item in data["streams"]:
            ctype = item.get("codec_type")
            if ctype == "video":
                streams.append(VideoStreamInfo.from_dict(item))
            elif ctype == "audio":
                streams.append(AudioStreamInfo.from_dict(item))
            else:
                raise VideoContractError(f"unsupported stream codec_type: {ctype}")
        warnings = tuple(ProbeWarning.from_dict(w) for w in data.get("warnings", []))
        return cls(
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
            source_id=str(data["source_id"]),
            source_sha256=str(data["source_sha256"]),
            probe_tool=str(data["probe_tool"]),
            probe_tool_version=str(data["probe_tool_version"]),
            probed_at_utc=str(data["probed_at_utc"]),
            container=data.get("container"),
            format_name=data.get("format_name"),
            duration_us=data.get("duration_us"),
            start_time_us=data.get("start_time_us"),
            bit_rate_bps=data.get("bit_rate_bps"),
            file_size_bytes=int(data["file_size_bytes"]),
            streams=tuple(streams),
            selected_video_stream_index=int(data["selected_video_stream_index"]),
            selected_audio_stream_index=data.get("selected_audio_stream_index"),
            warnings=warnings,
        )

    def fingerprint(self) -> str:
        return hash_canonical_json(self.to_dict())


@dataclass(frozen=True)
class NormalizePlan:
    plan_id: str
    source_id: str
    source_sha256: str
    policy_version: str
    required: bool
    reasons: tuple[str, ...]
    target_container: str
    target_video_codec: str
    target_audio_policy: str
    target_pixel_format: str
    target_width: int | None
    target_height: int | None
    resize_policy: str
    target_frame_rate: Rational | None
    frame_rate_policy: str
    target_time_base: Rational | None
    rotation_policy: str
    sar_policy: str
    audio_policy: str
    copy_metadata_policy: str
    estimated_output_path: str
    overwrite_policy: bool
    plan_fingerprint: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise VideoContractError("unsupported NormalizePlan schema_version")
        object.__setattr__(self, "plan_id", require_path_safe_id(self.plan_id, label="plan_id"))
        object.__setattr__(
            self, "source_id", require_path_safe_id(self.source_id, label="source_id")
        )
        object.__setattr__(
            self, "source_sha256", require_sha256(self.source_sha256, label="source_sha256")
        )
        object.__setattr__(self, "reasons", tuple(self.reasons))
        if not self.reasons:
            raise VideoContractError("normalize reasons empty")
        if not self.required and not self.reasons:
            raise VideoContractError("required=false needs reasons")
        if self.overwrite_policy is not False and self.overwrite_policy is not True:
            raise VideoContractError("overwrite_policy must be bool")
        if self.overwrite_policy is True:
            raise VideoContractError("overwrite_policy must be false in Stage 3A defaults")
        object.__setattr__(
            self,
            "plan_fingerprint",
            require_sha256(self.plan_fingerprint, label="plan_fingerprint"),
        )
        expected = self.compute_fingerprint()
        if self.plan_fingerprint != expected:
            raise VideoContractError("plan_fingerprint mismatch")

    def _fingerprint_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "source_id": self.source_id,
            "source_sha256": self.source_sha256,
            "policy_version": self.policy_version,
            "required": self.required,
            "reasons": list(self.reasons),
            "target_container": self.target_container,
            "target_video_codec": self.target_video_codec,
            "target_audio_policy": self.target_audio_policy,
            "target_pixel_format": self.target_pixel_format,
            "target_width": self.target_width,
            "target_height": self.target_height,
            "resize_policy": self.resize_policy,
            "target_frame_rate": (
                None if self.target_frame_rate is None else self.target_frame_rate.to_dict()
            ),
            "frame_rate_policy": self.frame_rate_policy,
            "target_time_base": (
                None if self.target_time_base is None else self.target_time_base.to_dict()
            ),
            "rotation_policy": self.rotation_policy,
            "sar_policy": self.sar_policy,
            "audio_policy": self.audio_policy,
            "copy_metadata_policy": self.copy_metadata_policy,
            "estimated_output_path": self.estimated_output_path,
            "overwrite_policy": self.overwrite_policy,
        }

    def compute_fingerprint(self) -> str:
        return hash_canonical_json(self._fingerprint_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._fingerprint_payload()
        payload["plan_fingerprint"] = self.plan_fingerprint
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NormalizePlan:
        fr = data.get("target_frame_rate")
        tb = data.get("target_time_base")
        return cls(
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
            plan_id=str(data["plan_id"]),
            source_id=str(data["source_id"]),
            source_sha256=str(data["source_sha256"]),
            policy_version=str(data["policy_version"]),
            required=bool(data["required"]),
            reasons=tuple(data["reasons"]),
            target_container=str(data["target_container"]),
            target_video_codec=str(data["target_video_codec"]),
            target_audio_policy=str(data["target_audio_policy"]),
            target_pixel_format=str(data["target_pixel_format"]),
            target_width=data.get("target_width"),
            target_height=data.get("target_height"),
            resize_policy=str(data["resize_policy"]),
            target_frame_rate=None if fr is None else Rational.from_dict(fr),
            frame_rate_policy=str(data["frame_rate_policy"]),
            target_time_base=None if tb is None else Rational.from_dict(tb),
            rotation_policy=str(data["rotation_policy"]),
            sar_policy=str(data["sar_policy"]),
            audio_policy=str(data["audio_policy"]),
            copy_metadata_policy=str(data["copy_metadata_policy"]),
            estimated_output_path=str(data["estimated_output_path"]),
            overwrite_policy=bool(data["overwrite_policy"]),
            plan_fingerprint=str(data["plan_fingerprint"]),
        )


@dataclass(frozen=True)
class ContractFingerprints:
    source: str
    request: str
    probe: str | None = None
    normalize_plan: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", require_sha256(self.source, label="fp.source"))
        object.__setattr__(self, "request", require_sha256(self.request, label="fp.request"))
        if self.probe is not None:
            object.__setattr__(self, "probe", require_sha256(self.probe, label="fp.probe"))
        if self.normalize_plan is not None:
            object.__setattr__(
                self,
                "normalize_plan",
                require_sha256(self.normalize_plan, label="fp.normalize_plan"),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "request": self.request,
            "probe": self.probe,
            "normalize_plan": self.normalize_plan,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ContractFingerprints:
        return cls(
            source=str(data["source"]),
            request=str(data["request"]),
            probe=data.get("probe"),
            normalize_plan=data.get("normalize_plan"),
        )


@dataclass(frozen=True)
class ReceiptProvenance:
    stage: str
    label: str
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.stage != "3A":
            raise VideoContractError("receipt provenance.stage must be 3A")
        if not isinstance(self.label, str) or not self.label:
            raise VideoContractError("receipt provenance.label empty")

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "label": self.label, "notes": self.notes}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReceiptProvenance:
        return cls(
            stage=str(data["stage"]),
            label=str(data["label"]),
            notes=data.get("notes"),
        )


@dataclass(frozen=True)
class IngestReceipt:
    receipt_id: str
    request_id: str
    run_id: str
    source_id: str
    source_sha256: str
    source_size_bytes: int
    status: ReceiptStatus
    started_at_utc: str
    completed_at_utc: str
    probe_record_ref: str | None
    normalize_plan_ref: str | None
    artifact_refs: tuple[str, ...]
    policy_version: str
    contract_fingerprints: ContractFingerprints
    warnings: tuple[Issue, ...]
    errors: tuple[Issue, ...]
    provenance: ReceiptProvenance
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise VideoContractError("unsupported IngestReceipt schema_version")
        object.__setattr__(
            self, "receipt_id", require_path_safe_id(self.receipt_id, label="receipt_id")
        )
        object.__setattr__(
            self, "request_id", require_path_safe_id(self.request_id, label="request_id")
        )
        object.__setattr__(self, "run_id", validate_run_id(self.run_id))
        object.__setattr__(
            self, "source_id", require_path_safe_id(self.source_id, label="source_id")
        )
        object.__setattr__(
            self, "source_sha256", require_sha256(self.source_sha256, label="source_sha256")
        )
        require_non_bool_int(self.source_size_bytes, label="source_size_bytes", minimum=0)
        if not isinstance(self.status, ReceiptStatus):
            raise VideoContractError("status invalid")
        # Stage 3A must never claim production success
        if self.status.value in {"succeeded", "completed", "success"}:
            raise VideoContractError("false-success receipt status forbidden in Stage 3A")
        object.__setattr__(
            self, "started_at_utc", require_utc_z(self.started_at_utc, label="started_at_utc")
        )
        object.__setattr__(
            self,
            "completed_at_utc",
            require_utc_z(self.completed_at_utc, label="completed_at_utc"),
        )
        object.__setattr__(self, "artifact_refs", tuple(self.artifact_refs))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "errors", tuple(self.errors))
        if self.status in {ReceiptStatus.REJECTED, ReceiptStatus.FAILED} and not self.errors:
            raise VideoContractError(f"{self.status.value} receipt requires errors")
        if self.status == ReceiptStatus.VALIDATED and self.errors:
            raise VideoContractError("validated receipt must not include errors")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "receipt_id": self.receipt_id,
            "request_id": self.request_id,
            "run_id": self.run_id,
            "source_id": self.source_id,
            "source_sha256": self.source_sha256,
            "source_size_bytes": self.source_size_bytes,
            "status": self.status.value,
            "started_at_utc": self.started_at_utc,
            "completed_at_utc": self.completed_at_utc,
            "probe_record_ref": self.probe_record_ref,
            "normalize_plan_ref": self.normalize_plan_ref,
            "artifact_refs": list(self.artifact_refs),
            "policy_version": self.policy_version,
            "contract_fingerprints": self.contract_fingerprints.to_dict(),
            "warnings": [w.to_dict() for w in self.warnings],
            "errors": [e.to_dict() for e in self.errors],
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> IngestReceipt:
        return cls(
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
            receipt_id=str(data["receipt_id"]),
            request_id=str(data["request_id"]),
            run_id=str(data["run_id"]),
            source_id=str(data["source_id"]),
            source_sha256=str(data["source_sha256"]),
            source_size_bytes=int(data["source_size_bytes"]),
            status=ReceiptStatus(str(data["status"])),
            started_at_utc=str(data["started_at_utc"]),
            completed_at_utc=str(data["completed_at_utc"]),
            probe_record_ref=data.get("probe_record_ref"),
            normalize_plan_ref=data.get("normalize_plan_ref"),
            artifact_refs=tuple(data.get("artifact_refs", [])),
            policy_version=str(data["policy_version"]),
            contract_fingerprints=ContractFingerprints.from_dict(data["contract_fingerprints"]),
            warnings=tuple(Issue.from_dict(w) for w in data.get("warnings", [])),
            errors=tuple(Issue.from_dict(e) for e in data.get("errors", [])),
            provenance=ReceiptProvenance.from_dict(data["provenance"]),
        )

    def fingerprint(self) -> str:
        return hash_canonical_json(self.to_dict())


class NormalizationStatus(str, Enum):
    PLANNED = "planned"
    SKIPPED = "skipped"
    SUCCEEDED = "succeeded"
    REJECTED = "rejected"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class NormalizationSelectedStreams:
    video_stream_index: int
    audio_stream_index: int | None

    def __post_init__(self) -> None:
        require_non_bool_int(self.video_stream_index, label="video_stream_index", minimum=0)
        if self.audio_stream_index is not None:
            require_non_bool_int(self.audio_stream_index, label="audio_stream_index", minimum=0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_stream_index": self.video_stream_index,
            "audio_stream_index": self.audio_stream_index,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NormalizationSelectedStreams:
        return cls(
            video_stream_index=int(data["video_stream_index"]),
            audio_stream_index=data.get("audio_stream_index"),
        )


@dataclass(frozen=True)
class FrameRateConversionInfo:
    performed: bool
    source_mode: str
    target_mode: str
    notes: str
    requires_stage3d_mapping: bool

    def __post_init__(self) -> None:
        if self.source_mode not in {"cfr", "vfr", "unknown"}:
            raise VideoContractError("frame_rate_conversion.source_mode invalid")
        if self.target_mode not in {"cfr", "vfr", "unknown", "unchanged"}:
            raise VideoContractError("frame_rate_conversion.target_mode invalid")
        if not isinstance(self.notes, str):
            raise VideoContractError("frame_rate_conversion.notes must be str")
        if not isinstance(self.performed, bool) or not isinstance(
            self.requires_stage3d_mapping, bool
        ):
            raise VideoContractError("frame_rate_conversion booleans invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "performed": self.performed,
            "source_mode": self.source_mode,
            "target_mode": self.target_mode,
            "notes": self.notes,
            "requires_stage3d_mapping": self.requires_stage3d_mapping,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FrameRateConversionInfo:
        return cls(
            performed=bool(data["performed"]),
            source_mode=str(data["source_mode"]),
            target_mode=str(data["target_mode"]),
            notes=str(data["notes"]),
            requires_stage3d_mapping=bool(data["requires_stage3d_mapping"]),
        )


@dataclass(frozen=True)
class RotationTransformInfo:
    performed: bool
    source_degrees: int
    output_degrees: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "performed": bool(self.performed),
            "source_degrees": int(self.source_degrees),
            "output_degrees": int(self.output_degrees),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RotationTransformInfo:
        return cls(
            performed=bool(data["performed"]),
            source_degrees=int(data["source_degrees"]),
            output_degrees=int(data["output_degrees"]),
        )


@dataclass(frozen=True)
class ResizeTransformInfo:
    performed: bool
    source_width: int
    source_height: int
    target_width: int | None
    target_height: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "performed": bool(self.performed),
            "source_width": int(self.source_width),
            "source_height": int(self.source_height),
            "target_width": self.target_width,
            "target_height": self.target_height,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ResizeTransformInfo:
        return cls(
            performed=bool(data["performed"]),
            source_width=int(data["source_width"]),
            source_height=int(data["source_height"]),
            target_width=data.get("target_width"),
            target_height=data.get("target_height"),
        )


@dataclass(frozen=True)
class AudioTransformInfo:
    policy: str
    action: str

    def __post_init__(self) -> None:
        if not isinstance(self.policy, str) or not self.policy:
            raise VideoContractError("audio_transform.policy empty")
        if self.action not in {"none", "copy", "transcode", "drop", "absent"}:
            raise VideoContractError("audio_transform.action invalid")

    def to_dict(self) -> dict[str, str]:
        return {"policy": self.policy, "action": self.action}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AudioTransformInfo:
        return cls(policy=str(data["policy"]), action=str(data["action"]))


@dataclass(frozen=True)
class NormalizationCleanup:
    temp_removed: bool
    lock_released: bool

    def to_dict(self) -> dict[str, bool]:
        return {
            "temp_removed": bool(self.temp_removed),
            "lock_released": bool(self.lock_released),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NormalizationCleanup:
        return cls(
            temp_removed=bool(data["temp_removed"]),
            lock_released=bool(data["lock_released"]),
        )


@dataclass(frozen=True)
class NormalizationProvenance:
    stage: str
    label: str
    notes: str | None = None
    sanitized_argv_summary: str | None = None

    def __post_init__(self) -> None:
        if self.stage != "3C":
            raise VideoContractError("normalization provenance.stage must be 3C")
        if not isinstance(self.label, str) or not self.label:
            raise VideoContractError("normalization provenance.label empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "label": self.label,
            "notes": self.notes,
            "sanitized_argv_summary": self.sanitized_argv_summary,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NormalizationProvenance:
        return cls(
            stage=str(data["stage"]),
            label=str(data["label"]),
            notes=data.get("notes"),
            sanitized_argv_summary=data.get("sanitized_argv_summary"),
        )


@dataclass(frozen=True)
class NormalizationReceipt:
    receipt_id: str
    run_id: str
    plan_id: str
    plan_fingerprint: str
    source_id: str
    source_sha256: str
    source_probe_fingerprint: str | None
    output_artifact: str | None
    output_sha256: str | None
    output_size_bytes: int | None
    output_probe_fingerprint: str | None
    status: NormalizationStatus
    started_at_utc: str
    completed_at_utc: str
    ffmpeg_path: str
    ffmpeg_version: str
    execution_profile: str
    selected_streams: NormalizationSelectedStreams
    applied_transforms: tuple[str, ...]
    frame_rate_conversion: FrameRateConversionInfo
    rotation_transform: RotationTransformInfo
    resize_transform: ResizeTransformInfo
    audio_transform: AudioTransformInfo
    duration_drift_us: int | None
    warnings: tuple[Issue, ...]
    errors: tuple[Issue, ...]
    cleanup: NormalizationCleanup
    provenance: NormalizationProvenance
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise VideoContractError("unsupported NormalizationReceipt schema_version")
        object.__setattr__(
            self, "receipt_id", require_path_safe_id(self.receipt_id, label="receipt_id")
        )
        object.__setattr__(self, "run_id", validate_run_id(self.run_id))
        object.__setattr__(self, "plan_id", require_path_safe_id(self.plan_id, label="plan_id"))
        object.__setattr__(
            self, "source_id", require_path_safe_id(self.source_id, label="source_id")
        )
        object.__setattr__(
            self, "source_sha256", require_sha256(self.source_sha256, label="source_sha256")
        )
        object.__setattr__(
            self,
            "plan_fingerprint",
            require_sha256(self.plan_fingerprint, label="plan_fingerprint"),
        )
        if self.source_probe_fingerprint is not None:
            object.__setattr__(
                self,
                "source_probe_fingerprint",
                require_sha256(self.source_probe_fingerprint, label="source_probe_fingerprint"),
            )
        if self.output_sha256 is not None:
            object.__setattr__(
                self,
                "output_sha256",
                require_sha256(self.output_sha256, label="output_sha256"),
            )
        if self.output_probe_fingerprint is not None:
            object.__setattr__(
                self,
                "output_probe_fingerprint",
                require_sha256(self.output_probe_fingerprint, label="output_probe_fingerprint"),
            )
        if not isinstance(self.status, NormalizationStatus):
            raise VideoContractError("status invalid")
        if self.status == NormalizationStatus.SUCCEEDED and (
            not self.output_artifact or not self.output_sha256
        ):
            raise VideoContractError("succeeded receipt requires output_artifact and output_sha256")
        if self.status == NormalizationStatus.SKIPPED and self.output_artifact is not None:
            raise VideoContractError("skipped receipt requires output_artifact is None")
        if self.output_size_bytes is not None:
            require_non_bool_int(self.output_size_bytes, label="output_size_bytes", minimum=0)
        object.__setattr__(
            self, "started_at_utc", require_utc_z(self.started_at_utc, label="started_at_utc")
        )
        object.__setattr__(
            self,
            "completed_at_utc",
            require_utc_z(self.completed_at_utc, label="completed_at_utc"),
        )
        if not isinstance(self.ffmpeg_path, str) or not self.ffmpeg_path:
            raise VideoContractError("ffmpeg_path empty")
        if not isinstance(self.ffmpeg_version, str) or not self.ffmpeg_version:
            raise VideoContractError("ffmpeg_version empty")
        if not isinstance(self.execution_profile, str) or not self.execution_profile:
            raise VideoContractError("execution_profile empty")
        object.__setattr__(self, "applied_transforms", tuple(self.applied_transforms))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "errors", tuple(self.errors))
        if (
            self.status
            in {
                NormalizationStatus.REJECTED,
                NormalizationStatus.FAILED,
            }
            and not self.errors
        ):
            raise VideoContractError(f"{self.status.value} receipt requires errors")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "receipt_id": self.receipt_id,
            "run_id": self.run_id,
            "plan_id": self.plan_id,
            "plan_fingerprint": self.plan_fingerprint,
            "source_id": self.source_id,
            "source_sha256": self.source_sha256,
            "source_probe_fingerprint": self.source_probe_fingerprint,
            "output_artifact": self.output_artifact,
            "output_sha256": self.output_sha256,
            "output_size_bytes": self.output_size_bytes,
            "output_probe_fingerprint": self.output_probe_fingerprint,
            "status": self.status.value,
            "started_at_utc": self.started_at_utc,
            "completed_at_utc": self.completed_at_utc,
            "ffmpeg_path": self.ffmpeg_path,
            "ffmpeg_version": self.ffmpeg_version,
            "execution_profile": self.execution_profile,
            "selected_streams": self.selected_streams.to_dict(),
            "applied_transforms": list(self.applied_transforms),
            "frame_rate_conversion": self.frame_rate_conversion.to_dict(),
            "rotation_transform": self.rotation_transform.to_dict(),
            "resize_transform": self.resize_transform.to_dict(),
            "audio_transform": self.audio_transform.to_dict(),
            "duration_drift_us": self.duration_drift_us,
            "warnings": [w.to_dict() for w in self.warnings],
            "errors": [e.to_dict() for e in self.errors],
            "cleanup": self.cleanup.to_dict(),
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NormalizationReceipt:
        return cls(
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
            receipt_id=str(data["receipt_id"]),
            run_id=str(data["run_id"]),
            plan_id=str(data["plan_id"]),
            plan_fingerprint=str(data["plan_fingerprint"]),
            source_id=str(data["source_id"]),
            source_sha256=str(data["source_sha256"]),
            source_probe_fingerprint=data.get("source_probe_fingerprint"),
            output_artifact=data.get("output_artifact"),
            output_sha256=data.get("output_sha256"),
            output_size_bytes=data.get("output_size_bytes"),
            output_probe_fingerprint=data.get("output_probe_fingerprint"),
            status=NormalizationStatus(str(data["status"])),
            started_at_utc=str(data["started_at_utc"]),
            completed_at_utc=str(data["completed_at_utc"]),
            ffmpeg_path=str(data["ffmpeg_path"]),
            ffmpeg_version=str(data["ffmpeg_version"]),
            execution_profile=str(data["execution_profile"]),
            selected_streams=NormalizationSelectedStreams.from_dict(data["selected_streams"]),
            applied_transforms=tuple(data.get("applied_transforms", [])),
            frame_rate_conversion=FrameRateConversionInfo.from_dict(data["frame_rate_conversion"]),
            rotation_transform=RotationTransformInfo.from_dict(data["rotation_transform"]),
            resize_transform=ResizeTransformInfo.from_dict(data["resize_transform"]),
            audio_transform=AudioTransformInfo.from_dict(data["audio_transform"]),
            duration_drift_us=data.get("duration_drift_us"),
            warnings=tuple(Issue.from_dict(w) for w in data.get("warnings", [])),
            errors=tuple(Issue.from_dict(e) for e in data.get("errors", [])),
            cleanup=NormalizationCleanup.from_dict(data["cleanup"]),
            provenance=NormalizationProvenance.from_dict(data["provenance"]),
        )

    def fingerprint(self) -> str:
        return hash_canonical_json(self.to_dict())


class FrameTimelineMode(str, Enum):
    TIMELINE_ONLY = "timeline_only"
    SAMPLED = "sampled"
    ALL_FRAMES = "all_frames"


class FrameTimelineStatus(str, Enum):
    SUCCEEDED = "succeeded"
    REJECTED = "rejected"
    FAILED = "failed"


class MappingQuality(str, Enum):
    EXACT = "exact"
    GOOD = "good"
    DEGRADED = "degraded"
    UNRELIABLE = "unreliable"
    FAILED = "failed"


@dataclass(frozen=True)
class FrameTimelineCleanup:
    temp_removed: bool

    def to_dict(self) -> dict[str, bool]:
        return {"temp_removed": self.temp_removed}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FrameTimelineCleanup:
        return cls(temp_removed=bool(data["temp_removed"]))


@dataclass(frozen=True)
class FrameTimelineProvenance:
    stage: str
    label: str
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.stage != "3D":
            raise VideoContractError("FrameTimelineProvenance.stage must be 3D")
        if not isinstance(self.label, str) or not self.label:
            raise VideoContractError("provenance.label empty")

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "label": self.label, "notes": self.notes}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FrameTimelineProvenance:
        return cls(
            stage=str(data["stage"]),
            label=str(data["label"]),
            notes=data.get("notes"),
        )


@dataclass(frozen=True)
class FrameTimelineReceipt:
    receipt_id: str
    run_id: str
    video_id: str
    source_path: str
    source_sha256: str
    mode: FrameTimelineMode
    status: FrameTimelineStatus
    started_at_utc: str
    completed_at_utc: str
    ffprobe_path: str
    ffprobe_version: str
    video_stream_index: int
    time_base: Rational
    frame_rate_mode: FrameRateMode
    frames_parquet: str | None
    frames_parquet_sha256: str | None
    frame_count: int
    ok_count: int
    skipped_count: int
    failed_count: int
    unknown_count: int
    missing_pts_count: int
    duplicate_pts_count: int
    non_monotonic_pts_count: int
    mapping_quality: MappingQuality
    materialized: bool
    artifact_manifest: str | None
    warnings: tuple[Issue, ...]
    errors: tuple[Issue, ...]
    cleanup: FrameTimelineCleanup
    provenance: FrameTimelineProvenance
    normalization_receipt_path: str | None = None
    sample_every: int | None = None
    materialized_frame_count: int | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise VideoContractError("unsupported FrameTimelineReceipt schema_version")
        object.__setattr__(
            self, "receipt_id", require_path_safe_id(self.receipt_id, label="receipt_id")
        )
        object.__setattr__(self, "run_id", validate_run_id(self.run_id))
        object.__setattr__(self, "video_id", require_path_safe_id(self.video_id, label="video_id"))
        object.__setattr__(
            self, "source_sha256", require_sha256(self.source_sha256, label="source_sha256")
        )
        if not isinstance(self.mode, FrameTimelineMode):
            raise VideoContractError("mode invalid")
        if not isinstance(self.status, FrameTimelineStatus):
            raise VideoContractError("status invalid")
        if not isinstance(self.mapping_quality, MappingQuality):
            raise VideoContractError("mapping_quality invalid")
        if not isinstance(self.frame_rate_mode, FrameRateMode):
            raise VideoContractError("frame_rate_mode invalid")
        object.__setattr__(
            self, "started_at_utc", require_utc_z(self.started_at_utc, label="started_at_utc")
        )
        object.__setattr__(
            self,
            "completed_at_utc",
            require_utc_z(self.completed_at_utc, label="completed_at_utc"),
        )
        require_non_bool_int(self.video_stream_index, label="video_stream_index", minimum=0)
        for name in (
            "frame_count",
            "ok_count",
            "skipped_count",
            "failed_count",
            "unknown_count",
            "missing_pts_count",
            "duplicate_pts_count",
            "non_monotonic_pts_count",
        ):
            require_non_bool_int(getattr(self, name), label=name, minimum=0)
        if self.frames_parquet_sha256 is not None:
            object.__setattr__(
                self,
                "frames_parquet_sha256",
                require_sha256(self.frames_parquet_sha256, label="frames_parquet_sha256"),
            )
        if self.sample_every is not None:
            require_non_bool_int(self.sample_every, label="sample_every", minimum=1)
        if self.materialized_frame_count is not None:
            require_non_bool_int(
                self.materialized_frame_count, label="materialized_frame_count", minimum=0
            )
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "errors", tuple(self.errors))
        if (
            self.status in {FrameTimelineStatus.REJECTED, FrameTimelineStatus.FAILED}
            and not self.errors
        ):
            raise VideoContractError(f"{self.status.value} receipt requires errors")
        if self.status == FrameTimelineStatus.SUCCEEDED and (
            not self.frames_parquet or not self.frames_parquet_sha256
        ):
            raise VideoContractError("succeeded receipt requires frames_parquet and sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "receipt_id": self.receipt_id,
            "run_id": self.run_id,
            "video_id": self.video_id,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "normalization_receipt_path": self.normalization_receipt_path,
            "mode": self.mode.value,
            "status": self.status.value,
            "started_at_utc": self.started_at_utc,
            "completed_at_utc": self.completed_at_utc,
            "ffprobe_path": self.ffprobe_path,
            "ffprobe_version": self.ffprobe_version,
            "video_stream_index": self.video_stream_index,
            "time_base": self.time_base.to_dict(),
            "frame_rate_mode": self.frame_rate_mode.value,
            "frames_parquet": self.frames_parquet,
            "frames_parquet_sha256": self.frames_parquet_sha256,
            "frame_count": self.frame_count,
            "ok_count": self.ok_count,
            "skipped_count": self.skipped_count,
            "failed_count": self.failed_count,
            "unknown_count": self.unknown_count,
            "missing_pts_count": self.missing_pts_count,
            "duplicate_pts_count": self.duplicate_pts_count,
            "non_monotonic_pts_count": self.non_monotonic_pts_count,
            "mapping_quality": self.mapping_quality.value,
            "sample_every": self.sample_every,
            "materialized": self.materialized,
            "materialized_frame_count": self.materialized_frame_count,
            "artifact_manifest": self.artifact_manifest,
            "warnings": [w.to_dict() for w in self.warnings],
            "errors": [e.to_dict() for e in self.errors],
            "cleanup": self.cleanup.to_dict(),
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FrameTimelineReceipt:
        return cls(
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
            receipt_id=str(data["receipt_id"]),
            run_id=str(data["run_id"]),
            video_id=str(data["video_id"]),
            source_path=str(data["source_path"]),
            source_sha256=str(data["source_sha256"]),
            normalization_receipt_path=data.get("normalization_receipt_path"),
            mode=FrameTimelineMode(str(data["mode"])),
            status=FrameTimelineStatus(str(data["status"])),
            started_at_utc=str(data["started_at_utc"]),
            completed_at_utc=str(data["completed_at_utc"]),
            ffprobe_path=str(data["ffprobe_path"]),
            ffprobe_version=str(data["ffprobe_version"]),
            video_stream_index=int(data["video_stream_index"]),
            time_base=Rational.from_dict(data["time_base"], label="time_base"),
            frame_rate_mode=FrameRateMode(str(data["frame_rate_mode"])),
            frames_parquet=data.get("frames_parquet"),
            frames_parquet_sha256=data.get("frames_parquet_sha256"),
            frame_count=int(data["frame_count"]),
            ok_count=int(data["ok_count"]),
            skipped_count=int(data["skipped_count"]),
            failed_count=int(data["failed_count"]),
            unknown_count=int(data["unknown_count"]),
            missing_pts_count=int(data["missing_pts_count"]),
            duplicate_pts_count=int(data["duplicate_pts_count"]),
            non_monotonic_pts_count=int(data["non_monotonic_pts_count"]),
            mapping_quality=MappingQuality(str(data["mapping_quality"])),
            sample_every=data.get("sample_every"),
            materialized=bool(data["materialized"]),
            materialized_frame_count=data.get("materialized_frame_count"),
            artifact_manifest=data.get("artifact_manifest"),
            warnings=tuple(Issue.from_dict(w) for w in data.get("warnings", [])),
            errors=tuple(Issue.from_dict(e) for e in data.get("errors", [])),
            cleanup=FrameTimelineCleanup.from_dict(data["cleanup"]),
            provenance=FrameTimelineProvenance.from_dict(data["provenance"]),
        )

    def fingerprint(self) -> str:
        return hash_canonical_json(self.to_dict())
