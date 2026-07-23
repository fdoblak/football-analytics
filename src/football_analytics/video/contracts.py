"""Schema/policy loading and contract helpers for Stage 3A video ingest."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from football_analytics.core.hashing import hash_canonical_json
from football_analytics.video.types import (
    SCHEMA_VERSION,
    IngestMode,
    IngestReceipt,
    IngestRequest,
    NormalizationReceipt,
    NormalizePlan,
    Rational,
    ReceiptStatus,
    SourceKind,
    VideoContractError,
    VideoPolicyError,
    VideoProbe,
    VideoSource,
)

SCHEMA_FILES = (
    "video_source.schema.json",
    "ingest_request.schema.json",
    "video_probe.schema.json",
    "normalize_plan.schema.json",
    "ingest_receipt.schema.json",
    "normalization_receipt.schema.json",
)

DEFAULT_POLICY_REL = Path("configs/video/ingest_policy.yaml")
DEFAULT_SCHEMA_REL = Path("schemas/video")


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_json_schema(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise VideoContractError(f"schema missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise VideoContractError(f"schema root must be object: {path}")
    return data


def load_all_video_schemas(schema_root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    ids: set[str] = set()
    for name in SCHEMA_FILES:
        path = schema_root / name
        schema = load_json_schema(path)
        schema_id = schema.get("$id")
        if not isinstance(schema_id, str) or not schema_id:
            raise VideoContractError(f"schema $id missing: {name}")
        if schema_id in ids:
            raise VideoContractError(f"duplicate schema $id: {schema_id}")
        ids.add(schema_id)
        out[name] = schema
    return out


def load_ingest_policy(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise VideoPolicyError(f"policy missing: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise VideoPolicyError("policy root must be mapping")
    if int(raw.get("schema_version", -1)) != SCHEMA_VERSION:
        raise VideoPolicyError("unsupported policy schema_version")
    required = {
        "policy_version",
        "allowed_source_kinds",
        "allowed_file_extensions",
        "allowed_container_names",
        "allowed_video_codecs",
        "allowed_audio_codecs",
        "allowed_pixel_formats",
        "maximum_source_size_bytes",
        "minimum_duration_us",
        "maximum_duration_us",
        "minimum_width",
        "minimum_height",
        "maximum_width",
        "maximum_height",
        "network_sources_allowed",
        "symlinks_allowed",
        "special_files_allowed",
        "overwrite_allowed",
        "hash_algorithm",
        "canonical_time_unit",
        "unknown_frame_count_allowed",
        "unknown_duration_allowed",
        "rotation_policy",
        "stream_selection_policy",
        "fixture_policy",
        "normalization_defaults",
        "ffprobe_policy",
        "ffmpeg_policy",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise VideoPolicyError(f"policy missing keys: {missing}")
    if raw["network_sources_allowed"] is not False:
        raise VideoPolicyError("network_sources_allowed must be false")
    if raw["symlinks_allowed"] is not False:
        raise VideoPolicyError("symlinks_allowed must be false")
    if raw["special_files_allowed"] is not False:
        raise VideoPolicyError("special_files_allowed must be false")
    if raw["overwrite_allowed"] is not False:
        raise VideoPolicyError("overwrite_allowed must be false")
    if raw["hash_algorithm"] != "sha256":
        raise VideoPolicyError("hash_algorithm must be sha256")
    if raw["canonical_time_unit"] != "microseconds":
        raise VideoPolicyError("canonical_time_unit must be microseconds")
    norms = raw["normalization_defaults"]
    if not isinstance(norms, dict):
        raise VideoPolicyError("normalization_defaults must be a mapping")
    if str(norms.get("hardware_acceleration", "")) != "disabled":
        raise VideoPolicyError("hardware_acceleration must be disabled")
    if norms.get("overwrite_policy") is not False:
        raise VideoPolicyError("normalization_defaults.overwrite_policy must be false")
    if norms.get("inplace_forbidden") is not True:
        raise VideoPolicyError("inplace_forbidden must be true")
    ffprobe = raw["ffprobe_policy"]
    if not isinstance(ffprobe, dict):
        raise VideoPolicyError("ffprobe_policy must be a mapping")
    ffprobe_required = {
        "ffprobe_binary",
        "allowed_binary_realpaths",
        "probe_timeout_seconds",
        "maximum_stdout_bytes",
        "maximum_stderr_bytes",
        "maximum_stream_count",
        "maximum_metadata_entries",
        "maximum_metadata_key_length",
        "maximum_metadata_value_length",
        "maximum_json_depth",
        "count_frames",
        "persist_raw_ffprobe_json",
        "protocol_whitelist",
        "runtime_root",
    }
    missing_ff = sorted(ffprobe_required - set(ffprobe))
    if missing_ff:
        raise VideoPolicyError(f"ffprobe_policy missing keys: {missing_ff}")
    if ffprobe.get("count_frames") is not False:
        raise VideoPolicyError("ffprobe_policy.count_frames must be false by default")
    if ffprobe.get("persist_raw_ffprobe_json") is not False:
        raise VideoPolicyError("persist_raw_ffprobe_json must be false by default")
    if str(ffprobe.get("ffprobe_binary")) != "/usr/bin/ffprobe":
        raise VideoPolicyError("ffprobe_binary must be /usr/bin/ffprobe")
    ffmpeg = raw["ffmpeg_policy"]
    if not isinstance(ffmpeg, dict):
        raise VideoPolicyError("ffmpeg_policy must be a mapping")
    ffmpeg_required = {
        "ffmpeg_binary",
        "allowed_binary_realpaths",
        "required_encoders",
        "required_filters",
        "timeout_base_seconds",
        "timeout_per_media_second",
        "maximum_timeout_seconds",
        "maximum_stderr_bytes",
        "maximum_progress_bytes",
        "maximum_parallel_normalizations",
        "ffmpeg_threads",
        "minimum_free_space_bytes",
        "minimum_free_space_after_estimate_bytes",
        "output_size_estimation_policy",
        "size_estimate_multiplier",
        "video_crf",
        "video_preset",
        "runtime_root",
        "protocol_whitelist",
    }
    missing_fm = sorted(ffmpeg_required - set(ffmpeg))
    if missing_fm:
        raise VideoPolicyError(f"ffmpeg_policy missing keys: {missing_fm}")
    if str(ffmpeg.get("ffmpeg_binary")) != "/usr/bin/ffmpeg":
        raise VideoPolicyError("ffmpeg_binary must be /usr/bin/ffmpeg")
    if int(ffmpeg.get("maximum_parallel_normalizations", -1)) != 1:
        raise VideoPolicyError("maximum_parallel_normalizations must be 1")
    # Enum alignment with Python
    for kind in raw["allowed_source_kinds"]:
        SourceKind(kind)
    return raw


def validate_payload_against_schema(payload: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    import jsonschema

    jsonschema.validate(instance=dict(payload), schema=dict(schema))


def build_normalize_plan(
    *,
    plan_id: str,
    source_id: str,
    source_sha256: str,
    policy_version: str,
    required: bool,
    reasons: tuple[str, ...],
    target_container: str,
    target_video_codec: str,
    target_audio_policy: str,
    target_pixel_format: str,
    target_width: int | None,
    target_height: int | None,
    resize_policy: str,
    target_frame_rate: Rational | None,
    frame_rate_policy: str,
    target_time_base: Rational | None,
    rotation_policy: str,
    sar_policy: str,
    audio_policy: str,
    copy_metadata_policy: str,
    estimated_output_path: str,
    overwrite_policy: bool = False,
) -> NormalizePlan:
    """Construct a NormalizePlan with deterministic fingerprint."""
    if overwrite_policy:
        raise VideoContractError("overwrite_policy must be false")
    draft = {
        "schema_version": SCHEMA_VERSION,
        "plan_id": plan_id,
        "source_id": source_id,
        "source_sha256": source_sha256,
        "policy_version": policy_version,
        "required": required,
        "reasons": list(reasons),
        "target_container": target_container,
        "target_video_codec": target_video_codec,
        "target_audio_policy": target_audio_policy,
        "target_pixel_format": target_pixel_format,
        "target_width": target_width,
        "target_height": target_height,
        "resize_policy": resize_policy,
        "target_frame_rate": None if target_frame_rate is None else target_frame_rate.to_dict(),
        "frame_rate_policy": frame_rate_policy,
        "target_time_base": None if target_time_base is None else target_time_base.to_dict(),
        "rotation_policy": rotation_policy,
        "sar_policy": sar_policy,
        "audio_policy": audio_policy,
        "copy_metadata_policy": copy_metadata_policy,
        "estimated_output_path": estimated_output_path,
        "overwrite_policy": overwrite_policy,
    }
    fingerprint = hash_canonical_json(draft)
    draft["plan_fingerprint"] = fingerprint
    return NormalizePlan.from_dict(draft)


def assert_schema_python_enum_alignment(policy: Mapping[str, Any]) -> None:
    """Ensure policy allowlists align with Python enums used by contracts."""
    for kind in policy["allowed_source_kinds"]:
        SourceKind(kind)
    for mode in IngestMode:
        assert mode.value
    for status in ReceiptStatus:
        if status.value in {"succeeded", "completed"}:
            raise VideoContractError("false-success status must not exist")


__all__ = [
    "SCHEMA_FILES",
    "DEFAULT_POLICY_REL",
    "DEFAULT_SCHEMA_REL",
    "default_repo_root",
    "load_json_schema",
    "load_all_video_schemas",
    "load_ingest_policy",
    "validate_payload_against_schema",
    "build_normalize_plan",
    "assert_schema_python_enum_alignment",
    "VideoSource",
    "IngestRequest",
    "VideoProbe",
    "NormalizePlan",
    "IngestReceipt",
    "NormalizationReceipt",
]
