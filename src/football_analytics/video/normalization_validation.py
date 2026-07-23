"""Post-normalization conformance checks (Stage 3C). Caller decides publish."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from football_analytics.video.normalization import PlannedNormalization
from football_analytics.video.types import (
    AudioStreamInfo,
    Issue,
    VideoProbe,
    VideoStreamInfo,
)


@dataclass
class ConformanceResult:
    ok: bool
    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)
    duration_drift_us: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "duration_drift_us": self.duration_drift_us,
        }


def _err(result: ConformanceResult, code: str, message: str) -> None:
    result.errors.append(Issue(code=code, message=message))
    result.ok = False


def _warn(result: ConformanceResult, code: str, message: str) -> None:
    result.warnings.append(Issue(code=code, message=message))


def _primary_video(probe: VideoProbe) -> VideoStreamInfo | None:
    for stream in probe.streams:
        if (
            isinstance(stream, VideoStreamInfo)
            and stream.stream_index == probe.selected_video_stream_index
        ):
            return stream
    return None


def _primary_audio(probe: VideoProbe) -> AudioStreamInfo | None:
    idx = probe.selected_audio_stream_index
    if idx is None:
        return None
    for stream in probe.streams:
        if isinstance(stream, AudioStreamInfo) and stream.stream_index == idx:
            return stream
    return None


def validate_normalized_output(
    *,
    plan: PlannedNormalization,
    source_probe: VideoProbe,
    output_probe: VideoProbe,
    policy: Mapping[str, Any],
) -> ConformanceResult:
    """Compare output probe against plan/policy. Does not publish artifacts."""
    result = ConformanceResult(ok=True)
    nd = policy["normalization_defaults"]
    np = plan.plan
    out_v = _primary_video(output_probe)
    if out_v is None:
        _err(result, "OUTPUT_NO_VIDEO_STREAM", "normalized output has no video stream")
        return result

    # Container
    target_c = str(np.target_container).lower()
    fmt = (output_probe.format_name or "").lower()
    container = (output_probe.container or "").lower()
    if target_c not in fmt.split(",") and container not in {target_c, "mov"}:
        _err(result, "OUTPUT_CONTAINER_MISMATCH", "output container not canonical")

    # Codec
    codec = out_v.codec_name.lower()
    target_codec = str(np.target_video_codec).lower()
    if target_codec in {"h264", "avc"} and codec not in {"h264", "avc", "avc1"}:
        _err(result, "OUTPUT_CODEC_MISMATCH", f"expected h264 got {codec}")
    elif target_codec not in {"h264", "avc"} and codec != target_codec:
        _err(result, "OUTPUT_CODEC_MISMATCH", f"expected {target_codec} got {codec}")

    # Pixel format
    pix = (out_v.pixel_format or "").lower()
    if pix != str(np.target_pixel_format).lower():
        _err(result, "OUTPUT_PIXEL_FORMAT_MISMATCH", f"expected {np.target_pixel_format} got {pix}")

    # Dimensions
    if np.target_width is not None and out_v.width != np.target_width:
        _err(
            result,
            "OUTPUT_DIMENSIONS_MISMATCH",
            f"width {out_v.width} != target {np.target_width}",
        )
    if np.target_height is not None and out_v.height != np.target_height:
        _err(
            result,
            "OUTPUT_DIMENSIONS_MISMATCH",
            f"height {out_v.height} != target {np.target_height}",
        )

    # Rotation baked → output should be 0
    if plan.bake_rotation and out_v.rotation_degrees != 0:
        _err(result, "OUTPUT_ROTATION_MISMATCH", "rotation not baked to 0")
    if not plan.bake_rotation and out_v.rotation_degrees not in {0, plan.source_rotation_degrees}:
        _warn(result, "OUTPUT_ROTATION_UNEXPECTED", "unexpected output rotation")

    # SAR
    if str(nd.get("sar_policy")) == "force_1_1":
        sar = out_v.sample_aspect_ratio
        if sar.numerator != 1 or sar.denominator != 1:
            _err(result, "OUTPUT_SAR_MISMATCH", "output SAR not 1/1")

    # Audio
    out_a = _primary_audio(output_probe)
    if plan.audio_action in {"copy", "transcode"}:
        if out_a is None:
            _err(result, "OUTPUT_AUDIO_MISMATCH", "expected audio stream missing")
        elif plan.audio_action == "transcode" and out_a.codec_name.lower() not in {"aac", "mp4a"}:
            _err(result, "OUTPUT_AUDIO_MISMATCH", "expected aac audio")
    elif plan.audio_action in {"drop", "absent", "none"} and out_a is not None:
        _err(result, "OUTPUT_AUDIO_MISMATCH", "audio should be absent")

    # Duration drift
    src_d = source_probe.duration_us
    out_d = output_probe.duration_us
    max_drift = int(nd.get("maximum_duration_drift_us", 100_000))
    if src_d is not None and out_d is not None:
        drift = abs(out_d - src_d)
        result.duration_drift_us = out_d - src_d
        if drift > max_drift:
            _err(
                result,
                "OUTPUT_DURATION_MISMATCH",
                f"duration drift {drift}us exceeds {max_drift}us",
            )
    elif src_d is not None and out_d is None:
        _warn(result, "OUTPUT_DURATION_UNKNOWN", "output duration unknown")

    return result
