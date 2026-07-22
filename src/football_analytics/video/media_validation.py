"""Policy-based media validation against Stage 3A/3B ingest policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from football_analytics.video.types import (
    AudioStreamInfo,
    Issue,
    VideoProbe,
    VideoStreamInfo,
)


@dataclass
class MediaValidationResult:
    status: str  # accepted | rejected
    accepted: bool
    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)
    policy_version: str = ""
    source_id: str = ""
    source_sha256: str = ""
    probe_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "accepted": self.accepted,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "policy_version": self.policy_version,
            "source_id": self.source_id,
            "source_sha256": self.source_sha256,
            "probe_fingerprint": self.probe_fingerprint,
        }


def _err(result: MediaValidationResult, code: str, message: str) -> None:
    result.errors.append(Issue(code=code, message=message))
    result.accepted = False
    result.status = "rejected"


def _warn(result: MediaValidationResult, code: str, message: str) -> None:
    result.warnings.append(Issue(code=code, message=message))


def validate_probe_against_policy(
    probe: VideoProbe,
    policy: Mapping[str, Any],
    *,
    source_size_bytes: int,
) -> MediaValidationResult:
    result = MediaValidationResult(
        status="accepted",
        accepted=True,
        policy_version=str(policy["policy_version"]),
        source_id=probe.source_id,
        source_sha256=probe.source_sha256,
        probe_fingerprint=probe.fingerprint(),
    )
    # Carry parser warnings forward as soft warnings when severity is warning/unsupported
    for pw in probe.warnings:
        if pw.severity.value == "hard_failure":
            _err(result, pw.code, pw.message)
        else:
            _warn(result, pw.code, pw.message)

    max_size = int(policy["maximum_source_size_bytes"])
    if source_size_bytes < 0 or source_size_bytes > max_size:
        _err(result, "SOURCE_SIZE_OUT_OF_RANGE", "source size outside policy")

    # Streams
    video = next(
        (
            s
            for s in probe.streams
            if isinstance(s, VideoStreamInfo)
            and s.stream_index == probe.selected_video_stream_index
        ),
        None,
    )
    if video is None:
        _err(result, "NO_USABLE_VIDEO_STREAM", "selected video stream missing")
        return result

    if video.disposition.attached_pic:
        _err(result, "NO_USABLE_VIDEO_STREAM", "attached picture selected as primary")

    # Dimensions
    if video.width < int(policy["minimum_width"]) or video.height < int(policy["minimum_height"]):
        _err(result, "DIMENSIONS_OUT_OF_RANGE", "dimensions below policy minimum")
    if video.width > int(policy["maximum_width"]) or video.height > int(policy["maximum_height"]):
        _err(result, "DIMENSIONS_OUT_OF_RANGE", "dimensions above policy maximum")

    # Duration
    duration = probe.duration_us if probe.duration_us is not None else video.duration_us
    if duration is None:
        if not policy.get("unknown_duration_allowed", True):
            _err(result, "DURATION_OUT_OF_RANGE", "unknown duration forbidden")
        else:
            _warn(result, "UNKNOWN_DURATION", "duration unknown")
    else:
        if duration < int(policy["minimum_duration_us"]) or duration > int(
            policy["maximum_duration_us"]
        ):
            _err(result, "DURATION_OUT_OF_RANGE", "duration outside policy")

    # Frame count
    if video.frame_count is None and not policy.get("unknown_frame_count_allowed", True):
        _err(result, "UNKNOWN_FRAME_COUNT", "unknown frame count forbidden")

    # Container / codec / pixel format allowlists
    containers = {str(x).lower() for x in policy["allowed_container_names"]}
    if probe.container is not None:
        # format_name may be "mov,mp4,..." — check any token
        tokens = {t.strip().lower() for t in (probe.format_name or probe.container).split(",")}
        if (
            not (tokens & containers)
            and probe.container.lower() not in containers
            and not any(t in containers for t in tokens)
        ):
            _err(result, "UNSUPPORTED_CONTAINER", f"container not allowed: {probe.container}")
    else:
        _warn(result, "UNKNOWN_CONTAINER", "container unknown")

    codecs = {str(x).lower() for x in policy["allowed_video_codecs"]}
    if video.codec_name.lower() not in codecs:
        _err(result, "UNSUPPORTED_VIDEO_CODEC", f"video codec not allowed: {video.codec_name}")

    pix = {str(x).lower() for x in policy["allowed_pixel_formats"]}
    if video.pixel_format is None:
        _warn(result, "UNKNOWN_PIXEL_FORMAT", "pixel format unknown")
    elif video.pixel_format.lower() not in pix:
        _err(result, "UNSUPPORTED_PIXEL_FORMAT", f"pixel format not allowed: {video.pixel_format}")

    # Audio optional
    audio = None
    if probe.selected_audio_stream_index is not None:
        audio = next(
            (
                s
                for s in probe.streams
                if isinstance(s, AudioStreamInfo)
                and s.stream_index == probe.selected_audio_stream_index
            ),
            None,
        )
    if audio is None:
        if not policy["stream_selection_policy"].get("audio_optional", True):
            _err(result, "AUDIO_REQUIRED", "audio stream required by policy")
    else:
        acodecs = {str(x).lower() for x in policy["allowed_audio_codecs"]}
        if audio.codec_name.lower() not in acodecs:
            _warn(
                result,
                "UNSUPPORTED_AUDIO_CODEC",
                f"audio codec not in allowlist: {audio.codec_name}",
            )

    # Rotation allowlist already normalized on model; conflicting soft warnings already attached
    allowed_rot = set(policy["rotation_policy"].get("normalize_to", [0, 90, 180, 270]))
    if video.rotation_degrees not in allowed_rot:
        _err(result, "UNSUPPORTED_ROTATION", "rotation not in policy allowlist")

    if policy.get("network_sources_allowed") is not False:
        _err(result, "NETWORK_SOURCES_FORBIDDEN", "network sources must remain disabled")

    if result.errors:
        result.accepted = False
        result.status = "rejected"
    else:
        result.accepted = True
        result.status = "accepted"
    return result
