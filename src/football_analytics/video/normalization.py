"""Pure normalization planner (Stage 3C). No subprocess or filesystem mutation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from football_analytics.video.contracts import build_normalize_plan
from football_analytics.video.types import (
    AudioStreamInfo,
    FrameRateMode,
    NormalizePlan,
    Rational,
    VideoProbe,
    VideoStreamInfo,
)

REASON_CONTAINER = "CONTAINER_NOT_CANONICAL"
REASON_CODEC = "VIDEO_CODEC_NOT_CANONICAL"
REASON_PIXFMT = "PIXEL_FORMAT_NOT_CANONICAL"
REASON_DIMS = "DIMENSIONS_REQUIRE_NORMALIZATION"
REASON_ROTATION = "ROTATION_REQUIRES_BAKE"
REASON_SAR = "SAR_REQUIRES_NORMALIZATION"
REASON_FPS = "FRAME_RATE_REQUIRES_NORMALIZATION"
REASON_AUDIO_TX = "AUDIO_REQUIRES_TRANSCODE"
REASON_AUDIO_DROP = "AUDIO_WILL_BE_DROPPED"
REASON_CANONICAL = "ALREADY_CANONICAL"


@dataclass(frozen=True)
class PlannedNormalization:
    """Plan plus stream indices / transform flags for argv building."""

    plan: NormalizePlan
    video_stream_index: int
    audio_stream_index: int | None
    video_stream_ordinal: int
    audio_stream_ordinal: int | None
    source_rotation_degrees: int
    bake_rotation: bool
    resize_performed: bool
    force_setsar: bool
    frame_rate_conversion: bool
    audio_action: str
    applied_transforms: tuple[str, ...]
    source_width: int
    source_height: int
    display_width: int
    display_height: int


def compute_target_dimensions(
    width: int,
    height: int,
    *,
    max_width: int,
    max_height: int,
    even: bool,
    upscale: bool,
) -> tuple[int, int]:
    """Aspect-preserving fit within max bounds; optional even rounding."""
    if width < 1 or height < 1:
        raise ValueError("dimensions must be >= 1")
    tw, th = width, height
    if not upscale:
        scale = min(1.0, max_width / tw, max_height / th)
    else:
        scale = min(max_width / tw, max_height / th)
    tw = max(1, int(tw * scale))
    th = max(1, int(th * scale))
    if even:
        tw = tw - (tw % 2)
        th = th - (th % 2)
        tw = max(2 if even else 1, tw)
        th = max(2 if even else 1, th)
    # Re-clamp after even rounding without upscaling past max
    if tw > max_width:
        tw = max_width - (max_width % 2 if even else 0)
    if th > max_height:
        th = max_height - (max_height % 2 if even else 0)
    return tw, th


def estimate_output_bytes(probe: VideoProbe, policy: Mapping[str, Any]) -> int:
    ff = policy["ffmpeg_policy"]
    multiplier = float(ff.get("size_estimate_multiplier", 2.0))
    est_policy = str(ff.get("output_size_estimation_policy", "bitrate_or_source_times_two"))
    source_bytes = int(probe.file_size_bytes)
    if est_policy == "bitrate_or_source_times_two" and probe.bit_rate_bps and probe.duration_us:
        # bitrate_bps * duration_s / 8
        duration_s = probe.duration_us / 1_000_000.0
        bitrate_est = int(probe.bit_rate_bps * duration_s / 8.0)
        return max(int(source_bytes * multiplier), bitrate_est, 1024)
    return max(int(source_bytes * multiplier), 1024)


def compute_timeout_seconds(duration_us: int | None, policy: Mapping[str, Any]) -> float:
    ff = policy["ffmpeg_policy"]
    base = float(ff["timeout_base_seconds"])
    per = float(ff["timeout_per_media_second"])
    maximum = float(ff["maximum_timeout_seconds"])
    media_s = 0.0 if duration_us is None else max(0.0, duration_us / 1_000_000.0)
    return min(maximum, base + per * media_s)


def _container_canonical(probe: VideoProbe, target: str) -> bool:
    target = target.lower()
    fmt = (probe.format_name or "").lower()
    container = (probe.container or "").lower()
    if target in fmt.split(","):
        return True
    if target == "mp4" and container in {"mp4", "mov"}:
        return True
    return container == target


def _codec_canonical(codec_name: str, target: str) -> bool:
    c = codec_name.lower()
    t = target.lower()
    if t in {"h264", "avc", "avc1"}:
        return c in {"h264", "avc", "avc1"} and c != "libx264"
    return c == t


def _display_dims(video: VideoStreamInfo) -> tuple[int, int]:
    w, h = video.width, video.height
    if video.rotation_degrees in {90, 270}:
        return h, w
    return w, h


def _video_ordinal(probe: VideoProbe, absolute_index: int) -> int:
    ordinal = 0
    for stream in probe.streams:
        if isinstance(stream, VideoStreamInfo):
            if stream.stream_index == absolute_index:
                return ordinal
            ordinal += 1
    return 0


def _audio_ordinal(probe: VideoProbe, absolute_index: int | None) -> int | None:
    if absolute_index is None:
        return None
    ordinal = 0
    for stream in probe.streams:
        if isinstance(stream, AudioStreamInfo):
            if stream.stream_index == absolute_index:
                return ordinal
            ordinal += 1
    return None


def _selected_video(probe: VideoProbe) -> VideoStreamInfo:
    for stream in probe.streams:
        if (
            isinstance(stream, VideoStreamInfo)
            and stream.stream_index == probe.selected_video_stream_index
        ):
            return stream
    raise ValueError("selected video stream missing")


def _selected_audio(probe: VideoProbe) -> AudioStreamInfo | None:
    idx = probe.selected_audio_stream_index
    if idx is None:
        return None
    for stream in probe.streams:
        if isinstance(stream, AudioStreamInfo) and stream.stream_index == idx:
            return stream
    return None


def plan_normalization(
    *,
    probe: VideoProbe,
    policy: Mapping[str, Any],
    output_path: str,
    plan_id: str,
    source_id: str,
) -> PlannedNormalization:
    """Derive a NormalizePlan + transform flags from probe and policy (pure)."""
    nd = policy["normalization_defaults"]
    target_container = str(nd["target_container"])
    target_codec = str(nd["target_video_codec"])
    target_pix = str(nd["target_pixel_format"])
    audio_policy = str(nd["target_audio_policy"])
    frame_rate_policy = str(nd["frame_rate_policy"])
    sar_policy = str(nd["sar_policy"])
    resize_policy = str(nd["resize_policy"])
    rotation_policy = str(nd["rotation_policy"])
    copy_metadata_policy = str(nd["copy_metadata_policy"])
    max_w = int(nd["maximum_target_width"])
    max_h = int(nd["maximum_target_height"])
    even = bool(nd.get("require_even_dimensions", True))
    upscale = bool(nd.get("upscaling_allowed", False))

    video = _selected_video(probe)
    audio = _selected_audio(probe)
    reasons: list[str] = []
    transforms: list[str] = []

    if not _container_canonical(probe, target_container):
        reasons.append(REASON_CONTAINER)
        transforms.append("remux_container")

    if not _codec_canonical(video.codec_name, target_codec):
        reasons.append(REASON_CODEC)
        transforms.append("transcode_video")

    pix = video.pixel_format or ""
    if pix.lower() != target_pix.lower():
        reasons.append(REASON_PIXFMT)
        transforms.append("convert_pixel_format")

    bake_rotation = False
    if video.rotation_degrees != 0:
        reasons.append(REASON_ROTATION)
        bake_rotation = True
        transforms.append(f"bake_rotation_{video.rotation_degrees}")

    disp_w, disp_h = _display_dims(video)
    target_w, target_h = compute_target_dimensions(
        disp_w,
        disp_h,
        max_width=max_w,
        max_height=max_h,
        even=even,
        upscale=upscale,
    )
    resize_performed = False
    odd = (disp_w % 2 != 0) or (disp_h % 2 != 0)
    if odd or target_w != disp_w or target_h != disp_h:
        reasons.append(REASON_DIMS)
        resize_performed = True
        transforms.append("resize")

    force_setsar = False
    sar = video.sample_aspect_ratio
    if sar_policy == "force_1_1" and (sar.numerator != 1 or sar.denominator != 1):
        reasons.append(REASON_SAR)
        force_setsar = True
        transforms.append("force_sar_1_1")

    frame_rate_conversion = False
    target_frame_rate: Rational | None = None
    if frame_rate_policy == "preserve_unless_explicit_cfr":
        if video.frame_rate_mode == FrameRateMode.VFR:
            reasons.append(REASON_FPS)
            frame_rate_conversion = True
            fr = nd.get("force_cfr_target_frame_rate") or {"numerator": 25, "denominator": 1}
            target_frame_rate = Rational(int(fr["numerator"]), int(fr["denominator"]))
            transforms.append("force_cfr")
        elif video.frame_rate_mode == FrameRateMode.UNKNOWN:
            # preserve — do not invent a rate
            frame_rate_conversion = False
            target_frame_rate = None
        else:
            # CFR preserve
            frame_rate_conversion = False
            target_frame_rate = None
    elif frame_rate_policy == "force_cfr":
        reasons.append(REASON_FPS)
        frame_rate_conversion = True
        fr = nd.get("force_cfr_target_frame_rate") or {"numerator": 25, "denominator": 1}
        target_frame_rate = Rational(int(fr["numerator"]), int(fr["denominator"]))
        transforms.append("force_cfr")

    audio_action = "absent"
    if audio_policy == "copy_if_present_else_drop":
        if audio is None:
            audio_action = "absent"
        else:
            codec = audio.codec_name.lower()
            if codec in {"aac", "mp4a"}:
                audio_action = "copy"
                transforms.append("copy_audio")
            else:
                reasons.append(REASON_AUDIO_TX)
                audio_action = "transcode"
                transforms.append("transcode_audio")
    elif audio_policy == "drop":
        if audio is not None:
            reasons.append(REASON_AUDIO_DROP)
            audio_action = "drop"
            transforms.append("drop_audio")
        else:
            audio_action = "absent"
    elif audio_policy == "copy":
        audio_action = "copy" if audio is not None else "absent"
    elif audio_policy == "transcode":
        if audio is not None:
            reasons.append(REASON_AUDIO_TX)
            audio_action = "transcode"
        else:
            audio_action = "absent"

    # Deduplicate reasons while preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            ordered.append(r)

    if not ordered:
        ordered = [REASON_CANONICAL]
        required = False
    else:
        required = True

    # required=false only with ALREADY_CANONICAL
    if not required:
        ordered = [REASON_CANONICAL]
        plan_tw, plan_th = disp_w, disp_h
    else:
        plan_tw, plan_th = target_w, target_h

    plan = build_normalize_plan(
        plan_id=plan_id,
        source_id=source_id,
        source_sha256=probe.source_sha256,
        policy_version=str(policy["policy_version"]),
        required=required,
        reasons=tuple(ordered),
        target_container=target_container,
        target_video_codec=target_codec,
        target_audio_policy=audio_policy,
        target_pixel_format=target_pix,
        target_width=plan_tw,
        target_height=plan_th,
        resize_policy=resize_policy,
        target_frame_rate=target_frame_rate,
        frame_rate_policy=frame_rate_policy,
        target_time_base=None,
        rotation_policy=rotation_policy,
        sar_policy=sar_policy,
        audio_policy=audio_policy,
        copy_metadata_policy=copy_metadata_policy,
        estimated_output_path=output_path,
        overwrite_policy=False,
    )

    # Always remux/encode when required — video encode is needed for most reasons
    # Container/pixfmt/dims/rotation/sar/fps still need encode path (safe CPU normalize)
    if (
        required
        and "transcode_video" not in transforms
        and any(
            r
            in {
                REASON_CONTAINER,
                REASON_PIXFMT,
                REASON_DIMS,
                REASON_ROTATION,
                REASON_SAR,
                REASON_FPS,
                REASON_CODEC,
            }
            for r in ordered
        )
    ):
        transforms.append("encode_libx264")

    return PlannedNormalization(
        plan=plan,
        video_stream_index=probe.selected_video_stream_index,
        audio_stream_index=probe.selected_audio_stream_index,
        video_stream_ordinal=_video_ordinal(probe, probe.selected_video_stream_index),
        audio_stream_ordinal=_audio_ordinal(probe, probe.selected_audio_stream_index),
        source_rotation_degrees=video.rotation_degrees,
        bake_rotation=bake_rotation,
        resize_performed=resize_performed,
        force_setsar=force_setsar or (sar_policy == "force_1_1" and required),
        frame_rate_conversion=frame_rate_conversion,
        audio_action=audio_action,
        applied_transforms=tuple(dict.fromkeys(transforms)),
        source_width=video.width,
        source_height=video.height,
        display_width=disp_w,
        display_height=disp_h,
    )
