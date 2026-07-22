"""Pure FFprobe JSON → Stage 3A VideoProbe mapping (no subprocess/filesystem)."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from typing import Any

from football_analytics.video.ffprobe import ProbeError
from football_analytics.video.types import (
    AudioStreamInfo,
    FrameCountSource,
    FrameRateMode,
    ProbeWarning,
    Rational,
    StreamDisposition,
    VideoProbe,
    VideoStreamInfo,
    WarningSeverity,
    normalize_rotation_degrees,
    select_primary_video_stream,
)

_NA = frozenset({"", "n/a", "na", "none", "null", "unknown"})
_RATIONAL_RE = re.compile(r"^(-?\d+)[/:](-?\d+)$")


class ProbeParseError(ProbeError):
    """Strict parser failure."""


def _is_na(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and value.strip().lower() in _NA


def parse_rational(value: Any, *, label: str) -> Rational | None:
    """Parse FFprobe rational; 0/0 and N/A → None; invalid den → error."""
    if _is_na(value):
        return None
    if isinstance(value, Rational):
        return value
    if isinstance(value, Mapping):
        num = value.get("numerator", value.get("num"))
        den = value.get("denominator", value.get("den"))
        if den == 0:
            raise ProbeParseError("PROBE_UNEXPECTED_STRUCTURE", f"invalid rational {label}")
        return Rational(int(num), int(den))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            raise ProbeParseError("PROBE_UNEXPECTED_STRUCTURE", f"non-finite rational {label}")
        # Prefer exact int representation when possible
        return Rational(int(value), 1)
    if not isinstance(value, str):
        raise ProbeParseError("PROBE_UNEXPECTED_STRUCTURE", f"unsupported rational {label}")
    text = value.strip()
    match = _RATIONAL_RE.fullmatch(text)
    if not match:
        raise ProbeParseError("PROBE_UNEXPECTED_STRUCTURE", f"unparseable rational {label}")
    num_s, den_s = match.group(1), match.group(2)
    num, den = int(num_s), int(den_s)
    if den == 0:
        if num == 0:
            return None  # FFprobe "0/0" → unknown
        raise ProbeParseError("PROBE_UNEXPECTED_STRUCTURE", f"invalid rational {label}")
    return Rational(num, den)


def seconds_to_us(value: Any, *, label: str) -> int | None:
    """Convert FFprobe duration/start_time seconds to integer microseconds."""
    if _is_na(value):
        return None
    with localcontext() as ctx:
        ctx.rounding = ROUND_HALF_EVEN
        ctx.prec = 50
        try:
            if isinstance(value, int) and not isinstance(value, bool):
                # Already microseconds? FFprobe uses seconds as string/float typically.
                # Treat bare int as seconds for format.duration consistency with strings.
                dec = Decimal(value)
            elif isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    raise ProbeParseError("PROBE_UNEXPECTED_STRUCTURE", f"non-finite {label}")
                dec = Decimal(str(value))
            elif isinstance(value, str):
                dec = Decimal(value.strip())
            else:
                raise ProbeParseError("PROBE_UNEXPECTED_STRUCTURE", f"unsupported {label}")
        except Exception as exc:  # noqa: BLE001
            raise ProbeParseError("PROBE_UNEXPECTED_STRUCTURE", f"bad {label}") from exc
        us = (dec * Decimal(1_000_000)).to_integral_value(rounding=ROUND_HALF_EVEN)
        return int(us)


def parse_int_or_none(value: Any, *, label: str, minimum: int | None = None) -> int | None:
    if _is_na(value):
        return None
    try:
        if isinstance(value, bool):
            raise ValueError("bool")
        if isinstance(value, int):
            num = value
        elif isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                raise ValueError("non-finite")
            if not value.is_integer():
                raise ValueError("non-integer float")
            num = int(value)
        elif isinstance(value, str):
            num = int(value.strip())
        else:
            raise ValueError("type")
    except ValueError as exc:
        raise ProbeParseError("PROBE_UNEXPECTED_STRUCTURE", f"bad int {label}") from exc
    if minimum is not None and num < minimum:
        raise ProbeParseError("PROBE_UNEXPECTED_STRUCTURE", f"{label} below minimum")
    return num


def parse_disposition(raw: Any) -> StreamDisposition:
    if not isinstance(raw, Mapping):
        return StreamDisposition(default=False, attached_pic=False, forced=False)

    def flag(name: str) -> bool:
        val = raw.get(name, 0)
        if isinstance(val, bool):
            return val
        try:
            return int(val) != 0
        except (TypeError, ValueError):
            return False

    return StreamDisposition(
        default=flag("default"),
        attached_pic=flag("attached_pic"),
        forced=flag("forced"),
    )


def _rotation_from_stream(stream: Mapping[str, Any]) -> tuple[int, list[ProbeWarning]]:
    warnings: list[ProbeWarning] = []
    candidates: list[int] = []
    tags_raw = stream.get("tags")
    tags: Mapping[str, Any] = tags_raw if isinstance(tags_raw, Mapping) else {}
    if "rotate" in tags and not _is_na(tags.get("rotate")):
        try:
            candidates.append(int(str(tags["rotate"]).strip()))
        except ValueError:
            warnings.append(
                ProbeWarning(
                    code="ROTATION_UNPARSEABLE",
                    message="rotate tag not integer",
                    severity=WarningSeverity.WARNING,
                )
            )
    side = stream.get("side_data_list")
    if isinstance(side, list):
        for item in side:
            if not isinstance(item, Mapping):
                continue
            if item.get("side_data_type") in {
                "Display Matrix",
                "displaymatrix",
                "Display matrix",
            }:
                rot = item.get("rotation")
                if not _is_na(rot):
                    try:
                        # FFprobe may emit float degrees
                        candidates.append(int(round(float(str(rot)))))
                    except (TypeError, ValueError):
                        warnings.append(
                            ProbeWarning(
                                code="ROTATION_UNPARSEABLE",
                                message="display matrix rotation unparseable",
                                severity=WarningSeverity.WARNING,
                            )
                        )
    if not candidates:
        return 0, warnings
    clean: list[int] = []
    for c in candidates:
        try:
            mod = ((c % 360) + 360) % 360
            if mod in {0, 90, 180, 270}:
                clean.append(normalize_rotation_degrees(mod))
            elif mod in {45, 135, 225, 315}:
                warnings.append(
                    ProbeWarning(
                        code="ROTATION_NON_RIGHT_ANGLE",
                        message="non-right-angle rotation rejected as hard warning",
                        severity=WarningSeverity.UNSUPPORTED,
                    )
                )
            else:
                clean.append(normalize_rotation_degrees(c))
        except Exception:  # noqa: BLE001
            warnings.append(
                ProbeWarning(
                    code="ROTATION_NON_RIGHT_ANGLE",
                    message="rotation not in allowlist",
                    severity=WarningSeverity.UNSUPPORTED,
                )
            )
    if not clean:
        return 0, warnings
    if len(set(clean)) > 1:
        warnings.append(
            ProbeWarning(
                code="ROTATION_CONFLICT",
                message="conflicting rotation sources",
                severity=WarningSeverity.WARNING,
            )
        )
    return clean[0], warnings


def _classify_frame_rate_mode(r: Rational | None, avg: Rational | None) -> FrameRateMode:
    if r is None or avg is None:
        return FrameRateMode.UNKNOWN
    if r.numerator * avg.denominator == avg.numerator * r.denominator:
        return FrameRateMode.CFR
    return FrameRateMode.VFR


def _parse_video_stream(
    stream: Mapping[str, Any], index: int
) -> tuple[VideoStreamInfo, list[ProbeWarning]]:
    warnings: list[ProbeWarning] = []
    width = parse_int_or_none(stream.get("width"), label="width", minimum=1)
    height = parse_int_or_none(stream.get("height"), label="height", minimum=1)
    if width is None or height is None:
        raise ProbeParseError("NO_USABLE_VIDEO_STREAM", "video stream missing dimensions")
    rotation, rot_warns = _rotation_from_stream(stream)
    warnings.extend(rot_warns)
    r_rate = parse_rational(stream.get("r_frame_rate"), label="r_frame_rate")
    avg_rate = parse_rational(stream.get("avg_frame_rate"), label="avg_frame_rate")
    if r_rate is None:
        r_rate = Rational(0, 1)
    if avg_rate is None:
        avg_rate = Rational(0, 1)
    r_for_class = None if r_rate.numerator == 0 else r_rate
    avg_for_class = None if avg_rate.numerator == 0 else avg_rate
    mode = _classify_frame_rate_mode(r_for_class, avg_for_class)
    nb = parse_int_or_none(stream.get("nb_frames"), label="nb_frames", minimum=0)
    nb_read = parse_int_or_none(stream.get("nb_read_frames"), label="nb_read_frames", minimum=0)
    if nb is not None:
        frame_count, frame_src = nb, FrameCountSource.NB_FRAMES
    elif nb_read is not None:
        frame_count, frame_src = nb_read, FrameCountSource.ESTIMATED
    else:
        frame_count, frame_src = None, FrameCountSource.UNKNOWN
    sar = parse_rational(stream.get("sample_aspect_ratio"), label="sar") or Rational(1, 1)
    dar = parse_rational(stream.get("display_aspect_ratio"), label="dar")
    if dar is None:
        dar = Rational(width, height)
    time_base = parse_rational(stream.get("time_base"), label="time_base") or Rational(1, 1_000_000)
    codec_tb = parse_rational(stream.get("codec_time_base"), label="codec_time_base")
    duration_us = seconds_to_us(stream.get("duration"), label="stream.duration")
    start_pts = parse_int_or_none(stream.get("start_pts"), label="start_pts")
    info = VideoStreamInfo(
        stream_index=index,
        codec_name=str(stream.get("codec_name") or "unknown"),
        codec_long_name=(
            None if _is_na(stream.get("codec_long_name")) else str(stream.get("codec_long_name"))
        ),
        profile=None if _is_na(stream.get("profile")) else str(stream.get("profile")),
        pixel_format=None if _is_na(stream.get("pix_fmt")) else str(stream.get("pix_fmt")),
        width=width,
        height=height,
        coded_width=parse_int_or_none(stream.get("coded_width"), label="coded_width", minimum=1),
        coded_height=parse_int_or_none(stream.get("coded_height"), label="coded_height", minimum=1),
        sample_aspect_ratio=sar,
        display_aspect_ratio=dar,
        rotation_degrees=rotation,
        time_base=time_base,
        codec_time_base=codec_tb,
        r_frame_rate=r_rate,
        avg_frame_rate=avg_rate,
        nominal_frame_rate=r_for_class,
        frame_rate_mode=mode,
        start_pts=start_pts,
        duration_ts=parse_int_or_none(stream.get("duration_ts"), label="duration_ts", minimum=0),
        duration_us=duration_us,
        frame_count=frame_count,
        frame_count_source=frame_src,
        bit_rate_bps=parse_int_or_none(stream.get("bit_rate"), label="bit_rate", minimum=0),
        color_range=None if _is_na(stream.get("color_range")) else str(stream.get("color_range")),
        color_space=None if _is_na(stream.get("color_space")) else str(stream.get("color_space")),
        color_transfer=(
            None if _is_na(stream.get("color_transfer")) else str(stream.get("color_transfer"))
        ),
        color_primaries=(
            None if _is_na(stream.get("color_primaries")) else str(stream.get("color_primaries"))
        ),
        field_order=None if _is_na(stream.get("field_order")) else str(stream.get("field_order")),
        disposition=parse_disposition(stream.get("disposition")),
    )
    return info, warnings


def _parse_audio_stream(stream: Mapping[str, Any], index: int) -> AudioStreamInfo:
    time_base = parse_rational(stream.get("time_base"), label="audio.time_base") or Rational(
        1, 48_000
    )
    return AudioStreamInfo(
        stream_index=index,
        codec_name=str(stream.get("codec_name") or "unknown"),
        sample_rate_hz=parse_int_or_none(stream.get("sample_rate"), label="sample_rate", minimum=1),
        channels=parse_int_or_none(stream.get("channels"), label="channels", minimum=1),
        channel_layout=(
            None if _is_na(stream.get("channel_layout")) else str(stream.get("channel_layout"))
        ),
        time_base=time_base,
        duration_us=seconds_to_us(stream.get("duration"), label="audio.duration"),
        bit_rate_bps=parse_int_or_none(stream.get("bit_rate"), label="audio.bit_rate", minimum=0),
        disposition=parse_disposition(stream.get("disposition")),
    )


def select_primary_audio_stream(
    streams: tuple[VideoStreamInfo | AudioStreamInfo, ...],
) -> int | None:
    audios = [s for s in streams if isinstance(s, AudioStreamInfo)]
    if not audios:
        return None
    defaults = [s for s in audios if s.disposition.default]
    pool = defaults or audios
    pool.sort(key=lambda s: s.stream_index)
    return pool[0].stream_index


def map_ffprobe_json_to_video_probe(
    data: Mapping[str, Any],
    *,
    source_id: str,
    source_sha256: str,
    file_size_bytes: int,
    probe_tool_version: str,
    probed_at_utc: str | None = None,
    max_stream_count: int = 32,
) -> VideoProbe:
    """Map decoded FFprobe JSON into a Stage 3A VideoProbe."""
    if "streams" not in data or not isinstance(data.get("streams"), list):
        raise ProbeParseError("PROBE_UNEXPECTED_STRUCTURE", "streams array required")
    if len(data["streams"]) > max_stream_count:
        raise ProbeParseError("TOO_MANY_STREAMS", "stream count exceeds policy maximum")
    fmt_raw = data.get("format")
    fmt: Mapping[str, Any] = fmt_raw if isinstance(fmt_raw, Mapping) else {}
    warnings: list[ProbeWarning] = []
    parsed: list[VideoStreamInfo | AudioStreamInfo] = []
    for raw in data["streams"]:
        if not isinstance(raw, Mapping):
            raise ProbeParseError("PROBE_UNEXPECTED_STRUCTURE", "stream must be object")
        index = parse_int_or_none(raw.get("index"), label="stream.index", minimum=0)
        if index is None:
            raise ProbeParseError("PROBE_UNEXPECTED_STRUCTURE", "stream index missing")
        ctype = str(raw.get("codec_type") or "")
        if ctype == "video":
            info, w = _parse_video_stream(raw, index)
            warnings.extend(w)
            parsed.append(info)
        elif ctype == "audio":
            parsed.append(_parse_audio_stream(raw, index))
        # subtitle/data ignored intentionally
    if not parsed:
        raise ProbeParseError("NO_USABLE_VIDEO_STREAM", "no streams mapped")
    try:
        selected_video = select_primary_video_stream(tuple(parsed))
    except Exception as exc:  # noqa: BLE001
        raise ProbeParseError("NO_USABLE_VIDEO_STREAM", "no usable video stream") from exc
    selected_audio = select_primary_audio_stream(tuple(parsed))
    duration_us = seconds_to_us(fmt.get("duration"), label="format.duration")
    start_time_us = seconds_to_us(fmt.get("start_time"), label="format.start_time")
    format_name = None if _is_na(fmt.get("format_name")) else str(fmt.get("format_name"))
    container = None
    if format_name:
        container = format_name.split(",")[0].strip() or None
    stamp = probed_at_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return VideoProbe(
        source_id=source_id,
        source_sha256=source_sha256,
        probe_tool="ffprobe",
        probe_tool_version=probe_tool_version,
        probed_at_utc=stamp,
        container=container,
        format_name=format_name,
        duration_us=duration_us,
        start_time_us=start_time_us,
        bit_rate_bps=parse_int_or_none(fmt.get("bit_rate"), label="format.bit_rate", minimum=0),
        file_size_bytes=file_size_bytes,
        streams=tuple(parsed),
        selected_video_stream_index=selected_video,
        selected_audio_stream_index=selected_audio,
        warnings=tuple(warnings),
    )
