"""Interval sweep fusion of shot + camera segments into atomic analysis windows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from football_analytics.broadcast.types import (
    FramingScale,
    GraphicsStatus,
    Playability,
    ReplayStatus,
    ViewFamily,
)
from football_analytics.video.types import MappingQuality

DECISION_ATTR_KEYS = (
    "shot_id",
    "view_family",
    "framing_scale",
    "replay_status",
    "graphics_status",
    "playability",
    "timeline_mapping_quality",
    "is_gap",
    "is_conflict",
    "coverage",
    "confidence",
)


class FusionError(ValueError):
    """Shot/camera interval fusion failure."""


@dataclass(frozen=True)
class FusedWindow:
    """Pre-routing fused interval (half-open)."""

    run_id: str
    video_id: str
    start_time_us: int
    end_time_us: int
    start_frame_index: int | None
    end_frame_index_exclusive: int | None
    shot_id: str | None
    camera_segment_ids: tuple[str, ...]
    view_family: str
    framing_scale: str
    replay_status: str
    graphics_status: str
    playability: str
    coverage: float
    confidence: float | None
    timeline_mapping_quality: str
    is_gap: bool
    is_conflict: bool
    source_refs: tuple[str, ...]

    def decision_key(self) -> tuple[Any, ...]:
        return (
            self.shot_id,
            self.view_family,
            self.framing_scale,
            self.replay_status,
            self.graphics_status,
            self.playability,
            self.timeline_mapping_quality,
            self.is_gap,
            self.is_conflict,
            round(float(self.coverage), 6),
            None if self.confidence is None else round(float(self.confidence), 6),
            self.camera_segment_ids,
        )


def _as_row(obj: Any) -> dict[str, Any]:
    if isinstance(obj, Mapping):
        return dict(obj)
    if hasattr(obj, "to_dict"):
        return dict(obj.to_dict())
    raise FusionError("shot/camera rows must be mappings or typed contracts")


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 < b1 and b0 < a1


def _label_tuple(cam: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(cam["view_family"]),
        str(cam["framing_scale"]),
        str(cam["replay_status"]),
        str(cam["graphics_status"]),
        str(cam["playability"]),
    )


def _validate_camera_in_shot(
    cam: Mapping[str, Any],
    shot_lookup: dict[tuple[str, str, str], Mapping[str, Any]],
) -> None:
    shot_id = cam.get("shot_id")
    if shot_id is None:
        return
    key = (str(cam["run_id"]), str(cam["video_id"]), str(shot_id))
    shot = shot_lookup.get(key)
    if shot is None:
        raise FusionError(f"camera segment references missing shot_id={shot_id}")
    c0, c1 = int(cam["start_time_us"]), int(cam["end_time_us"])
    s0, s1 = int(shot["start_time_us"]), int(shot["end_time_us"])
    if c0 < s0 or c1 > s1:
        raise FusionError(
            f"camera segment {cam.get('camera_segment_id')} outside shot {shot_id} bounds"
        )


def _covering_shot(
    shots: Sequence[Mapping[str, Any]], start: int, end: int
) -> Mapping[str, Any] | None:
    for shot in shots:
        s0, s1 = int(shot["start_time_us"]), int(shot["end_time_us"])
        if start >= s0 and end <= s1:
            return shot
    return None


def _covering_cameras(
    cameras: Sequence[Mapping[str, Any]], start: int, end: int
) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = []
    for cam in cameras:
        c0, c1 = int(cam["start_time_us"]), int(cam["end_time_us"])
        if _intervals_overlap(start, end, c0, c1):
            # Require full containment of atomic window in camera for attribution.
            if start >= c0 and end <= c1:
                out.append(cam)
            else:
                # Partial overlap of atomic window should not happen if boundaries
                # include all camera edges; treat as conflict-like integrity issue.
                out.append(cam)
    return out


def _interp_frames(
    shot: Mapping[str, Any] | None,
    cameras: Sequence[Mapping[str, Any]],
    start: int,
    end: int,
) -> tuple[int | None, int | None]:
    """Carry frame indices only when they align to segment endpoints."""
    if cameras:
        cam = cameras[0]
        sf = cam.get("start_frame_index")
        ef = cam.get("end_frame_index_exclusive")
        if int(cam["start_time_us"]) == start and int(cam["end_time_us"]) == end:
            return (
                None if sf is None else int(sf),
                None if ef is None else int(ef),
            )
    if shot is not None and int(shot["start_time_us"]) == start and int(shot["end_time_us"]) == end:
        sf = shot.get("start_frame_index")
        ef = shot.get("end_frame_index_exclusive")
        return (
            None if sf is None else int(sf),
            None if ef is None else int(ef),
        )
    return None, None


def _gap_window(
    *,
    run_id: str,
    video_id: str,
    start: int,
    end: int,
    shot: Mapping[str, Any],
) -> FusedWindow:
    mapping = str(shot["timeline_mapping_quality"])
    return FusedWindow(
        run_id=run_id,
        video_id=video_id,
        start_time_us=start,
        end_time_us=end,
        start_frame_index=None,
        end_frame_index_exclusive=None,
        shot_id=str(shot["shot_id"]),
        camera_segment_ids=(),
        view_family=ViewFamily.UNKNOWN.value,
        framing_scale=FramingScale.UNKNOWN.value,
        replay_status=ReplayStatus.UNKNOWN.value,
        graphics_status=GraphicsStatus.UNKNOWN.value,
        playability=Playability.UNCERTAIN.value,
        coverage=0.0,
        confidence=None,
        timeline_mapping_quality=mapping,
        is_gap=True,
        is_conflict=False,
        source_refs=(str(shot["shot_id"]),),
    )


def _from_cameras(
    *,
    run_id: str,
    video_id: str,
    start: int,
    end: int,
    shot: Mapping[str, Any],
    cameras: Sequence[Mapping[str, Any]],
) -> FusedWindow:
    labels = {_label_tuple(c) for c in cameras}
    conflict = len(labels) > 1
    ids = tuple(sorted({str(c["camera_segment_id"]) for c in cameras}))
    refs = list(ids)
    refs.append(str(shot["shot_id"]))
    sf, ef = _interp_frames(shot, cameras, start, end)
    mapping = str(shot["timeline_mapping_quality"])
    try:
        MappingQuality(mapping)
    except ValueError as exc:
        raise FusionError(f"invalid timeline_mapping_quality: {mapping}") from exc

    if conflict:
        return FusedWindow(
            run_id=run_id,
            video_id=video_id,
            start_time_us=start,
            end_time_us=end,
            start_frame_index=sf,
            end_frame_index_exclusive=ef,
            shot_id=str(shot["shot_id"]),
            camera_segment_ids=ids,
            view_family=ViewFamily.UNKNOWN.value,
            framing_scale=FramingScale.UNKNOWN.value,
            replay_status=ReplayStatus.UNKNOWN.value,
            graphics_status=GraphicsStatus.UNKNOWN.value,
            playability=Playability.UNCERTAIN.value,
            coverage=min(float(c["coverage"]) for c in cameras),
            confidence=None,
            timeline_mapping_quality=mapping,
            is_gap=False,
            is_conflict=True,
            source_refs=tuple(refs),
        )

    cam = cameras[0]
    confs = [c.get("confidence") for c in cameras]
    confidence: float | None
    if any(c is None for c in confs):
        confidence = None
    else:
        confidence = min(float(c) for c in confs if c is not None)

    return FusedWindow(
        run_id=run_id,
        video_id=video_id,
        start_time_us=start,
        end_time_us=end,
        start_frame_index=sf,
        end_frame_index_exclusive=ef,
        shot_id=str(shot["shot_id"]),
        camera_segment_ids=ids,
        view_family=str(cam["view_family"]),
        framing_scale=str(cam["framing_scale"]),
        replay_status=str(cam["replay_status"]),
        graphics_status=str(cam["graphics_status"]),
        playability=str(cam["playability"]),
        coverage=min(float(c["coverage"]) for c in cameras),
        confidence=confidence,
        timeline_mapping_quality=mapping,
        is_gap=False,
        is_conflict=False,
        source_refs=tuple(refs),
    )


def merge_adjacent_windows(
    windows: Sequence[FusedWindow], *, enabled: bool = True
) -> list[FusedWindow]:
    """Merge adjacent windows when all decision fields (incl. camera ids) match."""
    if not enabled or not windows:
        return list(windows)
    ordered = sorted(windows, key=lambda w: (w.start_time_us, w.end_time_us, w.shot_id or ""))
    merged: list[FusedWindow] = [ordered[0]]
    for cur in ordered[1:]:
        prev = merged[-1]
        if prev.end_time_us == cur.start_time_us and prev.decision_key() == cur.decision_key():
            merged[-1] = FusedWindow(
                run_id=prev.run_id,
                video_id=prev.video_id,
                start_time_us=prev.start_time_us,
                end_time_us=cur.end_time_us,
                start_frame_index=prev.start_frame_index,
                end_frame_index_exclusive=cur.end_frame_index_exclusive,
                shot_id=prev.shot_id,
                camera_segment_ids=prev.camera_segment_ids,
                view_family=prev.view_family,
                framing_scale=prev.framing_scale,
                replay_status=prev.replay_status,
                graphics_status=prev.graphics_status,
                playability=prev.playability,
                coverage=min(prev.coverage, cur.coverage),
                confidence=(
                    None
                    if prev.confidence is None or cur.confidence is None
                    else min(prev.confidence, cur.confidence)
                ),
                timeline_mapping_quality=prev.timeline_mapping_quality,
                is_gap=prev.is_gap,
                is_conflict=prev.is_conflict,
                source_refs=tuple(dict.fromkeys([*prev.source_refs, *cur.source_refs])),
            )
        else:
            merged.append(cur)
    return merged


def fuse_shot_camera_intervals(
    shots: Sequence[Any],
    cameras: Sequence[Any],
    *,
    merge_identical_adjacent: bool = True,
) -> list[FusedWindow]:
    """Fuse shot backbone + camera edges into non-overlapping atomic windows.

    - Splits at every shot/camera boundary inside shots.
    - Camera extending outside its shot raises ``FusionError``.
    - Gaps (no camera) become unknown/uncertain windows (no silent fill).
    - Conflicting overlapping camera labels mark ``is_conflict``.
    """
    shot_rows = [_as_row(s) for s in shots]
    cam_rows = [_as_row(c) for c in cameras]
    if not shot_rows:
        return []

    shot_lookup: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for s in shot_rows:
        key = (str(s["run_id"]), str(s["video_id"]), str(s["shot_id"]))
        if key in shot_lookup:
            raise FusionError(f"duplicate shot_id: {key}")
        if int(s["end_time_us"]) <= int(s["start_time_us"]):
            raise FusionError(f"invalid shot interval: {key}")
        shot_lookup[key] = s

    for cam in cam_rows:
        if int(cam["end_time_us"]) <= int(cam["start_time_us"]):
            raise FusionError(f"invalid camera interval: {cam.get('camera_segment_id')}")
        _validate_camera_in_shot(cam, shot_lookup)
        # Cameras without shot_id must still lie within some shot span.
        if cam.get("shot_id") is None:
            contained = any(
                int(cam["start_time_us"]) >= int(s["start_time_us"])
                and int(cam["end_time_us"]) <= int(s["end_time_us"])
                and str(cam["run_id"]) == str(s["run_id"])
                and str(cam["video_id"]) == str(s["video_id"])
                for s in shot_rows
            )
            if not contained:
                raise FusionError(
                    f"camera segment {cam.get('camera_segment_id')} outside all shots"
                )

    groups: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}
    for s in shot_rows:
        groups.setdefault((str(s["run_id"]), str(s["video_id"])), {"shots": [], "cameras": []})[
            "shots"
        ].append(s)
    for c in cam_rows:
        groups.setdefault((str(c["run_id"]), str(c["video_id"])), {"shots": [], "cameras": []})[
            "cameras"
        ].append(c)

    fused: list[FusedWindow] = []
    for (run_id, video_id), bundle in sorted(groups.items()):
        s_list = sorted(bundle["shots"], key=lambda r: (int(r["start_time_us"]), str(r["shot_id"])))
        c_list = sorted(
            bundle["cameras"],
            key=lambda r: (int(r["start_time_us"]), str(r["camera_segment_id"])),
        )
        for shot in s_list:
            s0, s1 = int(shot["start_time_us"]), int(shot["end_time_us"])
            boundaries: set[int] = {s0, s1}
            for cam in c_list:
                c0, c1 = int(cam["start_time_us"]), int(cam["end_time_us"])
                if _intervals_overlap(s0, s1, c0, c1):
                    if c0 > s0:
                        boundaries.add(c0)
                    if c1 < s1:
                        boundaries.add(c1)
                    boundaries.add(max(c0, s0))
                    boundaries.add(min(c1, s1))
            times = sorted(boundaries)
            for t0, t1 in zip(times, times[1:], strict=False):
                if t1 <= t0:
                    continue
                # Only emit windows fully inside this shot.
                if t0 < s0 or t1 > s1:
                    continue
                covering = [
                    c
                    for c in c_list
                    if int(c["start_time_us"]) <= t0 and int(c["end_time_us"]) >= t1
                ]
                # Also accept cameras that fully contain the atomic interval.
                if not covering:
                    # Partial overlaps should not remain after boundary sweep; if any
                    # camera overlaps incompletely, escalate as conflict window.
                    partial = [
                        c
                        for c in c_list
                        if _intervals_overlap(
                            t0, t1, int(c["start_time_us"]), int(c["end_time_us"])
                        )
                    ]
                    if partial:
                        # Integrity: camera not aligned to sweep boundaries.
                        raise FusionError(
                            f"camera partial overlap unresolved at [{t0},{t1}) "
                            f"for shot {shot['shot_id']}"
                        )
                    fused.append(
                        _gap_window(
                            run_id=run_id,
                            video_id=video_id,
                            start=t0,
                            end=t1,
                            shot=shot,
                        )
                    )
                else:
                    fused.append(
                        _from_cameras(
                            run_id=run_id,
                            video_id=video_id,
                            start=t0,
                            end=t1,
                            shot=shot,
                            cameras=covering,
                        )
                    )

    return merge_adjacent_windows(fused, enabled=merge_identical_adjacent)


__all__ = [
    "DECISION_ATTR_KEYS",
    "FusionError",
    "FusedWindow",
    "fuse_shot_camera_intervals",
    "merge_adjacent_windows",
]
