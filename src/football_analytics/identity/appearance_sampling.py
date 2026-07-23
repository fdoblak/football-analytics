"""Observed-only human crop sampling for Stage 7B appearance ReID."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from football_analytics.identity.appearance_descriptor import (
    AppearanceDescriptorError,
    crop_quality,
    extract_descriptor_from_bgr,
)


class AppearanceSamplingError(ValueError):
    """Crop sampling failure."""


@dataclass(frozen=True)
class SampledCrop:
    track_id: int
    frame_index: int
    detection_id: int | None
    bbox_xyxy: tuple[float, float, float, float]
    quality: float
    descriptor: tuple[float, ...]
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class TrackSamplingResult:
    track_id: int
    accepted: tuple[SampledCrop, ...]
    rejected_count: int
    reject_reasons: tuple[str, ...]
    entity_type: str
    start_frame: int | None
    end_frame: int | None
    start_time_us: int | None
    end_time_us: int | None


def _entity_for_track(
    track_id: int,
    observations: Sequence[Mapping[str, Any]],
    attributes_by_det: Mapping[tuple[int, int], Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]] | None,
) -> str:
    for s in summaries or ():
        if int(s["track_id"]) == track_id and s.get("entity_type"):
            return str(s["entity_type"])
    for obs in observations:
        if int(obs["track_id"]) != track_id:
            continue
        det_id = obs.get("detection_id")
        if det_id is None:
            continue
        attr = attributes_by_det.get((int(obs["frame_index"]), int(det_id)))
        if attr and attr.get("entity_type"):
            return str(attr["entity_type"])
    # Fallback: class_id 0 ≈ human in synthetic fixtures; ball often class_id 32/1.
    for obs in observations:
        if int(obs["track_id"]) == track_id:
            cid = int(obs.get("class_id", -1))
            if cid in {32, 37}:  # common ball ids in COCO-ish fixtures
                return "ball"
            return "human"
    return "unknown"


def _bbox_area(bbox: Sequence[float]) -> float:
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _crop_from_frame(frame_bgr: np.ndarray, bbox: Sequence[float]) -> np.ndarray | None:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return None
    return frame_bgr[y1:y2, x1:x2].copy()


def sample_tracklet_crops(
    *,
    track_id: int,
    observations: Sequence[Mapping[str, Any]],
    frames_bgr: Mapping[int, np.ndarray] | None,
    synthetic_crops: Mapping[tuple[int, int], np.ndarray] | None,
    attributes: Sequence[Mapping[str, Any]] | None,
    summaries: Sequence[Mapping[str, Any]] | None,
    config: Mapping[str, Any],
    frame_times_us: Mapping[int, int] | None = None,
) -> TrackSamplingResult:
    """Sample observed-only human crops; never persist crops to disk."""
    sampling = config["sampling"]
    if sampling["persist_crops"] or sampling["debug_crop_output"]:
        raise AppearanceSamplingError("crop persistence forbidden")

    attrs_by_det: dict[tuple[int, int], Mapping[str, Any]] = {}
    for a in attributes or ():
        attrs_by_det[(int(a["frame_index"]), int(a["detection_id"]))] = a

    track_obs = [o for o in observations if int(o["track_id"]) == track_id]
    track_obs = sorted(
        track_obs, key=lambda r: (int(r["frame_index"]), int(r.get("detection_id") or -1))
    )
    entity = _entity_for_track(track_id, track_obs, attrs_by_det, summaries)
    if sampling["human_only"] and entity != "human":
        return TrackSamplingResult(
            track_id=track_id,
            accepted=(),
            rejected_count=len(track_obs),
            reject_reasons=("HUMAN_ONLY_FILTER",),
            entity_type=entity,
            start_frame=None,
            end_frame=None,
            start_time_us=None,
            end_time_us=None,
        )

    if not track_obs:
        return TrackSamplingResult(
            track_id=track_id,
            accepted=(),
            rejected_count=0,
            reject_reasons=("NO_OBSERVATIONS",),
            entity_type=entity,
            start_frame=None,
            end_frame=None,
            start_time_us=None,
            end_time_us=None,
        )

    frames = [int(o["frame_index"]) for o in track_obs]
    f0, f1 = min(frames), max(frames)
    t0 = frame_times_us.get(f0) if frame_times_us else None
    t1 = frame_times_us.get(f1) if frame_times_us else None
    bin_count = int(sampling["temporal_bin_count"])
    span = max(1, f1 - f0)
    bin_counts: dict[int, int] = defaultdict(int)
    accepted: list[SampledCrop] = []
    rejected = 0
    reasons: list[str] = []
    stride = int(sampling["temporal_stride_frames"])
    max_total = int(sampling["max_samples_per_track"])
    max_per_bin = int(sampling["max_samples_per_temporal_bin"])
    min_area = float(sampling["min_bbox_area_px"])
    min_side = int(sampling["min_crop_side_px"])
    min_q = float(sampling["min_crop_quality"])

    for obs in track_obs:
        if len(accepted) >= max_total:
            break
        fi = int(obs["frame_index"])
        if (fi - f0) % stride != 0 and stride > 1 and fi not in {f0, f1}:
            rejected += 1
            reasons.append("TEMPORAL_STRIDE")
            continue
        state = str(obs.get("observation_state", ""))
        if sampling["observed_only"] and state != "observed":
            rejected += 1
            reasons.append("PREDICTED_OR_INTERPOLATED_REJECTED")
            continue
        bbox = (
            float(obs["bbox_x1"]),
            float(obs["bbox_y1"]),
            float(obs["bbox_x2"]),
            float(obs["bbox_y2"]),
        )
        area = _bbox_area(bbox)
        if area < min_area:
            rejected += 1
            reasons.append("BBOX_TOO_SMALL")
            continue
        if (bbox[2] - bbox[0]) < min_side or (bbox[3] - bbox[1]) < min_side:
            rejected += 1
            reasons.append("CROP_SIDE_TOO_SMALL")
            continue

        bin_idx = int(((fi - f0) / span) * bin_count)
        bin_idx = min(bin_count - 1, max(0, bin_idx))
        if bin_counts[bin_idx] >= max_per_bin:
            rejected += 1
            reasons.append("TEMPORAL_BIN_CAP")
            continue

        det_id = obs.get("detection_id")
        det_i = int(det_id) if det_id is not None else None
        crop: np.ndarray | None = None
        if synthetic_crops is not None and det_i is not None:
            crop = synthetic_crops.get((fi, det_i))
            if crop is None:
                crop = synthetic_crops.get((track_id, fi))
        if crop is None and frames_bgr is not None:
            frame = frames_bgr.get(fi)
            if frame is None:
                rejected += 1
                reasons.append("FRAME_MISSING")
                continue
            crop = _crop_from_frame(frame, bbox)
        if crop is None:
            rejected += 1
            reasons.append("CROP_UNAVAILABLE")
            continue
        if crop.shape[0] < min_side or crop.shape[1] < min_side:
            rejected += 1
            reasons.append("CROP_SIDE_TOO_SMALL")
            continue
        try:
            q = crop_quality(crop)
            if q < min_q:
                rejected += 1
                reasons.append("LOW_CROP_QUALITY")
                continue
            desc = extract_descriptor_from_bgr(crop, config=config)
        except (AppearanceDescriptorError, AppearanceSamplingError) as exc:
            rejected += 1
            reasons.append(f"DESCRIPTOR_FAIL:{type(exc).__name__}")
            continue

        accepted.append(
            SampledCrop(
                track_id=track_id,
                frame_index=fi,
                detection_id=det_i,
                bbox_xyxy=bbox,
                quality=float(desc.quality),
                descriptor=desc.vector,
                reason_codes=(),
            )
        )
        bin_counts[bin_idx] += 1

    # Unique reason codes, preserve order
    seen: set[str] = set()
    uniq_reasons: list[str] = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            uniq_reasons.append(r)

    return TrackSamplingResult(
        track_id=track_id,
        accepted=tuple(accepted),
        rejected_count=rejected,
        reject_reasons=tuple(uniq_reasons),
        entity_type=entity,
        start_frame=f0,
        end_frame=f1,
        start_time_us=t0,
        end_time_us=t1,
    )


__all__ = [
    "AppearanceSamplingError",
    "SampledCrop",
    "TrackSamplingResult",
    "sample_tracklet_crops",
]
