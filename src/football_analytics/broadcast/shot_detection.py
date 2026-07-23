"""Rule-based shot boundary detection from consecutive-frame features."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from football_analytics.broadcast.shot_config import shot_config_fingerprint
from football_analytics.broadcast.shot_features import FeatureFrame
from football_analytics.broadcast.types import (
    CONTRACT_VERSION,
    DetectionSource,
    ReviewStatus,
    SegmentStatus,
    ShotBoundary,
    ShotSegment,
    TransitionType,
)
from football_analytics.video.types import MappingQuality


class ShotDetectionError(ValueError):
    """Shot detection failure."""


@dataclass(frozen=True)
class ScoredFrame:
    frame_index: int
    video_time_us: int
    score: float
    luma_mae: float
    hist_distance: float
    edge_change_ratio: float
    mean_luma: float


@dataclass(frozen=True)
class RawBoundary:
    boundary_time_us: int
    left_frame_index: int
    right_frame_index: int
    transition_type: TransitionType
    transition_duration_us: int
    raw_score: float
    peak_sharpness: float


def score_features(
    features: Sequence[FeatureFrame], config: Mapping[str, Any]
) -> list[ScoredFrame]:
    w = config["feature_weights"]
    wl, wh, we = float(w["luma"]), float(w["hist"]), float(w["edge"])
    out: list[ScoredFrame] = []
    for f in features:
        score = wl * f.luma_mae + wh * f.hist_distance + we * f.edge_change_ratio
        out.append(
            ScoredFrame(
                frame_index=f.frame_index,
                video_time_us=f.video_time_us,
                score=float(score),
                luma_mae=f.luma_mae,
                hist_distance=f.hist_distance,
                edge_change_ratio=f.edge_change_ratio,
                mean_luma=f.mean_luma,
            )
        )
    return out


def _peak_sharpness(scores: Sequence[float], idx: int, window: int) -> float:
    if not scores:
        return 0.0
    peak = scores[idx]
    lo = max(0, idx - window)
    hi = min(len(scores), idx + window + 1)
    neighbors = [scores[i] for i in range(lo, hi) if i != idx]
    if not neighbors:
        return peak
    mean_n = sum(neighbors) / len(neighbors)
    return max(0.0, peak - mean_n)


def _flash_end_index(
    scored: Sequence[ScoredFrame],
    idx: int,
    *,
    config: Mapping[str, Any],
) -> int | None:
    """Return index of post-flash baseline frame if idx starts a flash; else None."""
    flash = config["flash_suppression"]
    intensity_min = float(flash["intensity_min"])
    max_dur = int(flash["max_duration_us"])
    baseline_max = float(flash["baseline_similarity_max"])
    if scored[idx].score < intensity_min or idx <= 0:
        return None
    pre = scored[idx - 1]
    for j in range(idx + 1, len(scored)):
        if scored[j].video_time_us - scored[idx].video_time_us > max_dur:
            break
        if scored[j].score > baseline_max + 0.05:
            continue
        luma_delta = abs(pre.mean_luma - scored[j].mean_luma)
        if luma_delta <= baseline_max + 0.08 and pre.score <= baseline_max + 0.05:
            return j
    return None


def _is_flash(
    scored: Sequence[ScoredFrame],
    idx: int,
    *,
    config: Mapping[str, Any],
) -> bool:
    return _flash_end_index(scored, idx, config=config) is not None


def _classify_gradual(scored: Sequence[ScoredFrame], start: int, end: int) -> TransitionType:
    """Heuristic dissolve vs fade; unknown if ambiguous."""
    window = scored[start : end + 1]
    if not window:
        return TransitionType.UNKNOWN
    mean_lumas = [s.mean_luma for s in window]
    min_luma = min(mean_lumas)
    max_luma = max(mean_lumas)
    start_luma = mean_lumas[0]
    end_luma = mean_lumas[-1]
    # Fade: passes near black or ends/starts near black with large luma swing.
    near_black = min_luma < 0.12
    large_swing = abs(end_luma - start_luma) > 0.35
    hist_mean = sum(s.hist_distance for s in window) / len(window)
    if near_black and large_swing:
        return TransitionType.FADE
    # Dissolve: sustained hist/luma change without near-black trough.
    if hist_mean > 0.08 and not near_black and max_luma - min_luma < 0.55:
        return TransitionType.DISSOLVE
    if near_black and not large_swing:
        return TransitionType.UNKNOWN
    if large_swing and hist_mean > 0.05:
        return TransitionType.DISSOLVE if not near_black else TransitionType.FADE
    return TransitionType.UNKNOWN


def detect_raw_boundaries(
    scored: Sequence[ScoredFrame], config: Mapping[str, Any]
) -> list[RawBoundary]:
    if len(scored) < 2:
        return []
    hard_th = float(config["hard_cut_threshold"])
    gradual = config["gradual"]
    elev_th = float(gradual["elevated_mean_threshold"])
    sharp_max = float(gradual["peak_sharpness_max"])
    win = int(gradual["window_frames"])
    min_elev = int(gradual["min_elevated_frames"])
    merge_tol = int(config["boundary_merge_tolerance_us"])
    scores = [s.score for s in scored]

    candidates: list[RawBoundary] = []
    i = 1
    while i < len(scored):
        flash_end = _flash_end_index(scored, i, config=config)
        if flash_end is not None:
            i = flash_end + 1
            continue
        sharpness = _peak_sharpness(scores, i, max(2, win // 2))
        # Gradual window centered-ish on i
        lo = max(1, i - win // 2)
        hi = min(len(scored) - 1, i + win // 2)
        window_scores = scores[lo : hi + 1]
        elev_count = sum(1 for v in window_scores if v >= elev_th)
        mean_elev = sum(window_scores) / max(1, len(window_scores))
        gradual_like = elev_count >= min_elev and mean_elev >= elev_th and sharpness < sharp_max

        if scores[i] >= hard_th and not gradual_like:
            candidates.append(
                RawBoundary(
                    boundary_time_us=scored[i].video_time_us,
                    left_frame_index=scored[i - 1].frame_index,
                    right_frame_index=scored[i].frame_index,
                    transition_type=TransitionType.HARD_CUT,
                    transition_duration_us=0,
                    raw_score=scores[i],
                    peak_sharpness=sharpness,
                )
            )
            i += max(1, win // 2)
            continue

        if gradual_like and scores[i] >= elev_th:
            # Expand elevated run
            left = i
            while left > 1 and scores[left - 1] >= elev_th * 0.4:
                left -= 1
            right = i
            while right + 1 < len(scored) and scores[right + 1] >= elev_th * 0.4:
                right += 1
            peak_i = max(range(left, right + 1), key=lambda j: scores[j])
            # Prefer temporal centroid of elevated mass for gradual timing stability
            mass = sum(max(0.0, scores[j] - elev_th * 0.3) for j in range(left, right + 1))
            if mass > 0:
                centroid = (
                    sum(
                        scored[j].video_time_us * max(0.0, scores[j] - elev_th * 0.3)
                        for j in range(left, right + 1)
                    )
                    / mass
                )
                peak_i = min(
                    range(left, right + 1),
                    key=lambda j: abs(scored[j].video_time_us - centroid),
                )
            ttype = _classify_gradual(scored, left, right)
            duration = scored[right].video_time_us - scored[left].video_time_us
            candidates.append(
                RawBoundary(
                    boundary_time_us=scored[peak_i].video_time_us,
                    left_frame_index=scored[left].frame_index,
                    right_frame_index=scored[right].frame_index,
                    transition_type=ttype,
                    transition_duration_us=max(0, duration),
                    raw_score=scores[peak_i],
                    peak_sharpness=_peak_sharpness(scores, peak_i, max(2, win // 2)),
                )
            )
            i = right + 1
            continue
        i += 1

    # Peak merge within tolerance (keep higher score); graduals may merge farther
    if not config["peak_suppression"]["enabled"] or not candidates:
        return candidates
    candidates.sort(key=lambda b: (b.boundary_time_us, -b.raw_score))
    merged: list[RawBoundary] = []
    gradual_types = {
        TransitionType.DISSOLVE,
        TransitionType.FADE,
        TransitionType.UNKNOWN,
    }
    for cand in candidates:
        if not merged:
            merged.append(cand)
            continue
        prev = merged[-1]
        tol = merge_tol
        if prev.transition_type in gradual_types and cand.transition_type in gradual_types:
            tol = max(merge_tol, 500_000)
        if abs(cand.boundary_time_us - prev.boundary_time_us) <= tol:
            # Prefer higher score; for gradual pairs prefer later mid-run only if much stronger
            if cand.raw_score > prev.raw_score * 1.15:
                merged[-1] = cand
            continue
        merged.append(cand)
    return merged


def _enforce_min_shot_duration(
    boundaries: Sequence[RawBoundary],
    *,
    duration_us: int,
    min_duration_us: int,
) -> list[RawBoundary]:
    if min_duration_us <= 0 or not boundaries:
        return list(boundaries)
    kept: list[RawBoundary] = []
    prev_t = 0
    for b in sorted(boundaries, key=lambda x: x.boundary_time_us):
        if b.boundary_time_us - prev_t < min_duration_us:
            # Drop weaker-looking close boundary (already merged; drop this one)
            continue
        if duration_us - b.boundary_time_us < min_duration_us:
            continue
        kept.append(b)
        prev_t = b.boundary_time_us
    return kept


def build_shot_rows(
    boundaries: Sequence[RawBoundary],
    *,
    run_id: str,
    video_id: str,
    duration_us: int,
    timeline: Sequence[tuple[int, int]],
    mapping_quality: MappingQuality,
    config_fingerprint: str,
) -> tuple[list[ShotBoundary], list[ShotSegment]]:
    if duration_us <= 0:
        raise ShotDetectionError("duration_us must be > 0")
    index_to_time = {idx: t for idx, t in timeline}
    max_frame = max((idx for idx, _ in timeline), default=-1)

    shot_boundaries: list[ShotBoundary] = []
    for i, b in enumerate(boundaries):
        bid = f"bnd_{i + 1:04d}"
        prov = {
            "detector": "shot_boundary_baseline",
            "raw_score": round(b.raw_score, 6),
            "peak_sharpness": round(b.peak_sharpness, 6),
            "config_fingerprint": config_fingerprint,
        }
        shot_boundaries.append(
            ShotBoundary(
                run_id=run_id,
                video_id=video_id,
                boundary_id=bid,
                boundary_time_us=b.boundary_time_us,
                left_frame_index=b.left_frame_index,
                right_frame_index=b.right_frame_index,
                transition_type=b.transition_type,
                transition_duration_us=b.transition_duration_us,
                confidence=None,
                detection_source=DetectionSource.RULE,
                evidence_ref=None,
                review_status=ReviewStatus.UNREVIEWED,
                provenance_json=json.dumps(prov, sort_keys=True, separators=(",", ":")),
                contract_version=CONTRACT_VERSION,
            )
        )

    # Segments covering [0, duration)
    times = [0] + [b.boundary_time_us for b in boundaries] + [duration_us]
    # Deduplicate identical consecutive times
    cleaned_times: list[int] = []
    for t in times:
        if not cleaned_times or t != cleaned_times[-1]:
            cleaned_times.append(t)
    if cleaned_times[-1] != duration_us:
        cleaned_times.append(duration_us)

    def _frame_at_or_after(t_us: int) -> int | None:
        for idx, vt in timeline:
            if vt >= t_us:
                return idx
        return max_frame + 1 if max_frame >= 0 else None

    def _frame_exclusive_at(t_us: int) -> int | None:
        # exclusive end: first frame with time >= end, or one past last
        for idx, vt in timeline:
            if vt >= t_us:
                return idx
        return max_frame + 1 if max_frame >= 0 else None

    segments: list[ShotSegment] = []
    for i in range(len(cleaned_times) - 1):
        start = cleaned_times[i]
        end = cleaned_times[i + 1]
        if end <= start:
            continue
        start_bid = None
        end_bid = None
        if i > 0 and i - 1 < len(shot_boundaries):
            start_bid = shot_boundaries[i - 1].boundary_id
        if i < len(shot_boundaries):
            end_bid = shot_boundaries[i].boundary_id
        sf = _frame_at_or_after(start)
        ef = _frame_exclusive_at(end)
        frame_count = None
        if sf is not None and ef is not None and ef > sf:
            frame_count = ef - sf
        # Verify start time matches timeline when possible
        if sf is not None and sf in index_to_time and start == 0:
            pass
        segments.append(
            ShotSegment(
                run_id=run_id,
                video_id=video_id,
                shot_id=f"shot_{i + 1:04d}",
                start_time_us=start,
                end_time_us=end,
                start_frame_index=sf,
                end_frame_index_exclusive=ef,
                start_boundary_id=start_bid,
                end_boundary_id=end_bid,
                duration_us=end - start,
                frame_count=frame_count,
                timeline_mapping_quality=mapping_quality,
                segment_status=SegmentStatus.ACTIVE,
                provenance_json=json.dumps(
                    {
                        "detector": "shot_boundary_baseline",
                        "config_fingerprint": config_fingerprint,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                contract_version=CONTRACT_VERSION,
            )
        )
    return shot_boundaries, segments


def detect_shots(
    features: Sequence[FeatureFrame],
    *,
    run_id: str,
    video_id: str,
    duration_us: int,
    timeline: Sequence[tuple[int, int]],
    config: Mapping[str, Any],
    mapping_quality: MappingQuality = MappingQuality.NOT_AVAILABLE,
    config_fingerprint: str | None = None,
) -> tuple[list[ShotBoundary], list[ShotSegment], list[ScoredFrame]]:
    scored = score_features(features, config)
    raw = detect_raw_boundaries(scored, config)
    raw = _enforce_min_shot_duration(
        raw,
        duration_us=duration_us,
        min_duration_us=int(config["minimum_shot_duration_us"]),
    )
    fp = config_fingerprint or shot_config_fingerprint(config)
    boundaries, segments = build_shot_rows(
        raw,
        run_id=run_id,
        video_id=video_id,
        duration_us=duration_us,
        timeline=timeline,
        mapping_quality=mapping_quality,
        config_fingerprint=fp,
    )
    return boundaries, segments, scored


__all__ = [
    "ShotDetectionError",
    "ScoredFrame",
    "RawBoundary",
    "score_features",
    "detect_raw_boundaries",
    "build_shot_rows",
    "detect_shots",
]
