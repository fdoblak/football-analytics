"""Sprint bout extraction with hysteresis (Stage 9C — project_generated, not official Opta)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from football_analytics.physical.speed import InstantSpeedSample, compute_segment_speeds


@dataclass(frozen=True)
class SprintBout:
    sprint_id: str
    start_time_us: int
    end_time_us: int
    duration_us: int
    distance_m: float
    peak_robust_speed_mps: float
    mean_robust_speed_mps: float
    supporting_sample_count: int
    observed_coverage_us: int
    derived_coverage_us: int
    uncertainty_m: float | None
    source_segment_id: str
    sample_layer: str
    evaluability: str
    reason_codes: tuple[str, ...]
    metric_origin: str
    definition_style: str
    config_fingerprint: str
    sprint_profile_version: int


def _interval_distance(samples: Sequence[InstantSpeedSample]) -> float:
    return float(sum(s.distance_m for s in samples if not s.rejected))


def extract_sprint_bouts_for_segment(
    points: Sequence[Mapping[str, Any]],
    *,
    trajectory_segment_id: str,
    sample_layer: str,
    config: Mapping[str, Any],
    config_fingerprint: str,
    bout_id_prefix: str = "sprint",
) -> list[SprintBout]:
    """Hysteresis sprint state machine within one continuous eligible segment."""
    sprint_cfg = config["sprint"]
    entry = float(sprint_cfg["entry_speed_mps"])
    exit_ = float(sprint_cfg["exit_speed_mps"])
    min_dur = int(sprint_cfg["min_duration_us"])
    min_samples = int(sprint_cfg["min_supporting_samples"])
    min_dist = float(sprint_cfg["min_distance_m"])
    max_gap = int(sprint_cfg["max_internal_sample_gap_us"])
    origin = str(config["metric_origin"])
    style = str(config["definition_style"])
    profile_v = int(sprint_cfg["profile_version"])

    speed_res = compute_segment_speeds(
        points,
        trajectory_segment_id=trajectory_segment_id,
        sample_layer=sample_layer,
        config=config,
        diagnostic=False,
    )
    accepted = [s for s in speed_res.instantaneous if not s.rejected and math.isfinite(s.speed_mps)]
    if not accepted:
        return []

    bouts: list[SprintBout] = []
    in_sprint = False
    start_idx = 0
    bout_seq = 0

    def _close(end_idx: int, *, force_reasons: Sequence[str] = ()) -> None:
        nonlocal bout_seq, in_sprint
        if not in_sprint:
            return
        window = accepted[start_idx : end_idx + 1]
        in_sprint = False
        if len(window) < min_samples:
            return
        start_us = window[0].t0_us
        end_us = window[-1].t1_us
        duration = end_us - start_us
        dist = _interval_distance(window)
        speeds = [s.speed_mps for s in window]
        reasons = list(force_reasons)
        evaluability = "evaluable"
        if duration < min_dur:
            evaluability = "not_evaluable"
            reasons.append("SPRINT_BELOW_MIN_DURATION")
        if dist < min_dist:
            evaluability = "not_evaluable"
            reasons.append("SPRINT_BELOW_MIN_DISTANCE")
        if sprint_cfg.get("single_frame_spike_not_sprint") and len(window) < 2:
            evaluability = "not_evaluable"
            reasons.append("SINGLE_SPIKE_NOT_SPRINT")
        # Uncertainty: reject if any sample uncertainty exceeds policy max
        unc_max = float(config["input_eligibility"]["uncertainty_max_m"])
        unc_vals = [s.uncertainty_m for s in window if s.uncertainty_m is not None]
        if any(u > unc_max for u in unc_vals):
            evaluability = "not_evaluable"
            reasons.append("UNCERTAIN_INTERVAL_NOT_SPRINT")
        bout_seq += 1
        unc = max(unc_vals) if unc_vals else None
        observed_us = sum(s.t1_us - s.t0_us for s in window)
        bouts.append(
            SprintBout(
                sprint_id=f"{bout_id_prefix}_{trajectory_segment_id}_{bout_seq:02d}",
                start_time_us=start_us,
                end_time_us=end_us,
                duration_us=duration,
                distance_m=float(dist),
                peak_robust_speed_mps=float(max(speeds)),
                mean_robust_speed_mps=float(sum(speeds) / len(speeds)),
                supporting_sample_count=len(window) + 1,  # intervals → points
                observed_coverage_us=observed_us,
                derived_coverage_us=0,
                uncertainty_m=unc,
                source_segment_id=trajectory_segment_id,
                sample_layer=sample_layer,
                evaluability=evaluability,
                reason_codes=tuple(sorted(set(reasons))),
                metric_origin=origin,
                definition_style=style,
                config_fingerprint=config_fingerprint,
                sprint_profile_version=profile_v,
            )
        )

    for i, sample in enumerate(accepted):
        # Gap between consecutive accepted intervals terminates sprint.
        if i > 0 and in_sprint:
            gap = sample.t0_us - accepted[i - 1].t1_us
            if gap > max_gap:
                _close(i - 1, force_reasons=("SPRINT_TERMINATED_BY_GAP",))
        speed = sample.speed_mps
        if not in_sprint:
            if speed >= entry:
                in_sprint = True
                start_idx = i
        else:
            if speed < exit_:
                _close(i - 1 if i > 0 else i, force_reasons=())
                # Re-check entry on current sample after exit
                if speed >= entry:
                    in_sprint = True
                    start_idx = i

    if in_sprint:
        _close(len(accepted) - 1, force_reasons=("SPRINT_ENDED_AT_SEGMENT_BOUNDARY",))

    return bouts


def sprint_bouts_to_dicts(bouts: Sequence[SprintBout]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for b in bouts:
        out.append(
            {
                "sprint_id": b.sprint_id,
                "start_time_us": b.start_time_us,
                "end_time_us": b.end_time_us,
                "duration_us": b.duration_us,
                "distance_m": b.distance_m,
                "peak_robust_speed_mps": b.peak_robust_speed_mps,
                "mean_robust_speed_mps": b.mean_robust_speed_mps,
                "supporting_sample_count": b.supporting_sample_count,
                "observed_coverage_us": b.observed_coverage_us,
                "derived_coverage_us": b.derived_coverage_us,
                "uncertainty_m": b.uncertainty_m,
                "source_segment_id": b.source_segment_id,
                "sample_layer": b.sample_layer,
                "evaluability": b.evaluability,
                "reason_codes": list(b.reason_codes),
                "metric_origin": b.metric_origin,
                "definition_style": b.definition_style,
                "config_fingerprint": b.config_fingerprint,
                "sprint_profile_version": b.sprint_profile_version,
            }
        )
    return out


def count_evaluable_sprints(bouts: Sequence[SprintBout]) -> dict[str, Any]:
    evaluable = [b for b in bouts if b.evaluability == "evaluable"]
    return {
        "sprint_count": len(evaluable),
        "sprint_distance_m": float(sum(b.distance_m for b in evaluable)),
        "sprint_duration_us": int(sum(b.duration_us for b in evaluable)),
        "not_evaluable_count": sum(1 for b in bouts if b.evaluability != "evaluable"),
    }


__all__ = [
    "SprintBout",
    "extract_sprint_bouts_for_segment",
    "sprint_bouts_to_dicts",
    "count_evaluable_sprints",
]
