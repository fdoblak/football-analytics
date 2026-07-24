"""Operational quality + review sampling for Stage 8D pitch projection."""

from __future__ import annotations

import json
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from football_analytics.calibration.types import MappingStatus, PhysicalMetricEligibility
from football_analytics.identity.review_audit import sample_review_items


def recount_projection_stats(projections: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    status_counts = {m.value: 0 for m in MappingStatus}
    human_n = 0
    ball_n = 0
    human_phys = 0
    target_customer = 0
    ball_phys = 0
    ball_event = 0
    predicted_phys_violations = 0
    extrapolated_phys_violations = 0
    rt_errors: list[float] = []
    unc_values: list[float] = []
    outside_coverage = 0
    for p in projections:
        st = str(p["mapping_status"])
        status_counts[st] = status_counts.get(st, 0) + 1
        et = str(p["entity_type"])
        if et == "human":
            human_n += 1
        elif et == "ball":
            ball_n += 1
        if p.get("physical_metric_eligibility") == PhysicalMetricEligibility.ELIGIBLE.value:
            if et == "human":
                human_phys += 1
            if et == "ball":
                ball_phys += 1
        if (
            str(p.get("observation_source")) in {"predicted", "interpolated"}
            and p.get("physical_metric_eligibility") == PhysicalMetricEligibility.ELIGIBLE.value
        ):
            predicted_phys_violations += 1
        if (
            p.get("is_extrapolated")
            and p.get("physical_metric_eligibility") == PhysicalMetricEligibility.ELIGIBLE.value
        ):
            extrapolated_phys_violations += 1
        prov = {}
        if p.get("provenance_json"):
            try:
                prov = json.loads(str(p["provenance_json"]))
            except json.JSONDecodeError:
                prov = {}
        if prov.get("target_customer_metric_eligible") is True:
            target_customer += 1
        if et == "ball" and prov.get("event_metric_eligible") is True:
            ball_event += 1
        if prov.get("outside_coverage") is True:
            outside_coverage += 1
        if prov.get("round_trip_error_px") is not None:
            rt_errors.append(float(prov["round_trip_error_px"]))
        if p.get("uncertainty_m") is not None:
            unc_values.append(float(p["uncertainty_m"]))
    return {
        "total": len(projections),
        "human_observation_count": human_n,
        "ball_observation_count": ball_n,
        "mapping_status_counts": status_counts,
        "human_physical_metric_eligible_count": human_phys,
        "target_customer_metric_eligible_count": target_customer,
        "ball_physical_metric_eligible_count": ball_phys,
        "ball_event_metric_eligible_count": ball_event,
        "predicted_physical_eligibility_violations": predicted_phys_violations,
        "extrapolated_physical_eligibility_violations": extrapolated_phys_violations,
        "outside_coverage_count": outside_coverage,
        "round_trip_error_px": {
            "n": len(rt_errors),
            "mean": statistics.fmean(rt_errors) if rt_errors else None,
            "max": max(rt_errors) if rt_errors else None,
        },
        "uncertainty_m": {
            "n": len(unc_values),
            "mean": statistics.fmean(unc_values) if unc_values else None,
            "max": max(unc_values) if unc_values else None,
        },
    }


def operational_quality_status(stats: Mapping[str, Any]) -> str:
    if (
        int(stats.get("ball_physical_metric_eligible_count", 0)) != 0
        or int(stats.get("ball_event_metric_eligible_count", 0)) != 0
        or int(stats.get("predicted_physical_eligibility_violations", 0)) > 0
        or int(stats.get("extrapolated_physical_eligibility_violations", 0)) > 0
    ):
        return "fail"
    if int(stats.get("total", 0)) == 0:
        return "not_evaluated"
    # Operational gates pass with findings when no reviewed GT (caller sets).
    return "pass_with_findings"


def build_projection_review_queue(
    projections: Sequence[Mapping[str, Any]],
    *,
    max_samples: int,
    enabled: bool,
    conflicts: Sequence[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for p in projections:
        reasons = [str(x) for x in (p.get("reason_codes") or [])]
        needs = bool(p.get("manual_review_required"))
        if not needs and str(p["mapping_status"]) not in {
            MappingStatus.UNCERTAIN.value,
            MappingStatus.FAILED.value,
            MappingStatus.EXTRAPOLATED.value,
            MappingStatus.OUTSIDE_PITCH.value,
        }:
            continue
        priority = "b"
        if (
            str(p["mapping_status"]) == MappingStatus.FAILED.value
            or "SEGMENT_OVERLAP_CONFLICT" in reasons
        ):
            priority = "a"
        candidates.append(
            {
                "item_id": str(p["projection_id"]),
                "priority": priority,
                "run_id": str(p["run_id"]),
                "video_id": str(p["video_id"]),
                "frame_index": int(p["frame_index"]),
                "entity_type": str(p["entity_type"]),
                "mapping_status": str(p["mapping_status"]),
                "reason_codes": reasons,
            }
        )
    for a, b in conflicts or []:
        candidates.append(
            {
                "item_id": f"conflict_{a}_{b}",
                "priority": "a",
                "run_id": "",
                "video_id": "",
                "frame_index": -1,
                "entity_type": "unknown",
                "mapping_status": "conflict",
                "reason_codes": ["SEGMENT_OVERLAP_CONFLICT", a, b],
            }
        )
    sampled = sample_review_items(candidates, max_items=max_samples if enabled else 0)
    return {
        "schema_version": 1,
        "enabled": enabled,
        "max_samples": max_samples,
        "candidate_count": len(candidates),
        "sampled_count": len(sampled),
        "items": [dict(x) for x in sampled],
        "notes": "sampled review only; not auto-accepted",
    }


def build_quality_report(
    *,
    run_id: str,
    video_id: str,
    stats: Mapping[str, Any],
    segment_usage: Mapping[str, int],
    conflict_count: int,
    duplicate_count: int,
    evaluation_status: str,
    config_fingerprint: str,
) -> dict[str, Any]:
    status = operational_quality_status(stats)
    return {
        "schema_version": 1,
        "run_id": run_id,
        "video_id": video_id,
        "status": status,
        "config_fingerprint": config_fingerprint,
        "stats": dict(stats),
        "segment_usage": dict(segment_usage),
        "conflict_count": int(conflict_count),
        "duplicate_count": int(duplicate_count),
        "evaluation_status": evaluation_status,
        "notes": [
            "operational quality only; not football metre accuracy",
            "ball physical/event eligible must be 0",
            "attack_direction=unknown",
            "no distance/speed/sprint/heatmap/events",
        ],
    }


__all__ = [
    "recount_projection_stats",
    "operational_quality_status",
    "build_projection_review_queue",
    "build_quality_report",
]
