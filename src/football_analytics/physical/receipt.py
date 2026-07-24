"""Physical metric request/receipt builders for contract fixtures only (Stage 9A)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from football_analytics.physical.contracts import (
    load_physical_json_schema,
    validate_against_json_schema,
)
from football_analytics.physical.evaluation import NOT_EVALUATED_PHYSICAL
from football_analytics.physical.types import PhysicalContractError
from football_analytics.physical.validation import recount_sample_layers


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def build_synthetic_request(
    *,
    run_id: str,
    video_id: str,
    target_player_id: str,
    trajectory_policy_fingerprint: str,
    metrics_policy_fingerprint: str,
    pitch_template_fingerprint: str,
    request_id: str = "phys_req_01",
    output_root: str = "/home/fdoblak/workspace/physical_metric_contract_checks",
    metric_selection: Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "request_id": request_id,
        "run_id": run_id,
        "video_id": video_id,
        "target_player_id": target_player_id,
        "identity_artifact_ref": "synthetic://identity/confirmed",
        "projected_positions_ref": "synthetic://calibration/projected_positions",
        "pitch_template_fingerprint": pitch_template_fingerprint,
        "trajectory_policy_fingerprint": trajectory_policy_fingerprint,
        "metrics_policy_fingerprint": metrics_policy_fingerprint,
        "metric_selection": list(metric_selection or ["distance", "speed", "sprint", "heatmap"]),
        "filter_resample_policy_ref": None,
        "output_root": output_root,
        "no_overwrite": True,
        "cache_enabled": False,
        "created_at_utc": _utc_now(),
        "provenance": {
            "stage": "9A",
            "label": "synthetic_contract_fixture",
            "notes": "no_real_metric_computation",
            "no_real_metric_computation": True,
        },
    }


def build_synthetic_receipt(
    *,
    run_id: str,
    video_id: str,
    target_player_id: str,
    trajectory_policy_fingerprint: str,
    metrics_policy_fingerprint: str,
    samples: Sequence[Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
    gaps: Sequence[Mapping[str, Any]],
    metric_results: Sequence[Mapping[str, Any]] | None = None,
    request_id: str = "phys_req_01",
    receipt_id: str = "phys_receipt_01",
    status: str = "contract_stub",
    review_count: int = 0,
    outlier_count: int = 0,
) -> dict[str, Any]:
    metric_results = list(metric_results or [])
    layers = recount_sample_layers(samples)
    eligible = sum(1 for s in samples if str(s.get("metric_eligibility")) == "eligible")
    rejected = len(samples) - eligible
    metric_status_counts: dict[str, int] = {}
    for m in metric_results:
        st = str(m.get("status"))
        metric_status_counts[st] = metric_status_counts.get(st, 0) + 1
    eligible_duration = sum(
        int(s.get("duration_us", 0))
        for s in segments
        if str(s.get("metric_eligibility")) == "eligible"
    )
    return {
        "schema_version": 1,
        "receipt_id": receipt_id,
        "request_id": request_id,
        "run_id": run_id,
        "video_id": video_id,
        "target_player_id": target_player_id,
        "status": status,
        "trajectory_policy_fingerprint": trajectory_policy_fingerprint,
        "metrics_policy_fingerprint": metrics_policy_fingerprint,
        "input_fingerprints": {
            "projected_positions": None,
            "identity": None,
            "pitch_template": None,
        },
        "output_fingerprints": {
            "target_trajectory_samples": None,
            "target_trajectory_segments": None,
            "trajectory_gaps": None,
            "physical_metric_results": None,
        },
        "eligible_sample_count": eligible,
        "rejected_sample_count": rejected,
        "segment_count": len(segments),
        "gap_count": len(gaps),
        "raw_sample_count": layers["raw_observed"],
        "filtered_sample_count": layers["filtered"],
        "resampled_sample_count": layers["resampled"],
        "metric_status_counts": metric_status_counts,
        "eligible_duration_us": eligible_duration,
        "excluded_duration_us": 0,
        "coverage_summary": {
            "eligible_sample_ratio": (eligible / len(samples)) if samples else 0.0,
            "note": "coverage_is_not_activity",
        },
        "outlier_count": outlier_count,
        "review_count": review_count,
        "evaluation_status": NOT_EVALUATED_PHYSICAL,
        "artifact_hashes": {},
        "warning_codes": [],
        "error_codes": [],
        "created_at_utc": _utc_now(),
        "provenance": {
            "stage": "9A",
            "label": "synthetic_contract_fixture",
            "notes": "no_real_metric_computation",
            "no_real_metric_computation": True,
        },
    }


def validate_request_payload(payload: Mapping[str, Any]) -> None:
    schema = load_physical_json_schema("physical_metric_request")
    try:
        validate_against_json_schema(dict(payload), schema)
    except Exception as exc:  # noqa: BLE001
        raise PhysicalContractError(f"request schema invalid: {exc}") from exc
    if payload.get("provenance", {}).get("no_real_metric_computation") is not True:
        raise PhysicalContractError("request must declare no_real_metric_computation")


def validate_receipt_payload(payload: Mapping[str, Any]) -> None:
    schema = load_physical_json_schema("physical_metric_run_receipt")
    try:
        validate_against_json_schema(dict(payload), schema)
    except Exception as exc:  # noqa: BLE001
        raise PhysicalContractError(f"receipt schema invalid: {exc}") from exc
    # Stage 9A contract fixtures without reviewed GT keep NOT_EVALUATED status.
    _ = payload.get("evaluation_status")


def recount_receipt_counts(
    *,
    samples: Sequence[Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
    gaps: Sequence[Mapping[str, Any]],
    receipt: Mapping[str, Any],
) -> list[str]:
    """Return mismatch reasons if receipt counts disagree with artifacts."""
    errors: list[str] = []
    layers = recount_sample_layers(samples)
    eligible = sum(1 for s in samples if str(s.get("metric_eligibility")) == "eligible")
    if int(receipt["eligible_sample_count"]) != eligible:
        errors.append("eligible_sample_count mismatch")
    if int(receipt["raw_sample_count"]) != layers["raw_observed"]:
        errors.append("raw_sample_count mismatch")
    if int(receipt["segment_count"]) != len(segments):
        errors.append("segment_count mismatch")
    if int(receipt["gap_count"]) != len(gaps):
        errors.append("gap_count mismatch")
    return errors


__all__ = [
    "build_synthetic_request",
    "build_synthetic_receipt",
    "validate_request_payload",
    "validate_receipt_payload",
    "recount_receipt_counts",
]
