"""Tracking request/receipt builders for synthetic contract runs only."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from football_analytics.tracking.contracts import (
    load_tracking_json_schema,
    validate_against_json_schema,
)
from football_analytics.tracking.evaluation import NOT_EVALUATED_TRACKING
from football_analytics.tracking.types import LifecycleState, TrackingContractError


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def build_synthetic_request(
    *,
    run_id: str,
    video_id: str,
    policy_fingerprint: str,
    tracker_config_fingerprint: str | None = None,
    output_root: str = "/tmp/tracking_contract_synth",
    entity_scope: Sequence[str] = ("human", "ball"),
    request_id: str = "trk_req_01",
) -> dict[str, Any]:
    fp = tracker_config_fingerprint or policy_fingerprint
    return {
        "schema_version": 1,
        "request_id": request_id,
        "run_id": run_id,
        "video_id": video_id,
        "detection_bundle_ref": "detections.parquet",
        "frames_ref": "frames.parquet",
        "analysis_windows_ref": "analysis_windows.parquet",
        "tracker_config_fingerprint": fp,
        "policy_fingerprint": policy_fingerprint,
        "entity_scope": list(entity_scope),
        "output_root": output_root,
        "no_overwrite": True,
        "cache_policy": "disabled",
        "source_sha256": "b" * 64,
        "timeline_fingerprint": "c" * 64,
        "detection_bundle_fingerprint": "d" * 64,
        "provenance": {
            "stage": "6A",
            "label": "synthetic_contract_fixture",
            "notes": "no_tracker_algorithm",
        },
    }


def recount_receipt_from_tables(
    *,
    observations: Sequence[Mapping[str, Any]],
    lifecycle: Sequence[Mapping[str, Any]],
    detections: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    observed = sum(1 for o in observations if o["observation_state"] == "observed")
    predicted = sum(1 for o in observations if o["observation_state"] == "predicted")
    interpolated = sum(1 for o in observations if o["observation_state"] == "interpolated")
    final: dict[tuple[Any, Any, int], str] = {}
    for ev in sorted(lifecycle, key=lambda e: int(e["event_index"])):
        final[(ev["run_id"], ev["video_id"], int(ev["track_id"]))] = str(ev["lifecycle_state"])
    track_counts = {s.value: 0 for s in LifecycleState}
    for st in final.values():
        track_counts[st] = track_counts.get(st, 0) + 1
    used = {
        (o["run_id"], o["video_id"], o["frame_index"], o["detection_id"])
        for o in observations
        if o.get("detection_id") is not None
    }
    total_dets = len(detections) if detections is not None else len(used)
    return {
        "observation_counts": {
            "detection_associated": observed,
            "predicted": predicted,
            "interpolated": interpolated,
            "observed": observed,
            "total": len(observations),
        },
        "track_counts": track_counts,
        "detections_used": len(used),
        "total_input_detections": total_dets,
        "unassigned_detection_count": max(0, total_dets - len(used)),
        "review_required_count": sum(1 for e in lifecycle if bool(e.get("manual_review_required"))),
    }


def build_synthetic_receipt(
    *,
    run_id: str,
    video_id: str,
    policy_fingerprint: str,
    observations: Sequence[Mapping[str, Any]],
    lifecycle: Sequence[Mapping[str, Any]],
    detections: Sequence[Mapping[str, Any]] | None = None,
    request_id: str = "trk_req_01",
    receipt_id: str = "trk_receipt_01",
    status: str = "succeeded",
) -> dict[str, Any]:
    counts = recount_receipt_from_tables(
        observations=observations, lifecycle=lifecycle, detections=detections
    )
    return {
        "schema_version": 1,
        "receipt_id": receipt_id,
        "run_id": run_id,
        "video_id": video_id,
        "request_id": request_id,
        "tracker_id": "contract_synth_v1",
        "tracker_version": "0.0.0",
        "config_fingerprint": policy_fingerprint,
        "policy_fingerprint": policy_fingerprint,
        "input_artifacts": {
            "detections": {
                "path": "detections.parquet",
                "sha256": "d" * 64,
                "size_bytes": 1,
            }
        },
        "output_artifacts": {
            "track_observations": {
                "path": "track_observations.parquet",
                "sha256": "e" * 64,
                "size_bytes": 1,
            },
            "track_lifecycle": {
                "path": "track_lifecycle.parquet",
                "sha256": "f" * 64,
                "size_bytes": 1,
            },
        },
        "total_input_detections": counts["total_input_detections"],
        "detections_used": counts["detections_used"],
        "detections_rejected": 0,
        "unassigned_detection_count": counts["unassigned_detection_count"],
        "track_counts": counts["track_counts"],
        "observation_counts": counts["observation_counts"],
        "invalid_transition_count": 0,
        "invalid_fk_count": 0,
        "duplicate_count": 0,
        "routing_gap_count": 0,
        "review_required_count": counts["review_required_count"],
        "ground_truth_evaluation_status": NOT_EVALUATED_TRACKING,
        "started_at_utc": "2026-07-23T00:00:00.000000Z",
        "completed_at_utc": _utc_now(),
        "status": status,
        "warnings": [],
        "errors": [],
        "environment_ref": None,
        "provenance": {
            "stage": "6A",
            "label": "synthetic_contract_fixture",
            "notes": "no_tracker_algorithm",
            "tracker_algorithm": "none_stage_6a_contracts_only",
            "merge_reid": False,
            "track_id_is_player_identity": False,
        },
    }


def validate_request_payload(payload: Mapping[str, Any], *, project_root: Any = None) -> None:
    schema = load_tracking_json_schema("tracking_request", project_root=project_root)
    try:
        validate_against_json_schema(dict(payload), schema)
    except Exception as exc:  # noqa: BLE001
        raise TrackingContractError(f"tracking_request schema invalid: {exc}") from exc


def validate_receipt_payload(payload: Mapping[str, Any], *, project_root: Any = None) -> None:
    schema = load_tracking_json_schema("tracking_run_receipt", project_root=project_root)
    try:
        validate_against_json_schema(dict(payload), schema)
    except Exception as exc:  # noqa: BLE001
        raise TrackingContractError(f"tracking_run_receipt schema invalid: {exc}") from exc


__all__ = [
    "build_synthetic_request",
    "build_synthetic_receipt",
    "recount_receipt_from_tables",
    "validate_request_payload",
    "validate_receipt_payload",
]
