"""Synthetic physical / trajectory contract fixtures only (Stage 9A)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pyarrow as pa

from football_analytics.core.run_id import generate_run_id
from football_analytics.data.compiler import compile_arrow_schema, get_contract
from football_analytics.physical.types import CONTRACT_VERSION


def _cast(name: str, rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    schema = compile_arrow_schema(get_contract(name, 1))
    return pa.Table.from_pylist([dict(r) for r in rows], schema=schema)


def base_ids() -> dict[str, str]:
    return {
        "run_id": generate_run_id(),
        "video_id": "video_synth_01",
        "target_player_id": "target_player_01",
        "identity_assignment_id": "asn_confirmed_01",
    }


def sample_row(
    run_id: str,
    video_id: str,
    target_player_id: str,
    sample_id: str,
    *,
    identity_assignment_id: str,
    track_id: int,
    frame_index: int,
    video_time_us: int,
    pitch_x_m: float,
    pitch_y_m: float,
    policy_fingerprint: str,
    sample_source: str = "raw_observed",
    eligibility_status: str = "eligible",
    metric_eligibility: str = "eligible",
    identity_quality: str = "confirmed",
    mapping_status: str = "mapped",
    gap_boundary_reason: str = "none",
    derived_from_sample_ids: Sequence[str] | None = None,
    trajectory_segment_id: str | None = "traj_seg_01",
    projection_id: str | None = "proj_01",
    reason_codes: Sequence[str] | None = None,
    pitch_coordinate_frame_id: str = "canonical_pitch",
    pitch_template_fingerprint: str = "a" * 64,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "target_player_id": target_player_id,
        "identity_assignment_id": identity_assignment_id,
        "sample_id": sample_id,
        "track_id": track_id,
        "observation_id": f"obs_{frame_index}",
        "detection_id": frame_index,
        "projection_id": projection_id,
        "frame_index": frame_index,
        "video_time_us": video_time_us,
        "pitch_x_m": pitch_x_m,
        "pitch_y_m": pitch_y_m,
        "pitch_coordinate_frame_id": pitch_coordinate_frame_id,
        "pitch_template_fingerprint": pitch_template_fingerprint,
        "calibration_id": 0,
        "segment_id": "cal_seg_01",
        "source_point_type": "bbox_bottom_centre",
        "sample_source": sample_source,
        "derived_from_sample_ids": list(derived_from_sample_ids or []),
        "mapping_status": mapping_status,
        "calibration_quality": "good",
        "identity_quality": identity_quality,
        "uncertainty_m": 0.4,
        "uncertainty_x_m": 0.3,
        "uncertainty_y_m": 0.3,
        "eligibility_status": eligibility_status,
        "gap_boundary_reason": gap_boundary_reason,
        "metric_eligibility": metric_eligibility,
        "trajectory_segment_id": trajectory_segment_id,
        "manual_review_required": False,
        "reason_codes": list(reason_codes or []),
        "quality_flags": [],
        "evidence_fingerprint": "b" * 64,
        "projected_positions_fingerprint": "c" * 64,
        "identity_artifact_fingerprint": "d" * 64,
        "calibration_artifact_fingerprint": "e" * 64,
        "policy_fingerprint": policy_fingerprint,
        "provenance_json": None,
        "contract_version": CONTRACT_VERSION,
    }


def segment_row(
    run_id: str,
    video_id: str,
    target_player_id: str,
    trajectory_segment_id: str,
    *,
    identity_assignment_id: str,
    track_id: int,
    start_time_us: int,
    end_time_us: int,
    raw_sample_count: int,
    eligible_sample_count: int,
    policy_fingerprint: str,
    segment_status: str = "continuous",
    metric_eligibility: str = "eligible",
    start_boundary_reason: str = "none",
    end_boundary_reason: str = "none",
    reason_codes: Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "target_player_id": target_player_id,
        "trajectory_segment_id": trajectory_segment_id,
        "identity_assignment_id": identity_assignment_id,
        "track_id": track_id,
        "start_time_us": start_time_us,
        "end_time_us": end_time_us,
        "raw_sample_count": raw_sample_count,
        "eligible_sample_count": eligible_sample_count,
        "duration_us": max(0, end_time_us - start_time_us),
        "calibration_segment_ids": ["cal_seg_01"],
        "start_boundary_reason": start_boundary_reason,
        "end_boundary_reason": end_boundary_reason,
        "coverage_ratio": 1.0 if eligible_sample_count >= 2 else 0.0,
        "max_sample_interval_us": 100_000,
        "uncertainty_summary_m": 0.4,
        "segment_status": segment_status,
        "metric_eligibility": metric_eligibility,
        "manual_review_required": False,
        "pitch_coordinate_frame_id": "canonical_pitch",
        "input_fingerprint": "f" * 64,
        "output_fingerprint": "0" * 64,
        "policy_fingerprint": policy_fingerprint,
        "reason_codes": list(reason_codes or []),
        "quality_flags": [],
        "provenance_json": None,
        "contract_version": CONTRACT_VERSION,
    }


def gap_row(
    run_id: str,
    video_id: str,
    target_player_id: str,
    gap_id: str,
    *,
    gap_type: str,
    start_time_us: int,
    end_time_us: int,
    policy_fingerprint: str,
    preceding_segment_id: str | None = None,
    following_segment_id: str | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "target_player_id": target_player_id,
        "gap_id": gap_id,
        "gap_type": gap_type,
        "start_time_us": start_time_us,
        "end_time_us": end_time_us,
        "duration_us": max(0, end_time_us - start_time_us),
        "preceding_segment_id": preceding_segment_id,
        "following_segment_id": following_segment_id,
        "track_id": 0,
        "identity_assignment_id": "asn_confirmed_01",
        "allows_distance_bridge": False,
        "allows_interpolation_default": False,
        "manual_review_required": False,
        "reason_codes": [],
        "quality_flags": [],
        "policy_fingerprint": policy_fingerprint,
        "provenance_json": None,
        "contract_version": CONTRACT_VERSION,
    }


def metric_result_row(
    run_id: str,
    video_id: str,
    target_player_id: str,
    metric_result_id: str,
    *,
    metric_name: str,
    unit: str,
    config_fingerprint: str,
    status: str = "contract_stub",
    value: float | None = None,
    sample_layer: str = "none",
    reason_codes: Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "target_player_id": target_player_id,
        "metric_result_id": metric_result_id,
        "metric_name": metric_name,
        "metric_version": 1,
        "time_scope_start_us": 0,
        "time_scope_end_us": 1_000_000,
        "value": value,
        "unit": unit,
        "status": status,
        "coverage_ratio": None,
        "confidence": None,
        "uncertainty": None,
        "included_sample_count": 0,
        "excluded_sample_count": 0,
        "included_duration_us": 0,
        "excluded_duration_us": 0,
        "sample_layer": sample_layer,
        "trajectory_segment_ids": [],
        "evidence_ids": [],
        "config_fingerprint": config_fingerprint,
        "trajectory_artifact_fingerprint": None,
        "calibration_artifact_fingerprint": None,
        "identity_artifact_fingerprint": None,
        "warning_codes": [],
        "reason_codes": list(reason_codes or ["STAGE_9A_CONTRACTS_ONLY"]),
        "review_status": "not_required",
        "producer": "physical_contract_synth",
        "producer_version": "0.0.0",
        "provenance_json": None,
        "contract_version": CONTRACT_VERSION,
    }


def confirmed_observed_bundle(policy_fingerprint: str) -> dict[str, Any]:
    ids = base_ids()
    rid, vid, tid = ids["run_id"], ids["video_id"], ids["target_player_id"]
    asn = ids["identity_assignment_id"]
    samples = [
        sample_row(
            rid,
            vid,
            tid,
            f"smp_{i:02d}",
            identity_assignment_id=asn,
            track_id=0,
            frame_index=i,
            video_time_us=i * 100_000,
            pitch_x_m=10.0 + i,
            pitch_y_m=20.0,
            policy_fingerprint=policy_fingerprint,
        )
        for i in range(5)
    ]
    segs = [
        segment_row(
            rid,
            vid,
            tid,
            "traj_seg_01",
            identity_assignment_id=asn,
            track_id=0,
            start_time_us=0,
            end_time_us=500_000,
            raw_sample_count=5,
            eligible_sample_count=5,
            policy_fingerprint=policy_fingerprint,
        )
    ]
    return {
        **ids,
        "sample_rows": samples,
        "segment_rows": segs,
        "gap_rows": [],
        "metric_rows": [
            metric_result_row(
                rid,
                vid,
                tid,
                "met_distance_01",
                metric_name="distance",
                unit="m",
                config_fingerprint=policy_fingerprint,
            )
        ],
        "target_trajectory_samples": _cast("target_trajectory_samples", samples),
        "target_trajectory_segments": _cast("target_trajectory_segments", segs),
        "trajectory_gaps": _cast("trajectory_gaps", []),
        "physical_metric_results": _cast(
            "physical_metric_results",
            [
                metric_result_row(
                    rid,
                    vid,
                    tid,
                    "met_distance_01",
                    metric_name="distance",
                    unit="m",
                    config_fingerprint=policy_fingerprint,
                )
            ],
        ),
    }


def single_sample_segment_bundle(policy_fingerprint: str) -> dict[str, Any]:
    ids = base_ids()
    rid, vid, tid = ids["run_id"], ids["video_id"], ids["target_player_id"]
    asn = ids["identity_assignment_id"]
    samples = [
        sample_row(
            rid,
            vid,
            tid,
            "smp_00",
            identity_assignment_id=asn,
            track_id=0,
            frame_index=0,
            video_time_us=0,
            pitch_x_m=10.0,
            pitch_y_m=20.0,
            policy_fingerprint=policy_fingerprint,
        )
    ]
    segs = [
        segment_row(
            rid,
            vid,
            tid,
            "traj_seg_single",
            identity_assignment_id=asn,
            track_id=0,
            start_time_us=0,
            end_time_us=100_000,
            raw_sample_count=1,
            eligible_sample_count=1,
            policy_fingerprint=policy_fingerprint,
            segment_status="insufficient",
            metric_eligibility="not_eligible",
            reason_codes=["SINGLE_SAMPLE_SEGMENT_INSUFFICIENT"],
        )
    ]
    return {
        **ids,
        "sample_rows": samples,
        "segment_rows": segs,
        "target_trajectory_samples": _cast("target_trajectory_samples", samples),
        "target_trajectory_segments": _cast("target_trajectory_segments", segs),
    }


def gap_bundle(policy_fingerprint: str, *, gap_type: str) -> dict[str, Any]:
    ids = base_ids()
    rid, vid, tid = ids["run_id"], ids["video_id"], ids["target_player_id"]
    asn = ids["identity_assignment_id"]
    samples = [
        sample_row(
            rid,
            vid,
            tid,
            "smp_00",
            identity_assignment_id=asn,
            track_id=0,
            frame_index=0,
            video_time_us=0,
            pitch_x_m=10.0,
            pitch_y_m=20.0,
            policy_fingerprint=policy_fingerprint,
            trajectory_segment_id="traj_seg_a",
        ),
        sample_row(
            rid,
            vid,
            tid,
            "smp_01",
            identity_assignment_id=asn,
            track_id=0,
            frame_index=20,
            video_time_us=2_000_000,
            pitch_x_m=30.0,
            pitch_y_m=20.0,
            policy_fingerprint=policy_fingerprint,
            trajectory_segment_id="traj_seg_b",
            gap_boundary_reason=gap_type if gap_type != "shot_boundary" else "shot_boundary",
        ),
    ]
    gaps = [
        gap_row(
            rid,
            vid,
            tid,
            "gap_01",
            gap_type=gap_type,
            start_time_us=100_000,
            end_time_us=2_000_000,
            policy_fingerprint=policy_fingerprint,
            preceding_segment_id="traj_seg_a",
            following_segment_id="traj_seg_b",
        )
    ]
    return {
        **ids,
        "sample_rows": samples,
        "gap_rows": gaps,
        "target_trajectory_samples": _cast("target_trajectory_samples", samples),
        "trajectory_gaps": _cast("trajectory_gaps", gaps),
    }


def provisional_exclusion_candidate() -> dict[str, Any]:
    return {
        "identity_status": "provisional",
        "entity_type": "human",
        "observation_source": "detection_associated",
        "mapping_status": "mapped",
        "physical_metric_eligibility": "eligible",
        "is_extrapolated": False,
        "assignment_revoked_or_conflicted": False,
        "playable_non_replay": True,
        "fingerprints_match": True,
    }


def predicted_exclusion_candidate() -> dict[str, Any]:
    return {
        "identity_status": "confirmed",
        "entity_type": "human",
        "observation_source": "predicted",
        "mapping_status": "mapped",
        "physical_metric_eligibility": "eligible",
        "is_extrapolated": False,
        "assignment_revoked_or_conflicted": False,
        "playable_non_replay": True,
        "fingerprints_match": True,
    }


def eligible_candidate() -> dict[str, Any]:
    return {
        "identity_status": "confirmed",
        "entity_type": "human",
        "observation_source": "detection_associated",
        "mapping_status": "mapped",
        "physical_metric_eligibility": "eligible",
        "is_extrapolated": False,
        "assignment_revoked_or_conflicted": False,
        "playable_non_replay": True,
        "fingerprints_match": True,
    }


__all__ = [
    "base_ids",
    "sample_row",
    "segment_row",
    "gap_row",
    "metric_result_row",
    "confirmed_observed_bundle",
    "single_sample_segment_bundle",
    "gap_bundle",
    "provisional_exclusion_candidate",
    "predicted_exclusion_candidate",
    "eligible_candidate",
]
