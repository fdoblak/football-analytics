"""Stage 9C distance / speed / sprint computation service."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa

from football_analytics.core.hashing import sha256_file
from football_analytics.core.records import write_json_record
from football_analytics.data.compiler import compile_arrow_schema, get_contract
from football_analytics.data.fingerprint import contract_fingerprint
from football_analytics.data.parquet import write_contract_parquet
from football_analytics.physical.distance import (
    aggregate_measured_distance,
    compute_segment_distance,
)
from football_analytics.physical.motion_config import (
    MotionConfigError,
    load_motion_baseline_config,
    motion_baseline_config_fingerprint,
)
from football_analytics.physical.motion_evaluation import (
    NOT_EVALUATED_MOTION,
    evaluate_motion_metrics,
)
from football_analytics.physical.motion_quality import (
    build_motion_quality_report,
    observed_coverage_from_points,
)
from football_analytics.physical.speed import (
    aggregate_speed_summary,
    compute_segment_speeds,
    mps_to_kmh,
)
from football_analytics.physical.sprint import (
    count_evaluable_sprints,
    extract_sprint_bouts_for_segment,
    sprint_bouts_to_dicts,
)
from football_analytics.physical.types import CONTRACT_VERSION


class MotionServiceError(RuntimeError):
    """Physical motion metric failure."""


@dataclass
class MotionServiceResult:
    accepted: bool
    exit_code: int
    error_code: str | None
    config_fingerprint: str
    segment_metrics_json: str | None
    sprint_bouts_json: str | None
    metric_results_parquet: str | None
    summary_json: str | None
    receipt_json: str | None
    quality_json: str | None
    evaluation_json: str | None
    summary: Mapping[str, Any]

    def to_summary(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "exit_code": self.exit_code,
            "error_code": self.error_code,
            "config_fingerprint": self.config_fingerprint,
            "segment_metrics_json": self.segment_metrics_json,
            "sprint_bouts_json": self.sprint_bouts_json,
            "metric_results_parquet": self.metric_results_parquet,
            "summary_json": self.summary_json,
            "receipt_json": self.receipt_json,
            "quality_json": self.quality_json,
            "evaluation_json": self.evaluation_json,
            **dict(self.summary),
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _cast(name: str, rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    schema = compile_arrow_schema(get_contract(name, 1))
    return pa.Table.from_pylist([dict(r) for r in rows], schema=schema)


def _group_by_segment(
    points: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for p in points:
        sid = str(p.get("trajectory_segment_id") or "traj_seg_unknown")
        groups.setdefault(sid, []).append(dict(p))
    return groups


def _metric_row(
    *,
    run_id: str,
    video_id: str,
    target_player_id: str,
    metric_result_id: str,
    metric_name: str,
    unit: str,
    value: float | None,
    status: str,
    sample_layer: str,
    config_fingerprint: str,
    time_start_us: int,
    time_end_us: int,
    coverage_ratio: float | None,
    uncertainty: float | None,
    included_sample_count: int,
    included_duration_us: int,
    segment_ids: Sequence[str],
    reason_codes: Sequence[str],
    warning_codes: Sequence[str] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "target_player_id": target_player_id,
        "metric_result_id": metric_result_id,
        "metric_name": metric_name,
        "metric_version": 1,
        "time_scope_start_us": int(time_start_us),
        "time_scope_end_us": int(time_end_us),
        "value": value,
        "unit": unit,
        "status": status,
        "coverage_ratio": coverage_ratio,
        "confidence": None,
        "uncertainty": uncertainty,
        "included_sample_count": int(included_sample_count),
        "excluded_sample_count": 0,
        "included_duration_us": int(included_duration_us),
        "excluded_duration_us": 0,
        "sample_layer": sample_layer,
        "trajectory_segment_ids": list(segment_ids),
        "evidence_ids": [],
        "config_fingerprint": config_fingerprint,
        "trajectory_artifact_fingerprint": None,
        "calibration_artifact_fingerprint": None,
        "identity_artifact_fingerprint": None,
        "warning_codes": list(warning_codes or []),
        "reason_codes": list(reason_codes),
        "review_status": "not_required",
        "producer": "distance_speed_sprint_baseline",
        "producer_version": "1",
        "provenance_json": json.dumps(dict(provenance or {}), sort_keys=True),
        "contract_version": CONTRACT_VERSION,
    }


def compute_physical_motion(
    *,
    primary_points: Sequence[Mapping[str, Any]],
    output_dir: Path,
    config: Mapping[str, Any] | None = None,
    config_path: Path | None = None,
    diagnostic_layers: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    analysis_window_us: int | None = None,
    cleanup_on_failure: bool = True,
) -> MotionServiceResult:
    """Compute measured distance / robust speed / sprint bouts from eligible trajectory points."""
    cfg = config or load_motion_baseline_config(config_path)
    cfg_fp = motion_baseline_config_fingerprint(cfg)
    out = Path(output_dir)
    created = False
    primary_layer = str(cfg["primary_sample_layer"])
    try:
        if out.exists() and any(out.iterdir()) and cfg.get("overwrite_allowed") is not True:
            raise MotionServiceError("output_dir not empty and overwrite forbidden")
        out.mkdir(parents=True, exist_ok=True)
        created = True

        groups = _group_by_segment(primary_points)
        segment_metrics: list[dict[str, Any]] = []
        dist_results = []
        speed_results = []
        all_bouts = []

        for seg_id, pts in sorted(groups.items()):
            d = compute_segment_distance(
                pts,
                trajectory_segment_id=seg_id,
                sample_layer=primary_layer,
                config=cfg,
                diagnostic=False,
            )
            s = compute_segment_speeds(
                pts,
                trajectory_segment_id=seg_id,
                sample_layer=primary_layer,
                config=cfg,
                diagnostic=False,
            )
            bouts = extract_sprint_bouts_for_segment(
                pts,
                trajectory_segment_id=seg_id,
                sample_layer=primary_layer,
                config=cfg,
                config_fingerprint=cfg_fp,
            )
            dist_results.append(d)
            speed_results.append(s)
            all_bouts.extend(bouts)
            segment_metrics.append(
                {
                    "trajectory_segment_id": seg_id,
                    "sample_layer": primary_layer,
                    "distance_m": d.distance_m,
                    "distance_status": d.status,
                    "distance_reason_codes": list(d.reason_codes),
                    "robust_mean_speed_mps": s.robust_mean_mps,
                    "robust_peak_speed_mps": s.robust_peak_mps,
                    "diagnostic_raw_peak_speed_mps": s.diagnostic_raw_peak_mps,
                    "speed_status": s.status,
                    "speed_reason_codes": list(s.reason_codes),
                    "sample_count": d.sample_count,
                    "measured_duration_us": d.measured_duration_us,
                    "sprint_bout_ids": [b.sprint_id for b in bouts],
                }
            )

        # Diagnostic layers (do not feed customer summary)
        diagnostic_segment_metrics: list[dict[str, Any]] = []
        for layer_name, layer_pts in (diagnostic_layers or {}).items():
            for seg_id, pts in sorted(_group_by_segment(layer_pts).items()):
                d = compute_segment_distance(
                    pts,
                    trajectory_segment_id=seg_id,
                    sample_layer=layer_name,
                    config=cfg,
                    diagnostic=True,
                )
                s = compute_segment_speeds(
                    pts,
                    trajectory_segment_id=seg_id,
                    sample_layer=layer_name,
                    config=cfg,
                    diagnostic=True,
                )
                diagnostic_segment_metrics.append(
                    {
                        "trajectory_segment_id": seg_id,
                        "sample_layer": layer_name,
                        "distance_m": d.distance_m,
                        "robust_mean_speed_mps": s.robust_mean_mps,
                        "diagnostic_raw_peak_speed_mps": s.diagnostic_raw_peak_mps,
                        "status_distance": d.status,
                        "status_speed": s.status,
                        "diagnostic": True,
                    }
                )

        if primary_points:
            t_min = min(int(p["video_time_us"]) for p in primary_points)
            t_max = max(int(p["video_time_us"]) for p in primary_points)
            run_id = str(primary_points[0]["run_id"])
            video_id = str(primary_points[0]["video_id"])
            target_id = str(primary_points[0]["target_player_id"])
        else:
            t_min, t_max = 0, 0
            run_id, video_id, target_id = "run_unknown", "video_unknown", "target_unknown"

        window_us = (
            int(analysis_window_us) if analysis_window_us is not None else max(1, t_max - t_min)
        )
        dist_agg = aggregate_measured_distance(
            dist_results,
            analysis_window_us=window_us,
            min_coverage_ratio=float(cfg["distance"]["min_coverage_ratio_for_computed"]),
        )
        speed_agg = aggregate_speed_summary(
            speed_results,
            analysis_window_us=window_us,
            min_coverage_ratio=float(cfg["speed"]["min_coverage_ratio_for_computed"]),
            min_eligible_duration_us=int(cfg["speed"]["min_eligible_duration_us"]),
        )
        sprint_stats = count_evaluable_sprints(all_bouts)
        bout_dicts = sprint_bouts_to_dicts(all_bouts)
        obs_cov = observed_coverage_from_points(primary_points)
        der_cov = 0
        if diagnostic_layers and "resampled" in diagnostic_layers:
            der_cov = observed_coverage_from_points(diagnostic_layers["resampled"])

        not_eval_reasons: list[str] = []
        not_eval_reasons.extend(dist_agg.get("reason_codes") or [])
        not_eval_reasons.extend(speed_agg.get("reason_codes") or [])
        for b in all_bouts:
            if b.evaluability != "evaluable":
                not_eval_reasons.extend(b.reason_codes)

        summary = {
            "schema_version": 1,
            "stage": "9C",
            "primary_sample_layer": primary_layer,
            "primary_layer_justification": (
                "Filtered layer is quality-gated observed pitch points with "
                "metric_eligibility=eligible; raw is pre-gate diagnostic; "
                "resampled is derived and not customer-primary."
            ),
            "measured_eligible_duration_us": int(
                dist_agg.get("measured_eligible_duration_us")
                or speed_agg.get("measured_eligible_duration_us")
                or 0
            ),
            "observed_coverage_us": obs_cov,
            "derived_coverage_us": der_cov,
            "measured_distance_m": dist_agg.get("measured_distance_m"),
            "distance_status": dist_agg.get("status"),
            "robust_mean_speed_mps": speed_agg.get("robust_mean_mps"),
            "robust_peak_speed_mps": speed_agg.get("robust_peak_mps"),
            "robust_mean_speed_kmh": (
                mps_to_kmh(float(speed_agg["robust_mean_mps"]))
                if speed_agg.get("robust_mean_mps") is not None
                else None
            ),
            "speed_status": speed_agg.get("status"),
            "sprint_count": sprint_stats["sprint_count"],
            "sprint_distance_m": sprint_stats["sprint_distance_m"],
            "sprint_duration_us": sprint_stats["sprint_duration_us"],
            "sprint_not_evaluable_count": sprint_stats["not_evaluable_count"],
            "segment_count": len(groups),
            "not_evaluable_reasons": sorted(set(str(r) for r in not_eval_reasons)),
            "coverage_ratio_distance": dist_agg.get("coverage_ratio"),
            "coverage_ratio_speed": speed_agg.get("coverage_ratio"),
            "metric_origin": cfg["metric_origin"],
            "definition_style": cfg["definition_style"],
            "config_fingerprint": cfg_fp,
            "attack_direction": cfg["attack_direction"],
            "evaluation_status": NOT_EVALUATED_MOTION,
            "no_coverage_extrapolation": True,
            "not_official_opta": True,
        }

        metric_rows = [
            _metric_row(
                run_id=run_id,
                video_id=video_id,
                target_player_id=target_id,
                metric_result_id="metric_distance_measured",
                metric_name="distance_measured",
                unit="m",
                value=dist_agg.get("value_m"),
                status=str(dist_agg.get("status")),
                sample_layer=primary_layer,
                config_fingerprint=cfg_fp,
                time_start_us=t_min,
                time_end_us=t_max,
                coverage_ratio=dist_agg.get("coverage_ratio"),
                uncertainty=None,
                included_sample_count=int(dist_agg.get("included_sample_count") or 0),
                included_duration_us=int(dist_agg.get("measured_eligible_duration_us") or 0),
                segment_ids=list(dist_agg.get("segment_ids") or []),
                reason_codes=list(dist_agg.get("reason_codes") or []),
                provenance={
                    "metric_origin": cfg["metric_origin"],
                    "semantics": "measured_eligible_distance_not_full_match",
                },
            ),
            _metric_row(
                run_id=run_id,
                video_id=video_id,
                target_player_id=target_id,
                metric_result_id="metric_speed_robust_mean",
                metric_name="speed_robust_mean",
                unit="m_s",
                value=speed_agg.get("robust_mean_mps"),
                status=str(speed_agg.get("status")),
                sample_layer=primary_layer,
                config_fingerprint=cfg_fp,
                time_start_us=t_min,
                time_end_us=t_max,
                coverage_ratio=speed_agg.get("coverage_ratio"),
                uncertainty=None,
                included_sample_count=int(speed_agg.get("included_sample_count") or 0),
                included_duration_us=int(speed_agg.get("measured_eligible_duration_us") or 0),
                segment_ids=list(speed_agg.get("segment_ids") or []),
                reason_codes=list(speed_agg.get("reason_codes") or []),
                provenance={"metric_origin": cfg["metric_origin"], "vs": "diagnostic_raw_speed"},
            ),
            _metric_row(
                run_id=run_id,
                video_id=video_id,
                target_player_id=target_id,
                metric_result_id="metric_speed_robust_peak",
                metric_name="speed_robust_peak",
                unit="m_s",
                value=speed_agg.get("robust_peak_mps"),
                status=(
                    "computed" if speed_agg.get("robust_peak_mps") is not None else "not_evaluable"
                ),
                sample_layer=primary_layer,
                config_fingerprint=cfg_fp,
                time_start_us=t_min,
                time_end_us=t_max,
                coverage_ratio=speed_agg.get("coverage_ratio"),
                uncertainty=None,
                included_sample_count=int(speed_agg.get("included_sample_count") or 0),
                included_duration_us=int(speed_agg.get("measured_eligible_duration_us") or 0),
                segment_ids=list(speed_agg.get("segment_ids") or []),
                reason_codes=list(speed_agg.get("reason_codes") or []),
                provenance={
                    "metric_origin": cfg["metric_origin"],
                    "peak_requires_min_support": True,
                },
            ),
            _metric_row(
                run_id=run_id,
                video_id=video_id,
                target_player_id=target_id,
                metric_result_id="metric_sprint_count",
                metric_name="sprint_count",
                unit="count",
                value=float(sprint_stats["sprint_count"]),
                status="computed",
                sample_layer=primary_layer,
                config_fingerprint=cfg_fp,
                time_start_us=t_min,
                time_end_us=t_max,
                coverage_ratio=None,
                uncertainty=None,
                included_sample_count=len(primary_points),
                included_duration_us=obs_cov,
                segment_ids=sorted(groups.keys()),
                reason_codes=["evaluable_bouts_only"],
                provenance={
                    "metric_origin": cfg["metric_origin"],
                    "definition_style": cfg["definition_style"],
                    "not_official_opta": True,
                },
            ),
        ]

        seg_path = out / "segment_distance_speed.json"
        write_json_record(
            seg_path,
            {
                "schema_version": 1,
                "primary": segment_metrics,
                "diagnostic": diagnostic_segment_metrics,
            },
            overwrite=False,
        )
        bout_path = out / "sprint_bouts.json"
        write_json_record(
            bout_path,
            {
                "schema_version": 1,
                "metric_origin": cfg["metric_origin"],
                "definition_style": cfg["definition_style"],
                "config_fingerprint": cfg_fp,
                "bouts": bout_dicts,
            },
            overwrite=False,
        )
        metrics_path = out / "physical_metric_results.parquet"
        write_contract_parquet(
            _cast("physical_metric_results", metric_rows),
            metrics_path,
            get_contract("physical_metric_results", 1),
            overwrite=False,
        )
        sum_path = out / "physical_motion_summary.json"
        write_json_record(sum_path, summary, overwrite=False)

        quality = build_motion_quality_report(
            primary_layer=primary_layer,
            segment_metric_count=len(segment_metrics),
            sprint_bout_count=len(all_bouts),
            evaluable_sprint_count=int(sprint_stats["sprint_count"]),
            measured_eligible_duration_us=int(summary["measured_eligible_duration_us"]),
            observed_coverage_us=obs_cov,
            derived_coverage_us=der_cov,
            measured_distance_m=summary.get("measured_distance_m"),
            robust_mean_mps=summary.get("robust_mean_speed_mps"),
            robust_peak_mps=summary.get("robust_peak_speed_mps"),
            not_evaluable_reasons=list(summary["not_evaluable_reasons"]),
            config_fingerprint=cfg_fp,
            findings=[
                "Synthetic/math pipeline; real football accuracy not claimed.",
            ],
        )
        q_path = out / "motion_quality.json"
        write_json_record(q_path, quality, overwrite=False)

        ev = evaluate_motion_metrics(metric_results=metric_rows, has_reviewed_ground_truth=False)
        e_path = out / "motion_evaluation.json"
        write_json_record(
            e_path,
            ev.to_dict(run_id=run_id, video_id=video_id, config_fingerprint=cfg_fp),
            overwrite=False,
        )

        artifact_hashes = {
            "segment_metrics": {"sha256": sha256_file(seg_path), "size": seg_path.stat().st_size},
            "sprint_bouts": {"sha256": sha256_file(bout_path), "size": bout_path.stat().st_size},
            "metric_results": {
                "sha256": sha256_file(metrics_path),
                "size": metrics_path.stat().st_size,
            },
            "summary": {"sha256": sha256_file(sum_path), "size": sum_path.stat().st_size},
            "quality": {"sha256": sha256_file(q_path), "size": q_path.stat().st_size},
            "evaluation": {"sha256": sha256_file(e_path), "size": e_path.stat().st_size},
        }
        receipt = {
            "schema_version": 1,
            "receipt_id": "motion_receipt_01",
            "request_id": "motion_req_01",
            "run_id": run_id,
            "video_id": video_id,
            "target_player_id": target_id,
            "status": "succeeded",
            "config_fingerprint": cfg_fp,
            "metrics_policy_fingerprint": cfg_fp,
            "trajectory_policy_fingerprint": cfg_fp,
            "input_fingerprints": {
                "physical_metric_results": contract_fingerprint(
                    get_contract("physical_metric_results", 1)
                ),
            },
            "output_fingerprints": {
                "physical_metric_results": contract_fingerprint(
                    get_contract("physical_metric_results", 1)
                ),
            },
            "eligible_sample_count": len(primary_points),
            "rejected_sample_count": 0,
            "segment_count": len(groups),
            "gap_count": 0,
            "metric_status_counts": {
                status: sum(1 for x in metric_rows if x["status"] == status)
                for status in sorted({str(r["status"]) for r in metric_rows})
            },
            "eligible_duration_us": obs_cov,
            "excluded_duration_us": 0,
            "coverage_summary": {
                "observed_coverage_us": obs_cov,
                "derived_coverage_us": der_cov,
                "measured_eligible_duration_us": summary["measured_eligible_duration_us"],
            },
            "outlier_count": 0,
            "review_count": 0,
            "evaluation_status": NOT_EVALUATED_MOTION,
            "artifact_hashes": artifact_hashes,
            "reason_code_distribution": {},
            "warning_codes": [],
            "error_codes": [],
            "created_at_utc": _utc_now(),
            "provenance": {
                "stage": "9C",
                "label": "distance_speed_sprint_baseline",
                "metric_origin": cfg["metric_origin"],
                "definition_style": cfg["definition_style"],
                "primary_sample_layer": primary_layer,
                "not_official_opta": True,
                "no_coverage_extrapolation": True,
            },
            "completion_status": "succeeded",
            "summary_metrics": {
                "measured_distance_m": summary.get("measured_distance_m"),
                "robust_mean_speed_mps": summary.get("robust_mean_speed_mps"),
                "robust_peak_speed_mps": summary.get("robust_peak_speed_mps"),
                "sprint_count": summary.get("sprint_count"),
            },
        }
        r_path = out / "motion_receipt.json"
        write_json_record(r_path, receipt, overwrite=False)

        return MotionServiceResult(
            accepted=True,
            exit_code=0,
            error_code=None,
            config_fingerprint=cfg_fp,
            segment_metrics_json=str(seg_path),
            sprint_bouts_json=str(bout_path),
            metric_results_parquet=str(metrics_path),
            summary_json=str(sum_path),
            receipt_json=str(r_path),
            quality_json=str(q_path),
            evaluation_json=str(e_path),
            summary=summary,
        )
    except (MotionServiceError, MotionConfigError, OSError, ValueError) as exc:
        if cleanup_on_failure and created and out.exists():
            shutil.rmtree(out, ignore_errors=True)
        return MotionServiceResult(
            accepted=False,
            exit_code=1,
            error_code=type(exc).__name__,
            config_fingerprint=cfg_fp,
            segment_metrics_json=None,
            sprint_bouts_json=None,
            metric_results_parquet=None,
            summary_json=None,
            receipt_json=None,
            quality_json=None,
            evaluation_json=None,
            summary={"error": str(exc)},
        )


__all__ = [
    "MotionServiceError",
    "MotionServiceResult",
    "compute_physical_motion",
]
