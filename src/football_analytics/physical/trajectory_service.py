"""Stage 9B target trajectory preparation service."""

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
from football_analytics.physical.trajectory_config import (
    TrajectoryConfigError,
    load_trajectory_baseline_config,
    trajectory_baseline_config_fingerprint,
)
from football_analytics.physical.trajectory_evaluation import (
    NOT_EVALUATED_TRAJECTORY,
    evaluate_trajectory_preparation,
)
from football_analytics.physical.trajectory_filter import filter_trajectory_points
from football_analytics.physical.trajectory_quality import (
    build_trajectory_quality_report,
    observed_coverage_us,
)
from football_analytics.physical.trajectory_resample import resample_all_segments
from football_analytics.physical.trajectory_segments import split_trajectory_segments
from football_analytics.physical.types import CONTRACT_VERSION


class TrajectoryServiceError(RuntimeError):
    """Trajectory preparation failure."""


@dataclass
class TrajectoryServiceResult:
    accepted: bool
    exit_code: int
    error_code: str | None
    config_fingerprint: str
    raw_parquet: str | None
    filtered_parquet: str | None
    resampled_parquet: str | None
    gaps_parquet: str | None
    segments_parquet: str | None
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
            "raw_parquet": self.raw_parquet,
            "filtered_parquet": self.filtered_parquet,
            "resampled_parquet": self.resampled_parquet,
            "gaps_parquet": self.gaps_parquet,
            "segments_parquet": self.segments_parquet,
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


def select_eligible_raw(
    candidates: Sequence[Mapping[str, Any]], *, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep only confirmed/observed/mapped/eligible points as immutable raw_observed."""
    elig = config["input_eligibility"]
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for c in candidates:
        reasons: list[str] = []
        if str(c.get("identity_quality", c.get("identity_status", ""))) != "confirmed":
            reasons.append("not_confirmed")
        if c.get("assignment_revoked") is True:
            reasons.append("revoked")
        if str(c.get("observation_source", "detection_associated")) in {
            "predicted",
            "interpolated",
        }:
            reasons.append("predicted_or_interpolated")
        if str(c.get("mapping_status")) != str(elig["require_mapping_status"]):
            reasons.append("not_mapped")
        phys = str(
            c.get("physical_metric_eligibility", c.get("metric_eligibility", "not_eligible"))
        )
        if phys != "eligible":
            reasons.append("not_physical_eligible")
        if c.get("calibration_invalid") is True:
            reasons.append("invalid_calibration")
        if c.get("non_playable") is True and elig.get("require_playable_non_replay"):
            reasons.append("non_playable")
        if reasons:
            rejected.append({"sample_id": c.get("sample_id"), "reasons": reasons})
            continue
        row = dict(c)
        row["sample_source"] = "raw_observed"
        row["derived_from_sample_ids"] = []
        row["eligibility_status"] = "eligible"
        row["metric_eligibility"] = "eligible"
        row["contract_version"] = CONTRACT_VERSION
        # Strip service-only hints from contract row later via field projection
        accepted.append(row)
    return accepted, rejected


def _contract_sample_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Project to target_trajectory_samples fields only."""
    spec = get_contract("target_trajectory_samples", 1)
    names = [f.name for f in spec.fields]
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append({n: r.get(n) for n in names})
    return out


def prepare_target_trajectory(
    *,
    candidates: Sequence[Mapping[str, Any]],
    output_dir: Path,
    config: Mapping[str, Any] | None = None,
    config_path: Path | None = None,
    cleanup_on_failure: bool = True,
) -> TrajectoryServiceResult:
    """Prepare raw/filtered/resampled trajectories + gaps/segments/receipt/quality."""
    cfg = config or load_trajectory_baseline_config(config_path)
    cfg_fp = trajectory_baseline_config_fingerprint(cfg)
    out = Path(output_dir)
    created = False
    try:
        if out.exists() and any(out.iterdir()) and cfg.get("overwrite_allowed") is not True:
            raise TrajectoryServiceError("output_dir not empty and overwrite forbidden")
        out.mkdir(parents=True, exist_ok=True)
        created = True

        raw_rows, input_rejects = select_eligible_raw(candidates, config=cfg)
        # Immutable raw layer
        raw_contract = _contract_sample_rows(raw_rows)

        filtered_rows, filter_rejections, reason_counts = filter_trajectory_points(
            raw_rows, config=cfg
        )
        for _r in input_rejects:
            reason_counts["input_ineligible"] = reason_counts.get("input_ineligible", 0) + 1

        segments, gaps, filtered_tagged = split_trajectory_segments(
            filtered_rows, config=cfg, policy_fingerprint=cfg_fp
        )
        filtered_rows = filtered_tagged
        filtered_contract = _contract_sample_rows(filtered_rows)

        resampled_rows = resample_all_segments(
            filtered_rows, segments, config=cfg, policy_fingerprint=cfg_fp
        )
        resampled_contract = _contract_sample_rows(resampled_rows)

        # Write artifacts
        paths: dict[str, Path] = {}
        if cfg["output_policy"].get("write_raw", True):
            paths["raw"] = out / "raw_trajectory_samples.parquet"
            write_contract_parquet(
                _cast("target_trajectory_samples", raw_contract),
                paths["raw"],
                get_contract("target_trajectory_samples", 1),
                overwrite=False,
            )
        if cfg["output_policy"].get("write_filtered", True):
            paths["filtered"] = out / "filtered_trajectory_samples.parquet"
            write_contract_parquet(
                _cast("target_trajectory_samples", filtered_contract),
                paths["filtered"],
                get_contract("target_trajectory_samples", 1),
                overwrite=False,
            )
        if cfg["output_policy"].get("write_resampled", True):
            paths["resampled"] = out / "resampled_trajectory_samples.parquet"
            write_contract_parquet(
                _cast("target_trajectory_samples", resampled_contract),
                paths["resampled"],
                get_contract("target_trajectory_samples", 1),
                overwrite=False,
            )
        if cfg["output_policy"].get("write_gaps", True):
            paths["gaps"] = out / "trajectory_gaps.parquet"
            write_contract_parquet(
                _cast("trajectory_gaps", gaps),
                paths["gaps"],
                get_contract("trajectory_gaps", 1),
                overwrite=False,
            )
        if cfg["output_policy"].get("write_segments", True):
            paths["segments"] = out / "target_trajectory_segments.parquet"
            write_contract_parquet(
                _cast("target_trajectory_segments", segments),
                paths["segments"],
                get_contract("target_trajectory_segments", 1),
                overwrite=False,
            )

        # Rejection sidecar (JSONL) — not a silent drop
        rej_path = out / "trajectory_rejections.jsonl"
        with rej_path.open("w", encoding="utf-8") as fh:
            for item in filter_rejections:
                fh.write(json.dumps(item, sort_keys=True) + "\n")
            for item in input_rejects:
                fh.write(
                    json.dumps({"reason_code": "input_ineligible", **item}, sort_keys=True) + "\n"
                )

        single_pts = sum(1 for s in segments if int(s.get("eligible_sample_count", 0)) < 2)
        obs_cov = observed_coverage_us(filtered_rows)
        der_cov = observed_coverage_us(resampled_rows)
        quality = build_trajectory_quality_report(
            raw_count=len(raw_contract),
            filtered_count=len(filtered_contract),
            resampled_count=len(resampled_contract),
            rejected_count=len(filter_rejections) + len(input_rejects),
            segment_count=len(segments),
            gap_count=len(gaps),
            reason_counts=reason_counts,
            single_point_segments=single_pts,
            observed_coverage_us=obs_cov,
            derived_coverage_us=der_cov,
            config_fingerprint=cfg_fp,
        )
        q_path = out / "trajectory_quality.json"
        write_json_record(q_path, quality, overwrite=False)

        run_id = str(raw_rows[0]["run_id"]) if raw_rows else "run_unknown"
        video_id = str(raw_rows[0]["video_id"]) if raw_rows else "video_unknown"
        target_id = str(raw_rows[0]["target_player_id"]) if raw_rows else "target_unknown"
        ev = evaluate_trajectory_preparation(
            raw=raw_contract,
            filtered=filtered_contract,
            resampled=resampled_contract,
            has_reviewed_ground_truth=False,
        )
        e_path = out / "trajectory_evaluation.json"
        write_json_record(
            e_path,
            ev.to_dict(run_id=run_id, video_id=video_id, config_fingerprint=cfg_fp),
            overwrite=False,
        )

        artifact_hashes = {
            name: {"sha256": sha256_file(path), "size": path.stat().st_size}
            for name, path in paths.items()
        }
        artifact_hashes["rejections"] = {
            "sha256": sha256_file(rej_path),
            "size": rej_path.stat().st_size,
        }
        artifact_hashes["quality"] = {
            "sha256": sha256_file(q_path),
            "size": q_path.stat().st_size,
        }
        artifact_hashes["evaluation"] = {
            "sha256": sha256_file(e_path),
            "size": e_path.stat().st_size,
        }

        receipt = {
            "schema_version": 1,
            "receipt_id": "traj_receipt_01",
            "request_id": "traj_req_01",
            "run_id": run_id,
            "video_id": video_id,
            "target_player_id": target_id,
            "status": "succeeded",
            "trajectory_policy_fingerprint": cfg_fp,
            "metrics_policy_fingerprint": cfg_fp,
            "config_fingerprint": cfg_fp,
            "input_fingerprints": {
                "target_trajectory_samples": contract_fingerprint(
                    get_contract("target_trajectory_samples", 1)
                ),
            },
            "output_fingerprints": {
                "target_trajectory_samples": contract_fingerprint(
                    get_contract("target_trajectory_samples", 1)
                ),
                "target_trajectory_segments": contract_fingerprint(
                    get_contract("target_trajectory_segments", 1)
                ),
                "trajectory_gaps": contract_fingerprint(get_contract("trajectory_gaps", 1)),
            },
            "eligible_sample_count": len(filtered_contract),
            "rejected_sample_count": len(filter_rejections) + len(input_rejects),
            "segment_count": len(segments),
            "gap_count": len(gaps),
            "raw_sample_count": len(raw_contract),
            "filtered_sample_count": len(filtered_contract),
            "resampled_sample_count": len(resampled_contract),
            "metric_status_counts": {"contract_stub": 0},
            "eligible_duration_us": obs_cov,
            "excluded_duration_us": 0,
            "coverage_summary": {
                "observed_coverage_us": obs_cov,
                "derived_coverage_us": der_cov,
            },
            "outlier_count": len(filter_rejections),
            "review_count": 0,
            "evaluation_status": NOT_EVALUATED_TRAJECTORY,
            "artifact_hashes": artifact_hashes,
            "reason_code_distribution": dict(reason_counts),
            "warning_codes": list(quality.get("findings") or []),
            "error_codes": [],
            "created_at_utc": _utc_now(),
            "provenance": {
                "stage": "9B",
                "label": "target_trajectory_preparation",
                "notes": "no_customer_physical_metrics",
                "no_real_metric_computation": True,
            },
            "completion_status": "succeeded",
        }
        r_path = out / "trajectory_receipt.json"
        write_json_record(r_path, receipt, overwrite=False)

        summary = {
            "raw_count": len(raw_contract),
            "filtered_count": len(filtered_contract),
            "resampled_count": len(resampled_contract),
            "rejected_count": len(filter_rejections) + len(input_rejects),
            "segment_count": len(segments),
            "gap_count": len(gaps),
            "evaluation_status": NOT_EVALUATED_TRAJECTORY,
            "customer_metrics_computed": False,
        }
        return TrajectoryServiceResult(
            accepted=True,
            exit_code=0,
            error_code=None,
            config_fingerprint=cfg_fp,
            raw_parquet=str(paths.get("raw")) if "raw" in paths else None,
            filtered_parquet=str(paths.get("filtered")) if "filtered" in paths else None,
            resampled_parquet=str(paths.get("resampled")) if "resampled" in paths else None,
            gaps_parquet=str(paths.get("gaps")) if "gaps" in paths else None,
            segments_parquet=str(paths.get("segments")) if "segments" in paths else None,
            receipt_json=str(r_path),
            quality_json=str(q_path),
            evaluation_json=str(e_path),
            summary=summary,
        )
    except (TrajectoryServiceError, TrajectoryConfigError, OSError, ValueError) as exc:
        if cleanup_on_failure and created and out.exists():
            shutil.rmtree(out, ignore_errors=True)
        return TrajectoryServiceResult(
            accepted=False,
            exit_code=1,
            error_code=type(exc).__name__,
            config_fingerprint=cfg_fp,
            raw_parquet=None,
            filtered_parquet=None,
            resampled_parquet=None,
            gaps_parquet=None,
            segments_parquet=None,
            receipt_json=None,
            quality_json=None,
            evaluation_json=None,
            summary={"error": str(exc)},
        )


__all__ = [
    "TrajectoryServiceError",
    "TrajectoryServiceResult",
    "select_eligible_raw",
    "prepare_target_trajectory",
]
