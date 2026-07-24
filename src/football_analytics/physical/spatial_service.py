"""Stage 9D heatmap / zones / activity computation service."""

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
from football_analytics.physical.activity import compute_activity_distribution
from football_analytics.physical.heatmap import (
    compute_time_weighted_heatmap,
    heatmap_to_dict,
)
from football_analytics.physical.spatial_config import (
    SpatialConfigError,
    load_spatial_baseline_config,
    spatial_baseline_config_fingerprint,
)
from football_analytics.physical.spatial_evaluation import (
    NOT_EVALUATED_SPATIAL,
    evaluate_spatial_metrics,
)
from football_analytics.physical.spatial_quality import (
    build_spatial_quality_report,
    observed_coverage_us,
)
from football_analytics.physical.types import CONTRACT_VERSION
from football_analytics.physical.zone_occupancy import compute_zone_occupancy


class SpatialServiceError(RuntimeError):
    """Spatial / activity metric failure."""


@dataclass
class SpatialServiceResult:
    accepted: bool
    exit_code: int
    error_code: str | None
    config_fingerprint: str
    heatmap_json: str | None
    zones_json: str | None
    activity_json: str | None
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
            "heatmap_json": self.heatmap_json,
            "zones_json": self.zones_json,
            "activity_json": self.activity_json,
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
    included_sample_count: int,
    included_duration_us: int,
    reason_codes: Sequence[str],
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
        "uncertainty": None,
        "included_sample_count": int(included_sample_count),
        "excluded_sample_count": 0,
        "included_duration_us": int(included_duration_us),
        "excluded_duration_us": 0,
        "sample_layer": sample_layer,
        "trajectory_segment_ids": [],
        "evidence_ids": [],
        "config_fingerprint": config_fingerprint,
        "trajectory_artifact_fingerprint": None,
        "calibration_artifact_fingerprint": None,
        "identity_artifact_fingerprint": None,
        "warning_codes": [],
        "reason_codes": list(reason_codes),
        "review_status": "not_required",
        "producer": "heatmap_activity_baseline",
        "producer_version": "1",
        "provenance_json": json.dumps(dict(provenance or {}), sort_keys=True),
        "contract_version": CONTRACT_VERSION,
    }


def compute_spatial_metrics(
    *,
    primary_points: Sequence[Mapping[str, Any]],
    output_dir: Path,
    config: Mapping[str, Any] | None = None,
    config_path: Path | None = None,
    analysis_window_us: int | None = None,
    gap_unobserved_us: int = 0,
    cleanup_on_failure: bool = True,
    temp_visual_dir: Path | None = None,
) -> SpatialServiceResult:
    """Compute time-weighted heatmap, zone occupancy, and activity distribution."""
    cfg = config or load_spatial_baseline_config(config_path)
    cfg_fp = spatial_baseline_config_fingerprint(cfg)
    out = Path(output_dir)
    created = False
    primary_layer = str(cfg["primary_sample_layer"])
    try:
        if out.exists() and any(out.iterdir()) and cfg.get("overwrite_allowed") is not True:
            raise SpatialServiceError("output_dir not empty and overwrite forbidden")
        out.mkdir(parents=True, exist_ok=True)
        created = True

        if primary_points:
            t_min = min(int(p["video_time_us"]) for p in primary_points)
            t_max = max(int(p["video_time_us"]) for p in primary_points)
            run_id = str(primary_points[0]["run_id"])
            video_id = str(primary_points[0]["video_id"])
            target_id = str(primary_points[0]["target_player_id"])
        else:
            t_min, t_max = 0, 0
            run_id, video_id, target_id = "run_unknown", "video_unknown", "target_unknown"

        window = (
            int(analysis_window_us) if analysis_window_us is not None else max(1, t_max - t_min)
        )
        hm = compute_time_weighted_heatmap(
            primary_points,
            config=cfg,
            contribution_source="observed",
            analysis_window_us=window,
        )
        zones = compute_zone_occupancy(primary_points, config=cfg, analysis_window_us=window)
        activity = compute_activity_distribution(
            primary_points,
            config=cfg,
            analysis_window_us=window,
            gap_unobserved_us=gap_unobserved_us,
        )
        hm_dict = heatmap_to_dict(hm, config_fingerprint=cfg_fp)
        zones["config_fingerprint"] = cfg_fp
        activity["config_fingerprint"] = cfg_fp

        obs_cov = observed_coverage_us(primary_points)
        mass_ok = abs(hm.mass_before_smooth - hm.mass_after_smooth) < 1e-6 or (
            hm.mass_before_smooth == 0 and hm.mass_after_smooth == 0
        )
        # After smooth with out-of-bound drop + renormalize, mass should match
        mass_ok = abs(hm.mass_before_smooth - hm.mass_after_smooth) < 1e-6

        summary = {
            "schema_version": 1,
            "stage": "9D",
            "primary_sample_layer": primary_layer,
            "attack_direction": "unknown",
            "heatmap_status": hm.status,
            "zone_status": zones["status"],
            "activity_status": activity["status"],
            "total_dwell_seconds": hm.total_dwell_seconds,
            "heatmap_percent_sum": hm_dict["percent_sum"],
            "eligible_observed_duration_us": activity["eligible_observed_duration_us"],
            "gap_or_not_observed_duration_us": activity["gap_or_not_observed_duration_us"],
            "missing_coverage_counted_as_inactive": False,
            "movement_activity_index": activity["movement_activity_index"]["value"],
            "movement_activity_index_status": activity["movement_activity_index"]["status"],
            "observed_coverage_us": obs_cov,
            "derived_coverage_us": 0,
            "coverage_ratio": hm.coverage_ratio,
            "metric_origin": cfg["metric_origin"],
            "definition_style": cfg["definition_style"],
            "config_fingerprint": cfg_fp,
            "evaluation_status": NOT_EVALUATED_SPATIAL,
            "not_official_opta": True,
            "penalty_is_not_touch": True,
            "visuals_committed_to_git": False,
        }

        hm_path = out / "heatmap_grid.json"
        write_json_record(hm_path, hm_dict, overwrite=False)
        z_path = out / "zone_occupancy.json"
        write_json_record(z_path, zones, overwrite=False)
        a_path = out / "activity_distribution.json"
        write_json_record(a_path, activity, overwrite=False)
        sum_path = out / "spatial_summary.json"
        write_json_record(sum_path, summary, overwrite=False)

        metric_rows = [
            _metric_row(
                run_id=run_id,
                video_id=video_id,
                target_player_id=target_id,
                metric_result_id="metric_heatmap_total_dwell",
                metric_name="heatmap_total_dwell",
                unit="s",
                value=hm.total_dwell_seconds if hm.status == "computed" else None,
                status=hm.status,
                sample_layer=primary_layer,
                config_fingerprint=cfg_fp,
                time_start_us=t_min,
                time_end_us=t_max,
                coverage_ratio=hm.coverage_ratio,
                included_sample_count=hm.sample_count,
                included_duration_us=hm.eligible_duration_us,
                reason_codes=hm.reason_codes,
                provenance={"weighting": "time_weighted", "not_frame_count": True},
            ),
            _metric_row(
                run_id=run_id,
                video_id=video_id,
                target_player_id=target_id,
                metric_result_id="metric_movement_activity_index",
                metric_name="movement_activity_index",
                unit="ratio",
                value=activity["movement_activity_index"]["value"],
                status=activity["movement_activity_index"]["status"],
                sample_layer=primary_layer,
                config_fingerprint=cfg_fp,
                time_start_us=t_min,
                time_end_us=t_max,
                coverage_ratio=activity["coverage_ratio"],
                included_sample_count=len(primary_points),
                included_duration_us=activity["eligible_observed_duration_us"],
                reason_codes=activity["reason_codes"],
                provenance={
                    "metric_origin": "project_generated",
                    "not_official_opta": True,
                    "not_possession_or_tactical": True,
                },
            ),
        ]
        metrics_path = out / "physical_metric_results.parquet"
        write_contract_parquet(
            _cast("physical_metric_results", metric_rows),
            metrics_path,
            get_contract("physical_metric_results", 1),
            overwrite=False,
        )

        quality = build_spatial_quality_report(
            heatmap_status=hm.status,
            zone_status=str(zones["status"]),
            activity_status=str(activity["status"]),
            coverage_ratio=float(hm.coverage_ratio),
            observed_coverage_us=obs_cov,
            derived_coverage_us=0,
            percent_sum=float(hm_dict["percent_sum"]),
            mass_conservation_ok=mass_ok,
            config_fingerprint=cfg_fp,
            findings=[
                "Synthetic/math pipeline; real football accuracy not claimed.",
                "No SVG/PNG written to git evidence.",
            ],
        )
        q_path = out / "spatial_quality.json"
        write_json_record(q_path, quality, overwrite=False)

        ev = evaluate_spatial_metrics(has_reviewed_ground_truth=False)
        e_path = out / "spatial_evaluation.json"
        write_json_record(
            e_path,
            ev.to_dict(run_id=run_id, video_id=video_id, config_fingerprint=cfg_fp),
            overwrite=False,
        )

        # Optional temporary visual for validator (workspace only; never git evidence)
        visual_meta: dict[str, Any] | None = None
        if temp_visual_dir is not None:
            temp_visual_dir = Path(temp_visual_dir)
            temp_visual_dir.mkdir(parents=True, exist_ok=True)
            svg = _render_temp_heatmap_svg(hm_dict)
            svg_path = temp_visual_dir / "temp_heatmap.svg"
            svg_path.write_text(svg, encoding="utf-8")
            visual_meta = {
                "path": str(svg_path),
                "sha256": sha256_file(svg_path),
                "size_bytes": svg_path.stat().st_size,
                "git_tracked": False,
                "ephemeral": True,
            }

        artifact_hashes = {
            "heatmap": {"sha256": sha256_file(hm_path), "size": hm_path.stat().st_size},
            "zones": {"sha256": sha256_file(z_path), "size": z_path.stat().st_size},
            "activity": {"sha256": sha256_file(a_path), "size": a_path.stat().st_size},
            "summary": {"sha256": sha256_file(sum_path), "size": sum_path.stat().st_size},
            "metric_results": {
                "sha256": sha256_file(metrics_path),
                "size": metrics_path.stat().st_size,
            },
            "quality": {"sha256": sha256_file(q_path), "size": q_path.stat().st_size},
            "evaluation": {"sha256": sha256_file(e_path), "size": e_path.stat().st_size},
        }
        if visual_meta:
            artifact_hashes["temp_visual"] = visual_meta

        receipt = {
            "schema_version": 1,
            "receipt_id": "spatial_receipt_01",
            "request_id": "spatial_req_01",
            "run_id": run_id,
            "video_id": video_id,
            "target_player_id": target_id,
            "status": "succeeded",
            "config_fingerprint": cfg_fp,
            "metrics_policy_fingerprint": cfg_fp,
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
            "segment_count": len({str(p.get("trajectory_segment_id")) for p in primary_points}),
            "gap_count": 0,
            "metric_status_counts": {
                s: sum(1 for r in metric_rows if r["status"] == s)
                for s in sorted({str(r["status"]) for r in metric_rows})
            },
            "eligible_duration_us": obs_cov,
            "excluded_duration_us": int(gap_unobserved_us),
            "coverage_summary": {
                "observed_coverage_us": obs_cov,
                "derived_coverage_us": 0,
                "missing_coverage_counted_as_inactive": False,
            },
            "outlier_count": 0,
            "review_count": 0,
            "evaluation_status": NOT_EVALUATED_SPATIAL,
            "artifact_hashes": artifact_hashes,
            "reason_code_distribution": {},
            "warning_codes": [],
            "error_codes": [],
            "created_at_utc": _utc_now(),
            "provenance": {
                "stage": "9D",
                "label": "heatmap_zones_activity_baseline",
                "metric_origin": cfg["metric_origin"],
                "attack_direction": "unknown",
                "not_official_opta": True,
                "visuals_to_git": False,
            },
            "completion_status": "succeeded",
        }
        r_path = out / "spatial_receipt.json"
        write_json_record(r_path, receipt, overwrite=False)

        return SpatialServiceResult(
            accepted=True,
            exit_code=0,
            error_code=None,
            config_fingerprint=cfg_fp,
            heatmap_json=str(hm_path),
            zones_json=str(z_path),
            activity_json=str(a_path),
            metric_results_parquet=str(metrics_path),
            summary_json=str(sum_path),
            receipt_json=str(r_path),
            quality_json=str(q_path),
            evaluation_json=str(e_path),
            summary=summary,
        )
    except (SpatialServiceError, SpatialConfigError, OSError, ValueError) as exc:
        if cleanup_on_failure and created and out.exists():
            shutil.rmtree(out, ignore_errors=True)
        return SpatialServiceResult(
            accepted=False,
            exit_code=1,
            error_code=type(exc).__name__,
            config_fingerprint=cfg_fp,
            heatmap_json=None,
            zones_json=None,
            activity_json=None,
            metric_results_parquet=None,
            summary_json=None,
            receipt_json=None,
            quality_json=None,
            evaluation_json=None,
            summary={"error": str(exc)},
        )


def _render_temp_heatmap_svg(hm: Mapping[str, Any]) -> str:
    """Ephemeral workspace visual — not for git evidence."""
    grid = hm.get("dwell_seconds") or []
    n_y = len(grid)
    n_x = len(grid[0]) if n_y else 0
    mx = max((max(row) if row else 0) for row in grid) if grid else 1.0
    mx = mx or 1.0
    cell = 8
    w, h = n_x * cell + 80, n_y * cell + 60
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        '<rect width="100%" height="100%" fill="#f7f7f5"/>',
        '<text x="8" y="18" font-size="12">Temp time-weighted heatmap (workspace only)</text>',
    ]
    for iy, row in enumerate(grid):
        for ix, v in enumerate(row):
            t = v / mx
            c = int(255 * (1 - t))
            parts.append(
                f'<rect x="{40 + ix * cell}" y="{30 + iy * cell}" width="{cell}" '
                f'height="{cell}" fill="rgb({c},{c},255)" />'
            )
    parts.append(
        '<text x="8" y="'
        + str(h - 8)
        + '" font-size="10" fill="#444">coverage note; not final customer visual</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


__all__ = [
    "SpatialServiceError",
    "SpatialServiceResult",
    "compute_spatial_metrics",
]
