"""Stage 8C homography solve + calibration segment service (no projected_positions)."""

from __future__ import annotations

import math
import shutil
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa

from football_analytics.calibration.correspondence import build_correspondences_from_features
from football_analytics.calibration.homography_config import homography_config_fingerprint
from football_analytics.calibration.homography_evaluation import (
    NOT_EVALUATED_HOMOGRAPHY,
    evaluate_homography,
)
from football_analytics.calibration.homography_segments import (
    FrameCalibrationCandidate,
    build_calibration_segments,
    pitch_test_points,
)
from football_analytics.calibration.homography_solve import (
    HomographyQuality,
    calibration_row_from_solution,
    solve_frame_homography,
)
from football_analytics.calibration.pitch_template import (
    build_pitch_template,
    pitch_template_fingerprint,
)
from football_analytics.calibration.types import CONTRACT_VERSION
from football_analytics.core.hashing import sha256_file
from football_analytics.core.records import write_json_record
from football_analytics.core.run_id import generate_run_id, validate_run_id
from football_analytics.data.compiler import compile_arrow_schema, get_contract
from football_analytics.data.parquet import read_contract_parquet, write_contract_parquet
from football_analytics.data.types import SAFE_ID_RE
from football_analytics.video.validation import (
    reject_unsafe_path_string,
    require_absolute_path,
)


class HomographyServiceError(RuntimeError):
    """Homography service failure."""


@dataclass
class HomographyServiceResult:
    accepted: bool
    exit_code: int
    error_code: str | None
    config_fingerprint: str
    calibrations_parquet: str | None
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
            "calibrations_parquet": self.calibrations_parquet,
            "segments_parquet": self.segments_parquet,
            "receipt_json": self.receipt_json,
            "quality_json": self.quality_json,
            "evaluation_json": self.evaluation_json,
            **dict(self.summary),
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _fail(
    *,
    error_code: str,
    exit_code: int,
    config_fingerprint: str,
    cleanup: Sequence[Path] | None = None,
) -> HomographyServiceResult:
    if cleanup:
        for p in cleanup:
            try:
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
            except OSError:
                pass
    return HomographyServiceResult(
        accepted=False,
        exit_code=exit_code,
        error_code=error_code,
        config_fingerprint=config_fingerprint,
        calibrations_parquet=None,
        segments_parquet=None,
        receipt_json=None,
        quality_json=None,
        evaluation_json=None,
        summary={"status": "failed", "error_code": error_code},
    )


def _rows_to_table(rows: list[dict[str, Any]], contract_name: str) -> Any:
    contract = get_contract(contract_name, 1)
    schema = compile_arrow_schema(contract)
    if not rows:
        return schema.empty_table()
    return pa.Table.from_pylist(rows, schema=schema)


def _load_feature_rows(
    path: Path | None, in_memory: Sequence[Mapping[str, Any]] | None
) -> list[dict[str, Any]]:
    if in_memory is not None:
        return [dict(r) for r in in_memory]
    if path is None:
        raise HomographyServiceError("FEATURES_REQUIRED")
    table = read_contract_parquet(path, get_contract("calibration_features", 1))
    return list(table.to_pylist())


def _group_by_frame(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    out: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (str(r["run_id"]), str(r["video_id"]), int(r["frame_index"]))
        out[key].append(dict(r))
    return out


def recount_calibration_quality(calibrations: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {
        "total": len(calibrations),
        "valid": 0,
        "degraded": 0,
        "uncertain": 0,
        "invalid": 0,
        "not_available": 0,
        "is_valid_true": 0,
    }
    for row in calibrations:
        if row.get("is_valid"):
            counts["is_valid_true"] += 1
        flags = [str(x) for x in (row.get("quality_flags") or [])]
        q = "invalid"
        for f in flags:
            if f.startswith("quality:"):
                q = f.split(":", 1)[1]
                break
        if q in counts:
            counts[q] += 1
        else:
            counts["invalid"] += 1
    return counts


def run_homography_solve(
    *,
    output_dir: str | Path,
    config: Mapping[str, Any],
    contain_root: Path | str | None = None,
    run_id: str | None = None,
    video_id: str | None = None,
    features_path: str | Path | None = None,
    features_rows: Sequence[Mapping[str, Any]] | None = None,
    timeline_path: str | Path | None = None,
    analysis_windows_path: str | Path | None = None,
    shot_cuts_us: Sequence[int] | None = None,
    build_segments: bool = True,
    image_width: float | None = None,
    image_height: float | None = None,
    correspondence_mode: str = "hybrid",
) -> HomographyServiceResult:
    """Solve frame calibrations (+ optional segments) from calibration_features."""
    cfg_fp = homography_config_fingerprint(config)
    if config.get("auto_project_positions") is True:
        return _fail(
            error_code="PROJECTED_POSITIONS_FORBIDDEN", exit_code=2, config_fingerprint=cfg_fp
        )
    if config.get("output_policy", {}).get("write_projected_positions") is True:
        return _fail(
            error_code="PROJECTED_POSITIONS_FORBIDDEN", exit_code=2, config_fingerprint=cfg_fp
        )
    if str(config.get("attack_direction", "unknown")) != "unknown":
        return _fail(
            error_code="ATTACK_DIRECTION_FORBIDDEN", exit_code=2, config_fingerprint=cfg_fp
        )

    try:
        reject_unsafe_path_string(str(output_dir), label="output_dir")
        out = require_absolute_path(str(output_dir), label="output_dir")
    except Exception as exc:  # noqa: BLE001
        return _fail(
            error_code=f"OUTPUT_PATH:{type(exc).__name__}",
            exit_code=2,
            config_fingerprint=cfg_fp,
        )

    root = Path(contain_root) if contain_root is not None else Path(str(config["runtime_root"]))
    try:
        from football_analytics.utils.archive_safety import (
            assert_contained,
            assert_not_dangerous_operation_root,
            resolve_strict,
        )

        root = require_absolute_path(str(root), label="contain_root")
        assert_not_dangerous_operation_root(root)
        resolved_out = resolve_strict(out) if out.exists() else out.resolve()
        assert_contained(resolved_out, resolve_strict(root), label="output_root")
    except Exception as exc:  # noqa: BLE001
        return _fail(
            error_code=f"CONTAINMENT:{type(exc).__name__}",
            exit_code=2,
            config_fingerprint=cfg_fp,
        )

    out.mkdir(parents=True, exist_ok=True, mode=0o700)

    cal_out = out / "calibrations.parquet"
    seg_out = out / "calibration_segments.parquet"
    receipt_out = out / "homography_receipt.json"
    quality_out = out / "homography_quality.json"
    eval_out = out / "homography_evaluation.json"
    artifacts = [cal_out, seg_out, receipt_out, quality_out, eval_out]
    if any(p.exists() for p in artifacts) and not config.get("overwrite_allowed", False):
        return _fail(error_code="NO_OVERWRITE", exit_code=1, config_fingerprint=cfg_fp)

    try:
        features = _load_feature_rows(Path(features_path) if features_path else None, features_rows)
    except Exception as exc:  # noqa: BLE001
        return _fail(
            error_code=f"FEATURES_LOAD:{type(exc).__name__}",
            exit_code=1,
            config_fingerprint=cfg_fp,
            cleanup=[out],
        )
    if not features:
        return _fail(
            error_code="EMPTY_FEATURES",
            exit_code=1,
            config_fingerprint=cfg_fp,
            cleanup=[out],
        )

    rid = run_id or str(features[0]["run_id"])
    vid = video_id or str(features[0]["video_id"])
    try:
        validate_run_id(rid)
    except Exception:
        rid = generate_run_id()
    if not SAFE_ID_RE.match(vid):
        return _fail(
            error_code="INVALID_VIDEO_ID", exit_code=2, config_fingerprint=cfg_fp, cleanup=[out]
        )

    # Optional windows/timeline — used only for eligibility/cuts when present.
    _ = (timeline_path, analysis_windows_path)

    template = build_pitch_template(
        length_m=float(config["pitch"]["length_m"]),
        width_m=float(config["pitch"]["width_m"]),
        real_size_known=bool(config["pitch"]["real_size_known"]),
    )
    t_fp = pitch_template_fingerprint(template)
    grouped = _group_by_frame(features)

    calibrations: list[dict[str, Any]] = []
    candidates: list[FrameCalibrationCandidate] = []
    corr_accepted = 0
    corr_rejected = 0
    quality_counts = {
        "solved": 0,
        "valid": 0,
        "degraded": 0,
        "uncertain": 0,
        "invalid": 0,
        "not_available": 0,
        "no_solution": 0,
    }
    reprojs: list[float] = []
    conditions: list[float] = []
    coverages: list[float] = []
    inlier_ratios: list[float] = []
    cal_id = 0

    for (_r, _v, frame_index), feats in sorted(grouped.items(), key=lambda kv: kv[0][2]):
        video_time_us = int(feats[0].get("video_time_us", 0))
        built = build_correspondences_from_features(
            feats,
            template=template,
            config=config,
            image_width=image_width,
            image_height=image_height,
            mode=correspondence_mode,
        )
        corr_accepted += built.stats["accepted"]
        corr_rejected += built.stats["rejected"]
        sol = solve_frame_homography(
            built.accepted,
            config=config,
            image_width=image_width,
            image_height=image_height,
            pitch_length_m=template.length_m,
            pitch_width_m=template.width_m,
        )
        q = sol.quality.value
        quality_counts[q] = quality_counts.get(q, 0) + 1
        if sol.status == "solved":
            quality_counts["solved"] += 1
        else:
            quality_counts["no_solution"] += 1
        if sol.mean_reprojection_error_px is not None:
            reprojs.append(float(sol.mean_reprojection_error_px))
        if sol.condition_number is not None and math.isfinite(sol.condition_number):
            conditions.append(float(sol.condition_number))
        if sol.coverage_hull_fraction is not None:
            coverages.append(float(sol.coverage_hull_fraction))
        if sol.inlier_ratio is not None:
            inlier_ratios.append(float(sol.inlier_ratio))

        # Only persist a calibrations row when a matrix exists (valid/degraded/uncertain),
        # or an explicit invalid marker without H (is_valid=false).
        if sol.H is not None or sol.quality in {
            HomographyQuality.INVALID,
            HomographyQuality.NOT_AVAILABLE,
        }:
            row = calibration_row_from_solution(
                run_id=rid,
                video_id=vid,
                frame_index=frame_index,
                calibration_id=cal_id,
                solution=sol,
                pitch_length_m=template.length_m,
                pitch_width_m=template.width_m,
            )
            # Contract: valid requires H — keep is_valid false when H missing.
            if row["homography_image_to_pitch"] is None:
                row["is_valid"] = False
            calibrations.append(row)
            h_rm = sol.matrix_row_major()
            hinv_rm = sol.inverse_row_major()
            candidates.append(
                FrameCalibrationCandidate(
                    frame_index=frame_index,
                    video_time_us=video_time_us,
                    calibration_id=cal_id,
                    quality=sol.quality.value,
                    H_row_major=tuple(h_rm) if h_rm is not None else None,
                    H_inv_row_major=tuple(hinv_rm) if hinv_rm is not None else None,
                    correspondence_count=sol.correspondence_count,
                    inlier_count=sol.inlier_count,
                    inlier_ratio=sol.inlier_ratio,
                    mean_reprojection_error_px=sol.mean_reprojection_error_px,
                    condition_number=sol.condition_number,
                    determinant=sol.determinant,
                    coverage_hull_fraction=sol.coverage_hull_fraction,
                    solver_method=sol.solver_method,
                    solver_version=sol.solver_version,
                    physical_mapping_eligible=sol.physical_mapping_eligible,
                    reason_codes=sol.reason_codes,
                    quality_flags=sol.quality_flags,
                )
            )
            cal_id += 1

    segments: list[dict[str, Any]] = []
    seg_stats: dict[str, int] = {}
    gaps: list[tuple[int, int]] = []
    overlaps: list[tuple[str, str]] = []
    review: list[str] = []
    if build_segments and config["output_policy"]["write_segments"]:
        built_seg = build_calibration_segments(
            candidates,
            run_id=rid,
            video_id=vid,
            config=config,
            pitch_template_fingerprint=t_fp,
            pitch_length_m=template.length_m,
            pitch_width_m=template.width_m,
            shot_cuts_us=shot_cuts_us,
            test_points=pitch_test_points(template),
        )
        segments = built_seg.segments
        seg_stats = built_seg.stats
        gaps = built_seg.gaps
        overlaps = built_seg.overlaps
        review = built_seg.review_required

    try:
        if config["output_policy"]["write_calibrations"]:
            table = _rows_to_table(calibrations, "calibrations")
            write_contract_parquet(
                table, cal_out, get_contract("calibrations", 1), contain_root=root
            )
            cal_out.chmod(0o600)
        if build_segments and config["output_policy"]["write_segments"]:
            seg_table = _rows_to_table(segments, "calibration_segments")
            write_contract_parquet(
                seg_table, seg_out, get_contract("calibration_segments", 1), contain_root=root
            )
            seg_out.chmod(0o600)
    except Exception as exc:  # noqa: BLE001
        return _fail(
            error_code=f"WRITE_FAILED:{type(exc).__name__}",
            exit_code=1,
            config_fingerprint=cfg_fp,
            cleanup=artifacts + [out],
        )

    # Receipt recount from canonical artifacts.
    recounted = recount_calibration_quality(calibrations)
    if cal_out.exists():
        disk_cal = read_contract_parquet(cal_out, get_contract("calibrations", 1)).to_pylist()
        recounted_disk = recount_calibration_quality(disk_cal)
        if recounted_disk["total"] != recounted["total"]:
            return _fail(
                error_code="RECEIPT_RECOUNT_MISMATCH",
                exit_code=1,
                config_fingerprint=cfg_fp,
                cleanup=artifacts + [out],
            )

    eval_report = evaluate_homography(
        calibrations=calibrations, segments=segments, has_reviewed_ground_truth=False
    )
    quality = {
        "schema_version": 1,
        "run_id": rid,
        "video_id": vid,
        "correspondence_counts": {
            "accepted": corr_accepted,
            "rejected": corr_rejected,
            "total": corr_accepted + corr_rejected,
        },
        "calibration_quality_counts": quality_counts,
        "recounted_calibrations": recounted,
        "reprojection_summary": {
            "n": len(reprojs),
            "mean": statistics.fmean(reprojs) if reprojs else None,
            "median": statistics.median(reprojs) if reprojs else None,
        },
        "condition_summary": {
            "n": len(conditions),
            "mean": statistics.fmean(conditions) if conditions else None,
        },
        "coverage_summary": {
            "n": len(coverages),
            "mean": statistics.fmean(coverages) if coverages else None,
        },
        "inlier_ratio_summary": {
            "n": len(inlier_ratios),
            "mean": statistics.fmean(inlier_ratios) if inlier_ratios else None,
        },
        "segment_stats": seg_stats,
        "gaps": [{"start_us": a, "end_us": b} for a, b in gaps],
        "overlaps": [{"a": a, "b": b} for a, b in overlaps],
        "notes": [
            "operational quality only; not football accuracy",
            "no projected_positions in Stage 8C",
            "attack_direction=unknown",
        ],
        "created_at_utc": _utc_now(),
    }
    output_artifacts: dict[str, Any] = {}
    if cal_out.exists():
        output_artifacts["calibrations"] = {
            "path": str(cal_out),
            "sha256": sha256_file(cal_out),
            "size_bytes": int(cal_out.stat().st_size),
        }
    if seg_out.exists():
        output_artifacts["calibration_segments"] = {
            "path": str(seg_out),
            "sha256": sha256_file(seg_out),
            "size_bytes": int(seg_out.stat().st_size),
        }
    receipt = {
        "schema_version": 1,
        "receipt_id": f"homo_{rid[-12:]}",
        "run_id": rid,
        "video_id": vid,
        "status": "succeeded",
        "stage": "8C",
        "solver_id": config["solver_id"],
        "solver_version": config["solver_version"],
        "method": config["method"],
        "config_fingerprint": cfg_fp,
        "pitch_template_fingerprint": t_fp,
        "pitch_length_m": template.length_m,
        "pitch_width_m": template.width_m,
        "real_pitch_size_known": bool(config["pitch"]["real_size_known"]),
        "attack_direction": "unknown",
        "auto_project_positions": False,
        "no_overwrite": True,
        "atomic_writes": True,
        "evaluation_status": NOT_EVALUATED_HOMOGRAPHY,
        "correspondence_counts": quality["correspondence_counts"],
        "calibration_quality_counts": quality_counts,
        "recounted_calibrations": recounted,
        "segment_stats": seg_stats,
        "gap_count": len(gaps),
        "overlap_count": len(overlaps),
        "physical_mapping_eligible_segment_count": int(
            seg_stats.get("physical_eligible_segments", 0)
        ),
        "review_count": len(review),
        "output_artifacts": output_artifacts,
        "contract_version": CONTRACT_VERSION,
        "created_at_utc": _utc_now(),
        "warnings": [
            "REAL_FOOTBALL_ACCURACY_NOT_YET_VALIDATED",
            "HOMOGRAPHY_IS_PITCH_PLANE_ONLY",
            "FEATURE_DETECTION_NOT_HOMOGRAPHY_GUARANTEE",
        ],
        "errors": [],
        "provenance": {
            "stage": "8C",
            "notes": config["notes"],
            "gpl_adapter_unchanged": True,
            "projected_positions": False,
        },
    }

    try:
        write_json_record(receipt_out, receipt, overwrite=False, contain_root=root)
        write_json_record(quality_out, quality, overwrite=False, contain_root=root)
        write_json_record(
            eval_out,
            eval_report.to_dict(run_id=rid, video_id=vid, config_fingerprint=cfg_fp),
            overwrite=False,
            contain_root=root,
        )
        for p in (receipt_out, quality_out, eval_out):
            p.chmod(0o600)
    except Exception as exc:  # noqa: BLE001
        return _fail(
            error_code=f"RECEIPT_WRITE_FAILED:{type(exc).__name__}",
            exit_code=1,
            config_fingerprint=cfg_fp,
            cleanup=artifacts + [out],
        )

    return HomographyServiceResult(
        accepted=True,
        exit_code=0,
        error_code=None,
        config_fingerprint=cfg_fp,
        calibrations_parquet=str(cal_out) if cal_out.exists() else None,
        segments_parquet=str(seg_out) if seg_out.exists() else None,
        receipt_json=str(receipt_out),
        quality_json=str(quality_out),
        evaluation_json=str(eval_out),
        summary={
            "status": "succeeded",
            "calibration_count": len(calibrations),
            "segment_count": len(segments),
            "evaluation_status": NOT_EVALUATED_HOMOGRAPHY,
            "quality_counts": quality_counts,
            "calibrations": calibrations,
            "segments": segments,
        },
    )


def run_segments_build(
    *,
    output_dir: str | Path,
    config: Mapping[str, Any],
    contain_root: Path | str | None = None,
    calibrations_path: str | Path,
    run_id: str | None = None,
    video_id: str | None = None,
    shot_cuts_us: Sequence[int] | None = None,
    frame_times_us: Mapping[int, int] | None = None,
) -> HomographyServiceResult:
    """Build segments from an existing calibrations parquet (no re-solve)."""
    cfg_fp = homography_config_fingerprint(config)
    try:
        reject_unsafe_path_string(str(output_dir), label="output_dir")
        out = require_absolute_path(str(output_dir), label="output_dir")
    except Exception as exc:  # noqa: BLE001
        return _fail(
            error_code=f"OUTPUT_PATH:{type(exc).__name__}",
            exit_code=2,
            config_fingerprint=cfg_fp,
        )
    root = Path(contain_root) if contain_root is not None else Path(str(config["runtime_root"]))
    try:
        from football_analytics.utils.archive_safety import (
            assert_contained,
            assert_not_dangerous_operation_root,
            resolve_strict,
        )

        root = require_absolute_path(str(root), label="contain_root")
        assert_not_dangerous_operation_root(root)
        resolved_out = resolve_strict(out) if out.exists() else out.resolve()
        assert_contained(resolved_out, resolve_strict(root), label="output_root")
    except Exception as exc:  # noqa: BLE001
        return _fail(
            error_code=f"CONTAINMENT:{type(exc).__name__}",
            exit_code=2,
            config_fingerprint=cfg_fp,
        )
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    seg_out = out / "calibration_segments.parquet"
    receipt_out = out / "segments_receipt.json"
    if seg_out.exists() or receipt_out.exists():
        return _fail(error_code="NO_OVERWRITE", exit_code=1, config_fingerprint=cfg_fp)

    table = read_contract_parquet(Path(calibrations_path), get_contract("calibrations", 1))
    rows = table.to_pylist()
    if not rows:
        return _fail(error_code="EMPTY_CALIBRATIONS", exit_code=1, config_fingerprint=cfg_fp)
    rid = run_id or str(rows[0]["run_id"])
    vid = video_id or str(rows[0]["video_id"])
    template = build_pitch_template(
        length_m=float(config["pitch"]["length_m"]),
        width_m=float(config["pitch"]["width_m"]),
    )
    t_fp = pitch_template_fingerprint(template)
    candidates: list[FrameCalibrationCandidate] = []
    for row in rows:
        flags = [str(x) for x in (row.get("quality_flags") or [])]
        quality = "invalid"
        for f in flags:
            if f.startswith("quality:"):
                quality = f.split(":", 1)[1]
                break
        if row.get("is_valid"):
            quality = HomographyQuality.VALID.value
        H = row.get("homography_image_to_pitch")
        fi = int(row["frame_index"])
        vt = int((frame_times_us or {}).get(fi, fi * 40_000))
        candidates.append(
            FrameCalibrationCandidate(
                frame_index=fi,
                video_time_us=vt,
                calibration_id=int(row["calibration_id"]),
                quality=quality,
                H_row_major=tuple(float(x) for x in H) if H is not None else None,
                H_inv_row_major=None,
                correspondence_count=4,
                inlier_count=4,
                inlier_ratio=1.0,
                mean_reprojection_error_px=(
                    float(row["reprojection_error_px"])
                    if row.get("reprojection_error_px") is not None
                    else None
                ),
                condition_number=None,
                determinant=None,
                coverage_hull_fraction=None,
                solver_method=str(row.get("method") or "unknown"),
                solver_version="1",
                physical_mapping_eligible=bool(row.get("is_valid")),
            )
        )
    built = build_calibration_segments(
        candidates,
        run_id=rid,
        video_id=vid,
        config=config,
        pitch_template_fingerprint=t_fp,
        pitch_length_m=template.length_m,
        pitch_width_m=template.width_m,
        shot_cuts_us=shot_cuts_us,
        test_points=pitch_test_points(template),
    )
    try:
        write_contract_parquet(
            _rows_to_table(built.segments, "calibration_segments"),
            seg_out,
            get_contract("calibration_segments", 1),
            contain_root=root,
        )
        seg_out.chmod(0o600)
        write_json_record(
            receipt_out,
            {
                "schema_version": 1,
                "run_id": rid,
                "video_id": vid,
                "status": "succeeded",
                "stage": "8C",
                "config_fingerprint": cfg_fp,
                "segment_stats": built.stats,
                "evaluation_status": NOT_EVALUATED_HOMOGRAPHY,
                "created_at_utc": _utc_now(),
            },
            overwrite=False,
            contain_root=root,
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(
            error_code=f"WRITE_FAILED:{type(exc).__name__}",
            exit_code=1,
            config_fingerprint=cfg_fp,
            cleanup=[seg_out, receipt_out, out],
        )
    return HomographyServiceResult(
        accepted=True,
        exit_code=0,
        error_code=None,
        config_fingerprint=cfg_fp,
        calibrations_parquet=str(calibrations_path),
        segments_parquet=str(seg_out),
        receipt_json=str(receipt_out),
        quality_json=None,
        evaluation_json=None,
        summary={
            "status": "succeeded",
            "segment_count": len(built.segments),
            "segment_stats": built.stats,
            "evaluation_status": NOT_EVALUATED_HOMOGRAPHY,
        },
    )


__all__ = [
    "HomographyServiceError",
    "HomographyServiceResult",
    "recount_calibration_quality",
    "run_homography_solve",
    "run_segments_build",
]
