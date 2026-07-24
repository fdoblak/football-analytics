"""Stage 8D pitch projection service — projected_positions + receipt/quality/eval."""

from __future__ import annotations

import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa

from football_analytics.calibration.pitch_projection import (
    PitchProjectionError,
    assert_fingerprints_aligned,
    build_projection_for_observation,
    covering_segment_conflicts,
    find_duplicate_projection_keys,
)
from football_analytics.calibration.pitch_projection_config import (
    pitch_projection_config_fingerprint,
)
from football_analytics.calibration.pitch_projection_evaluation import (
    NOT_EVALUATED_PROJECTED_POS,
    evaluate_pitch_projection,
)
from football_analytics.calibration.pitch_projection_quality import (
    build_projection_review_queue,
    build_quality_report,
    recount_projection_stats,
)
from football_analytics.calibration.pitch_template import (
    build_pitch_template,
    pitch_template_fingerprint,
)
from football_analytics.calibration.types import CONTRACT_VERSION
from football_analytics.core.hashing import sha256_file
from football_analytics.core.records import write_json_record
from football_analytics.core.run_id import validate_run_id
from football_analytics.data.compiler import compile_arrow_schema, get_contract
from football_analytics.data.parquet import read_contract_parquet, write_contract_parquet
from football_analytics.data.types import SAFE_ID_RE
from football_analytics.video.validation import (
    reject_unsafe_path_string,
    require_absolute_path,
)


class PitchProjectionServiceError(RuntimeError):
    """Pitch projection service failure."""


@dataclass
class PitchProjectionServiceResult:
    accepted: bool
    exit_code: int
    error_code: str | None
    config_fingerprint: str
    projected_positions_parquet: str | None
    receipt_json: str | None
    quality_json: str | None
    evaluation_json: str | None
    review_queue_json: str | None
    summary: Mapping[str, Any]

    def to_summary(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "exit_code": self.exit_code,
            "error_code": self.error_code,
            "config_fingerprint": self.config_fingerprint,
            "projected_positions_parquet": self.projected_positions_parquet,
            "receipt_json": self.receipt_json,
            "quality_json": self.quality_json,
            "evaluation_json": self.evaluation_json,
            "review_queue_json": self.review_queue_json,
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
) -> PitchProjectionServiceResult:
    if cleanup:
        for p in cleanup:
            try:
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
            except OSError:
                pass
    return PitchProjectionServiceResult(
        accepted=False,
        exit_code=exit_code,
        error_code=error_code,
        config_fingerprint=config_fingerprint,
        projected_positions_parquet=None,
        receipt_json=None,
        quality_json=None,
        evaluation_json=None,
        review_queue_json=None,
        summary={"status": "failed", "error_code": error_code},
    )


def _rows_to_table(rows: list[dict[str, Any]], contract_name: str) -> Any:
    contract = get_contract(contract_name, 1)
    schema = compile_arrow_schema(contract)
    if not rows:
        return schema.empty_table()
    return pa.Table.from_pylist(rows, schema=schema)


def _load_obs(
    path: Path | None, in_memory: Sequence[Mapping[str, Any]] | None
) -> list[dict[str, Any]]:
    if in_memory is not None:
        return [dict(r) for r in in_memory]
    if path is None:
        raise PitchProjectionServiceError("OBSERVATIONS_REQUIRED")
    table = read_contract_parquet(path, get_contract("track_observations", 1))
    return list(table.to_pylist())


def _load_segments(
    path: Path | None, in_memory: Sequence[Mapping[str, Any]] | None
) -> list[dict[str, Any]]:
    if in_memory is not None:
        return [dict(r) for r in in_memory]
    if path is None:
        raise PitchProjectionServiceError("SEGMENTS_REQUIRED")
    table = read_contract_parquet(path, get_contract("calibration_segments", 1))
    return list(table.to_pylist())


def run_pitch_projection(
    *,
    output_dir: str | Path,
    config: Mapping[str, Any],
    contain_root: Path | str | None = None,
    run_id: str | None = None,
    video_id: str | None = None,
    observations_path: str | Path | None = None,
    observations_rows: Sequence[Mapping[str, Any]] | None = None,
    segments_path: str | Path | None = None,
    segments_rows: Sequence[Mapping[str, Any]] | None = None,
    frame_times: Mapping[int, int] | None = None,
    coverage_hulls: Mapping[str, Sequence[Sequence[float]]] | None = None,
    eligibility_timeline: Mapping[str, Any] | None = None,
    analysis_windows: Sequence[Mapping[str, Any]] | None = None,
    fingerprints: Mapping[str, Any] | None = None,
    frame_width: float | None = None,
    frame_height: float | None = None,
    ambiguous_ball_track_ids: Sequence[int] | None = None,
    force_conflict_fail: bool = True,
) -> PitchProjectionServiceResult:
    """Project track observations onto pitch metres via calibration_segments."""
    cfg_fp = pitch_projection_config_fingerprint(config)
    if config.get("compute_physical_metrics") is True:
        return _fail(
            error_code="PHYSICAL_METRICS_FORBIDDEN", exit_code=2, config_fingerprint=cfg_fp
        )
    if config.get("compute_events") is True:
        return _fail(error_code="EVENTS_FORBIDDEN", exit_code=2, config_fingerprint=cfg_fp)
    if str(config.get("attack_direction", "unknown")) != "unknown":
        return _fail(
            error_code="ATTACK_DIRECTION_MUST_BE_UNKNOWN",
            exit_code=2,
            config_fingerprint=cfg_fp,
        )
    if config.get("ball_source", {}).get("physical_metric_eligible") is True:
        return _fail(
            error_code="BALL_PHYSICAL_ELIGIBLE_FORBIDDEN",
            exit_code=2,
            config_fingerprint=cfg_fp,
        )
    if config.get("ball_source", {}).get("event_metric_eligible") is True:
        return _fail(
            error_code="BALL_EVENT_ELIGIBLE_FORBIDDEN",
            exit_code=2,
            config_fingerprint=cfg_fp,
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

    proj_out = out / "projected_positions.parquet"
    receipt_out = out / "projection_receipt.json"
    quality_out = out / "projection_quality.json"
    eval_out = out / "projection_evaluation.json"
    review_out = out / "review_queue.json"
    artifacts = [proj_out, receipt_out, quality_out, eval_out, review_out]

    if any(p.exists() for p in artifacts):
        return _fail(error_code="NO_OVERWRITE", exit_code=1, config_fingerprint=cfg_fp)

    out.mkdir(parents=True, exist_ok=True, mode=0o700)

    try:
        observations = _load_obs(
            Path(observations_path) if observations_path else None, observations_rows
        )
        segments = _load_segments(Path(segments_path) if segments_path else None, segments_rows)
    except Exception as exc:  # noqa: BLE001
        return _fail(
            error_code=f"INPUT_LOAD:{type(exc).__name__}",
            exit_code=1,
            config_fingerprint=cfg_fp,
            cleanup=artifacts + [out],
        )

    if not observations:
        return _fail(
            error_code="EMPTY_OBSERVATIONS",
            exit_code=1,
            config_fingerprint=cfg_fp,
            cleanup=artifacts + [out],
        )
    if not segments:
        return _fail(
            error_code="EMPTY_SEGMENTS",
            exit_code=1,
            config_fingerprint=cfg_fp,
            cleanup=artifacts + [out],
        )

    rid = run_id or str(observations[0]["run_id"])
    vid = video_id or str(observations[0]["video_id"])
    try:
        validate_run_id(rid)
    except Exception as exc:  # noqa: BLE001
        return _fail(
            error_code=f"RUN_ID:{type(exc).__name__}",
            exit_code=2,
            config_fingerprint=cfg_fp,
            cleanup=artifacts + [out],
        )
    if not SAFE_ID_RE.match(vid):
        return _fail(
            error_code="VIDEO_ID_UNSAFE",
            exit_code=2,
            config_fingerprint=cfg_fp,
            cleanup=artifacts + [out],
        )

    # Align fingerprints / IDs across inputs.
    for row in observations + segments:
        if str(row.get("run_id")) != rid or str(row.get("video_id")) != vid:
            return _fail(
                error_code="FK_MISMATCH",
                exit_code=1,
                config_fingerprint=cfg_fp,
                cleanup=artifacts + [out],
            )
    if eligibility_timeline is not None and (
        str(eligibility_timeline.get("run_id")) != rid
        or str(eligibility_timeline.get("video_id")) != vid
    ):
        return _fail(
            error_code="FK_MISMATCH:identity_timeline",
            exit_code=1,
            config_fingerprint=cfg_fp,
            cleanup=artifacts + [out],
        )

    template = build_pitch_template(
        length_m=float(config["pitch"]["length_m"]),
        width_m=float(config["pitch"]["width_m"]),
    )
    t_fp = pitch_template_fingerprint(template)
    for seg in segments:
        if str(seg.get("pitch_template_fingerprint")) != t_fp:
            return _fail(
                error_code="PITCH_TEMPLATE_MISMATCH",
                exit_code=1,
                config_fingerprint=cfg_fp,
                cleanup=artifacts + [out],
            )

    if fingerprints is not None:
        actual_fps = {
            "run_id": rid,
            "video_id": vid,
            "pitch_template": t_fp,
            "coordinate_frame": "source_image",
        }
        for k in (
            "source_video_sha",
            "frame_timeline",
            "tracking_bundle",
            "calibration_artifact",
        ):
            if k in fingerprints:
                actual_fps[k] = fingerprints[k]
        # Optional keys must match when both sides declare them.
        try:
            for k, expected_v in fingerprints.items():
                if k in actual_fps and str(actual_fps[k]) != str(expected_v):
                    raise PitchProjectionError(f"FINGERPRINT_MISMATCH:{k}")
            assert_fingerprints_aligned(
                expected=fingerprints,
                actual=actual_fps,
                keys=("run_id", "video_id", "pitch_template", "coordinate_frame"),
            )
        except PitchProjectionError as exc:
            return _fail(
                error_code=str(exc),
                exit_code=1,
                config_fingerprint=cfg_fp,
                cleanup=artifacts + [out],
            )

    conflicts = covering_segment_conflicts(segments, run_id=rid, video_id=vid)
    if conflicts and force_conflict_fail:
        return _fail(
            error_code="SEGMENT_OVERLAP_CONFLICT",
            exit_code=1,
            config_fingerprint=cfg_fp,
            cleanup=artifacts + [out],
        )

    times = dict(frame_times or {})
    amb_ids = {int(x) for x in (ambiguous_ball_track_ids or [])}
    projections: list[dict[str, Any]] = []
    seen_obs: set[tuple[int, int]] = set()  # (frame, track)
    try:
        for i, obs in enumerate(observations):
            fi = int(obs["frame_index"])
            tid = int(obs["track_id"])
            key = (fi, tid)
            if key in seen_obs:
                return _fail(
                    error_code="DUPLICATE_OBSERVATION",
                    exit_code=1,
                    config_fingerprint=cfg_fp,
                    cleanup=artifacts + [out],
                )
            seen_obs.add(key)
            vt = int(times.get(fi, fi * 40_000))
            proj_id = f"proj_{fi:06d}_{tid:04d}_{i:04d}"
            row = build_projection_for_observation(
                observation=obs,
                segments=segments,
                config=config,
                run_id=rid,
                video_id=vid,
                video_time_us=vt,
                projection_id=proj_id,
                pitch_template_fingerprint=t_fp,
                coverage_hulls=coverage_hulls,
                eligibility_timeline=eligibility_timeline,
                analysis_windows=analysis_windows,
                frame_width=frame_width,
                frame_height=frame_height,
                ambiguous_ball=tid in amb_ids,
                force_conflict_fail=force_conflict_fail,
            )
            projections.append(row)
    except PitchProjectionError as exc:
        return _fail(
            error_code=str(exc)[:120],
            exit_code=1,
            config_fingerprint=cfg_fp,
            cleanup=artifacts + [out],
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(
            error_code=f"PROJECTION_FAILED:{type(exc).__name__}",
            exit_code=1,
            config_fingerprint=cfg_fp,
            cleanup=artifacts + [out],
        )

    dups = find_duplicate_projection_keys(projections)
    if dups:
        return _fail(
            error_code="DUPLICATE_PROJECTION_KEYS",
            exit_code=1,
            config_fingerprint=cfg_fp,
            cleanup=artifacts + [out],
        )

    stats = recount_projection_stats(projections)
    if int(stats["ball_physical_metric_eligible_count"]) != 0:
        return _fail(
            error_code="BALL_PHYSICAL_ELIGIBLE_NONZERO",
            exit_code=1,
            config_fingerprint=cfg_fp,
            cleanup=artifacts + [out],
        )
    if int(stats["ball_event_metric_eligible_count"]) != 0:
        return _fail(
            error_code="BALL_EVENT_ELIGIBLE_NONZERO",
            exit_code=1,
            config_fingerprint=cfg_fp,
            cleanup=artifacts + [out],
        )
    if int(stats["predicted_physical_eligibility_violations"]) > 0:
        return _fail(
            error_code="PREDICTED_PHYSICAL_ELIGIBLE",
            exit_code=1,
            config_fingerprint=cfg_fp,
            cleanup=artifacts + [out],
        )
    if int(stats["extrapolated_physical_eligibility_violations"]) > 0:
        return _fail(
            error_code="EXTRAPOLATED_PHYSICAL_ELIGIBLE",
            exit_code=1,
            config_fingerprint=cfg_fp,
            cleanup=artifacts + [out],
        )

    segment_usage = Counter(
        str(p["segment_id"]) for p in projections if p.get("segment_id") is not None
    )
    eval_report = evaluate_pitch_projection(
        projections=projections, has_reviewed_ground_truth=False
    )
    review = build_projection_review_queue(
        projections,
        max_samples=int(config["review_sampling"]["max_samples"]),
        enabled=bool(config["review_sampling"]["enabled"]),
        conflicts=conflicts,
    )
    quality = build_quality_report(
        run_id=rid,
        video_id=vid,
        stats=stats,
        segment_usage=dict(segment_usage),
        conflict_count=len(conflicts),
        duplicate_count=len(dups),
        evaluation_status=NOT_EVALUATED_PROJECTED_POS,
        config_fingerprint=cfg_fp,
    )

    try:
        if config["output_policy"]["write_projected_positions"]:
            write_contract_parquet(
                _rows_to_table(projections, "projected_positions"),
                proj_out,
                get_contract("projected_positions", 1),
                contain_root=root,
            )
            proj_out.chmod(0o600)
    except Exception as exc:  # noqa: BLE001
        return _fail(
            error_code=f"WRITE_FAILED:{type(exc).__name__}",
            exit_code=1,
            config_fingerprint=cfg_fp,
            cleanup=artifacts + [out],
        )

    # Recount from canonical artifact.
    if proj_out.exists():
        disk_rows = read_contract_parquet(
            proj_out, get_contract("projected_positions", 1)
        ).to_pylist()
        disk_stats = recount_projection_stats(disk_rows)
        if disk_stats["total"] != stats["total"]:
            return _fail(
                error_code="RECEIPT_RECOUNT_MISMATCH",
                exit_code=1,
                config_fingerprint=cfg_fp,
                cleanup=artifacts + [out],
            )
        if int(disk_stats["ball_physical_metric_eligible_count"]) != 0:
            return _fail(
                error_code="BALL_PHYSICAL_ELIGIBLE_NONZERO_DISK",
                exit_code=1,
                config_fingerprint=cfg_fp,
                cleanup=artifacts + [out],
            )
        stats = disk_stats

    output_artifacts: dict[str, Any] = {}
    if proj_out.exists():
        output_artifacts["projected_positions"] = {
            "path": str(proj_out),
            "sha256": sha256_file(proj_out),
            "size_bytes": int(proj_out.stat().st_size),
        }

    receipt = {
        "schema_version": 1,
        "receipt_id": f"proj_{rid[-12:]}",
        "run_id": rid,
        "video_id": vid,
        "status": "succeeded",
        "stage": "8D",
        "pipeline_id": config["pipeline_id"],
        "pipeline_version": config["pipeline_version"],
        "config_fingerprint": cfg_fp,
        "pitch_template_fingerprint": t_fp,
        "pitch_length_m": template.length_m,
        "pitch_width_m": template.width_m,
        "real_pitch_size_known": bool(config["pitch"]["real_size_known"]),
        "attack_direction": "unknown",
        "input_fingerprints": dict(fingerprints or {}),
        "human_observation_count": int(stats["human_observation_count"]),
        "ball_observation_count": int(stats["ball_observation_count"]),
        "mapping_status_counts": dict(stats["mapping_status_counts"]),
        "human_physical_metric_eligible_count": int(stats["human_physical_metric_eligible_count"]),
        "target_customer_metric_eligible_count": int(
            stats["target_customer_metric_eligible_count"]
        ),
        "ball_physical_metric_eligible_count": int(stats["ball_physical_metric_eligible_count"]),
        "ball_event_metric_eligible_count": int(stats["ball_event_metric_eligible_count"]),
        "segment_usage": dict(segment_usage),
        "conflict_count": len(conflicts),
        "duplicate_count": len(dups),
        "round_trip_summary": dict(stats["round_trip_error_px"]),
        "uncertainty_summary": dict(stats["uncertainty_m"]),
        "review_count": int(review["sampled_count"]),
        "evaluation_status": NOT_EVALUATED_PROJECTED_POS,
        "no_overwrite": True,
        "atomic_writes": True,
        "output_artifacts": output_artifacts,
        "contract_version": CONTRACT_VERSION,
        "created_at_utc": _utc_now(),
        "warnings": [
            "REAL_FOOTBALL_ACCURACY_NOT_YET_VALIDATED",
            "HUMAN_FOOTPOINT_IS_APPROXIMATION",
            "BALL_AIRBORNE_STATUS_UNKNOWN",
            "HOMOGRAPHY_IS_PITCH_PLANE_ONLY",
            "ATTACK_DIRECTION_UNKNOWN",
            "NO_DISTANCE_SPEED_SPRINT_HEATMAP_EVENTS",
        ],
        "errors": [],
        "provenance": {
            "stage": "8D",
            "notes": config["notes"],
            "gpl_adapter_unchanged": True,
            "compute_physical_metrics": False,
            "compute_events": False,
        },
    }

    # Hard invariant in receipt.
    if (
        receipt["ball_physical_metric_eligible_count"] != 0
        or receipt["ball_event_metric_eligible_count"] != 0
    ):
        return _fail(
            error_code="BALL_ELIGIBILITY_RECEIPT_NONZERO",
            exit_code=1,
            config_fingerprint=cfg_fp,
            cleanup=artifacts + [out],
        )

    try:
        write_json_record(receipt_out, receipt, overwrite=False, contain_root=root)
        if config["output_policy"]["write_quality_json"]:
            write_json_record(quality_out, quality, overwrite=False, contain_root=root)
        if config["output_policy"]["write_evaluation_json"]:
            write_json_record(
                eval_out,
                eval_report.to_dict(run_id=rid, video_id=vid, config_fingerprint=cfg_fp),
                overwrite=False,
                contain_root=root,
            )
        if config["output_policy"]["write_review_queue"]:
            write_json_record(review_out, review, overwrite=False, contain_root=root)
        for p in (receipt_out, quality_out, eval_out, review_out):
            if p.exists():
                p.chmod(0o600)
    except Exception as exc:  # noqa: BLE001
        return _fail(
            error_code=f"RECEIPT_WRITE_FAILED:{type(exc).__name__}",
            exit_code=1,
            config_fingerprint=cfg_fp,
            cleanup=artifacts + [out],
        )

    return PitchProjectionServiceResult(
        accepted=True,
        exit_code=0,
        error_code=None,
        config_fingerprint=cfg_fp,
        projected_positions_parquet=str(proj_out) if proj_out.exists() else None,
        receipt_json=str(receipt_out),
        quality_json=str(quality_out) if quality_out.exists() else None,
        evaluation_json=str(eval_out) if eval_out.exists() else None,
        review_queue_json=str(review_out) if review_out.exists() else None,
        summary={
            "status": "succeeded",
            "projection_count": len(projections),
            "evaluation_status": NOT_EVALUATED_PROJECTED_POS,
            "stats": stats,
            "projections": projections,
            "receipt": receipt,
        },
    )


__all__ = [
    "PitchProjectionServiceError",
    "PitchProjectionServiceResult",
    "run_pitch_projection",
]
