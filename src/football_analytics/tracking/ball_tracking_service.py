"""Stage 6C ball tracking service: load → track → validate → atomic publish."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa

from football_analytics.core.hashing import hash_canonical_json, sha256_file
from football_analytics.core.records import write_json_record
from football_analytics.core.run_id import generate_run_id, validate_run_id
from football_analytics.data.compiler import compile_arrow_schema, get_contract
from football_analytics.data.parquet import read_contract_parquet, write_contract_parquet
from football_analytics.data.types import SAFE_ID_RE
from football_analytics.tracking.ball_tracker import run_ball_tracker
from football_analytics.tracking.ball_tracking_config import ball_tracking_config_fingerprint
from football_analytics.tracking.ball_tracking_evaluation import (
    NOT_EVALUATED_BALL_TRACKING,
    compute_synthetic_ball_metrics,
    evaluate_ball_tracking,
)
from football_analytics.tracking.policy import load_tracking_policy, policy_fingerprint
from football_analytics.tracking.receipt import (
    recount_receipt_from_tables,
    validate_receipt_payload,
)
from football_analytics.tracking.validation import validate_track_bundle


class BallTrackingServiceError(RuntimeError):
    """Ball tracking service failure."""


@dataclass
class BallTrackingServiceResult:
    accepted: bool
    exit_code: int
    error_code: str | None
    config_fingerprint: str
    observations_parquet: str | None
    summaries_parquet: str | None
    lifecycle_parquet: str | None
    receipt_json: str | None
    evaluation_json: str | None
    quality_json: str | None
    primary_sidecar_json: str | None
    summary: Mapping[str, Any]

    def to_summary(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "exit_code": self.exit_code,
            "error_code": self.error_code,
            "config_fingerprint": self.config_fingerprint,
            "observations_parquet": self.observations_parquet,
            "summaries_parquet": self.summaries_parquet,
            "lifecycle_parquet": self.lifecycle_parquet,
            "receipt_json": self.receipt_json,
            "evaluation_json": self.evaluation_json,
            "quality_json": self.quality_json,
            "primary_sidecar_json": self.primary_sidecar_json,
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
) -> BallTrackingServiceResult:
    if cleanup:
        for p in cleanup:
            try:
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
            except OSError:
                pass
    return BallTrackingServiceResult(
        accepted=False,
        exit_code=exit_code,
        error_code=error_code,
        config_fingerprint=config_fingerprint,
        observations_parquet=None,
        summaries_parquet=None,
        lifecycle_parquet=None,
        receipt_json=None,
        evaluation_json=None,
        quality_json=None,
        primary_sidecar_json=None,
        summary={"status": "failed", "error_code": error_code},
    )


def _rows_to_table(rows: list[dict[str, Any]], contract_name: str) -> Any:
    contract = get_contract(contract_name, 1)
    schema = compile_arrow_schema(contract)
    if not rows:
        return schema.empty_table()
    return pa.Table.from_pylist(rows, schema=schema)


def _artifact_meta(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def run_ball_tracking(
    *,
    detections: str | Path,
    frames: str | Path,
    analysis_windows: str | Path,
    output_dir: str | Path,
    config: Mapping[str, Any],
    detection_attributes: str | Path | None = None,
    videos: str | Path | None = None,
    contain_root: Path | str | None = None,
    run_id: str | None = None,
    video_id: str | None = None,
    ground_truth: str | Path | None = None,
    policy_path: Path | str | None = None,
    request_id: str = "trk_req_ball_01",
    in_memory_bundle: Mapping[str, Any] | None = None,
) -> BallTrackingServiceResult:
    """Track balls from detection bundle; atomic no-overwrite publish."""
    cfg_fp = ball_tracking_config_fingerprint(config)
    root = Path(contain_root) if contain_root is not None else Path(str(config["runtime_root"]))
    out = Path(output_dir)
    written: list[Path] = []

    try:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        out.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not str(out.resolve()).startswith(str(root.resolve())):
            return _fail(error_code="PATH_SECURITY", exit_code=3, config_fingerprint=cfg_fp)
    except Exception:  # noqa: BLE001
        return _fail(error_code="PATH_SECURITY", exit_code=3, config_fingerprint=cfg_fp)

    obs_out = out / "track_observations.parquet"
    sum_out = out / "track_summaries.parquet"
    life_out = out / "track_lifecycle.parquet"
    receipt_out = out / "tracking_run_receipt.json"
    eval_out = out / "tracking_evaluation.json"
    quality_out = out / "tracking_quality_report.json"
    primary_out = out / "ball_primary_candidates.json"
    for p in (obs_out, sum_out, life_out, receipt_out, eval_out, quality_out, primary_out):
        if p.exists():
            return _fail(error_code="OVERWRITE_FORBIDDEN", exit_code=3, config_fingerprint=cfg_fp)

    rid = run_id or generate_run_id()
    try:
        validate_run_id(rid)
    except Exception:  # noqa: BLE001
        return _fail(error_code="INVALID_RUN_ID", exit_code=3, config_fingerprint=cfg_fp)
    vid = video_id or "video_01"
    if not SAFE_ID_RE.match(vid):
        return _fail(error_code="INVALID_VIDEO_ID", exit_code=3, config_fingerprint=cfg_fp)

    started = _utc_now()
    try:
        if in_memory_bundle is not None:
            det_rows = list(in_memory_bundle["detections"].to_pylist())
            frame_rows = list(in_memory_bundle["frames"].to_pylist())
            window_rows = list(in_memory_bundle["analysis_windows"].to_pylist())
            attr_rows = (
                list(in_memory_bundle["detection_attributes"].to_pylist())
                if in_memory_bundle.get("detection_attributes") is not None
                else []
            )
            videos_table = in_memory_bundle.get("videos")
            det_path = Path(str(detections)) if detections else out / "_mem_detections"
            frames_path = Path(str(frames)) if frames else out / "_mem_frames"
            windows_path = Path(str(analysis_windows)) if analysis_windows else out / "_mem_windows"
            attr_path = Path(str(detection_attributes)) if detection_attributes else None
            input_hashes = {
                "detections": "a" * 64,
                "frames": "b" * 64,
                "analysis_windows": "c" * 64,
            }
        else:
            det_path = Path(detections)
            frames_path = Path(frames)
            windows_path = Path(analysis_windows)
            for p in (det_path, frames_path, windows_path):
                if not p.is_file() or p.is_symlink():
                    return _fail(error_code="INPUT_MISSING", exit_code=3, config_fingerprint=cfg_fp)
            det_table = read_contract_parquet(det_path, get_contract("detections", 1))
            frames_table = read_contract_parquet(frames_path, get_contract("frames", 1))
            windows_table = read_contract_parquet(windows_path, get_contract("analysis_windows", 1))
            det_rows = det_table.to_pylist()
            frame_rows = frames_table.to_pylist()
            window_rows = windows_table.to_pylist()
            attr_rows = []
            attr_path = Path(detection_attributes) if detection_attributes else None
            if attr_path is not None:
                if not attr_path.is_file() or attr_path.is_symlink():
                    return _fail(
                        error_code="ATTRIBUTES_MISSING", exit_code=3, config_fingerprint=cfg_fp
                    )
                attr_rows = read_contract_parquet(
                    attr_path, get_contract("detection_attributes", 1)
                ).to_pylist()
            videos_table = None
            if videos is not None:
                vp = Path(videos)
                if vp.is_file() and not vp.is_symlink():
                    videos_table = read_contract_parquet(vp, get_contract("videos", 1))
            input_hashes = {
                "detections": sha256_file(det_path),
                "frames": sha256_file(frames_path),
                "analysis_windows": sha256_file(windows_path),
            }
            if attr_path is not None:
                input_hashes["detection_attributes"] = sha256_file(attr_path)
    except Exception:  # noqa: BLE001
        return _fail(error_code="INPUT_READ_FAILED", exit_code=3, config_fingerprint=cfg_fp)

    if det_rows:
        rid = str(det_rows[0].get("run_id", rid))
        vid = str(det_rows[0].get("video_id", vid))

    try:
        pol_path = Path(policy_path) if policy_path else None
        if pol_path is None:
            from football_analytics.data.registry import default_project_root

            pol_path = default_project_root() / str(config["policy_path"])
        policy = load_tracking_policy(pol_path)
        pol_fp = policy_fingerprint(policy)
    except Exception:  # noqa: BLE001
        return _fail(error_code="POLICY_LOAD_FAILED", exit_code=2, config_fingerprint=cfg_fp)

    try:
        result = run_ball_tracker(
            run_id=rid,
            video_id=vid,
            frames=frame_rows,
            detections=det_rows,
            analysis_windows=window_rows,
            config=config,
            detection_attributes=attr_rows or None,
            policy=policy,
            tracker_model_id=str(config["tracker_id"]),
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(
            error_code=f"TRACKER_FAILED:{type(exc).__name__}",
            exit_code=1,
            config_fingerprint=cfg_fp,
        )

    obs_table = _rows_to_table(result.observations, "track_observations")
    sum_table = _rows_to_table(result.summaries, "track_summaries")
    life_table = _rows_to_table(result.lifecycle, "track_lifecycle")
    det_table_out = _rows_to_table(det_rows, "detections")
    frames_table_out = _rows_to_table(frame_rows, "frames")
    windows_table_out = _rows_to_table(window_rows, "analysis_windows")
    attr_table_out = _rows_to_table(attr_rows, "detection_attributes") if attr_rows else None

    specs = {
        "track_observations": get_contract("track_observations", 1),
        "track_summaries": get_contract("track_summaries", 1),
        "track_lifecycle": get_contract("track_lifecycle", 1),
        "frames": get_contract("frames", 1),
        "detections": get_contract("detections", 1),
        "detection_attributes": get_contract("detection_attributes", 1),
        "analysis_windows": get_contract("analysis_windows", 1),
        "videos": get_contract("videos", 1),
    }
    if videos_table is None and in_memory_bundle is not None:
        videos_table = in_memory_bundle.get("videos")

    counts = recount_receipt_from_tables(
        observations=result.observations,
        lifecycle=result.lifecycle,
        detections=[d for d in det_rows],
    )
    counts["detections_used"] = result.stats["detections_used"]
    counts["unassigned_detection_count"] = result.stats["unassigned_detection_count"]
    counts["total_input_detections"] = result.stats["ball_input_detections"]
    counts["track_counts"] = result.stats["track_counts"]
    counts["observation_counts"] = result.stats["observation_counts"]
    counts["review_required_count"] = result.stats["review_required_count"]

    receipt_provisional = {
        "schema_version": 1,
        "receipt_id": "trk_receipt_ball_01",
        "run_id": rid,
        "video_id": vid,
        "request_id": request_id,
        "tracker_id": str(config["tracker_id"]),
        "tracker_version": str(config["tracker_version"]),
        "config_fingerprint": cfg_fp,
        "policy_fingerprint": pol_fp,
        "input_artifacts": {
            "detections": {
                "path": str(det_path),
                "sha256": input_hashes["detections"],
                "size_bytes": 1,
            }
        },
        "output_artifacts": {},
        "total_input_detections": counts["total_input_detections"],
        "detections_used": counts["detections_used"],
        "detections_rejected": int(result.stats["rejected_non_ball"]),
        "unassigned_detection_count": counts["unassigned_detection_count"],
        "track_counts": counts["track_counts"],
        "observation_counts": counts["observation_counts"],
        "invalid_transition_count": 0,
        "invalid_fk_count": 0,
        "duplicate_count": 0,
        "routing_gap_count": int(result.stats["routing_gap_count"]),
        "review_required_count": counts["review_required_count"],
        "ground_truth_evaluation_status": NOT_EVALUATED_BALL_TRACKING,
        "started_at_utc": started,
        "completed_at_utc": started,
        "status": "succeeded",
        "warnings": [],
        "errors": [],
        "environment_ref": None,
        "provenance": {
            "stage": "6C",
            "label": "ball_tracking_baseline",
            "notes": "motion_first_constant_velocity_no_reid; primary_candidate_not_identity",
            "tracker_algorithm": str(config["tracker_algorithm"]),
            "merge_reid": False,
            "track_id_is_player_identity": False,
        },
    }

    vr = validate_track_bundle(
        track_observations=obs_table,
        track_summaries=sum_table,
        track_lifecycle=life_table,
        frames=frames_table_out,
        detections=det_table_out,
        detection_attributes=attr_table_out,
        videos=videos_table,
        analysis_windows=windows_table_out,
        specs=specs,
        policy=policy,
        receipt=receipt_provisional,
        frame_width=int(config["frame_geometry"]["frame_width"]),
        frame_height=int(config["frame_geometry"]["frame_height"]),
    )
    if vr.status == "FAIL":
        return _fail(
            error_code="BUNDLE_VALIDATION_FAILED",
            exit_code=1,
            config_fingerprint=cfg_fp,
            cleanup=written,
        )

    gt_rows = None
    has_reviewed = False
    if ground_truth is not None:
        gp = Path(ground_truth)
        try:
            if gp.suffix.lower() == ".json":
                payload = json.loads(gp.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and payload.get("reviewed") is True:
                    gt_rows = list(payload.get("tracks") or payload.get("ground_truth") or [])
                    has_reviewed = True
        except Exception:  # noqa: BLE001
            gt_rows = None
            has_reviewed = False

    synth = compute_synthetic_ball_metrics(
        observations=result.observations,
        lifecycle=result.lifecycle,
        recoveries=int(result.stats["recoveries"]),
        fragmentations=int(result.stats["fragmentations"]),
        max_gap_us=int(result.stats["max_gap_us"]),
        primary_frames=int(result.stats["primary_frames"]),
        ambiguous_frames=int(result.stats["ambiguous_frames"]),
        no_candidate_frames=int(result.stats["no_candidate_frames"]),
    )
    eval_report = evaluate_ball_tracking(
        track_observations=result.observations,
        track_summaries=result.summaries,
        ground_truth=gt_rows,
        has_reviewed_ground_truth=has_reviewed,
        synthetic_metrics=synth,
        findings=result.findings,
        primary_sidecar=result.primary_sidecar,
    )
    eval_payload = eval_report.to_dict(run_id=rid, video_id=vid, config_fingerprint=cfg_fp)

    try:
        write_contract_parquet(
            obs_table, obs_out, get_contract("track_observations", 1), contain_root=root
        )
        written.append(obs_out)
        write_contract_parquet(
            sum_table, sum_out, get_contract("track_summaries", 1), contain_root=root
        )
        written.append(sum_out)
        write_contract_parquet(
            life_table, life_out, get_contract("track_lifecycle", 1), contain_root=root
        )
        written.append(life_out)
    except Exception:  # noqa: BLE001
        return _fail(
            error_code="OUTPUT_WRITE_FAILED",
            exit_code=3,
            config_fingerprint=cfg_fp,
            cleanup=written,
        )

    completed = _utc_now()
    output_artifacts = {
        "track_observations": _artifact_meta(obs_out),
        "track_summaries": _artifact_meta(sum_out),
        "track_lifecycle": _artifact_meta(life_out),
    }
    receipt = dict(receipt_provisional)
    receipt["completed_at_utc"] = completed
    receipt["ground_truth_evaluation_status"] = eval_report.ground_truth_evaluation_status
    receipt["output_artifacts"] = output_artifacts
    receipt["input_artifacts"] = {
        "detections": {
            "path": str(det_path),
            "sha256": input_hashes["detections"],
            "size_bytes": (int(det_path.stat().st_size) if det_path.is_file() else 0),
        },
        "frames": {
            "path": str(frames_path),
            "sha256": input_hashes["frames"],
            "size_bytes": (int(frames_path.stat().st_size) if frames_path.is_file() else 0),
        },
        "analysis_windows": {
            "path": str(windows_path),
            "sha256": input_hashes["analysis_windows"],
            "size_bytes": (int(windows_path.stat().st_size) if windows_path.is_file() else 0),
        },
    }

    recounted = recount_receipt_from_tables(
        observations=result.observations,
        lifecycle=result.lifecycle,
        detections=det_rows,
    )
    if receipt["observation_counts"] != recounted["observation_counts"]:
        return _fail(
            error_code="RECEIPT_COUNT_MISMATCH",
            exit_code=3,
            config_fingerprint=cfg_fp,
            cleanup=written,
        )
    if receipt["track_counts"] != recounted["track_counts"]:
        return _fail(
            error_code="RECEIPT_COUNT_MISMATCH",
            exit_code=3,
            config_fingerprint=cfg_fp,
            cleanup=written,
        )

    try:
        validate_receipt_payload(receipt)
    except Exception:  # noqa: BLE001
        return _fail(
            error_code="RECEIPT_SCHEMA_INVALID",
            exit_code=1,
            config_fingerprint=cfg_fp,
            cleanup=written,
        )

    primary_payload = {
        "schema_version": 1,
        "run_id": rid,
        "video_id": vid,
        "config_fingerprint": cfg_fp,
        "note": "primary_ball_candidate is not a guarantee of true ball identity",
        "frames": result.primary_sidecar,
        "created_at_utc": completed,
    }

    quality = {
        "schema_version": 1,
        "run_id": rid,
        "video_id": vid,
        "config_fingerprint": cfg_fp,
        "policy_fingerprint": pol_fp,
        "input_hashes": input_hashes,
        "output_fingerprints": {k: v["sha256"] for k, v in output_artifacts.items()},
        "deterministic_output_fingerprint": hash_canonical_json(
            {
                "observations": result.observations,
                "lifecycle": result.lifecycle,
                "summaries": result.summaries,
                "primary": result.primary_sidecar,
            }
        ),
        "stats": {
            k: v
            for k, v in result.stats.items()
            if k != "review_samples"  # keep quality lean; samples in review only
        },
        "review_samples": result.stats.get("review_samples", []),
        "findings": result.findings,
        "ground_truth_evaluation_status": eval_report.ground_truth_evaluation_status,
        "created_at_utc": completed,
    }

    try:
        if config["output_policy"]["emit_primary_sidecar"]:
            write_json_record(primary_out, primary_payload, contain_root=root, overwrite=False)
            written.append(primary_out)
            output_artifacts["ball_primary_candidates"] = _artifact_meta(primary_out)
            receipt["output_artifacts"] = output_artifacts
        write_json_record(eval_out, eval_payload, contain_root=root, overwrite=False)
        written.append(eval_out)
        write_json_record(quality_out, quality, contain_root=root, overwrite=False)
        written.append(quality_out)
        write_json_record(receipt_out, receipt, contain_root=root, overwrite=False)
        written.append(receipt_out)
    except Exception:  # noqa: BLE001
        return _fail(
            error_code="JSON_WRITE_FAILED",
            exit_code=3,
            config_fingerprint=cfg_fp,
            cleanup=written,
        )

    summary = {
        "status": "succeeded",
        "run_id": rid,
        "video_id": vid,
        "track_counts": receipt["track_counts"],
        "observation_counts": receipt["observation_counts"],
        "detections_used": receipt["detections_used"],
        "unassigned_detection_count": receipt["unassigned_detection_count"],
        "primary_frames": result.stats["primary_frames"],
        "ambiguous_frames": result.stats["ambiguous_frames"],
        "no_candidate_frames": result.stats["no_candidate_frames"],
        "evaluation_status": eval_report.ground_truth_evaluation_status,
        "config_fingerprint": cfg_fp,
        "deterministic_output_fingerprint": quality["deterministic_output_fingerprint"],
        "findings": result.findings,
    }
    return BallTrackingServiceResult(
        accepted=True,
        exit_code=0,
        error_code=None,
        config_fingerprint=cfg_fp,
        observations_parquet=str(obs_out),
        summaries_parquet=str(sum_out),
        lifecycle_parquet=str(life_out),
        receipt_json=str(receipt_out),
        evaluation_json=str(eval_out),
        quality_json=str(quality_out),
        primary_sidecar_json=str(primary_out) if primary_out.exists() else None,
        summary=summary,
    )


__all__ = [
    "BallTrackingServiceError",
    "BallTrackingServiceResult",
    "run_ball_tracking",
]
