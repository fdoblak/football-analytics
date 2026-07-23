"""Stage 6D tracking pipeline: fuse human+ball tracks, quality gates, atomic publish."""

from __future__ import annotations

import contextlib
import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa

from football_analytics.core.hashing import hash_canonical_json, sha256_file
from football_analytics.core.records import RecordError, write_json_record
from football_analytics.core.run_id import generate_run_id, validate_run_id
from football_analytics.data.compiler import compile_arrow_schema, get_contract
from football_analytics.data.parquet import read_contract_parquet, write_contract_parquet
from football_analytics.data.types import SAFE_ID_RE
from football_analytics.tracking.evaluation import NOT_EVALUATED_TRACKING
from football_analytics.tracking.tracking_fusion import (
    TrackingFusionError,
    fuse_tracking_bundle,
)
from football_analytics.tracking.tracking_pipeline_config import (
    tracking_pipeline_config_fingerprint,
)
from football_analytics.tracking.tracking_quality import (
    NOT_EVALUATED_BALL_TRACKING,
    NOT_EVALUATED_HUMAN_TRACKING,
    build_tracking_review_queue,
    evaluate_tracking_quality,
)
from football_analytics.utils.archive_safety import (
    assert_contained,
    assert_not_dangerous_operation_root,
    resolve_strict,
)
from football_analytics.video.types import VideoSourceError
from football_analytics.video.validation import (
    reject_unsafe_path_string,
    require_absolute_path,
)


class TrackingPipelineError(ValueError):
    """Tracking integrate pipeline failure."""


@dataclass
class TrackingPipelineResult:
    accepted: bool
    exit_code: int
    track_observations_parquet: str | None
    track_summaries_parquet: str | None
    track_lifecycle_parquet: str | None
    primary_sidecar_json: str | None
    pipeline_receipt_json: str | None
    quality_report_json: str | None
    review_queue_json: str | None
    bundle_manifest_json: str | None
    error_code: str | None
    quality_status: str | None
    config_fingerprint: str | None
    total_track_count: int
    review_count: int

    def to_summary(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "exit_code": self.exit_code,
            "track_observations_parquet": self.track_observations_parquet,
            "track_summaries_parquet": self.track_summaries_parquet,
            "track_lifecycle_parquet": self.track_lifecycle_parquet,
            "primary_sidecar_json": self.primary_sidecar_json,
            "pipeline_receipt_json": self.pipeline_receipt_json,
            "quality_report_json": self.quality_report_json,
            "review_queue_json": self.review_queue_json,
            "bundle_manifest_json": self.bundle_manifest_json,
            "error_code": self.error_code,
            "quality_status": self.quality_status,
            "config_fingerprint": self.config_fingerprint,
            "total_track_count": self.total_track_count,
            "review_count": self.review_count,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _fail(
    *,
    error_code: str,
    exit_code: int = 1,
    config_fingerprint: str | None = None,
) -> TrackingPipelineResult:
    return TrackingPipelineResult(
        accepted=False,
        exit_code=exit_code,
        track_observations_parquet=None,
        track_summaries_parquet=None,
        track_lifecycle_parquet=None,
        primary_sidecar_json=None,
        pipeline_receipt_json=None,
        quality_report_json=None,
        review_queue_json=None,
        bundle_manifest_json=None,
        error_code=error_code,
        quality_status=None,
        config_fingerprint=config_fingerprint,
        total_track_count=0,
        review_count=0,
    )


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TrackingPipelineError("json root must be object")
    return data


def _rows_to_table(rows: list[dict[str, Any]], contract_name: str) -> Any:
    schema = compile_arrow_schema(get_contract(contract_name, 1))
    return pa.Table.from_pylist(rows, schema=schema) if rows else schema.empty_table()


def _cleanup_tmpdir(tmp: Path | None) -> None:
    if tmp is None:
        return
    if tmp.exists() and tmp.is_dir() and not tmp.is_symlink():
        shutil.rmtree(tmp, ignore_errors=True)


def _artifact_meta(path: Path) -> dict[str, Any]:
    return {"path": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def run_tracking_integrate(
    *,
    detections: str,
    detection_attributes: str,
    detection_receipt: str,
    human_observations: str,
    human_summaries: str,
    human_lifecycle: str,
    human_receipt: str,
    ball_observations: str,
    ball_summaries: str,
    ball_lifecycle: str,
    ball_receipt: str,
    output_dir: str,
    config: Mapping[str, Any],
    contain_root: Path | str | None = None,
    frames: str | None = None,
    analysis_windows: str | None = None,
    ball_primary_sidecar: str | None = None,
    run_id: str | None = None,
    video_id: str | None = None,
    expected_source_sha: str | None = None,
    expected_timeline_fp: str | None = None,
    expected_detection_fp: str | None = None,
    expected_analysis_window_fp: str | None = None,
) -> TrackingPipelineResult:
    """Fuse human/ball tracking artifacts into one tracking bundle with quality gates."""
    cfg_fp = tracking_pipeline_config_fingerprint(config)
    started = _utc_now()
    tmp_dir: Path | None = None

    try:
        required = (
            ("detections", detections),
            ("detection_attributes", detection_attributes),
            ("detection_receipt", detection_receipt),
            ("human_observations", human_observations),
            ("human_summaries", human_summaries),
            ("human_lifecycle", human_lifecycle),
            ("human_receipt", human_receipt),
            ("ball_observations", ball_observations),
            ("ball_summaries", ball_summaries),
            ("ball_lifecycle", ball_lifecycle),
            ("ball_receipt", ball_receipt),
            ("output_dir", output_dir),
        )
        for label, raw in required:
            reject_unsafe_path_string(raw, label=label)
        paths: dict[str, Path] = {
            label: require_absolute_path(raw, label=label) for label, raw in required
        }
        aw_path: Path | None = None
        frames_path: Path | None = None
        primary_path: Path | None = None
        if analysis_windows:
            reject_unsafe_path_string(analysis_windows, label="analysis_windows")
            aw_path = require_absolute_path(analysis_windows, label="analysis_windows")
        if frames:
            reject_unsafe_path_string(frames, label="frames")
            frames_path = require_absolute_path(frames, label="frames")
        if ball_primary_sidecar:
            reject_unsafe_path_string(ball_primary_sidecar, label="ball_primary_sidecar")
            primary_path = require_absolute_path(ball_primary_sidecar, label="ball_primary_sidecar")
    except (VideoSourceError, Exception):  # noqa: BLE001
        return _fail(error_code="UNSAFE_PATH", exit_code=3, config_fingerprint=cfg_fp)

    root = Path(contain_root) if contain_root is not None else Path(str(config["runtime_root"]))
    try:
        root = require_absolute_path(str(root), label="contain_root")
        assert_not_dangerous_operation_root(root)
        out = paths["output_dir"]
        out_resolved = resolve_strict(out) if out.exists() else out.resolve()
        assert_contained(out_resolved, resolve_strict(root), label="output_dir")
        for key, path_item in paths.items():
            if key == "output_dir":
                continue
            if path_item.is_symlink():
                raise VideoSourceError(f"{key} must not be a symlink")
            if not path_item.is_file():
                raise VideoSourceError(f"{key} missing")
            assert_contained(resolve_strict(path_item), resolve_strict(root), label=key)
        for opt_path, label in (
            (aw_path, "analysis_windows"),
            (frames_path, "frames"),
            (primary_path, "ball_primary_sidecar"),
        ):
            if opt_path is None:
                continue
            if opt_path.is_symlink() or not opt_path.is_file():
                raise VideoSourceError(f"{label} invalid")
            assert_contained(resolve_strict(opt_path), resolve_strict(root), label=label)
    except Exception:  # noqa: BLE001
        return _fail(error_code="CONTAINMENT_FAILURE", exit_code=3, config_fingerprint=cfg_fp)

    if config.get("overwrite_allowed") is not False:
        return _fail(error_code="OVERWRITE_POLICY", exit_code=2, config_fingerprint=cfg_fp)

    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    obs_out = out / "track_observations.parquet"
    sum_out = out / "track_summaries.parquet"
    life_out = out / "track_lifecycle.parquet"
    primary_out = out / "ball_primary_candidates.json"
    receipt_out = out / "tracking_pipeline_receipt.json"
    quality_out = out / "tracking_quality_report.json"
    review_out = out / "review_queue.json"
    manifest_out = out / "tracking_bundle_manifest.json"
    published = [
        obs_out,
        sum_out,
        life_out,
        primary_out,
        receipt_out,
        quality_out,
        review_out,
        manifest_out,
    ]
    for p in published:
        if p.exists():
            return _fail(error_code="OVERWRITE_FORBIDDEN", exit_code=3, config_fingerprint=cfg_fp)

    try:
        det = read_contract_parquet(
            paths["detections"], get_contract("detections", 1), contain_root=root
        )
        attrs = read_contract_parquet(
            paths["detection_attributes"],
            get_contract("detection_attributes", 1),
            contain_root=root,
        )
        h_obs = read_contract_parquet(
            paths["human_observations"], get_contract("track_observations", 1), contain_root=root
        )
        h_sum = read_contract_parquet(
            paths["human_summaries"], get_contract("track_summaries", 1), contain_root=root
        )
        h_life = read_contract_parquet(
            paths["human_lifecycle"], get_contract("track_lifecycle", 1), contain_root=root
        )
        b_obs = read_contract_parquet(
            paths["ball_observations"], get_contract("track_observations", 1), contain_root=root
        )
        b_sum = read_contract_parquet(
            paths["ball_summaries"], get_contract("track_summaries", 1), contain_root=root
        )
        b_life = read_contract_parquet(
            paths["ball_lifecycle"], get_contract("track_lifecycle", 1), contain_root=root
        )
        det_rec = _load_json(paths["detection_receipt"])
        h_rec = _load_json(paths["human_receipt"])
        b_rec = _load_json(paths["ball_receipt"])
        aw_table = None
        frames_table = None
        if aw_path is not None:
            aw_table = read_contract_parquet(
                aw_path, get_contract("analysis_windows", 1), contain_root=root
            )
            if expected_analysis_window_fp is None:
                expected_analysis_window_fp = sha256_file(aw_path)
        if frames_path is not None:
            frames_table = read_contract_parquet(
                frames_path, get_contract("frames", 1), contain_root=root
            )
            if expected_timeline_fp is None:
                expected_timeline_fp = sha256_file(frames_path)
        primary_frames: list[dict[str, Any]] = []
        if primary_path is not None:
            primary_payload = _load_json(primary_path)
            frames_list = primary_payload.get("frames")
            if isinstance(frames_list, list):
                primary_frames = [dict(x) for x in frames_list if isinstance(x, dict)]
        if expected_detection_fp is None:
            expected_detection_fp = sha256_file(paths["detections"])
    except Exception:  # noqa: BLE001
        return _fail(error_code="INPUT_READ_FAIL", exit_code=1, config_fingerprint=cfg_fp)

    rid = run_id
    vid = video_id
    try:
        if rid is not None:
            validate_run_id(rid)
        if vid is not None and not SAFE_ID_RE.fullmatch(vid):
            raise TrackingPipelineError("invalid video_id")
        fused = fuse_tracking_bundle(
            human_observations=h_obs,
            human_summaries=h_sum,
            human_lifecycle=h_life,
            ball_observations=b_obs,
            ball_summaries=b_sum,
            ball_lifecycle=b_life,
            primary_sidecar=primary_frames,
            detections=det,
            detection_attributes=attrs,
            frames=frames_table,
            analysis_windows=aw_table,
            detection_receipt=det_rec,
            human_receipt=h_rec,
            ball_receipt=b_rec,
            config=config,
            expected_run_id=rid,
            expected_video_id=vid,
            expected_source_sha=expected_source_sha,
            expected_timeline_fp=expected_timeline_fp,
            expected_detection_fp=expected_detection_fp,
            expected_analysis_window_fp=expected_analysis_window_fp,
            validate=True,
        )
    except TrackingFusionError as exc:
        return _fail(error_code=exc.code, exit_code=1, config_fingerprint=cfg_fp)
    except Exception:  # noqa: BLE001
        return _fail(error_code="FUSION_ERROR", exit_code=1, config_fingerprint=cfg_fp)

    quality = evaluate_tracking_quality(
        observations=fused.observations,
        summaries=fused.summaries,
        lifecycle=fused.lifecycle,
        detection_attributes=attrs.to_pylist(),
        primary_sidecar=fused.primary_sidecar,
        frames=frames_table.to_pylist() if frames_table is not None else None,
        analysis_windows=aw_table.to_pylist() if aw_table is not None else None,
        config=config,
        receipt_counts=fused.counts,
        has_reviewed_ground_truth=False,
    )
    if quality.status == "fail":
        return _fail(error_code="QUALITY_GATE_FAIL", exit_code=1, config_fingerprint=cfg_fp)

    review = build_tracking_review_queue(
        observations=fused.observations,
        lifecycle=fused.lifecycle,
        detection_attributes=attrs.to_pylist(),
        primary_sidecar=fused.primary_sidecar,
        config=config,
        quality=quality,
        run_id=fused.run_id,
        video_id=fused.video_id,
        policy_version=str(config.get("pipeline_version", "1")),
    )

    try:
        tmp_dir = out / f".tmp_trk_fusion_{generate_run_id()[:12]}"
        tmp_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
        obs_table = _rows_to_table(fused.observations, "track_observations")
        sum_table = _rows_to_table(fused.summaries, "track_summaries")
        life_table = _rows_to_table(fused.lifecycle, "track_lifecycle")

        tmp_obs = tmp_dir / "track_observations.parquet"
        tmp_sum = tmp_dir / "track_summaries.parquet"
        tmp_life = tmp_dir / "track_lifecycle.parquet"
        write_contract_parquet(
            obs_table,
            tmp_obs,
            get_contract("track_observations", 1),
            contain_root=root,
            overwrite=False,
        )
        write_contract_parquet(
            sum_table,
            tmp_sum,
            get_contract("track_summaries", 1),
            contain_root=root,
            overwrite=False,
        )
        write_contract_parquet(
            life_table,
            tmp_life,
            get_contract("track_lifecycle", 1),
            contain_root=root,
            overwrite=False,
        )

        for src, dst in ((tmp_obs, obs_out), (tmp_sum, sum_out), (tmp_life, life_out)):
            if dst.exists():
                raise RecordError("overwrite forbidden during publish")
            src.replace(dst)

        output_hashes = {
            "track_observations_sha256": sha256_file(obs_out),
            "track_summaries_sha256": sha256_file(sum_out),
            "track_lifecycle_sha256": sha256_file(life_out),
            "track_observations_size_bytes": obs_out.stat().st_size,
            "track_summaries_size_bytes": sum_out.stat().st_size,
            "track_lifecycle_size_bytes": life_out.stat().st_size,
        }

        completed = _utc_now()
        source_sha = expected_source_sha
        if source_sha is None:
            for rec in (det_rec, h_rec, b_rec):
                for key in ("source_video_sha256", "source_sha256"):
                    if rec.get(key):
                        source_sha = str(rec[key]).lower()
                        break
                    arts = rec.get("artifacts")
                    if isinstance(arts, dict) and arts.get(key):
                        source_sha = str(arts[key]).lower()
                        break
                if source_sha:
                    break

        timeline_fp = expected_timeline_fp
        if timeline_fp is None and frames_path is not None:
            timeline_fp = sha256_file(frames_path)
        detection_fp = expected_detection_fp or sha256_file(paths["detections"])
        window_fp = expected_analysis_window_fp
        if window_fp is None and aw_path is not None:
            window_fp = sha256_file(aw_path)

        primary_payload = {
            "schema_version": 1,
            "run_id": fused.run_id,
            "video_id": fused.video_id,
            "config_fingerprint": cfg_fp,
            "note": "primary_ball_candidate is not a guarantee of true ball identity",
            "frames": fused.primary_sidecar,
            "created_at_utc": completed,
        }

        if config["output_policy"]["emit_primary_sidecar"]:
            write_json_record(primary_out, primary_payload, contain_root=root, overwrite=False)
            output_hashes["ball_primary_candidates_sha256"] = sha256_file(primary_out)
            output_hashes["ball_primary_candidates_size_bytes"] = primary_out.stat().st_size

        # Recount from written tables must match receipt.
        written_obs = read_contract_parquet(
            obs_out, get_contract("track_observations", 1), contain_root=root
        )
        written_sum = read_contract_parquet(
            sum_out, get_contract("track_summaries", 1), contain_root=root
        )
        written_life = read_contract_parquet(
            life_out, get_contract("track_lifecycle", 1), contain_root=root
        )
        recalc_observed = sum(
            1 for r in written_obs.to_pylist() if r["observation_state"] == "observed"
        )
        if (
            written_obs.num_rows != len(fused.observations)
            or written_sum.num_rows != len(fused.summaries)
            or written_life.num_rows != len(fused.lifecycle)
            or recalc_observed != fused.counts["observed_count"]
        ):
            for pub in (obs_out, sum_out, life_out, primary_out):
                if pub.exists():
                    pub.unlink()
            raise TrackingPipelineError("RECEIPT_COUNT_MISMATCH")

        receipt = {
            "schema_version": 1,
            "receipt_id": f"trk_pipe_{fused.run_id[:16]}",
            "run_id": fused.run_id,
            "video_id": fused.video_id,
            "pipeline_id": config["pipeline_id"],
            "pipeline_version": config["pipeline_version"],
            "config_fingerprint": cfg_fp,
            "source_video_sha256": source_sha,
            "timeline_fingerprint": timeline_fp,
            "detection_bundle_fingerprint": detection_fp,
            "analysis_window_fingerprint": window_fp,
            "human_config_fingerprint": h_rec.get("config_fingerprint"),
            "ball_config_fingerprint": b_rec.get("config_fingerprint"),
            "detection_config_fingerprint": det_rec.get("config_fingerprint"),
            "started_at_utc": started,
            "completed_at_utc": completed,
            "status": "succeeded",
            "quality_gate_status": quality.status,
            "ground_truth_evaluation_status": quality.ground_truth_evaluation_status,
            "human_input_detection_count": fused.counts["human_input_detection_count"],
            "ball_input_detection_count": fused.counts["ball_input_detection_count"],
            "assigned_detection_count": fused.counts["assigned_detection_count"],
            "unassigned_detection_count": fused.counts["unassigned_detection_count"],
            "human_track_count": fused.human_track_count,
            "ball_track_count": fused.ball_track_count,
            "total_track_count": fused.counts["total_track_count"],
            "tentative_count": fused.counts["tentative_count"],
            "confirmed_count": fused.counts["confirmed_count"],
            "lost_count": fused.counts["lost_count"],
            "terminated_count": fused.counts["terminated_count"],
            "observed_count": fused.counts["observed_count"],
            "predicted_count": fused.counts["predicted_count"],
            "interpolated_count": fused.counts["interpolated_count"],
            "fragmentation_count": int(
                round(
                    float(quality.metrics.get("track_fragmentation", 0))
                    * fused.counts["total_track_count"]
                )
            ),
            "recovery_count": fused.counts["confirmed_count"],
            "cross_cut_violation_count": fused.counts["cross_cut_violation_count"],
            "invalid_fk_count": fused.counts["invalid_fk_count"],
            "duplicate_count": fused.counts["duplicate_count"],
            "role_abstention_count": quality.metrics.get("role_abstention_count", 0),
            "review_required_count": len(review["items"]),
            "primary_ball_frames": fused.counts["primary_ball_frames"],
            "ambiguous_ball_frames": fused.counts["ambiguous_ball_frames"],
            "no_candidate_ball_frames": fused.counts["no_candidate_ball_frames"],
            "inputs": {
                "detections": paths["detections"].name,
                "detection_attributes": paths["detection_attributes"].name,
                "human_observations": paths["human_observations"].name,
                "ball_observations": paths["ball_observations"].name,
                "human_receipt": paths["human_receipt"].name,
                "ball_receipt": paths["ball_receipt"].name,
                "detection_receipt": paths["detection_receipt"].name,
            },
            "outputs": {
                "track_observations": obs_out.name,
                "track_summaries": sum_out.name,
                "track_lifecycle": life_out.name,
                "ball_primary_candidates": primary_out.name if primary_out.exists() else None,
                "tracking_quality_report": quality_out.name,
                "review_queue": review_out.name,
                "tracking_bundle_manifest": manifest_out.name,
            },
            "output_hashes": {k: str(v) for k, v in output_hashes.items()},
            "warnings": [{"code": f, "message": f} for f in quality.findings],
            "errors": [],
            "provenance": {
                "stage": "6D",
                "label": "tracking_fusion",
                "track_id_is_player_identity": False,
                "no_reid": True,
                "no_human_ball_relationship_table": True,
                "primary_ball_not_identity": True,
                "id_remap_count": len(fused.track_id_remap),
                "human_not_evaluated_code": NOT_EVALUATED_HUMAN_TRACKING,
                "ball_not_evaluated_code": NOT_EVALUATED_BALL_TRACKING,
                "tracking_not_evaluated_code": NOT_EVALUATED_TRACKING,
                "receipt_fingerprint": hash_canonical_json(
                    {
                        "config_fingerprint": cfg_fp,
                        "counts": fused.counts,
                        "quality_status": quality.status,
                    }
                ),
            },
        }

        manifest = {
            "schema_version": 1,
            "run_id": fused.run_id,
            "video_id": fused.video_id,
            "pipeline_id": config["pipeline_id"],
            "pipeline_version": config["pipeline_version"],
            "config_fingerprint": cfg_fp,
            "source_video_sha256": source_sha,
            "timeline_fingerprint": timeline_fp,
            "detection_bundle_fingerprint": detection_fp,
            "analysis_window_fingerprint": window_fp,
            "artifacts": {
                "track_observations": _artifact_meta(obs_out),
                "track_summaries": _artifact_meta(sum_out),
                "track_lifecycle": _artifact_meta(life_out),
            },
            "created_at_utc": completed,
        }
        if primary_out.exists():
            manifest["artifacts"]["ball_primary_candidates"] = _artifact_meta(primary_out)

        if config["output_policy"]["write_quality_report"]:
            write_json_record(
                quality_out,
                quality.to_dict(
                    run_id=fused.run_id, video_id=fused.video_id, config_fingerprint=cfg_fp
                ),
                contain_root=root,
                overwrite=False,
            )
        if config["output_policy"]["write_review_queue"]:
            write_json_record(review_out, review, contain_root=root, overwrite=False)
        if config["output_policy"]["write_bundle_manifest"]:
            write_json_record(manifest_out, manifest, contain_root=root, overwrite=False)
        if config["output_policy"]["write_pipeline_receipt"]:
            write_json_record(receipt_out, receipt, contain_root=root, overwrite=False)
    except Exception:  # noqa: BLE001
        _cleanup_tmpdir(tmp_dir)
        for pub in published:
            if pub.exists() and pub.is_file() and not pub.is_symlink():
                with contextlib.suppress(OSError):
                    pub.unlink()
        return _fail(error_code="WRITE_FAIL", exit_code=1, config_fingerprint=cfg_fp)
    finally:
        _cleanup_tmpdir(tmp_dir)

    return TrackingPipelineResult(
        accepted=True,
        exit_code=0,
        track_observations_parquet=str(obs_out),
        track_summaries_parquet=str(sum_out),
        track_lifecycle_parquet=str(life_out),
        primary_sidecar_json=str(primary_out) if primary_out.exists() else None,
        pipeline_receipt_json=str(receipt_out),
        quality_report_json=str(quality_out),
        review_queue_json=str(review_out),
        bundle_manifest_json=str(manifest_out) if manifest_out.exists() else None,
        error_code=None,
        quality_status=quality.status,
        config_fingerprint=cfg_fp,
        total_track_count=fused.counts["total_track_count"],
        review_count=len(review["items"]),
    )


def ensure_run_id(run_id: str | None) -> str:
    if run_id is None:
        return generate_run_id()
    validate_run_id(run_id)
    return run_id


__all__ = [
    "TrackingPipelineError",
    "TrackingPipelineResult",
    "run_tracking_integrate",
    "ensure_run_id",
    "NOT_EVALUATED_TRACKING",
]
