"""Stage 5E detection pipeline: fuse upstream artifacts, quality gates, atomic publish."""

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
from football_analytics.perception.detection_fusion import (
    DetectionFusionError,
    fuse_detection_bundle,
)
from football_analytics.perception.detection_pipeline_config import (
    detection_pipeline_config_fingerprint,
)
from football_analytics.perception.detection_quality import (
    NOT_EVALUATED_DETECTION,
    build_detection_review_queue,
    compute_role_counts,
    evaluate_detection_quality,
)
from football_analytics.perception.types import ProcessingStatus
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


class DetectionPipelineError(ValueError):
    """Detection integrate pipeline failure."""


@dataclass
class DetectionPipelineResult:
    accepted: bool
    exit_code: int
    detections_parquet: str | None
    detection_frame_status_parquet: str | None
    detection_attributes_parquet: str | None
    pipeline_receipt_json: str | None
    quality_report_json: str | None
    review_queue_json: str | None
    error_code: str | None
    quality_status: str | None
    config_fingerprint: str | None
    total_detection_count: int
    review_count: int

    def to_summary(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "exit_code": self.exit_code,
            "detections_parquet": self.detections_parquet,
            "detection_frame_status_parquet": self.detection_frame_status_parquet,
            "detection_attributes_parquet": self.detection_attributes_parquet,
            "pipeline_receipt_json": self.pipeline_receipt_json,
            "quality_report_json": self.quality_report_json,
            "review_queue_json": self.review_queue_json,
            "error_code": self.error_code,
            "quality_status": self.quality_status,
            "config_fingerprint": self.config_fingerprint,
            "total_detection_count": self.total_detection_count,
            "review_count": self.review_count,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _fail(
    *,
    error_code: str,
    exit_code: int = 1,
    config_fingerprint: str | None = None,
) -> DetectionPipelineResult:
    return DetectionPipelineResult(
        accepted=False,
        exit_code=exit_code,
        detections_parquet=None,
        detection_frame_status_parquet=None,
        detection_attributes_parquet=None,
        pipeline_receipt_json=None,
        quality_report_json=None,
        review_queue_json=None,
        error_code=error_code,
        quality_status=None,
        config_fingerprint=config_fingerprint,
        total_detection_count=0,
        review_count=0,
    )


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise DetectionPipelineError("json root must be object")
    return data


def _rows_to_table(rows: list[dict[str, Any]], contract_name: str) -> Any:
    schema = compile_arrow_schema(get_contract(contract_name, 1))
    return pa.Table.from_pylist(rows, schema=schema) if rows else schema.empty_table()


def _cleanup_tmpdir(tmp: Path | None) -> None:
    if tmp is None:
        return
    if tmp.exists() and tmp.is_dir() and not tmp.is_symlink():
        shutil.rmtree(tmp, ignore_errors=True)


def run_detection_integrate(
    *,
    human_detections: str,
    human_frame_status: str,
    human_attributes: str,
    human_receipt: str,
    ball_detections: str,
    ball_frame_status: str,
    ball_attributes: str,
    ball_receipt: str,
    role_attributes: str,
    role_receipt: str,
    output_dir: str,
    config: Mapping[str, Any],
    contain_root: Path | str | None = None,
    analysis_windows: str | None = None,
    frames: str | None = None,
    run_id: str | None = None,
    video_id: str | None = None,
    expected_source_sha: str | None = None,
    expected_timeline_fp: str | None = None,
) -> DetectionPipelineResult:
    """Fuse human/ball/role artifacts into one detection bundle with quality gates."""
    cfg_fp = detection_pipeline_config_fingerprint(config)
    started = _utc_now()
    tmp_dir: Path | None = None

    try:
        for label, raw in (
            ("human_detections", human_detections),
            ("human_frame_status", human_frame_status),
            ("human_attributes", human_attributes),
            ("human_receipt", human_receipt),
            ("ball_detections", ball_detections),
            ("ball_frame_status", ball_frame_status),
            ("ball_attributes", ball_attributes),
            ("ball_receipt", ball_receipt),
            ("role_attributes", role_attributes),
            ("role_receipt", role_receipt),
            ("output_dir", output_dir),
        ):
            reject_unsafe_path_string(raw, label=label)
        paths: dict[str, Path] = {
            "human_detections": require_absolute_path(human_detections, label="human_detections"),
            "human_frame_status": require_absolute_path(
                human_frame_status, label="human_frame_status"
            ),
            "human_attributes": require_absolute_path(human_attributes, label="human_attributes"),
            "human_receipt": require_absolute_path(human_receipt, label="human_receipt"),
            "ball_detections": require_absolute_path(ball_detections, label="ball_detections"),
            "ball_frame_status": require_absolute_path(
                ball_frame_status, label="ball_frame_status"
            ),
            "ball_attributes": require_absolute_path(ball_attributes, label="ball_attributes"),
            "ball_receipt": require_absolute_path(ball_receipt, label="ball_receipt"),
            "role_attributes": require_absolute_path(role_attributes, label="role_attributes"),
            "role_receipt": require_absolute_path(role_receipt, label="role_receipt"),
            "output_dir": require_absolute_path(output_dir, label="output_dir"),
        }
        aw_path: Path | None = None
        frames_path: Path | None = None
        if analysis_windows:
            reject_unsafe_path_string(analysis_windows, label="analysis_windows")
            aw_path = require_absolute_path(analysis_windows, label="analysis_windows")
        if frames:
            reject_unsafe_path_string(frames, label="frames")
            frames_path = require_absolute_path(frames, label="frames")
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
        for opt_path, label in ((aw_path, "analysis_windows"), (frames_path, "frames")):
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
    det_out = out / "detections.parquet"
    status_out = out / "detection_frame_status.parquet"
    attr_out = out / "detection_attributes.parquet"
    receipt_out = out / "detection_pipeline_receipt.json"
    quality_out = out / "detection_quality_report.json"
    review_out = out / "review_queue.json"
    for p in (det_out, status_out, attr_out, receipt_out, quality_out, review_out):
        if p.exists():
            return _fail(error_code="OVERWRITE_FORBIDDEN", exit_code=3, config_fingerprint=cfg_fp)

    try:
        h_det = read_contract_parquet(
            paths["human_detections"], get_contract("detections", 1), contain_root=root
        )
        h_status = read_contract_parquet(
            paths["human_frame_status"],
            get_contract("detection_frame_status", 1),
            contain_root=root,
        )
        h_attr = read_contract_parquet(
            paths["human_attributes"],
            get_contract("detection_attributes", 1),
            contain_root=root,
        )
        b_det = read_contract_parquet(
            paths["ball_detections"], get_contract("detections", 1), contain_root=root
        )
        b_status = read_contract_parquet(
            paths["ball_frame_status"],
            get_contract("detection_frame_status", 1),
            contain_root=root,
        )
        b_attr = read_contract_parquet(
            paths["ball_attributes"],
            get_contract("detection_attributes", 1),
            contain_root=root,
        )
        r_attr = read_contract_parquet(
            paths["role_attributes"],
            get_contract("detection_attributes", 1),
            contain_root=root,
        )
        h_rec = _load_json(paths["human_receipt"])
        b_rec = _load_json(paths["ball_receipt"])
        r_rec = _load_json(paths["role_receipt"])
        aw_table = None
        frames_table = None
        if aw_path is not None:
            aw_table = read_contract_parquet(
                aw_path, get_contract("analysis_windows", 1), contain_root=root
            )
        if frames_path is not None:
            frames_table = read_contract_parquet(
                frames_path, get_contract("frames", 1), contain_root=root
            )
            if expected_timeline_fp is None:
                expected_timeline_fp = sha256_file(frames_path)
    except Exception:  # noqa: BLE001
        return _fail(error_code="INPUT_READ_FAIL", exit_code=1, config_fingerprint=cfg_fp)

    rid = run_id
    vid = video_id
    try:
        if rid is not None:
            validate_run_id(rid)
        if vid is not None and not SAFE_ID_RE.fullmatch(vid):
            raise DetectionPipelineError("invalid video_id")
        fused = fuse_detection_bundle(
            human_detections=h_det,
            human_frame_status=h_status,
            human_attributes=h_attr,
            ball_detections=b_det,
            ball_frame_status=b_status,
            ball_attributes=b_attr,
            role_attributes=r_attr,
            human_receipt=h_rec,
            ball_receipt=b_rec,
            role_receipt=r_rec,
            config=config,
            frames=frames_table,
            analysis_windows=aw_table,
            expected_run_id=rid,
            expected_video_id=vid,
            expected_source_sha=expected_source_sha,
            expected_timeline_fp=expected_timeline_fp,
            validate=True,
        )
    except DetectionFusionError as exc:
        return _fail(error_code=exc.code, exit_code=1, config_fingerprint=cfg_fp)
    except Exception:  # noqa: BLE001
        return _fail(error_code="FUSION_ERROR", exit_code=1, config_fingerprint=cfg_fp)

    # Build counts from fused tables for receipt + quality.
    status_rows = fused.frame_status
    processed = sum(
        1
        for r in status_rows
        if r["processing_status"]
        in {
            ProcessingStatus.PROCESSED.value,
            ProcessingStatus.PROCESSED_NO_DETECTIONS.value,
        }
    )
    skipped = sum(
        1 for r in status_rows if r["processing_status"] == ProcessingStatus.SKIPPED.value
    )
    failed = sum(1 for r in status_rows if r["processing_status"] == ProcessingStatus.FAILED.value)
    no_det = sum(
        1
        for r in status_rows
        if r["processing_status"] == ProcessingStatus.PROCESSED_NO_DETECTIONS.value
    )
    eligible = sum(
        1
        for r in status_rows
        if r.get("eligibility") in {"eligible", "conditionally_eligible"}
        or r["processing_status"]
        in {
            ProcessingStatus.PROCESSED.value,
            ProcessingStatus.PROCESSED_NO_DETECTIONS.value,
            ProcessingStatus.SKIPPED.value,
        }
    )
    role_counts_raw = compute_role_counts(fused.attributes)
    role_counts = {k: v for k, v in role_counts_raw.items() if not k.startswith("__")}
    abstain = int(role_counts_raw.get("__abstention__", 0))

    receipt_counts = {
        "processed_frame_count": processed,
        "skipped_frame_count": skipped,
        "failed_frame_count": failed,
        "processed_no_detection_count": no_det,
        "total_detection_count": len(fused.detections),
        "human_detection_count": fused.human_detection_count,
        "ball_detection_count": fused.ball_detection_count,
    }

    quality = evaluate_detection_quality(
        detections=fused.detections,
        frame_status=status_rows,
        attributes=fused.attributes,
        config=config,
        receipt_counts=receipt_counts,
        has_reviewed_ground_truth=False,
    )
    if quality.status == "fail":
        return _fail(error_code="QUALITY_GATE_FAIL", exit_code=1, config_fingerprint=cfg_fp)

    review = build_detection_review_queue(
        attributes=fused.attributes,
        frame_status=status_rows,
        detections=fused.detections,
        config=config,
        quality=quality,
        run_id=fused.run_id,
        video_id=fused.video_id,
        policy_version=str(config.get("pipeline_version", "1")),
    )

    # Atomic publish via temp dir then rename individual files through writers.
    try:
        tmp_dir = out / f".tmp_fusion_{generate_run_id()[:12]}"
        tmp_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
        det_table = _rows_to_table(fused.detections, "detections")
        status_table = _rows_to_table(fused.frame_status, "detection_frame_status")
        attr_table = _rows_to_table(fused.attributes, "detection_attributes")

        tmp_det = tmp_dir / "detections.parquet"
        tmp_status = tmp_dir / "detection_frame_status.parquet"
        tmp_attr = tmp_dir / "detection_attributes.parquet"
        write_contract_parquet(
            det_table, tmp_det, get_contract("detections", 1), contain_root=root, overwrite=False
        )
        write_contract_parquet(
            status_table,
            tmp_status,
            get_contract("detection_frame_status", 1),
            contain_root=root,
            overwrite=False,
        )
        write_contract_parquet(
            attr_table,
            tmp_attr,
            get_contract("detection_attributes", 1),
            contain_root=root,
            overwrite=False,
        )

        # Move into final paths (no overwrite).
        for src, dst in (
            (tmp_det, det_out),
            (tmp_status, status_out),
            (tmp_attr, attr_out),
        ):
            if dst.exists():
                raise RecordError("overwrite forbidden during publish")
            src.replace(dst)

        output_hashes = {
            "detections_sha256": sha256_file(det_out),
            "detection_frame_status_sha256": sha256_file(status_out),
            "detection_attributes_sha256": sha256_file(attr_out),
            "detections_size_bytes": det_out.stat().st_size,
            "detection_frame_status_size_bytes": status_out.stat().st_size,
            "detection_attributes_size_bytes": attr_out.stat().st_size,
        }

        completed = _utc_now()
        source_sha = expected_source_sha
        if source_sha is None:
            source_sha = None
            for rec in (h_rec, b_rec):
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

        receipt = {
            "schema_version": 1,
            "receipt_id": f"pipe_{fused.run_id[:16]}",
            "run_id": fused.run_id,
            "video_id": fused.video_id,
            "pipeline_id": config["pipeline_id"],
            "pipeline_version": config["pipeline_version"],
            "config_fingerprint": cfg_fp,
            "source_video_sha256": source_sha,
            "timeline_fingerprint": timeline_fp,
            "human_config_fingerprint": h_rec.get("config_fingerprint"),
            "ball_config_fingerprint": b_rec.get("config_fingerprint"),
            "role_config_fingerprint": r_rec.get("config_fingerprint"),
            "started_at_utc": started,
            "completed_at_utc": completed,
            "status": "succeeded",
            "quality_gate_status": quality.status,
            "ground_truth_evaluation_status": quality.ground_truth_evaluation_status,
            "eligible_frame_count": eligible,
            "processed_frame_count": processed,
            "skipped_frame_count": skipped,
            "failed_frame_count": failed,
            "processed_no_detection_count": no_det,
            "total_detection_count": len(fused.detections),
            "human_detection_count": fused.human_detection_count,
            "ball_detection_count": fused.ball_detection_count,
            "role_counts": role_counts,
            "role_abstention_count": abstain,
            "review_required_count": len(review["items"]),
            "inputs": {
                "human_detections": paths["human_detections"].name,
                "ball_detections": paths["ball_detections"].name,
                "role_attributes": paths["role_attributes"].name,
                "human_receipt": paths["human_receipt"].name,
                "ball_receipt": paths["ball_receipt"].name,
                "role_receipt": paths["role_receipt"].name,
            },
            "outputs": {
                "detections": det_out.name,
                "detection_frame_status": status_out.name,
                "detection_attributes": attr_out.name,
                "detection_quality_report": quality_out.name,
                "review_queue": review_out.name,
            },
            "output_hashes": {k: str(v) for k, v in output_hashes.items()},
            "warnings": [{"code": f, "message": f} for f in quality.findings],
            "errors": [],
            "provenance": {
                "stage": "5E",
                "label": "detection_fusion",
                "cross_class_nms": False,
                "detections_contract_unchanged": True,
                "id_remap_count": len(fused.id_remap),
                "receipt_fingerprint": hash_canonical_json(
                    {
                        "config_fingerprint": cfg_fp,
                        "counts": receipt_counts,
                        "role_counts": role_counts,
                    }
                ),
            },
        }

        # Recalculate counts from written tables must match receipt.
        written_det = read_contract_parquet(
            det_out, get_contract("detections", 1), contain_root=root
        )
        written_status = read_contract_parquet(
            status_out, get_contract("detection_frame_status", 1), contain_root=root
        )
        written_attr = read_contract_parquet(
            attr_out, get_contract("detection_attributes", 1), contain_root=root
        )
        recalc_processed = sum(
            1
            for r in written_status.to_pylist()
            if r["processing_status"]
            in {
                ProcessingStatus.PROCESSED.value,
                ProcessingStatus.PROCESSED_NO_DETECTIONS.value,
            }
        )
        if (
            written_det.num_rows != receipt["total_detection_count"]
            or recalc_processed != receipt["processed_frame_count"]
            or written_attr.num_rows != written_det.num_rows
        ):
            # Cleanup published artifacts on integrity failure.
            for pub in (det_out, status_out, attr_out):
                if pub.exists():
                    pub.unlink()
            raise DetectionPipelineError("RECEIPT_COUNT_MISMATCH")

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
        if config["output_policy"]["write_pipeline_receipt"]:
            write_json_record(receipt_out, receipt, contain_root=root, overwrite=False)
    except Exception:  # noqa: BLE001
        _cleanup_tmpdir(tmp_dir)
        for pub in (det_out, status_out, attr_out, receipt_out, quality_out, review_out):
            if pub.exists() and pub.is_file() and not pub.is_symlink():
                with contextlib.suppress(OSError):
                    pub.unlink()
        return _fail(error_code="WRITE_FAIL", exit_code=1, config_fingerprint=cfg_fp)
    finally:
        _cleanup_tmpdir(tmp_dir)

    return DetectionPipelineResult(
        accepted=True,
        exit_code=0,
        detections_parquet=str(det_out),
        detection_frame_status_parquet=str(status_out),
        detection_attributes_parquet=str(attr_out),
        pipeline_receipt_json=str(receipt_out),
        quality_report_json=str(quality_out),
        review_queue_json=str(review_out),
        error_code=None,
        quality_status=quality.status,
        config_fingerprint=cfg_fp,
        total_detection_count=len(fused.detections),
        review_count=len(review["items"]),
    )


def ensure_run_id(run_id: str | None) -> str:
    if run_id is None:
        return generate_run_id()
    validate_run_id(run_id)
    return run_id


__all__ = [
    "DetectionPipelineError",
    "DetectionPipelineResult",
    "run_detection_integrate",
    "ensure_run_id",
    "NOT_EVALUATED_DETECTION",
]
