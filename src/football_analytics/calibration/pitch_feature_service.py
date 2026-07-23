"""Stage 8B pitch keypoint/line detection service (no Stage 8C homography)."""

from __future__ import annotations

import json
import math
import shutil
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import yaml

from football_analytics.calibration.features import feature_row, validate_feature_rows
from football_analytics.calibration.pitch_feature_adapter import (
    NbjwHrnetPitchFeatureAdapter,
    PitchFeatureAdapterError,
    verify_weight_file,
)
from football_analytics.calibration.pitch_feature_config import pitch_feature_config_fingerprint
from football_analytics.calibration.pitch_feature_evaluation import (
    NOT_EVALUATED_PITCH_FEATURES,
    evaluate_pitch_features,
)
from football_analytics.calibration.pitch_feature_mapping import (
    keypoint_mapping,
    line_mapping,
)
from football_analytics.calibration.pitch_feature_postprocess import (
    accepted_keypoints,
    accepted_lines,
)
from football_analytics.calibration.types import (
    CONTRACT_VERSION,
    FeatureStatus,
    FeatureType,
    Suitability,
)
from football_analytics.core.hashing import sha256_file
from football_analytics.core.records import write_json_record
from football_analytics.core.run_id import generate_run_id, validate_run_id
from football_analytics.data.compiler import compile_arrow_schema, get_contract
from football_analytics.data.parquet import write_contract_parquet
from football_analytics.data.types import SAFE_ID_RE
from football_analytics.video.validation import (
    assert_safe_output_root,
    reject_unsafe_path_string,
    require_absolute_path,
)


class PitchFeatureServiceError(RuntimeError):
    """Pitch feature service failure."""


@dataclass
class PitchFeatureServiceResult:
    accepted: bool
    exit_code: int
    error_code: str | None
    config_fingerprint: str
    features_parquet: str | None
    receipt_json: str | None
    quality_json: str | None
    evaluation_json: str | None
    frame_status_json: str | None
    summary: Mapping[str, Any]

    def to_summary(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "exit_code": self.exit_code,
            "error_code": self.error_code,
            "config_fingerprint": self.config_fingerprint,
            "features_parquet": self.features_parquet,
            "receipt_json": self.receipt_json,
            "quality_json": self.quality_json,
            "evaluation_json": self.evaluation_json,
            "frame_status_json": self.frame_status_json,
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
) -> PitchFeatureServiceResult:
    if cleanup:
        for p in cleanup:
            try:
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
            except OSError:
                pass
    return PitchFeatureServiceResult(
        accepted=False,
        exit_code=exit_code,
        error_code=error_code,
        config_fingerprint=config_fingerprint,
        features_parquet=None,
        receipt_json=None,
        quality_json=None,
        evaluation_json=None,
        frame_status_json=None,
        summary={"status": "failed", "error_code": error_code},
    )


def _rows_to_table(rows: list[dict[str, Any]], contract_name: str) -> Any:
    contract = get_contract(contract_name, 1)
    schema = compile_arrow_schema(contract)
    if not rows:
        return schema.empty_table()
    return pa.Table.from_pylist(rows, schema=schema)


def _load_model_registry_entry(registry_path: Path, model_id: str) -> dict[str, Any]:
    raw = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise PitchFeatureServiceError("MODEL_REGISTRY_INVALID")
    models = raw.get("models")
    if not isinstance(models, list):
        raise PitchFeatureServiceError("MODEL_REGISTRY_INVALID")
    for item in models:
        if isinstance(item, Mapping) and item.get("id") == model_id:
            return dict(item)
    raise PitchFeatureServiceError(f"MODEL_NOT_AVAILABLE:{model_id}")


def _window_eligible(
    window: Mapping[str, Any] | None, *, config: Mapping[str, Any]
) -> tuple[bool, str]:
    if window is None:
        return False, "no_analysis_window"
    elig = str(window.get("calibration_eligibility", "")).lower()
    if config["routing"]["require_calibration_eligible"] and elig not in {
        "eligible",
        "suitable",
    }:
        return False, f"calibration_eligibility={elig or 'missing'}"
    if config["routing"].get("reject_graphics_replay_closeup", True):
        graphics = str(window.get("graphics_status", "clean")).lower()
        playability = str(window.get("playability", "playable")).lower()
        view = str(window.get("view_class", window.get("camera_view", "main_broadcast"))).lower()
        if graphics in {"graphics", "replay_overlay", "heavy_graphics"}:
            return False, "graphics"
        if playability in {"replay", "non_playable", "not_playable"}:
            return False, "replay_or_non_playable"
        if view in {"close_up", "close-up", "crowd", "bench", "tunnel"}:
            return False, f"view={view}"
    return True, "eligible"


def _feature_rows_from_inference(
    *,
    run_id: str,
    video_id: str,
    frame_index: int,
    video_time_us: int,
    inference: Any,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    stats = {
        "keypoints_accepted": 0,
        "lines_accepted": 0,
        "keypoints_rejected": 0,
        "lines_rejected": 0,
        "mapped": 0,
        "unmapped": 0,
        "out_of_bounds": 0,
        "duplicates": 0,
    }
    kp_model_ref = f"{config['kp_model_registry_id']}@{inference.kp_model_sha256[:12]}"
    line_model_ref = f"{config['lines_model_registry_id']}@{inference.lines_model_sha256[:12]}"
    tf_fp = inference.transform.fingerprint()

    for kp in inference.keypoints:
        mapping = keypoint_mapping(kp.channel_index)
        if kp.rejected:
            stats["keypoints_rejected"] += 1
            if kp.reason == "out_of_bounds":
                stats["out_of_bounds"] += 1
            if kp.reason == "duplicate":
                stats["duplicates"] += 1
            continue
        stats["keypoints_accepted"] += 1
        if mapping.mapped:
            stats["mapped"] += 1
        else:
            stats["unmapped"] += 1
        status = FeatureStatus.MATCHED.value if mapping.mapped else FeatureStatus.UNMATCHED.value
        prov = {
            "channel_index": kp.channel_index,
            "source_name": mapping.source_name,
            "raw_score": kp.score,
            "model_space": {"x": kp.x_model, "y": kp.y_model},
            "preprocess_fingerprint": tf_fp,
            "model_sha256": inference.kp_model_sha256,
            "device": inference.device,
        }
        rows.append(
            feature_row(
                run_id=run_id,
                video_id=video_id,
                frame_index=frame_index,
                video_time_us=video_time_us,
                feature_id=f"f{frame_index:04d}_{mapping.sanitized_id}",
                feature_type=FeatureType.KEYPOINT.value,
                image_x=kp.x_source,
                image_y=kp.y_source,
                canonical_pitch_feature_id=mapping.canonical_pitch_feature_id,
                score=float(kp.score) if math.isfinite(kp.score) else None,
                confidence=None,
                source="sv_kp",
                model_ref=kp_model_ref,
                suitability=Suitability.MARGINAL.value,
                status=status,
                manual_review_required=True,
                reason_codes=["stage8b_detection_baseline"],
                quality_flags=["confidence_null", "evaluation_only"],
                provenance_json=json.dumps(prov, sort_keys=True, separators=(",", ":")),
            )
        )

    for ln in inference.lines:
        mapping = line_mapping(ln.channel_index)
        if ln.rejected:
            stats["lines_rejected"] += 1
            if ln.reason == "out_of_bounds":
                stats["out_of_bounds"] += 1
            if ln.reason == "duplicate":
                stats["duplicates"] += 1
            continue
        stats["lines_accepted"] += 1
        if mapping.mapped:
            stats["mapped"] += 1
        else:
            stats["unmapped"] += 1
        status = FeatureStatus.MATCHED.value if mapping.mapped else FeatureStatus.UNMATCHED.value
        prov = {
            "channel_index": ln.channel_index,
            "source_name": mapping.source_name,
            "raw_score": ln.score,
            "model_space": {
                "x1": ln.x1_model,
                "y1": ln.y1_model,
                "x2": ln.x2_model,
                "y2": ln.y2_model,
            },
            "length_source_px": ln.length_source,
            "preprocess_fingerprint": tf_fp,
            "model_sha256": inference.lines_model_sha256,
            "device": inference.device,
            "note": (
                "trailing_space_on_goal_left_post_left"
                if mapping.source_name.endswith(" ")
                else None
            ),
        }
        rows.append(
            feature_row(
                run_id=run_id,
                video_id=video_id,
                frame_index=frame_index,
                video_time_us=video_time_us,
                feature_id=f"f{frame_index:04d}_{mapping.sanitized_id}",
                feature_type=FeatureType.LINE.value,
                line_x1=ln.x1_source,
                line_y1=ln.y1_source,
                line_x2=ln.x2_source,
                line_y2=ln.y2_source,
                canonical_pitch_feature_id=mapping.canonical_pitch_feature_id,
                score=float(ln.score) if math.isfinite(ln.score) else None,
                confidence=None,
                source="sv_lines",
                model_ref=line_model_ref,
                suitability=Suitability.MARGINAL.value,
                status=status,
                manual_review_required=True,
                reason_codes=["stage8b_detection_baseline"],
                quality_flags=["confidence_null", "evaluation_only"],
                provenance_json=json.dumps(prov, sort_keys=True, separators=(",", ":")),
            )
        )

    # Bound features per frame
    max_f = int(config["max_features_per_frame"])
    if len(rows) > max_f:
        rows = sorted(rows, key=lambda r: float(r.get("score") or 0.0), reverse=True)[:max_f]
    return rows, stats


def run_pitch_feature_detect(
    *,
    output_dir: str,
    config: Mapping[str, Any],
    contain_root: Path | str | None = None,
    run_id: str | None = None,
    video_id: str | None = None,
    project_root: Path | str | None = None,
    in_memory_bundle: Mapping[str, Any] | None = None,
    source: str | None = None,
    timeline: str | None = None,
    analysis_windows: str | None = None,
    adapter: NbjwHrnetPitchFeatureAdapter | None = None,
    inject_failure: bool = False,
) -> PitchFeatureServiceResult:
    """Detect pitch keypoints/lines; write calibration_features + receipts.

    Prefer ``in_memory_bundle`` for fixture/smoke. Video decode path is optional and
    still bounded by ``maximum_frames_per_run``. Homography is never solved here.
    """
    cfg_fp = pitch_feature_config_fingerprint(config)
    try:
        reject_unsafe_path_string(output_dir, label="output_dir")
        out = require_absolute_path(output_dir, label="output_dir")
    except Exception:  # noqa: BLE001
        return _fail(error_code="UNSAFE_PATH", exit_code=3, config_fingerprint=cfg_fp)

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
        if source:
            assert_safe_output_root(
                str(out),
                contain_root=root,
                source_path=source,
                overwrite_allowed=False,
            )
    except Exception:  # noqa: BLE001
        return _fail(error_code="PATH_SECURITY", exit_code=3, config_fingerprint=cfg_fp)

    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    feat_out = out / "calibration_features.parquet"
    receipt_out = out / "pitch_feature_run_receipt.json"
    quality_out = out / "pitch_feature_quality.json"
    eval_out = out / "pitch_feature_evaluation.json"
    status_out = out / "feature_frame_status.json"
    artifacts = [feat_out, receipt_out, quality_out, eval_out, status_out]
    for p in artifacts:
        if p.exists():
            return _fail(error_code="OVERWRITE_FORBIDDEN", exit_code=3, config_fingerprint=cfg_fp)

    if inject_failure:
        return _fail(
            error_code="INJECTED_FAILURE",
            exit_code=1,
            config_fingerprint=cfg_fp,
            cleanup=[out],
        )

    rid = (
        run_id
        or (str(in_memory_bundle.get("run_id")) if in_memory_bundle else None)
        or generate_run_id()
    )
    try:
        validate_run_id(rid)
    except Exception:  # noqa: BLE001
        return _fail(error_code="INVALID_RUN_ID", exit_code=3, config_fingerprint=cfg_fp)

    vid = (
        video_id
        or (str(in_memory_bundle.get("video_id")) if in_memory_bundle else None)
        or "video_01"
    )
    if not SAFE_ID_RE.match(vid):
        return _fail(error_code="INVALID_VIDEO_ID", exit_code=3, config_fingerprint=cfg_fp)

    if config.get("auto_homography") is not False or config["routing"].get("auto_homography"):
        return _fail(error_code="AUTO_HOMOGRAPHY_FORBIDDEN", exit_code=3, config_fingerprint=cfg_fp)
    if config.get("network_sources_allowed") is not False:
        return _fail(error_code="NETWORK_FORBIDDEN", exit_code=3, config_fingerprint=cfg_fp)

    repo = Path(project_root) if project_root else Path(__file__).resolve().parents[3]
    registry_path = repo / "model_registry.yaml"

    try:
        kp_entry = _load_model_registry_entry(registry_path, str(config["kp_model_registry_id"]))
        lines_entry = _load_model_registry_entry(
            registry_path, str(config["lines_model_registry_id"])
        )
        kp_path, kp_sha = verify_weight_file(
            str(kp_entry["file_path"]),
            expected_sha256=str(kp_entry["sha256"]),
            expected_size=int(kp_entry.get("size_bytes") or 0) or None,
        )
        lines_path, lines_sha = verify_weight_file(
            str(lines_entry["file_path"]),
            expected_sha256=str(lines_entry["sha256"]),
            expected_size=int(lines_entry.get("size_bytes") or 0) or None,
        )
    except (PitchFeatureServiceError, PitchFeatureAdapterError) as exc:
        return _fail(
            error_code=str(exc).split(":")[0] if str(exc) else "MODEL_NOT_AVAILABLE",
            exit_code=3,
            config_fingerprint=cfg_fp,
            cleanup=[out],
        )

    # Build frame worklist
    work: list[dict[str, Any]] = []
    if in_memory_bundle is not None:
        for item in list(in_memory_bundle.get("images") or []):
            work.append(
                {
                    "frame_index": int(item["frame_index"]),
                    "video_time_us": int(item.get("video_time_us", 0)),
                    "rgb": item["rgb"],
                    "eligible": bool(item.get("eligible", True)),
                    "skip_reason": item.get("skip_reason"),
                }
            )
    elif source and timeline and analysis_windows:
        # Bounded optional video path — decode via OpenCV when provided.
        try:
            from football_analytics.data.parquet import read_contract_parquet

            reject_unsafe_path_string(source, label="source")
            reject_unsafe_path_string(timeline, label="timeline")
            reject_unsafe_path_string(analysis_windows, label="analysis_windows")
            src = require_absolute_path(source, label="source")
            tl_path = require_absolute_path(timeline, label="timeline")
            aw_path = require_absolute_path(analysis_windows, label="analysis_windows")
            frames = read_contract_parquet(tl_path, get_contract("frames", 1)).to_pylist()
            windows = read_contract_parquet(
                aw_path, get_contract("analysis_windows", 1)
            ).to_pylist()
            import cv2

            cap = cv2.VideoCapture(str(src))
            if not cap.isOpened():
                raise PitchFeatureServiceError("SOURCE_OPEN_FAILED")
            try:
                for fr in frames:
                    fi = int(fr["frame_index"])
                    win = None
                    for w in windows:
                        if int(w["start_frame_index"]) <= fi < int(w["end_frame_index_exclusive"]):
                            win = w
                            break
                    ok, reason = _window_eligible(win, config=config)
                    if not ok:
                        work.append(
                            {
                                "frame_index": fi,
                                "video_time_us": int(fr["video_time_us"]),
                                "rgb": None,
                                "eligible": False,
                                "skip_reason": reason,
                            }
                        )
                        continue
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                    ret, bgr = cap.read()
                    if not ret:
                        work.append(
                            {
                                "frame_index": fi,
                                "video_time_us": int(fr["video_time_us"]),
                                "rgb": None,
                                "eligible": True,
                                "failed": True,
                                "skip_reason": "decode_failed",
                            }
                        )
                        continue
                    rgb = bgr[:, :, ::-1].copy()
                    work.append(
                        {
                            "frame_index": fi,
                            "video_time_us": int(fr["video_time_us"]),
                            "rgb": rgb,
                            "eligible": True,
                            "skip_reason": None,
                        }
                    )
            finally:
                cap.release()
        except Exception as exc:  # noqa: BLE001
            return _fail(
                error_code=f"VIDEO_INPUT_FAILED:{type(exc).__name__}",
                exit_code=1,
                config_fingerprint=cfg_fp,
                cleanup=[out],
            )
    else:
        return _fail(
            error_code="INPUT_REQUIRED",
            exit_code=2,
            config_fingerprint=cfg_fp,
            cleanup=[out],
        )

    max_frames = int(config["maximum_frames_per_run"])
    eligible_work: list[dict[str, Any]] = []
    other_work: list[dict[str, Any]] = []
    for w in work:
        if w.get("eligible") and w.get("rgb") is not None and not w.get("failed"):
            eligible_work.append(w)
        else:
            other_work.append(w)
    capped = eligible_work[:max_frames]
    for w in eligible_work[max_frames:]:
        other_work.append(
            {
                "frame_index": int(w["frame_index"]),
                "video_time_us": int(w["video_time_us"]),
                "rgb": None,
                "eligible": False,
                "skip_reason": "max_frames_cap",
            }
        )

    try:
        loaded = adapter or NbjwHrnetPitchFeatureAdapter.load(
            config=config,
            kp_weights_path=kp_path,
            lines_weights_path=lines_path,
            kp_expected_sha256=kp_sha,
            lines_expected_sha256=lines_sha,
            kp_expected_size=int(kp_entry.get("size_bytes") or 0) or None,
            lines_expected_size=int(lines_entry.get("size_bytes") or 0) or None,
            device_policy=str(config["device_policy"]),
        )
    except PitchFeatureAdapterError as exc:
        code = str(exc)
        if "INCOMPATIBLE" in code or "get_cls_net" in code:
            return _fail(
                error_code="CALIBRATION_MODEL_ENVIRONMENT_INCOMPATIBLE",
                exit_code=3,
                config_fingerprint=cfg_fp,
                cleanup=[out],
            )
        return _fail(
            error_code=code.split(":")[0][:64],
            exit_code=3,
            config_fingerprint=cfg_fp,
            cleanup=[out],
        )

    feature_rows: list[dict[str, Any]] = []
    frame_statuses: list[dict[str, Any]] = []
    counts = {
        "eligible": 0,
        "processed": 0,
        "processed_no_features": 0,
        "not_eligible": 0,
        "skipped": 0,
        "failed": 0,
        "keypoints": 0,
        "lines": 0,
        "mapped": 0,
        "unmapped": 0,
        "duplicates": 0,
        "out_of_bounds": 0,
        "rejected": 0,
    }
    scores: list[float] = []
    device_used = loaded.device

    for item in other_work:
        reason = str(item.get("skip_reason") or "not_eligible")
        status = "skipped" if reason == "max_frames_cap" else "not_eligible"
        if item.get("failed"):
            status = "failed"
            counts["failed"] += 1
        elif status == "skipped":
            counts["skipped"] += 1
        else:
            counts["not_eligible"] += 1
        frame_statuses.append(
            {
                "frame_index": int(item["frame_index"]),
                "video_time_us": int(item["video_time_us"]),
                "status": status,
                "reason": reason,
                "feature_count": 0,
            }
        )

    for item in capped:
        counts["eligible"] += 1
        fi = int(item["frame_index"])
        vt = int(item["video_time_us"])
        try:
            inference = loaded.infer_rgb(item["rgb"])
            rows, st = _feature_rows_from_inference(
                run_id=rid,
                video_id=vid,
                frame_index=fi,
                video_time_us=vt,
                inference=inference,
                config=config,
            )
            feature_rows.extend(rows)
            n_acc = len(accepted_keypoints(inference.keypoints)) + len(
                accepted_lines(inference.lines)
            )
            counts["keypoints"] += st["keypoints_accepted"]
            counts["lines"] += st["lines_accepted"]
            counts["mapped"] += st["mapped"]
            counts["unmapped"] += st["unmapped"]
            counts["duplicates"] += st["duplicates"]
            counts["out_of_bounds"] += st["out_of_bounds"]
            counts["rejected"] += st["keypoints_rejected"] + st["lines_rejected"]
            for r in rows:
                if r.get("score") is not None:
                    scores.append(float(r["score"]))
            if n_acc == 0:
                counts["processed_no_features"] += 1
                frame_statuses.append(
                    {
                        "frame_index": fi,
                        "video_time_us": vt,
                        "status": "processed_no_features",
                        "reason": "no_features_above_threshold",
                        "feature_count": 0,
                    }
                )
            else:
                counts["processed"] += 1
                frame_statuses.append(
                    {
                        "frame_index": fi,
                        "video_time_us": vt,
                        "status": "processed",
                        "reason": None,
                        "feature_count": len(rows),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            counts["failed"] += 1
            frame_statuses.append(
                {
                    "frame_index": fi,
                    "video_time_us": vt,
                    "status": "failed",
                    "reason": type(exc).__name__,
                    "feature_count": 0,
                }
            )

    try:
        validated = validate_feature_rows(feature_rows)
        table = _rows_to_table(validated, "calibration_features")
        write_contract_parquet(
            table, feat_out, get_contract("calibration_features", 1), contain_root=root
        )
        feat_out.chmod(0o600)
    except Exception as exc:  # noqa: BLE001
        return _fail(
            error_code=f"WRITE_FAILED:{type(exc).__name__}",
            exit_code=1,
            config_fingerprint=cfg_fp,
            cleanup=artifacts + [out],
        )

    eval_report = evaluate_pitch_features(predictions=validated, has_reviewed_ground_truth=False)
    quality = {
        "schema_version": 1,
        "run_id": rid,
        "video_id": vid,
        "frame_counts": {
            "eligible": counts["eligible"],
            "processed": counts["processed"],
            "processed_no_features": counts["processed_no_features"],
            "not_eligible": counts["not_eligible"],
            "skipped": counts["skipped"],
            "failed": counts["failed"],
        },
        "feature_counts": {
            "keypoints": counts["keypoints"],
            "lines": counts["lines"],
            "mapped": counts["mapped"],
            "unmapped": counts["unmapped"],
            "duplicates": counts["duplicates"],
            "out_of_bounds": counts["out_of_bounds"],
            "rejected": counts["rejected"],
            "total_rows": len(validated),
        },
        "score_summary": {
            "n": len(scores),
            "mean": statistics.fmean(scores) if scores else None,
            "median": statistics.median(scores) if scores else None,
        },
        "no_feature_rate": (
            counts["processed_no_features"] / counts["eligible"] if counts["eligible"] else None
        ),
        "notes": [
            "operational quality only; not football accuracy",
            "confidence always null; raw scores in provenance",
            "homography deferred to Stage 8C",
        ],
        "created_at_utc": _utc_now(),
    }
    receipt = {
        "schema_version": 1,
        "receipt_id": f"pitch_feat_{rid[-12:]}",
        "run_id": rid,
        "video_id": vid,
        "status": "succeeded" if counts["failed"] == 0 else "partial",
        "stage": "8B",
        "adapter_id": config["adapter_id"],
        "adapter_version": config["adapter_version"],
        "adapter_choice": config["adapter_choice"],
        "config_fingerprint": cfg_fp,
        "kp_model_registry_id": config["kp_model_registry_id"],
        "lines_model_registry_id": config["lines_model_registry_id"],
        "kp_model_sha256": kp_sha,
        "lines_model_sha256": lines_sha,
        "kp_model_size_bytes": int(kp_path.stat().st_size),
        "lines_model_size_bytes": int(lines_path.stat().st_size),
        "device": device_used,
        "batch_size": 1,
        "auto_homography": False,
        "network_sources_allowed": False,
        "no_overwrite": True,
        "atomic_writes": True,
        "evaluation_status": NOT_EVALUATED_PITCH_FEATURES,
        "production_approved": False,
        "gpl_linking_risk": True,
        "license_note": config["license_note"],
        "frame_counts": quality["frame_counts"],
        "feature_counts": quality["feature_counts"],
        "score_summary": quality["score_summary"],
        "artifacts": {
            "calibration_features": {
                "path": str(feat_out),
                "sha256": sha256_file(feat_out),
                "size_bytes": int(feat_out.stat().st_size),
            }
        },
        "contract_version": CONTRACT_VERSION,
        "created_at_utc": _utc_now(),
        "warnings": [
            "REAL_FOOTBALL_ACCURACY_NOT_YET_VALIDATED",
            "GPL_2_0_LINKING_RISK_EVALUATION_ONLY",
        ],
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
        write_json_record(
            status_out,
            {"schema_version": 1, "frames": frame_statuses},
            overwrite=False,
            contain_root=root,
        )
        for p in (receipt_out, quality_out, eval_out, status_out):
            p.chmod(0o600)
    except Exception as exc:  # noqa: BLE001
        return _fail(
            error_code=f"RECEIPT_WRITE_FAILED:{type(exc).__name__}",
            exit_code=1,
            config_fingerprint=cfg_fp,
            cleanup=artifacts + [out],
        )

    return PitchFeatureServiceResult(
        accepted=True,
        exit_code=0,
        error_code=None,
        config_fingerprint=cfg_fp,
        features_parquet=str(feat_out),
        receipt_json=str(receipt_out),
        quality_json=str(quality_out),
        evaluation_json=str(eval_out),
        frame_status_json=str(status_out),
        summary={
            "status": receipt["status"],
            "feature_count": len(validated),
            "frame_counts": quality["frame_counts"],
            "kp_model_sha256": kp_sha,
            "lines_model_sha256": lines_sha,
            "device": device_used,
            "evaluation_status": NOT_EVALUATED_PITCH_FEATURES,
            "feature_rows": validated,
        },
    )


__all__ = [
    "PitchFeatureServiceError",
    "PitchFeatureServiceResult",
    "run_pitch_feature_detect",
]
