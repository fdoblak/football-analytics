"""Stage 5C ball detection service: route → decode → detect → contract outputs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import yaml

from football_analytics.core.hashing import sha256_file
from football_analytics.core.records import write_json_record
from football_analytics.core.run_id import generate_run_id, validate_run_id
from football_analytics.data.compiler import compile_arrow_schema, get_contract
from football_analytics.data.parquet import read_contract_parquet, write_contract_parquet
from football_analytics.data.types import SAFE_ID_RE
from football_analytics.perception.ball_detection import (
    build_attribute_rows,
    build_detection_rows,
    candidates_to_mapped,
    coverage_from_boxes,
    filter_raw_ball_boxes,
    mapped_to_candidates,
)
from football_analytics.perception.ball_detector_config import ball_detector_config_fingerprint
from football_analytics.perception.ball_evaluation import (
    evaluate_ball_from_rows,
    not_evaluated_ball_metrics,
)
from football_analytics.perception.candidate_merge import merge_ball_candidates
from football_analytics.perception.policy import (
    load_detection_policy,
    policy_fingerprint,
    resolve_frame_routing,
)
from football_analytics.perception.taxonomy import load_detection_taxonomy, taxonomy_fingerprint
from football_analytics.perception.tiling import crop_tile, generate_tiles, map_tile_bbox_to_source
from football_analytics.perception.transforms import build_preprocessing_transform
from football_analytics.perception.types import (
    ColorSpace,
    DetectionRunReceipt,
    Eligibility,
    ProcessingStatus,
    ReceiptStatus,
)
from football_analytics.perception.validation import validate_detection_bundle
from football_analytics.utils.archive_safety import assert_contained, resolve_strict
from football_analytics.video.probe_service import assert_snapshots_equal, snapshot_source
from football_analytics.video.types import VideoSourceError
from football_analytics.video.validation import (
    assert_safe_output_root,
    assert_safe_source_path,
    reject_unsafe_path_string,
    require_absolute_path,
)


class BallDetectionServiceError(ValueError):
    """Ball detection service failure."""


@dataclass
class BallDetectionServiceResult:
    accepted: bool
    exit_code: int
    detections_parquet: str | None
    frame_status_parquet: str | None
    attributes_parquet: str | None
    receipt_json: str | None
    evaluation_json: str | None
    error_code: str | None
    detection_count: int
    human_detection_count: int
    ball_detection_count: int
    processed_frame_count: int
    config_fingerprint: str | None
    model_sha256: str | None

    def to_summary(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "exit_code": self.exit_code,
            "detections_parquet": self.detections_parquet,
            "frame_status_parquet": self.frame_status_parquet,
            "attributes_parquet": self.attributes_parquet,
            "receipt_json": self.receipt_json,
            "evaluation_json": self.evaluation_json,
            "error_code": self.error_code,
            "detection_count": self.detection_count,
            "human_detection_count": self.human_detection_count,
            "ball_detection_count": self.ball_detection_count,
            "processed_frame_count": self.processed_frame_count,
            "config_fingerprint": self.config_fingerprint,
            "model_sha256": self.model_sha256,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _fail(
    *,
    error_code: str,
    exit_code: int = 1,
    config_fingerprint: str | None = None,
    model_sha256: str | None = None,
) -> BallDetectionServiceResult:
    return BallDetectionServiceResult(
        accepted=False,
        exit_code=exit_code,
        detections_parquet=None,
        frame_status_parquet=None,
        attributes_parquet=None,
        receipt_json=None,
        evaluation_json=None,
        error_code=error_code,
        detection_count=0,
        human_detection_count=0,
        ball_detection_count=0,
        processed_frame_count=0,
        config_fingerprint=config_fingerprint,
        model_sha256=model_sha256,
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
        raise BallDetectionServiceError("MODEL_REGISTRY_INVALID")
    models = raw.get("models")
    if not isinstance(models, list):
        raise BallDetectionServiceError("MODEL_REGISTRY_INVALID")
    for item in models:
        if isinstance(item, Mapping) and item.get("id") == model_id:
            return dict(item)
    raise BallDetectionServiceError("MODEL_NOT_AVAILABLE")


def _resolve_device(policy: str) -> tuple[str, bool]:
    prefer_cuda = policy == "prefer_cuda_else_cpu"
    require_cuda = policy == "cuda_required"
    try:
        import torch

        cuda_ok = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        cuda_ok = False
    if require_cuda and not cuda_ok:
        raise BallDetectionServiceError("CUDA_REQUIRED_UNAVAILABLE")
    if (prefer_cuda or require_cuda) and cuda_ok:
        return "cuda:0", True
    return "cpu", False


def _window_for_frame(windows: list[dict[str, Any]], frame_index: int) -> dict[str, Any] | None:
    for w in windows:
        start = w.get("start_frame_index")
        end = w.get("end_frame_index_exclusive")
        if start is None or end is None:
            continue
        if int(start) <= int(frame_index) < int(end):
            return w
    return windows[0] if len(windows) == 1 else None


def _status_row(
    *,
    run_id: str,
    video_id: str,
    frame_index: int,
    video_time_us: int,
    analysis_window_id: str | None,
    processing_status: str,
    eligibility: str,
    detector_id: str,
    detection_count: int,
    human_count: int,
    ball_count: int,
    skip_reason: str | None,
    error_code: str | None,
    coverage: float,
    provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "frame_index": int(frame_index),
        "video_time_us": int(video_time_us),
        "analysis_window_id": analysis_window_id,
        "processing_status": processing_status,
        "eligibility": eligibility,
        "detector_id": detector_id,
        "input_artifact_ref": None,
        "detection_count": int(detection_count),
        "human_count": int(human_count),
        "ball_count": int(ball_count),
        "skip_reason": skip_reason,
        "error_code": error_code,
        "coverage": float(coverage),
        "provenance_json": (
            None
            if provenance is None
            else json.dumps(dict(provenance), sort_keys=True, separators=(",", ":"))
        ),
        "contract_version": 1,
    }


def _predict_frame_candidates(
    *,
    frame: Any,
    ball_adapter: Any,
    config: Mapping[str, Any],
    device: str,
    half_ok: bool,
    taxonomy: Mapping[str, Any],
) -> list[Any]:
    """Run full_frame / tiled / hybrid inference and return merged MappedBallDetection list."""
    h, w = int(frame.shape[0]), int(frame.shape[1])
    imgsz = int(config["input_size"])
    ball_ids = list(config["sports_ball_class"]["class_ids"])
    ball_names = list(config["sports_ball_class"]["class_names"])
    mode = str(config["inference_mode"])
    filters = config["filters"]
    max_det = int(config["resource_limits"]["max_detections_per_frame"])
    conf = float(config["confidence_threshold"])
    nms = float(config["nms_iou"])
    merge_iou = float(config["merge_iou"])

    full_mapped = []
    tile_mapped = []

    def _run_predict(img: Any) -> list[Any]:
        return ball_adapter.predict_balls(
            img,
            conf=conf,
            iou=nms,
            imgsz=imgsz,
            device=device,
            half=half_ok,
            class_ids=ball_ids,
            class_names=ball_names,
            channel_order="bgr",
        )

    if mode in {"full_frame", "hybrid"}:
        raw = _run_predict(frame)
        full_mapped = filter_raw_ball_boxes(
            raw,
            confidence_threshold=conf,
            filters=filters,
            frame_width=w,
            frame_height=h,
            model_input_size=imgsz,
            boxes_in_source_space=True,
            taxonomy=taxonomy,
            max_detections=max_det,
            candidate_source="full_frame",
        )

    if mode in {"tiled", "hybrid"}:
        tiling = config["tiling"]
        tiles = generate_tiles(
            w,
            h,
            tile_width=int(tiling["tile_width"]),
            tile_height=int(tiling["tile_height"]),
            overlap_x=int(tiling["overlap_x"]),
            overlap_y=int(tiling["overlap_y"]),
            max_tiles=int(tiling["max_tiles"]),
        )
        for tile in tiles:
            crop = crop_tile(frame, tile)
            if crop is None or getattr(crop, "size", 1) == 0:
                continue
            raw = _run_predict(crop)
            # Map tile-local boxes → source before geometry filters.
            from football_analytics.perception.adapters.base import RawDetectionBox

            mapped_raw = []
            for box in raw:
                sx1, sy1, sx2, sy2 = map_tile_bbox_to_source(
                    (box.x1, box.y1, box.x2, box.y2),
                    tile,
                    coordinate_space="tile_local",
                )
                mapped_raw.append(
                    RawDetectionBox(
                        x1=sx1,
                        y1=sy1,
                        x2=sx2,
                        y2=sy2,
                        score=box.score,
                        class_id=box.class_id,
                        class_name=box.class_name,
                    )
                )
            tile_mapped.extend(
                filter_raw_ball_boxes(
                    mapped_raw,
                    confidence_threshold=conf,
                    filters=filters,
                    frame_width=w,
                    frame_height=h,
                    model_input_size=imgsz,
                    boxes_in_source_space=True,
                    taxonomy=taxonomy,
                    max_detections=max_det,
                    candidate_source=f"tile:{tile.tile_id}",
                )
            )

    if mode == "full_frame":
        return full_mapped[:max_det]
    if mode == "tiled":
        merged = merge_ball_candidates(
            [],
            mapped_to_candidates(tile_mapped),
            merge_iou=merge_iou,
            class_aware=True,
        )
        return candidates_to_mapped(merged, taxonomy=taxonomy)[:max_det]

    merged = merge_ball_candidates(
        mapped_to_candidates(full_mapped),
        mapped_to_candidates(tile_mapped),
        merge_iou=merge_iou,
        class_aware=True,
    )
    return candidates_to_mapped(merged, taxonomy=taxonomy)[:max_det]


def run_ball_detection(
    *,
    source: str,
    timeline: str,
    analysis_windows: str,
    output_dir: str,
    config: Mapping[str, Any],
    contain_root: Path | str | None = None,
    run_id: str | None = None,
    video_id: str | None = None,
    project_root: Path | str | None = None,
    ground_truth: str | None = None,
    adapter: Any | None = None,
) -> BallDetectionServiceResult:
    """Run bounded ball detection for eligible frames; write Stage 5A contracts."""
    cfg_fp = ball_detector_config_fingerprint(config)
    try:
        reject_unsafe_path_string(source, label="source")
        reject_unsafe_path_string(timeline, label="timeline")
        reject_unsafe_path_string(analysis_windows, label="analysis_windows")
        reject_unsafe_path_string(output_dir, label="output_dir")
        src = require_absolute_path(source, label="source")
        tl_path = require_absolute_path(timeline, label="timeline")
        aw_path = require_absolute_path(analysis_windows, label="analysis_windows")
        out = require_absolute_path(output_dir, label="output_dir")
    except (VideoSourceError, Exception):  # noqa: BLE001
        code = "UNSAFE_PATH"
        if "://" in str(source):
            code = "NETWORK_SOURCE_FORBIDDEN"
        return _fail(error_code=code, exit_code=3, config_fingerprint=cfg_fp)

    root = Path(contain_root) if contain_root is not None else Path(str(config["runtime_root"]))
    policy_paths = {
        "symlinks_allowed": False,
        "network_sources_allowed": False,
        "allowed_file_extensions": [".mp4", ".mkv", ".mov", ".webm", ".avi"],
    }
    try:
        from football_analytics.video.validation import assert_extension_allowed

        root = require_absolute_path(str(root), label="contain_root")
        src = assert_safe_source_path(str(src), contain_root=root, policy=policy_paths)
        assert_extension_allowed(src, policy_paths)
        assert_safe_output_root(
            str(out), contain_root=root, source_path=str(src), overwrite_allowed=False
        )
        for p, label in ((tl_path, "timeline"), (aw_path, "analysis_windows")):
            if p.is_symlink():
                raise VideoSourceError(f"{label} must not be a symlink")
            if not p.is_file():
                raise VideoSourceError(f"{label} missing")
            assert_contained(resolve_strict(p), resolve_strict(root), label=label)
    except Exception:  # noqa: BLE001
        return _fail(error_code="PATH_SECURITY", exit_code=3, config_fingerprint=cfg_fp)

    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    det_out = out / "detections.parquet"
    status_out = out / "detection_frame_status.parquet"
    attr_out = out / "detection_attributes.parquet"
    receipt_out = out / "ball_detection_run_receipt.json"
    eval_out = out / "ball_evaluation.json"
    for p in (det_out, status_out, attr_out, receipt_out, eval_out):
        if p.exists():
            return _fail(error_code="OVERWRITE_FORBIDDEN", exit_code=3, config_fingerprint=cfg_fp)

    try:
        before = snapshot_source(src)
    except Exception:  # noqa: BLE001
        return _fail(error_code="SOURCE_SNAPSHOT_FAILED", exit_code=3, config_fingerprint=cfg_fp)

    rid = run_id or generate_run_id()
    try:
        validate_run_id(rid)
    except Exception:  # noqa: BLE001
        return _fail(error_code="INVALID_RUN_ID", exit_code=3, config_fingerprint=cfg_fp)

    vid = video_id or "video_01"
    if not SAFE_ID_RE.match(vid):
        return _fail(error_code="INVALID_VIDEO_ID", exit_code=3, config_fingerprint=cfg_fp)

    repo = Path(project_root) if project_root else Path(__file__).resolve().parents[3]
    tax_rel = Path(str(config["taxonomy_path"]))
    pol_rel = Path(str(config["policy_path"]))
    tax_path = tax_rel if tax_rel.is_absolute() else repo / tax_rel
    pol_path = pol_rel if pol_rel.is_absolute() else repo / pol_rel
    try:
        taxonomy = load_detection_taxonomy(tax_path)
        det_policy = load_detection_policy(pol_path)
    except Exception:  # noqa: BLE001
        return _fail(error_code="CONFIG_LOAD_FAILED", exit_code=2, config_fingerprint=cfg_fp)

    tax_fp = taxonomy_fingerprint(taxonomy)
    pol_fp = policy_fingerprint(det_policy)
    thr_fp = ball_detector_config_fingerprint(
        {
            "confidence_threshold": config["confidence_threshold"],
            "nms_iou": config["nms_iou"],
            "merge_iou": config["merge_iou"],
            "filters": dict(config["filters"]),
            "inference_mode": config["inference_mode"],
            "tiling": dict(config["tiling"]),
        }
    )

    registry_path = repo / "model_registry.yaml"
    model_sha: str | None = None
    try:
        entry = _load_model_registry_entry(registry_path, str(config["model_registry_id"]))
        weights = Path(str(entry["file_path"]))
        expected_sha = str(entry["sha256"]).lower()
        if weights.is_symlink() or not weights.is_file():
            raise BallDetectionServiceError("MODEL_NOT_AVAILABLE")
        if "://" in str(weights):
            raise BallDetectionServiceError("NETWORK_WEIGHTS_FORBIDDEN")
        model_sha = sha256_file(weights).lower()
        if model_sha != expected_sha:
            raise BallDetectionServiceError("MODEL_HASH_MISMATCH")
    except BallDetectionServiceError as exc:
        return _fail(
            error_code=str(exc),
            exit_code=3,
            config_fingerprint=cfg_fp,
            model_sha256=model_sha,
        )
    except Exception:  # noqa: BLE001
        return _fail(error_code="MODEL_NOT_AVAILABLE", exit_code=3, config_fingerprint=cfg_fp)

    try:
        frames_table = read_contract_parquet(tl_path, get_contract("frames", 1))
        windows_table = read_contract_parquet(aw_path, get_contract("analysis_windows", 1))
    except Exception:  # noqa: BLE001
        return _fail(error_code="INPUT_CONTRACT_FAILED", exit_code=3, config_fingerprint=cfg_fp)

    frame_rows = frames_table.to_pylist()
    window_rows = windows_table.to_pylist()
    frame_rows.sort(key=lambda r: int(r["frame_index"]))

    try:
        device, half_ok = _resolve_device(str(config["device_policy"]))
        if str(config["precision_policy"]) == "fp32_only":
            half_ok = False
    except BallDetectionServiceError as exc:
        return _fail(error_code=str(exc), exit_code=3, config_fingerprint=cfg_fp)

    from football_analytics.perception.adapters.ultralytics_ball import (
        UltralyticsBallAdapter,
        get_ball_adapter,
    )

    owns_adapter = adapter is None
    ball_adapter = adapter or get_ball_adapter(str(config["adapter_id"]))
    try:
        if not getattr(ball_adapter, "is_loaded", lambda: False)():
            ball_adapter.load(str(weights), expected_sha)
    except Exception as exc:  # noqa: BLE001
        code = "MODEL_NOT_AVAILABLE"
        msg = str(exc)
        if "HASH" in msg:
            code = "MODEL_HASH_MISMATCH"
        if "SPORTS_BALL_CLASS_MISSING" in msg:
            code = "SPORTS_BALL_CLASS_MISSING"
        if owns_adapter:
            ball_adapter.unload()
        return _fail(
            error_code=code, exit_code=3, config_fingerprint=cfg_fp, model_sha256=model_sha
        )

    detector_id = "ball_yolo11n_v1"
    started = _utc_now()
    max_frames = int(config["maximum_frames_per_run"])
    imgsz = int(config["input_size"])
    skip_identity_only = bool(config["routing"].get("skip_identity_only", True))

    import cv2

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        if owns_adapter:
            ball_adapter.unload()
        return _fail(
            error_code="FRAME_DECODE_FAILED",
            exit_code=3,
            config_fingerprint=cfg_fp,
            model_sha256=model_sha,
        )

    detection_rows: list[dict[str, Any]] = []
    attribute_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    eligible_count = 0
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    empty_count = 0
    warnings: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    transform_fp: str | None = None
    frames_considered = 0
    last_seek = -1

    try:
        for fr in frame_rows:
            if frames_considered >= max_frames:
                break
            frames_considered += 1
            fi = int(fr["frame_index"])
            t_us = int(fr.get("video_time_us", 0))
            window = _window_for_frame(window_rows, fi)
            routing = resolve_frame_routing(window, policy=det_policy, detect_ball=True)
            wid = None if window is None else window.get("analysis_window_id")

            process_ball = bool(routing.get("process_ball"))
            if skip_identity_only and routing.get("identity_only"):
                process_ball = False
                skip_reason = "IDENTITY_ONLY_SKIP_BALL"
                status_rows.append(
                    _status_row(
                        run_id=rid,
                        video_id=vid,
                        frame_index=fi,
                        video_time_us=t_us,
                        analysis_window_id=wid,
                        processing_status=ProcessingStatus.SKIPPED.value,
                        eligibility=str(routing.get("eligibility", Eligibility.INELIGIBLE.value)),
                        detector_id=detector_id,
                        detection_count=0,
                        human_count=0,
                        ball_count=0,
                        skip_reason=skip_reason,
                        error_code=None,
                        coverage=0.0,
                        provenance={"stage": "5C", "route": "identity_only_skip"},
                    )
                )
                continue

            if not process_ball:
                status = str(routing["processing_status"])
                if status == "processed":
                    status = ProcessingStatus.NOT_ELIGIBLE.value
                if status not in {s.value for s in ProcessingStatus}:
                    status = ProcessingStatus.NOT_ELIGIBLE.value
                status_rows.append(
                    _status_row(
                        run_id=rid,
                        video_id=vid,
                        frame_index=fi,
                        video_time_us=t_us,
                        analysis_window_id=wid,
                        processing_status=status,
                        eligibility=str(routing.get("eligibility", Eligibility.INELIGIBLE.value)),
                        detector_id=detector_id,
                        detection_count=0,
                        human_count=0,
                        ball_count=0,
                        skip_reason=routing.get("skip_reason") or "BALL_ANALYSIS_NOT_ELIGIBLE",
                        error_code=routing.get("error_code"),
                        coverage=0.0,
                        provenance={"stage": "5C", "route": "skip"},
                    )
                )
                continue

            eligible_count += 1
            if fi != last_seek + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
            ok, frame = cap.read()
            last_seek = fi
            if not ok or frame is None:
                failed_count += 1
                status_rows.append(
                    _status_row(
                        run_id=rid,
                        video_id=vid,
                        frame_index=fi,
                        video_time_us=t_us,
                        analysis_window_id=wid,
                        processing_status=ProcessingStatus.FAILED.value,
                        eligibility=str(routing.get("eligibility", "eligible")),
                        detector_id=detector_id,
                        detection_count=0,
                        human_count=0,
                        ball_count=0,
                        skip_reason=None,
                        error_code="FRAME_DECODE_FAILED",
                        coverage=0.0,
                        provenance={"stage": "5C", "route": "decode_failed"},
                    )
                )
                errors.append({"code": "FRAME_DECODE_FAILED", "message": f"frame {fi}"})
                continue

            fh, fw = int(frame.shape[0]), int(frame.shape[1])
            transform = build_preprocessing_transform(
                source_width=fw,
                source_height=fh,
                model_input_width=imgsz,
                model_input_height=imgsz,
                color_space=ColorSpace.BGR,
            )
            transform_fp = transform.transform_fingerprint

            try:
                mapped = _predict_frame_candidates(
                    frame=frame,
                    ball_adapter=ball_adapter,
                    config=config,
                    device=device,
                    half_ok=half_ok,
                    taxonomy=taxonomy,
                )
            except Exception as exc:  # noqa: BLE001
                failed_count += 1
                status_rows.append(
                    _status_row(
                        run_id=rid,
                        video_id=vid,
                        frame_index=fi,
                        video_time_us=t_us,
                        analysis_window_id=wid,
                        processing_status=ProcessingStatus.FAILED.value,
                        eligibility=str(routing.get("eligibility", "eligible")),
                        detector_id=detector_id,
                        detection_count=0,
                        human_count=0,
                        ball_count=0,
                        skip_reason=None,
                        error_code="INFERENCE_FAILED",
                        coverage=0.0,
                        provenance={"stage": "5C", "error": type(exc).__name__},
                    )
                )
                errors.append({"code": "INFERENCE_FAILED", "message": type(exc).__name__})
                continue

            d_rows = build_detection_rows(
                mapped, run_id=rid, video_id=vid, frame_index=fi, model_id=detector_id
            )
            a_rows = build_attribute_rows(mapped, run_id=rid, video_id=vid, frame_index=fi)
            detection_rows.extend(d_rows)
            attribute_rows.extend(a_rows)
            ball_n = len(mapped)
            cov = coverage_from_boxes(
                [(m.bbox_x1, m.bbox_y1, m.bbox_x2, m.bbox_y2) for m in mapped],
                frame_width=fw,
                frame_height=fh,
            )
            if ball_n == 0:
                empty_count += 1
                pstatus = ProcessingStatus.PROCESSED_NO_DETECTIONS.value
            else:
                pstatus = ProcessingStatus.PROCESSED.value
            processed_count += 1
            status_rows.append(
                _status_row(
                    run_id=rid,
                    video_id=vid,
                    frame_index=fi,
                    video_time_us=t_us,
                    analysis_window_id=wid,
                    processing_status=pstatus,
                    eligibility=str(routing.get("eligibility", "eligible")),
                    detector_id=detector_id,
                    detection_count=ball_n,
                    human_count=0,
                    ball_count=ball_n,
                    skip_reason=None,
                    error_code=None,
                    coverage=cov,
                    provenance={
                        "stage": "5C",
                        "inference_mode": str(config["inference_mode"]),
                        "boxes_in_source_space": True,
                        "transform_fingerprint": transform_fp,
                        "device": device,
                    },
                )
            )
    finally:
        cap.release()
        if owns_adapter:
            ball_adapter.unload()

    skipped_count = sum(
        1 for r in status_rows if r["processing_status"] == ProcessingStatus.SKIPPED.value
    )
    not_eligible_count = sum(
        1 for r in status_rows if r["processing_status"] == ProcessingStatus.NOT_ELIGIBLE.value
    )

    try:
        assert_snapshots_equal(before, snapshot_source(src))
    except Exception:  # noqa: BLE001
        warnings.append({"code": "SOURCE_CHANGED", "message": "source snapshot drifted"})

    det_table = _rows_to_table(detection_rows, "detections")
    status_table = _rows_to_table(status_rows, "detection_frame_status")
    attr_table = _rows_to_table(attribute_rows, "detection_attributes")

    bundle = validate_detection_bundle(
        detections=det_table,
        frame_status=status_table,
        attributes=attr_table,
        frames=frames_table,
        analysis_windows=windows_table,
        specs={
            "detections": get_contract("detections", 1),
            "detection_frame_status": get_contract("detection_frame_status", 1),
            "detection_attributes": get_contract("detection_attributes", 1),
            "frames": get_contract("frames", 1),
            "analysis_windows": get_contract("analysis_windows", 1),
        },
    )
    if bundle.status == "FAIL":
        return _fail(
            error_code="OUTPUT_INTEGRITY_FAILED",
            exit_code=3,
            config_fingerprint=cfg_fp,
            model_sha256=model_sha,
        )

    try:
        write_contract_parquet(det_table, det_out, get_contract("detections", 1), contain_root=root)
        write_contract_parquet(
            status_table, status_out, get_contract("detection_frame_status", 1), contain_root=root
        )
        write_contract_parquet(
            attr_table, attr_out, get_contract("detection_attributes", 1), contain_root=root
        )
    except Exception:  # noqa: BLE001
        return _fail(
            error_code="OUTPUT_WRITE_FAILED",
            exit_code=3,
            config_fingerprint=cfg_fp,
            model_sha256=model_sha,
        )

    completed = _utc_now()
    ball_total = len(detection_rows)
    receipt_status = ReceiptStatus.SUCCEEDED
    if failed_count and processed_count:
        receipt_status = ReceiptStatus.PARTIAL
    elif failed_count and not processed_count:
        receipt_status = ReceiptStatus.FAILED

    soft_versions: dict[str, str] = {}
    if isinstance(ball_adapter, UltralyticsBallAdapter) or hasattr(
        ball_adapter, "software_versions"
    ):
        try:
            soft_versions = dict(ball_adapter.software_versions())
        except Exception:  # noqa: BLE001
            soft_versions = {}

    receipt = DetectionRunReceipt(
        receipt_id=f"receipt_{rid[:16]}",
        run_id=rid,
        detector_id=detector_id,
        model_registry_id=str(config["model_registry_id"]),
        model_sha256=model_sha,
        adapter_id=str(config["adapter_id"]),
        adapter_version=str(config["adapter_version"]),
        config_fingerprint=cfg_fp,
        taxonomy_version=str(taxonomy["taxonomy_version"]),
        source_video_ref=str(src),
        frames_ref=str(tl_path),
        analysis_windows_ref=str(aw_path),
        eligible_frame_count=eligible_count,
        processed_frame_count=processed_count,
        skipped_frame_count=skipped_count + not_eligible_count,
        failed_frame_count=failed_count,
        processed_no_detection_count=empty_count,
        total_detection_count=ball_total,
        human_detection_count=0,
        ball_detection_count=ball_total,
        pre_nms_count=None,
        post_nms_count=ball_total,
        started_at_utc=started,
        completed_at_utc=completed,
        status=receipt_status,
        warnings=tuple(warnings),
        errors=tuple(errors),
        artifacts={
            "detections_parquet": str(det_out),
            "detection_frame_status_parquet": str(status_out),
            "detection_attributes_parquet": str(attr_out),
            "ball_detection_run_receipt_json": str(receipt_out),
        },
        environment_ref=None,
        provenance={
            "stage": "5C",
            "label": "ball_detection_baseline_5c",
            "inference_mode": str(config["inference_mode"]),
            "notes": (
                "Stage 5C ball-only baseline; human_detection_count=0; "
                f"taxonomy_fp={tax_fp[:12]}; policy_fp={pol_fp[:12]}; thr_fp={thr_fp[:12]}"
            ),
        },
        execution_provider=device,
        precision="fp16" if half_ok and str(device).startswith("cuda") else "fp32",
        software_versions=soft_versions or None,
        transform_fingerprint=transform_fp,
        threshold_config_fingerprint=thr_fp,
    )
    try:
        write_json_record(receipt_out, receipt.to_dict(), contain_root=root, overwrite=False)
    except Exception:  # noqa: BLE001
        return _fail(
            error_code="RECEIPT_WRITE_FAILED",
            exit_code=3,
            config_fingerprint=cfg_fp,
            model_sha256=model_sha,
        )

    eval_path: str | None = None
    if config["output_policy"].get("write_evaluation_json", True):
        gt_rows = None
        if ground_truth:
            gt_p = Path(ground_truth)
            if gt_p.is_file():
                payload = json.loads(gt_p.read_text(encoding="utf-8"))
                gt_rows = list(payload.get("detections") or payload.get("ground_truth") or [])
        metrics = (
            evaluate_ball_from_rows(detection_rows, gt_rows)
            if gt_rows is not None
            else not_evaluated_ball_metrics()
        )
        try:
            write_json_record(eval_out, metrics.to_dict(), contain_root=root, overwrite=False)
            eval_path = str(eval_out)
        except Exception:  # noqa: BLE001
            warnings.append(
                {"code": "EVAL_WRITE_FAILED", "message": "ball_evaluation.json skipped"}
            )

    return BallDetectionServiceResult(
        accepted=True,
        exit_code=0 if receipt_status != ReceiptStatus.FAILED else 1,
        detections_parquet=str(det_out),
        frame_status_parquet=str(status_out),
        attributes_parquet=str(attr_out),
        receipt_json=str(receipt_out),
        evaluation_json=eval_path,
        error_code=None,
        detection_count=ball_total,
        human_detection_count=0,
        ball_detection_count=ball_total,
        processed_frame_count=processed_count,
        config_fingerprint=cfg_fp,
        model_sha256=model_sha,
    )


__all__ = [
    "BallDetectionServiceError",
    "BallDetectionServiceResult",
    "run_ball_detection",
]
