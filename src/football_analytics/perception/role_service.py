"""Stage 5D human role classification service (weightless, deterministic)."""

from __future__ import annotations

import json
from collections import defaultdict
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
from football_analytics.perception.role_classification import (
    AssignmentStatus,
    RoleAssignment,
    assign_roles_for_frame,
    make_non_human_skip,
    make_not_eligible,
)
from football_analytics.perception.role_clustering import cluster_kit_colors
from football_analytics.perception.role_config import human_role_config_fingerprint
from football_analytics.perception.role_evaluation import (
    NOT_EVALUATED_ROLE,
    count_roles,
    evaluate_roles_from_rows,
    not_evaluated_role_metrics,
)
from football_analytics.perception.role_features import (
    RoleFeatures,
    extract_features_from_crop,
    extract_synthetic_features,
)
from football_analytics.perception.types import (
    DetectionAttributes,
    ReviewStatus,
    RoleLabel,
    RoleSource,
)


class RoleServiceError(RuntimeError):
    """Role classification service failure."""


@dataclass(frozen=True)
class RoleServiceResult:
    accepted: bool
    exit_code: int
    error_code: str | None
    config_fingerprint: str
    attributes_parquet: str | None
    receipt_json: str | None
    evaluation_json: str | None
    summary: Mapping[str, Any]

    def to_summary(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "exit_code": self.exit_code,
            "error_code": self.error_code,
            "config_fingerprint": self.config_fingerprint,
            "attributes_parquet": self.attributes_parquet,
            "receipt_json": self.receipt_json,
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
) -> RoleServiceResult:
    return RoleServiceResult(
        accepted=False,
        exit_code=exit_code,
        error_code=error_code,
        config_fingerprint=config_fingerprint,
        attributes_parquet=None,
        receipt_json=None,
        evaluation_json=None,
        summary={"status": "failed", "error_code": error_code},
    )


def _rows_to_table(rows: list[dict[str, Any]], contract_name: str) -> Any:
    contract = get_contract(contract_name, 1)
    schema = compile_arrow_schema(contract)
    if not rows:
        return schema.empty_table()
    return pa.Table.from_pylist(rows, schema=schema)


def _window_for_frame(
    frame_index: int, windows: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    for w in windows:
        start = int(w["start_frame_index"])
        end = int(w["end_frame_index_exclusive"])
        if start <= frame_index < end:
            return w
    return None


def _read_frame_bgr(source: Path, frame_index: int) -> Any | None:
    import cv2

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        return frame
    finally:
        cap.release()


def _crop_bgr(frame: Any, bbox: Sequence[float]) -> Any | None:
    import numpy as np

    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, int(np.floor(bbox[0]))))
    y1 = max(0, min(h - 1, int(np.floor(bbox[1]))))
    x2 = max(0, min(w, int(np.ceil(bbox[2]))))
    y2 = max(0, min(h, int(np.ceil(bbox[3]))))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def classify_from_synthetic_humans(
    *,
    run_id: str,
    video_id: str,
    frame_index: int,
    humans: Sequence[Mapping[str, Any]],
    frame_width: float,
    frame_height: float,
    config: Mapping[str, Any],
    config_fingerprint: str,
) -> list[RoleAssignment]:
    """Fixture-only classification path (no video)."""
    features: list[RoleFeatures] = []
    for h in humans:
        bbox = list(h["bbox"])
        features.append(
            extract_synthetic_features(
                detection_id=int(h["detection_id"]),
                frame_index=frame_index,
                bbox_xyxy=bbox,
                frame_width=frame_width,
                frame_height=frame_height,
                config=config,
                kit_hue=float(h.get("kit_hue", 0.0)),
                kit_saturation=float(h.get("kit_saturation", 0.6)),
                kit_value=float(h.get("kit_value", 0.6)),
                lower_hue=h.get("lower_hue"),
                crop_quality=float(h.get("crop_quality", 0.8)),
            )
        )
    clusters, assignments = cluster_kit_colors(features, config=config)
    return assign_roles_for_frame(
        run_id=run_id,
        video_id=video_id,
        frame_index=frame_index,
        features=features,
        clusters=clusters,
        cluster_assignments=assignments,
        config=config,
        config_fingerprint=config_fingerprint,
    )


def run_human_role_classification(
    *,
    detections: str,
    detection_attributes: str,
    output_dir: str,
    config: Mapping[str, Any],
    detection_frame_status: str | None = None,
    analysis_windows: str | None = None,
    source: str | None = None,
    contain_root: Path | str | None = None,
    run_id: str | None = None,
    video_id: str | None = None,
    ground_truth: str | None = None,
    synthetic_humans: Sequence[Mapping[str, Any]] | None = None,
    synthetic_frame_size: tuple[float, float] | None = None,
    allow_synthetic_without_video: bool = False,
) -> RoleServiceResult:
    """Classify human roles; write updated detection_attributes + role receipt."""
    cfg_fp = human_role_config_fingerprint(config)
    try:
        det_path = Path(detections)
        attr_path = Path(detection_attributes)
        out = Path(output_dir)
        if not det_path.is_file() or det_path.is_symlink():
            return _fail(error_code="DETECTIONS_MISSING", exit_code=3, config_fingerprint=cfg_fp)
        if not attr_path.is_file() or attr_path.is_symlink():
            return _fail(error_code="ATTRIBUTES_MISSING", exit_code=3, config_fingerprint=cfg_fp)
    except Exception:  # noqa: BLE001
        return _fail(error_code="PATH_SECURITY", exit_code=3, config_fingerprint=cfg_fp)

    root = Path(contain_root) if contain_root is not None else Path(str(config["runtime_root"]))
    try:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        out.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not str(out.resolve()).startswith(str(root.resolve())):
            return _fail(error_code="PATH_SECURITY", exit_code=3, config_fingerprint=cfg_fp)
    except Exception:  # noqa: BLE001
        return _fail(error_code="PATH_SECURITY", exit_code=3, config_fingerprint=cfg_fp)

    attr_out = out / "detection_attributes.parquet"
    receipt_out = out / "role_run_receipt.json"
    eval_out = out / "role_evaluation.json"
    for p in (attr_out, receipt_out, eval_out):
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
        det_table = read_contract_parquet(det_path, get_contract("detections", 1))
        attr_table = read_contract_parquet(attr_path, get_contract("detection_attributes", 1))
        det_rows = det_table.to_pylist()
        attr_rows = attr_table.to_pylist()
    except Exception:  # noqa: BLE001
        return _fail(error_code="INPUT_CONTRACT_FAILED", exit_code=3, config_fingerprint=cfg_fp)

    status_by_frame: dict[int, Mapping[str, Any]] = {}
    if detection_frame_status:
        try:
            st = read_contract_parquet(
                Path(detection_frame_status), get_contract("detection_frame_status", 1)
            )
            for row in st.to_pylist():
                status_by_frame[int(row["frame_index"])] = row
        except Exception:  # noqa: BLE001
            return _fail(error_code="FRAME_STATUS_FAILED", exit_code=3, config_fingerprint=cfg_fp)

    windows: list[Mapping[str, Any]] = []
    if analysis_windows:
        try:
            wt = read_contract_parquet(Path(analysis_windows), get_contract("analysis_windows", 1))
            windows = wt.to_pylist()
        except Exception:  # noqa: BLE001
            return _fail(error_code="WINDOWS_FAILED", exit_code=3, config_fingerprint=cfg_fp)

    attr_by_key = {(int(r["frame_index"]), int(r["detection_id"])): r for r in attr_rows}
    elig = config["eligibility"]
    max_frames = int(config["maximum_frames_per_run"])

    # Group detections by frame
    by_frame: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for d in det_rows:
        by_frame[int(d["frame_index"])].append(d)

    frame_indices = sorted(by_frame.keys())[:max_frames]
    all_assignments: list[RoleAssignment] = []
    failed = 0
    src_path = Path(source) if source else None

    # Synthetic-only shortcut for unit tests (no video required).
    if allow_synthetic_without_video and synthetic_humans is not None:
        fw, fh = synthetic_frame_size or (200.0, 120.0)
        assigned = classify_from_synthetic_humans(
            run_id=rid,
            video_id=vid,
            frame_index=0,
            humans=synthetic_humans,
            frame_width=fw,
            frame_height=fh,
            config=config,
            config_fingerprint=cfg_fp,
        )
        all_assignments.extend(assigned)
    else:
        for fi in frame_indices:
            st = status_by_frame.get(fi)
            window = _window_for_frame(fi, windows) if windows else None
            dets = by_frame[fi]

            # Frame-level eligibility
            frame_ok = True
            frame_reason = "FRAME_NOT_ELIGIBLE"
            if elig["require_processed_frames"] and st is not None:
                ps = str(st.get("processing_status", ""))
                if ps not in {"processed", "processed_no_detections"}:
                    frame_ok = False
                    frame_reason = f"FRAME_STATUS_{ps.upper()}"
            if elig["require_analysis_window"]:
                if window is None and windows:
                    frame_ok = False
                    frame_reason = "NO_ANALYSIS_WINDOW"
                elif window is not None:
                    if str(window.get("playability")) not in set(elig["playability_allowed"]):
                        frame_ok = False
                        frame_reason = "NON_PLAYABLE_WINDOW"
                    if str(window.get("tracking_eligibility")) not in set(
                        elig["tracking_eligibility_allowed"]
                    ):
                        frame_ok = False
                        frame_reason = "TRACKING_INELIGIBLE"

            if not frame_ok:
                for d in dets:
                    did = int(d["detection_id"])
                    attr = attr_by_key.get((fi, did))
                    entity = str(attr.get("entity_type", "unknown")) if attr else "unknown"
                    if entity != "human":
                        all_assignments.append(
                            make_non_human_skip(
                                run_id=rid,
                                video_id=vid,
                                frame_index=fi,
                                detection_id=did,
                                entity_type=entity,
                                config=config,
                                config_fingerprint=cfg_fp,
                            )
                        )
                    else:
                        all_assignments.append(
                            make_not_eligible(
                                run_id=rid,
                                video_id=vid,
                                frame_index=fi,
                                detection_id=did,
                                reason=frame_reason,
                                config=config,
                                config_fingerprint=cfg_fp,
                            )
                        )
                continue

            # Build features for humans in this frame
            human_feats: list[RoleFeatures] = []
            human_meta: list[tuple[int, Mapping[str, Any]]] = []
            frame_bgr = None
            fw = fh = 0.0
            if src_path is not None and src_path.is_file():
                frame_bgr = _read_frame_bgr(src_path, fi)
                if frame_bgr is not None:
                    fh, fw = float(frame_bgr.shape[0]), float(frame_bgr.shape[1])

            for d in dets:
                did = int(d["detection_id"])
                attr = attr_by_key.get((fi, did))
                entity = str(attr.get("entity_type", "human")) if attr else "human"
                if entity != "human":
                    all_assignments.append(
                        make_non_human_skip(
                            run_id=rid,
                            video_id=vid,
                            frame_index=fi,
                            detection_id=did,
                            entity_type=entity,
                            config=config,
                            config_fingerprint=cfg_fp,
                        )
                    )
                    continue
                bbox = [
                    float(d["bbox_x1"]),
                    float(d["bbox_y1"]),
                    float(d["bbox_x2"]),
                    float(d["bbox_y2"]),
                ]
                try:
                    if frame_bgr is not None and fw > 0 and fh > 0:
                        crop = _crop_bgr(frame_bgr, bbox)
                        if crop is None or crop.size == 0:
                            all_assignments.append(
                                make_not_eligible(
                                    run_id=rid,
                                    video_id=vid,
                                    frame_index=fi,
                                    detection_id=did,
                                    reason="INVALID_CROP",
                                    config=config,
                                    config_fingerprint=cfg_fp,
                                )
                            )
                            continue
                        # Crops are NOT persisted by default.
                        feat = extract_features_from_crop(
                            crop,
                            detection_id=did,
                            frame_index=fi,
                            bbox_xyxy=bbox,
                            frame_width=fw,
                            frame_height=fh,
                            config=config,
                        )
                    elif allow_synthetic_without_video:
                        fw2, fh2 = synthetic_frame_size or (1920.0, 1080.0)
                        feat = extract_synthetic_features(
                            detection_id=did,
                            frame_index=fi,
                            bbox_xyxy=bbox,
                            frame_width=fw2,
                            frame_height=fh2,
                            config=config,
                        )
                    else:
                        all_assignments.append(
                            make_not_eligible(
                                run_id=rid,
                                video_id=vid,
                                frame_index=fi,
                                detection_id=did,
                                reason="NO_VIDEO_FOR_CROP",
                                config=config,
                                config_fingerprint=cfg_fp,
                            )
                        )
                        continue
                    human_feats.append(feat)
                    human_meta.append((did, d))
                except Exception:  # noqa: BLE001
                    failed += 1
                    all_assignments.append(
                        RoleAssignment(
                            run_id=rid,
                            video_id=vid,
                            frame_index=fi,
                            detection_id=did,
                            role_label=RoleLabel.UNKNOWN,
                            role_source=RoleSource.DOWNSTREAM_CLASSIFIER,
                            role_score=None,
                            assignment_status=AssignmentStatus.FAILED,
                            evidence_codes=("FEATURE_EXTRACTION_FAILED",),
                            review_status=ReviewStatus.NEEDS_REVIEW,
                            review_required=True,
                            crop_quality=0.0,
                            margin=0.0,
                            cluster_id=None,
                            config_fingerprint=cfg_fp,
                            classifier_id=str(config["classifier_id"]),
                            classifier_version=str(config["classifier_version"]),
                            provenance={"stage": "5D", "error": "feature_failed"},
                        )
                    )

            if human_feats:
                clusters, cass = cluster_kit_colors(human_feats, config=config)
                assigned = assign_roles_for_frame(
                    run_id=rid,
                    video_id=vid,
                    frame_index=fi,
                    features=human_feats,
                    clusters=clusters,
                    cluster_assignments=cass,
                    config=config,
                    config_fingerprint=cfg_fp,
                )
                all_assignments.extend(assigned)

    # Merge: prefer role assignments for humans; keep non-updated attrs for missing keys.
    out_attr_rows: list[dict[str, Any]] = []
    assigned_keys = {(a.frame_index, a.detection_id): a for a in all_assignments}
    for key, attr in attr_by_key.items():
        if key in assigned_keys:
            row = assigned_keys[key].to_attribute_row()
            # Validate contract shape
            DetectionAttributes.from_dict(row)
            out_attr_rows.append(row)
        else:
            out_attr_rows.append(dict(attr))
    # Include assignments for keys not in input attributes (synthetic path).
    for key, a in assigned_keys.items():
        if key not in attr_by_key:
            row = a.to_attribute_row()
            DetectionAttributes.from_dict(row)
            out_attr_rows.append(row)

    out_attr_rows.sort(key=lambda r: (int(r["frame_index"]), int(r["detection_id"])))

    # Counts
    status_counts = {
        "classified": 0,
        "abstained": 0,
        "not_eligible": 0,
        "skipped": 0,
        "failed": 0,
    }
    for a in all_assignments:
        status_counts[a.assignment_status.value] = (
            status_counts.get(a.assignment_status.value, 0) + 1
        )
    role_counts = count_roles([{"role_label": a.role_label.value} for a in all_assignments])
    review_required = sum(1 for a in all_assignments if a.review_required)
    human_in = sum(1 for r in attr_rows if str(r.get("entity_type")) == "human")

    # Consistency: classified+abstained+not_eligible+skipped+failed == len(assignments)
    total_status = sum(status_counts.values())
    if total_status != len(all_assignments):
        return _fail(error_code="RECEIPT_COUNT_MISMATCH", exit_code=3, config_fingerprint=cfg_fp)
    if failed and status_counts["classified"] == len(all_assignments) and len(all_assignments) > 0:
        return _fail(error_code="FALSE_SUCCESS", exit_code=3, config_fingerprint=cfg_fp)

    try:
        table = _rows_to_table(out_attr_rows, "detection_attributes")
        write_contract_parquet(
            table, attr_out, get_contract("detection_attributes", 1), contain_root=root
        )
    except Exception:  # noqa: BLE001
        return _fail(error_code="OUTPUT_WRITE_FAILED", exit_code=3, config_fingerprint=cfg_fp)

    # Evaluation
    gt_rows = None
    if ground_truth:
        gp = Path(ground_truth)
        try:
            if gp.suffix.lower() == ".json":
                payload = json.loads(gp.read_text(encoding="utf-8"))
                gt_rows = list(payload.get("roles") or payload.get("ground_truth") or [])
            else:
                gt_rows = read_contract_parquet(
                    gp, get_contract("detection_attributes", 1)
                ).to_pylist()
        except Exception:  # noqa: BLE001
            gt_rows = None
    pred_eval_rows = []
    for a in all_assignments:
        pred_eval_rows.append(
            {
                "frame_index": a.frame_index,
                "detection_id": a.detection_id,
                "role_label": a.role_label.value,
                "assignment_status": a.assignment_status.value,
            }
        )
    metrics = (
        evaluate_roles_from_rows(pred_eval_rows, gt_rows)
        if gt_rows is not None
        else not_evaluated_role_metrics()
    )
    if metrics.status != "EVALUATED":
        # Ensure exact code
        metrics = not_evaluated_role_metrics(reason=NOT_EVALUATED_ROLE)

    try:
        write_json_record(eval_out, metrics.to_dict(), contain_root=root, overwrite=False)
    except Exception:  # noqa: BLE001
        return _fail(error_code="EVAL_WRITE_FAILED", exit_code=3, config_fingerprint=cfg_fp)

    completed = _utc_now()
    det_hash = sha256_file(det_path)
    attr_in_hash = sha256_file(attr_path)
    attr_out_hash = sha256_file(attr_out)
    receipt = {
        "schema_version": 1,
        "receipt_id": f"role_{rid}",
        "run_id": rid,
        "video_id": vid,
        "classifier_id": config["classifier_id"],
        "classifier_version": config["classifier_version"],
        "config_fingerprint": cfg_fp,
        "started_at_utc": started,
        "completed_at_utc": completed,
        "status": "failed" if failed and not status_counts["classified"] else "succeeded",
        "input_hashes": {
            "detections_sha256": det_hash,
            "detection_attributes_sha256": attr_in_hash,
        },
        "output_hashes": {"detection_attributes_sha256": attr_out_hash},
        "human_detection_count": human_in,
        "assignment_counts": status_counts,
        "role_counts": role_counts,
        "review_required_count": review_required,
        "ground_truth_evaluation_status": metrics.status,
        "other_maps_to": "staff",
        "crops_persisted": False,
        "team_id": None,
        "artifacts": {
            "detection_attributes_parquet": str(attr_out),
            "role_evaluation_json": str(eval_out),
        },
        "provenance": {
            "stage": "5D",
            "label": "human_role_baseline",
            "weightless": True,
            "detections_contract_unchanged": True,
        },
        "errors": (
            [{"code": "PARTIAL_FEATURE_FAILURES", "message": f"failed={failed}"}] if failed else []
        ),
    }
    # Count consistency check for receipt
    if sum(status_counts.values()) != len(all_assignments):
        return _fail(error_code="RECEIPT_COUNT_MISMATCH", exit_code=3, config_fingerprint=cfg_fp)
    try:
        write_json_record(receipt_out, receipt, contain_root=root, overwrite=False)
    except Exception:  # noqa: BLE001
        return _fail(error_code="RECEIPT_WRITE_FAILED", exit_code=3, config_fingerprint=cfg_fp)

    # Ensure no crop artifacts left under output by default.
    crops_dir = out / "crops"
    if crops_dir.exists() and not config["output_policy"]["persist_crops"]:
        import shutil

        shutil.rmtree(crops_dir, ignore_errors=True)

    summary = {
        "status": receipt["status"],
        "assignment_counts": status_counts,
        "role_counts": role_counts,
        "review_required_count": review_required,
        "evaluation_status": metrics.status,
        "config_fingerprint": cfg_fp,
        "receipt_fingerprint": hash_canonical_json(
            {k: receipt[k] for k in ("config_fingerprint", "assignment_counts", "role_counts")}
        ),
    }
    return RoleServiceResult(
        accepted=True,
        exit_code=0,
        error_code=None,
        config_fingerprint=cfg_fp,
        attributes_parquet=str(attr_out),
        receipt_json=str(receipt_out),
        evaluation_json=str(eval_out),
        summary=summary,
    )


__all__ = [
    "RoleServiceError",
    "RoleServiceResult",
    "classify_from_synthetic_humans",
    "run_human_role_classification",
]
