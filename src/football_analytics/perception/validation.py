"""Detection bundle validation (detections + frame_status + attributes + windows)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from football_analytics.data.types import ValidationResult
from football_analytics.data.validation import validate_table
from football_analytics.perception.types import (
    UNPROCESSED_STATUSES,
    EntityType,
    PerceptionContractError,
    ProcessingStatus,
    RoleLabel,
)


def _keys(table: Any, cols: list[str]) -> set[tuple[Any, ...]]:
    if table is None or table.num_rows == 0:
        return set()
    arrays = [table.column(c).to_pylist() for c in cols]
    out = set()
    for i in range(table.num_rows):
        out.add(tuple(a[i] for a in arrays))
    return out


def _rows(table: Any | None) -> list[dict[str, Any]]:
    if table is None:
        return []
    return table.to_pylist()


def validate_detection_bundle(
    *,
    detections: Any | None,
    frame_status: Any | None,
    attributes: Any | None,
    frames: Any | None = None,
    videos: Any | None = None,
    analysis_windows: Any | None = None,
    specs: Mapping[str, Any] | None = None,
    receipt: Mapping[str, Any] | None = None,
) -> ValidationResult:
    """Validate detection sidecar consistency against detections v1 and windows."""
    result = ValidationResult(contract="detection_bundle", version=1)

    if specs is not None:
        for name, table in (
            ("detections", detections),
            ("detection_frame_status", frame_status),
            ("detection_attributes", attributes),
            ("frames", frames),
            ("videos", videos),
            ("analysis_windows", analysis_windows),
        ):
            if table is None or name not in specs:
                continue
            vr = validate_table(table, specs[name])
            if vr.status == "FAIL":
                for e in vr.errors[:10]:
                    result.err(f"{name}: {e}")

    frame_keys = (
        _keys(frames, ["run_id", "video_id", "frame_index"]) if frames is not None else None
    )
    det_keys = (
        _keys(detections, ["run_id", "video_id", "frame_index", "detection_id"])
        if detections is not None
        else set()
    )
    window_keys = (
        _keys(analysis_windows, ["run_id", "video_id", "analysis_window_id"])
        if analysis_windows is not None
        else None
    )

    # Duplicate detection IDs already covered by PK semantic validate_table; reinforce.
    if detections is not None and len(det_keys) != detections.num_rows:
        result.err("duplicate detection primary keys")

    det_by_frame: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in _rows(detections):
        det_by_frame[(r["run_id"], r["video_id"], r["frame_index"])].append(r)

    attr_rows = _rows(attributes)
    attr_keys = {
        (r["run_id"], r["video_id"], r["frame_index"], r["detection_id"]) for r in attr_rows
    }
    if attributes is not None and len(attr_keys) != attributes.num_rows:
        result.err("duplicate detection_attributes primary keys")

    for r in attr_rows:
        key = (r["run_id"], r["video_id"], r["frame_index"], r["detection_id"])
        if key not in det_keys:
            result.err(f"attributes FK missing detection {key}")
            break
        if r["entity_type"] == EntityType.BALL.value and r["role_label"] != RoleLabel.UNKNOWN.value:
            result.err(f"ball role forbidden for {key}")
            break

    # Optional: every detection should have attributes when attributes table present.
    if attributes is not None and detections is not None:
        missing_attr = det_keys - attr_keys
        if missing_attr:
            result.err(f"detections missing attributes: {next(iter(missing_attr))}")

    status_rows = _rows(frame_status)
    status_keys = {(r["run_id"], r["video_id"], r["frame_index"]) for r in status_rows}
    if frame_status is not None and len(status_keys) != frame_status.num_rows:
        result.err("duplicate detection_frame_status primary keys")

    for r in status_rows:
        fkey = (r["run_id"], r["video_id"], r["frame_index"])
        if frame_keys is not None and fkey not in frame_keys:
            result.err(f"frame_status FK missing frame {fkey}")
            break
        wid = r.get("analysis_window_id")
        if wid is not None and window_keys is not None:
            wkey = (r["run_id"], r["video_id"], wid)
            if wkey not in window_keys:
                result.err(f"frame_status FK missing analysis_window {wkey}")
                break

        status = r["processing_status"]
        dets = det_by_frame.get(fkey, [])
        actual_count = len(dets)
        human = 0
        ball = 0
        for d in dets:
            # Prefer attributes when present.
            akey = (d["run_id"], d["video_id"], d["frame_index"], d["detection_id"])
            attr = next(
                (
                    a
                    for a in attr_rows
                    if (a["run_id"], a["video_id"], a["frame_index"], a["detection_id"]) == akey
                ),
                None,
            )
            if attr is not None:
                if attr["entity_type"] == EntityType.HUMAN.value:
                    human += 1
                elif attr["entity_type"] == EntityType.BALL.value:
                    ball += 1
            else:
                cname = str(d.get("class_name", "")).lower()
                if cname in {"ball", "sports_ball", "soccer_ball", "football"}:
                    ball += 1
                else:
                    human += 1

        if status == ProcessingStatus.PROCESSED_NO_DETECTIONS.value:
            if actual_count != 0 or r["detection_count"] != 0:
                result.err(f"processed_no_detections must have zero rows/counts for {fkey}")
                break
            if r["human_count"] != 0 or r["ball_count"] != 0:
                result.err(f"processed_no_detections non-zero human/ball for {fkey}")
                break
        elif status == ProcessingStatus.PROCESSED.value:
            if actual_count <= 0 or r["detection_count"] != actual_count:
                result.err(f"processed count mismatch for {fkey}")
                break
            if r["human_count"] + r["ball_count"] > r["detection_count"]:
                result.err(f"human+ball exceeds detection_count for {fkey}")
                break
        elif status in {s.value for s in UNPROCESSED_STATUSES}:
            if actual_count != 0:
                result.err(f"{status} must not have detection rows for {fkey}")
                break
            if r["detection_count"] != 0 or r["human_count"] != 0 or r["ball_count"] != 0:
                result.err(f"{status} must not invent zero-as-processed counts for {fkey}")
                break
            if status == ProcessingStatus.FAILED.value and not r.get("error_code"):
                result.err(f"failed frame missing error_code for {fkey}")
                break

        # Score bounds on detections for this frame
        for d in dets:
            conf = d.get("confidence")
            if conf is None or not (0.0 <= float(conf) <= 1.0):
                result.err(f"confidence out of bounds for detection {d.get('detection_id')}")
                break

    # Detections without frame_status when status table present → error
    if frame_status is not None and detections is not None:
        for fkey in det_by_frame:
            if fkey not in status_keys:
                result.err(f"detections without frame_status for {fkey}")
                break

    if receipt is not None:
        try:
            _validate_receipt_totals(
                receipt,
                status_rows=status_rows,
                detections=_rows(detections),
                attributes=attr_rows,
            )
        except PerceptionContractError as exc:
            result.err(str(exc))

    result.statistics["detection_rows"] = 0 if detections is None else detections.num_rows
    result.statistics["frame_status_rows"] = 0 if frame_status is None else frame_status.num_rows
    result.statistics["attribute_rows"] = 0 if attributes is None else attributes.num_rows
    return result.finalize()


def _validate_receipt_totals(
    receipt: Mapping[str, Any],
    *,
    status_rows: list[dict[str, Any]],
    detections: list[dict[str, Any]],
    attributes: list[dict[str, Any]],
) -> None:
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
    if receipt.get("processed_frame_count") != processed:
        raise PerceptionContractError("receipt processed_frame_count mismatch")
    if receipt.get("skipped_frame_count") != skipped:
        raise PerceptionContractError("receipt skipped_frame_count mismatch")
    if receipt.get("failed_frame_count") != failed:
        raise PerceptionContractError("receipt failed_frame_count mismatch")
    if receipt.get("processed_no_detection_count") != no_det:
        raise PerceptionContractError("receipt processed_no_detection_count mismatch")
    if receipt.get("total_detection_count") != len(detections):
        raise PerceptionContractError("receipt total_detection_count mismatch")
    human = sum(1 for a in attributes if a["entity_type"] == EntityType.HUMAN.value)
    ball = sum(1 for a in attributes if a["entity_type"] == EntityType.BALL.value)
    if attributes:
        if receipt.get("human_detection_count") != human:
            raise PerceptionContractError("receipt human_detection_count mismatch")
        if receipt.get("ball_detection_count") != ball:
            raise PerceptionContractError("receipt ball_detection_count mismatch")
    if receipt.get("model_sha256") is not None:
        sha = receipt["model_sha256"]
        if not isinstance(sha, str) or len(sha) != 64:
            raise PerceptionContractError("receipt model_sha256 invalid")


__all__ = ["validate_detection_bundle"]
