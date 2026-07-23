"""Stage 5E detection fusion: merge human + ball + role into one validated bundle.

Frame status merge policy (deterministic):
- Union frame keys from human and ball status tables.
- Prefer processed / processed_no_detections over skipped when either detector ran.
- Preserve not_eligible when both sides are not_eligible (or sole side is).
- Eligibility conflict: one not_eligible and the other processed* → hard fail.
- Counts: human_count from human status contribution, ball_count from ball;
  detection_count = human_count + ball_count when status is processed*.
- Never invent zero-as-processed counts for unprocessed statuses.
- No cross-class NMS between human and ball.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from football_analytics.perception.types import (
    EntityType,
    ProcessingStatus,
    RoleLabel,
)
from football_analytics.perception.validation import validate_detection_bundle

PROCESSED_SET = {
    ProcessingStatus.PROCESSED.value,
    ProcessingStatus.PROCESSED_NO_DETECTIONS.value,
}
UNPROCESSED_SET = {
    ProcessingStatus.SKIPPED.value,
    ProcessingStatus.FAILED.value,
    ProcessingStatus.NOT_ELIGIBLE.value,
}


class DetectionFusionError(ValueError):
    """Fusion alignment or merge failure with explicit error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class FusionResult:
    detections: list[dict[str, Any]]
    frame_status: list[dict[str, Any]]
    attributes: list[dict[str, Any]]
    id_remap: dict[tuple[Any, ...], int]
    run_id: str
    video_id: str
    human_detection_count: int
    ball_detection_count: int


def _rows(table_or_rows: Any) -> list[dict[str, Any]]:
    if table_or_rows is None:
        return []
    if hasattr(table_or_rows, "to_pylist"):
        return list(table_or_rows.to_pylist())
    return [dict(r) for r in table_or_rows]


def _receipt_get(receipt: Mapping[str, Any] | None, *keys: str) -> Any:
    if receipt is None:
        return None
    for k in keys:
        if k in receipt and receipt[k] is not None:
            return receipt[k]
    artifacts = receipt.get("artifacts")
    if isinstance(artifacts, Mapping):
        for k in keys:
            if k in artifacts and artifacts[k] is not None:
                return artifacts[k]
    provenance = receipt.get("provenance")
    if isinstance(provenance, Mapping):
        for k in keys:
            if k in provenance and provenance[k] is not None:
                return provenance[k]
    return None


def align_upstream_receipts(
    *,
    human_receipt: Mapping[str, Any] | None,
    ball_receipt: Mapping[str, Any] | None,
    role_receipt: Mapping[str, Any] | None,
    config: Mapping[str, Any],
    expected_run_id: str | None = None,
    expected_video_id: str | None = None,
    expected_source_sha: str | None = None,
    expected_timeline_fp: str | None = None,
) -> dict[str, Any]:
    """Align run/video/timeline/source SHA across upstream receipts or raise."""
    align = config["alignment"]
    if align["fail_on_missing_receipt"] and (
        human_receipt is None or ball_receipt is None or role_receipt is None
    ):
        raise DetectionFusionError("MISSING_RECEIPT", "human/ball/role receipts required")

    run_ids = [
        r.get("run_id")
        for r in (human_receipt, ball_receipt, role_receipt)
        if r is not None and r.get("run_id") is not None
    ]
    if align["require_run_id_match"] and run_ids:
        if len(set(str(x) for x in run_ids)) != 1:
            raise DetectionFusionError("RUN_ID_MISMATCH", f"run_id mismatch: {run_ids}")
        if expected_run_id is not None and str(run_ids[0]) != expected_run_id:
            raise DetectionFusionError(
                "RUN_ID_MISMATCH", f"run_id != expected ({run_ids[0]} vs {expected_run_id})"
            )

    video_ids = []
    for r in (human_receipt, ball_receipt, role_receipt):
        if r is None:
            continue
        vid = r.get("video_id")
        if vid is None:
            # detection receipts may omit video_id; try artifacts
            vid = _receipt_get(r, "video_id")
        if vid is not None:
            video_ids.append(str(vid))
    if align["require_video_id_match"] and video_ids:
        if len(set(video_ids)) != 1:
            raise DetectionFusionError("VIDEO_ID_MISMATCH", f"video_id mismatch: {video_ids}")
        if expected_video_id is not None and video_ids[0] != expected_video_id:
            raise DetectionFusionError(
                "VIDEO_ID_MISMATCH",
                f"video_id != expected ({video_ids[0]} vs {expected_video_id})",
            )

    frames_refs = [
        str(_receipt_get(r, "frames_ref"))
        for r in (human_receipt, ball_receipt)
        if r is not None and _receipt_get(r, "frames_ref") is not None
    ]
    if align["require_frames_ref_match"] and len(frames_refs) >= 2:
        # Compare basenames so absolute path variants still align when identical timeline.
        bases = {PathName(x) for x in frames_refs}
        if len(bases) != 1:
            raise DetectionFusionError("TIMELINE_MISMATCH", f"frames_ref mismatch: {frames_refs}")

    shas = []
    for r in (human_receipt, ball_receipt):
        if r is None:
            continue
        sha = _receipt_get(r, "source_video_sha256", "source_sha256")
        if sha is not None:
            shas.append(str(sha).lower())
    if expected_source_sha is not None:
        shas.append(expected_source_sha.lower())
    if align["require_source_sha_match"] and len(shas) >= 2 and len(set(shas)) != 1:
        raise DetectionFusionError("SOURCE_SHA_MISMATCH", f"source sha mismatch: {shas}")

    timeline_fps = []
    for r in (human_receipt, ball_receipt, role_receipt):
        if r is None:
            continue
        tfp = _receipt_get(r, "timeline_fingerprint", "frames_fingerprint")
        if tfp is not None:
            timeline_fps.append(str(tfp).lower())
    if expected_timeline_fp is not None:
        timeline_fps.append(expected_timeline_fp.lower())
    if align["require_frames_ref_match"] and len(timeline_fps) >= 2 and len(set(timeline_fps)) != 1:
        raise DetectionFusionError(
            "TIMELINE_FINGERPRINT_MISMATCH",
            f"timeline fingerprint mismatch: {timeline_fps}",
        )

    return {
        "run_id": str(run_ids[0]) if run_ids else expected_run_id,
        "video_id": video_ids[0] if video_ids else expected_video_id,
        "source_video_sha256": shas[0] if shas else expected_source_sha,
        "timeline_fingerprint": timeline_fps[0] if timeline_fps else expected_timeline_fp,
        "frames_ref": frames_refs[0] if frames_refs else None,
    }


def PathName(path: str) -> str:
    """Basename helper for frames_ref comparison."""
    return path.replace("\\", "/").rstrip("/").split("/")[-1]


def _status_rank(status: str) -> int:
    order = {
        ProcessingStatus.FAILED.value: 50,
        ProcessingStatus.PROCESSED.value: 40,
        ProcessingStatus.PROCESSED_NO_DETECTIONS.value: 30,
        ProcessingStatus.SKIPPED.value: 20,
        ProcessingStatus.NOT_ELIGIBLE.value: 10,
    }
    return order.get(status, 0)


def merge_frame_status(
    human_status: Sequence[Mapping[str, Any]],
    ball_status: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Merge per-frame status rows with documented deterministic policy."""
    fusion = config["fusion"]
    by_key: dict[tuple[Any, Any, Any], dict[str, dict[str, Any]]] = {}
    for side, rows in (("human", human_status), ("ball", ball_status)):
        for r in rows:
            key = (r["run_id"], r["video_id"], r["frame_index"])
            by_key.setdefault(key, {})[side] = dict(r)

    out: list[dict[str, Any]] = []
    for key in sorted(by_key.keys(), key=lambda k: (str(k[0]), str(k[1]), int(k[2]))):
        sides = by_key[key]
        h = sides.get("human")
        b = sides.get("ball")
        h_st = h["processing_status"] if h else None
        b_st = b["processing_status"] if b else None

        if fusion["eligibility_conflict_fails"] and (
            (h_st == ProcessingStatus.NOT_ELIGIBLE.value and b_st in PROCESSED_SET)
            or (b_st == ProcessingStatus.NOT_ELIGIBLE.value and h_st in PROCESSED_SET)
        ):
            raise DetectionFusionError(
                "ELIGIBILITY_CONFLICT",
                f"not_eligible vs processed for frame {key}",
            )

        # Choose base row template.
        base = copy.deepcopy(h or b)
        assert base is not None

        human_count = int(h["human_count"]) if h and h_st in PROCESSED_SET else 0
        ball_count = int(b["ball_count"]) if b and b_st in PROCESSED_SET else 0
        # Prefer human_count from human table even if stored under detection_count only.
        if h and h_st in PROCESSED_SET:
            human_count = int(h.get("human_count") or 0)
        if b and b_st in PROCESSED_SET:
            ball_count = int(b.get("ball_count") or 0)

        candidates = [s for s in (h_st, b_st) if s is not None]
        if not candidates:
            continue

        if fusion["prefer_processed_over_skipped"]:
            if any(s in PROCESSED_SET for s in candidates):
                if human_count + ball_count > 0:
                    status = ProcessingStatus.PROCESSED.value
                elif any(s == ProcessingStatus.PROCESSED.value for s in candidates):
                    # One side claimed processed with dets that may be remapped later —
                    # use counts; if both empty → processed_no_detections.
                    status = (
                        ProcessingStatus.PROCESSED.value
                        if human_count + ball_count > 0
                        else ProcessingStatus.PROCESSED_NO_DETECTIONS.value
                    )
                else:
                    status = ProcessingStatus.PROCESSED_NO_DETECTIONS.value
            elif ProcessingStatus.FAILED.value in candidates:
                status = ProcessingStatus.FAILED.value
            elif (
                (
                    fusion["preserve_not_eligible"]
                    and all(s == ProcessingStatus.NOT_ELIGIBLE.value for s in candidates)
                )
                or ProcessingStatus.NOT_ELIGIBLE.value in candidates
                and all(
                    s in {ProcessingStatus.NOT_ELIGIBLE.value, ProcessingStatus.SKIPPED.value}
                    for s in candidates
                )
            ):
                status = ProcessingStatus.NOT_ELIGIBLE.value
            else:
                status = ProcessingStatus.SKIPPED.value
        else:
            status = max(candidates, key=_status_rank)

        det_count = human_count + ball_count
        if status in UNPROCESSED_SET or status == ProcessingStatus.PROCESSED_NO_DETECTIONS.value:
            human_count = 0
            ball_count = 0
            det_count = 0
        elif status == ProcessingStatus.PROCESSED.value and det_count == 0:
            status = ProcessingStatus.PROCESSED_NO_DETECTIONS.value

        error_code = None
        if status == ProcessingStatus.FAILED.value:
            error_code = (
                (h or {}).get("error_code")
                or (b or {}).get("error_code")
                or "FUSION_UPSTREAM_FAILED"
            )

        skip_reason = None
        if status in {
            ProcessingStatus.SKIPPED.value,
            ProcessingStatus.NOT_ELIGIBLE.value,
        }:
            skip_reason = (h or {}).get("skip_reason") or (b or {}).get("skip_reason")

        aw = (h or {}).get("analysis_window_id")
        if aw is None:
            aw = (b or {}).get("analysis_window_id")

        out.append(
            {
                "run_id": key[0],
                "video_id": key[1],
                "frame_index": key[2],
                "video_time_us": base.get("video_time_us"),
                "analysis_window_id": aw,
                "processing_status": status,
                "eligibility": base.get("eligibility")
                or (h or {}).get("eligibility")
                or (b or {}).get("eligibility"),
                "detector_id": "detection_fusion_v1",
                "input_artifact_ref": None,
                "detection_count": det_count,
                "human_count": human_count,
                "ball_count": ball_count,
                "skip_reason": skip_reason,
                "error_code": error_code,
                "coverage": base.get("coverage"),
                "provenance_json": '{"stage":"5E","fusion":"frame_status_v1"}',
                "contract_version": 1,
            }
        )
    return out


def _validate_bbox_row(row: Mapping[str, Any]) -> None:
    for key in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "confidence"):
        v = row.get(key)
        if v is None or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise DetectionFusionError("INVALID_BBOX", f"non-finite {key} in detection")
    x1, y1, x2, y2 = (
        float(row["bbox_x1"]),
        float(row["bbox_y1"]),
        float(row["bbox_x2"]),
        float(row["bbox_y2"]),
    )
    if not (x1 < x2 and y1 < y2):
        raise DetectionFusionError("INVALID_BBOX", "zero/negative area bbox")
    conf = float(row["confidence"])
    if not (0.0 <= conf <= 1.0):
        raise DetectionFusionError("INVALID_BBOX", "confidence out of bounds")


def merge_detections_and_attributes(
    *,
    human_detections: Sequence[Mapping[str, Any]],
    ball_detections: Sequence[Mapping[str, Any]],
    human_attributes: Sequence[Mapping[str, Any]],
    ball_attributes: Sequence[Mapping[str, Any]],
    role_attributes: Sequence[Mapping[str, Any]] | None,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[Any, ...], int]]:
    """Merge detections + attributes; remap ball IDs for uniqueness. No cross-class NMS."""
    if config["fusion"].get("cross_class_nms") is not False:
        raise DetectionFusionError("CROSS_CLASS_NMS_FORBIDDEN", "cross-class NMS must be false")

    role_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for r in role_attributes or []:
        key = (r["run_id"], r["video_id"], r["frame_index"], r["detection_id"])
        role_by_key[key] = dict(r)

    human_attrs_by_key = {
        (r["run_id"], r["video_id"], r["frame_index"], r["detection_id"]): dict(r)
        for r in human_attributes
    }

    fused_dets: list[dict[str, Any]] = []
    fused_attrs: list[dict[str, Any]] = []
    id_remap: dict[tuple[Any, ...], int] = {}

    # Humans first — keep IDs.
    for d in sorted(
        human_detections,
        key=lambda r: (r["run_id"], r["video_id"], int(r["frame_index"]), int(r["detection_id"])),
    ):
        _validate_bbox_row(d)
        row = dict(d)
        fused_dets.append(row)
        key = (row["run_id"], row["video_id"], row["frame_index"], row["detection_id"])
        attr = role_by_key.get(key) or human_attrs_by_key.get(key)
        if attr is None:
            raise DetectionFusionError("DANGLING_FK", f"human detection missing attributes {key}")
        if attr.get("entity_type") != EntityType.HUMAN.value:
            raise DetectionFusionError(
                "ENTITY_MISMATCH", f"human detection attribute entity_type invalid for {key}"
            )
        a = dict(attr)
        a["detection_id"] = row["detection_id"]
        fused_attrs.append(a)

    # Occupied IDs per frame after humans.
    occupied: dict[tuple[Any, Any, Any], set[int]] = {}
    for d in fused_dets:
        fkey = (d["run_id"], d["video_id"], d["frame_index"])
        occupied.setdefault(fkey, set()).add(int(d["detection_id"]))

    ball_attr_by_key = {
        (r["run_id"], r["video_id"], r["frame_index"], r["detection_id"]): dict(r)
        for r in ball_attributes
    }

    for d in sorted(
        ball_detections,
        key=lambda r: (r["run_id"], r["video_id"], int(r["frame_index"]), int(r["detection_id"])),
    ):
        _validate_bbox_row(d)
        row = dict(d)
        fkey = (row["run_id"], row["video_id"], row["frame_index"])
        old_id = int(row["detection_id"])
        new_id = old_id
        if config["fusion"]["remap_ball_detection_ids"]:
            used = occupied.setdefault(fkey, set())
            if new_id in used:
                new_id = (max(used) + 1) if used else 0
                while new_id in used:
                    new_id += 1
            used.add(new_id)
        else:
            used = occupied.setdefault(fkey, set())
            if new_id in used:
                raise DetectionFusionError(
                    "DUPLICATE_DETECTION_ID",
                    f"detection_id collision at {fkey + (old_id,)}",
                )
            used.add(new_id)

        if new_id != old_id:
            id_remap[fkey + (old_id,)] = new_id
        row["detection_id"] = new_id
        fused_dets.append(row)

        akey = (row["run_id"], row["video_id"], row["frame_index"], old_id)
        attr = ball_attr_by_key.get(akey)
        if attr is None:
            raise DetectionFusionError("DANGLING_FK", f"ball detection missing attributes {akey}")
        if attr.get("entity_type") != EntityType.BALL.value:
            raise DetectionFusionError(
                "ENTITY_MISMATCH", f"ball attribute entity_type invalid for {akey}"
            )
        if attr.get("role_label") != RoleLabel.UNKNOWN.value:
            raise DetectionFusionError(
                "BALL_ROLE_FORBIDDEN",
                f"ball must have role_label=unknown for {akey}",
            )
        a = dict(attr)
        a["detection_id"] = new_id
        a["role_label"] = RoleLabel.UNKNOWN.value
        fused_attrs.append(a)

    # Ensure unique detection PKs.
    seen: set[tuple[Any, ...]] = set()
    for d in fused_dets:
        pk = (d["run_id"], d["video_id"], d["frame_index"], d["detection_id"])
        if pk in seen:
            raise DetectionFusionError("DUPLICATE_DETECTION_ID", f"duplicate PK {pk}")
        seen.add(pk)

    return fused_dets, fused_attrs, id_remap


def fuse_detection_bundle(
    *,
    human_detections: Any,
    human_frame_status: Any,
    human_attributes: Any,
    ball_detections: Any,
    ball_frame_status: Any,
    ball_attributes: Any,
    role_attributes: Any,
    human_receipt: Mapping[str, Any] | None,
    ball_receipt: Mapping[str, Any] | None,
    role_receipt: Mapping[str, Any] | None,
    config: Mapping[str, Any],
    frames: Any | None = None,
    analysis_windows: Any | None = None,
    expected_run_id: str | None = None,
    expected_video_id: str | None = None,
    expected_source_sha: str | None = None,
    expected_timeline_fp: str | None = None,
    validate: bool = True,
) -> FusionResult:
    """Full fusion path: align → merge dets/attrs/status → optional validate."""
    align_meta = align_upstream_receipts(
        human_receipt=human_receipt,
        ball_receipt=ball_receipt,
        role_receipt=role_receipt,
        config=config,
        expected_run_id=expected_run_id,
        expected_video_id=expected_video_id,
        expected_source_sha=expected_source_sha,
        expected_timeline_fp=expected_timeline_fp,
    )

    h_dets = _rows(human_detections)
    b_dets = _rows(ball_detections)
    h_status = _rows(human_frame_status)
    b_status = _rows(ball_frame_status)
    h_attrs = _rows(human_attributes)
    b_attrs = _rows(ball_attributes)
    r_attrs = _rows(role_attributes)

    # Table-level run/video alignment
    all_rows = h_dets + b_dets + h_status + b_status + h_attrs + b_attrs + r_attrs
    if all_rows:
        run_ids = {str(r["run_id"]) for r in all_rows if "run_id" in r}
        video_ids = {str(r["video_id"]) for r in all_rows if "video_id" in r}
        if len(run_ids) != 1:
            raise DetectionFusionError("RUN_ID_MISMATCH", f"table run_id mismatch: {run_ids}")
        if len(video_ids) != 1:
            raise DetectionFusionError("VIDEO_ID_MISMATCH", f"table video_id mismatch: {video_ids}")
        rid = next(iter(run_ids))
        vid = next(iter(video_ids))
        if expected_run_id is not None and rid != expected_run_id:
            raise DetectionFusionError("RUN_ID_MISMATCH", "table run_id != expected")
        if expected_video_id is not None and vid != expected_video_id:
            raise DetectionFusionError("VIDEO_ID_MISMATCH", "table video_id != expected")
    else:
        rid = str(align_meta.get("run_id") or expected_run_id or "")
        vid = str(align_meta.get("video_id") or expected_video_id or "")
        if not rid or not vid:
            raise DetectionFusionError("ID_INFERENCE_FAIL", "cannot infer run_id/video_id")

    dets, attrs, id_remap = merge_detections_and_attributes(
        human_detections=h_dets,
        ball_detections=b_dets,
        human_attributes=h_attrs,
        ball_attributes=b_attrs,
        role_attributes=r_attrs,
        config=config,
    )
    status = merge_frame_status(h_status, b_status, config=config)

    # Recompute status counts from fused detections to stay consistent.
    from collections import defaultdict

    by_frame: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    attr_lookup = {
        (a["run_id"], a["video_id"], a["frame_index"], a["detection_id"]): a for a in attrs
    }
    for d in dets:
        by_frame[(d["run_id"], d["video_id"], d["frame_index"])].append(d)

    for s in status:
        fkey = (s["run_id"], s["video_id"], s["frame_index"])
        frame_dets = by_frame.get(fkey, [])
        human_n = 0
        ball_n = 0
        for d in frame_dets:
            a = attr_lookup.get((d["run_id"], d["video_id"], d["frame_index"], d["detection_id"]))
            if a and a["entity_type"] == EntityType.BALL.value:
                ball_n += 1
            else:
                human_n += 1
        if s["processing_status"] in PROCESSED_SET:
            if human_n + ball_n == 0:
                s["processing_status"] = ProcessingStatus.PROCESSED_NO_DETECTIONS.value
                s["detection_count"] = 0
                s["human_count"] = 0
                s["ball_count"] = 0
            else:
                s["processing_status"] = ProcessingStatus.PROCESSED.value
                s["detection_count"] = human_n + ball_n
                s["human_count"] = human_n
                s["ball_count"] = ball_n
        else:
            if frame_dets:
                raise DetectionFusionError(
                    "FRAME_STATUS_INCONSISTENT",
                    f"unprocessed frame has detections: {fkey}",
                )
            s["detection_count"] = 0
            s["human_count"] = 0
            s["ball_count"] = 0

    human_n = sum(1 for a in attrs if a["entity_type"] == EntityType.HUMAN.value)
    ball_n = sum(1 for a in attrs if a["entity_type"] == EntityType.BALL.value)

    if validate:
        # Build receipt stub for count check from tables.
        processed = sum(
            1
            for r in status
            if r["processing_status"]
            in {
                ProcessingStatus.PROCESSED.value,
                ProcessingStatus.PROCESSED_NO_DETECTIONS.value,
            }
        )
        skipped = sum(1 for r in status if r["processing_status"] == ProcessingStatus.SKIPPED.value)
        failed = sum(1 for r in status if r["processing_status"] == ProcessingStatus.FAILED.value)
        no_det = sum(
            1
            for r in status
            if r["processing_status"] == ProcessingStatus.PROCESSED_NO_DETECTIONS.value
        )
        receipt_stub = {
            "processed_frame_count": processed,
            "skipped_frame_count": skipped,
            "failed_frame_count": failed,
            "processed_no_detection_count": no_det,
            "total_detection_count": len(dets),
            "human_detection_count": human_n,
            "ball_detection_count": ball_n,
        }
        import pyarrow as pa

        from football_analytics.data.compiler import compile_arrow_schema, get_contract

        def _tbl(name: str, rows: list[dict[str, Any]]) -> Any:
            schema = compile_arrow_schema(get_contract(name, 1))
            return pa.Table.from_pylist(rows, schema=schema) if rows else schema.empty_table()

        vr = validate_detection_bundle(
            detections=_tbl("detections", dets),
            frame_status=_tbl("detection_frame_status", status),
            attributes=_tbl("detection_attributes", attrs),
            frames=frames,
            analysis_windows=analysis_windows,
            receipt=receipt_stub,
        )
        if vr.status == "FAIL":
            raise DetectionFusionError(
                "BUNDLE_INVALID",
                "; ".join(vr.errors[:5]) if vr.errors else "validate_detection_bundle failed",
            )

    return FusionResult(
        detections=dets,
        frame_status=status,
        attributes=attrs,
        id_remap=id_remap,
        run_id=rid,
        video_id=vid,
        human_detection_count=human_n,
        ball_detection_count=ball_n,
    )


__all__ = [
    "DetectionFusionError",
    "FusionResult",
    "align_upstream_receipts",
    "merge_frame_status",
    "merge_detections_and_attributes",
    "fuse_detection_bundle",
]
