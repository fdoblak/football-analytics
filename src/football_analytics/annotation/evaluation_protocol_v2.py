"""Evaluation Protocol v2 — on-pitch primary metrics with uncertain ignore regions.

Holdout v1 is consumed/failed and must not be used for selection or acceptance.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from football_analytics.annotation.independent_gt import utc_now
from football_analytics.perception.detection_evaluation import (
    BBoxDetection,
    bbox_iou,
    evaluate_human_detections,
    greedy_match_iou,
)

PROTOCOL_ID = "own_video_human_eval_protocol_v2"
EXPECTED_FROZEN_FP = "4e4e46d9edabd98aad53ea2538a2a67cd5cfeb6e0444abf7254b12f01ca4f9f1"
HOLDOUT_V1_STATUS = "CONSUMED_FAILED_EVALUATION"

HEIGHT_BINS: tuple[tuple[str, float, float], ...] = (
    ("h_lt_16", 0.0, 16.0),
    ("h_16_24", 16.0, 24.0),
    ("h_24_40", 24.0, 40.0),
    ("h_40_64", 40.0, 64.0),
    ("h_ge_64", 64.0, 1e9),
)

PRIMARY_ROLES = frozenset({"player", "goalkeeper", "referee", "unknown"})
ON_PITCH_ELIG = frozenset({"on_pitch"})
IGNORE_ELIG = frozenset({"uncertain"})


def protocol_v2_definition() -> dict[str, Any]:
    body = {
        "schema": "evaluation_protocol_v2",
        "protocol_id": PROTOCOL_ID,
        "frozen_gt_fingerprint_required": EXPECTED_FROZEN_FP,
        "primary_scope": "on_pitch_human",
        "primary_roles": sorted(PRIMARY_ROLES),
        "primary_eligibility": sorted(ON_PITCH_ELIG),
        "ignore_eligibility": sorted(IGNORE_ELIG),
        "ignore_prediction_iou": 0.5,
        "match_iou": 0.5,
        "matching": "greedy_one_to_one",
        "secondary_scopes": ["all_human", "off_pitch_human", "uncertain_region_predictions"],
        "holdout_v1": {
            "status": HOLDOUT_V1_STATUS,
            "acceptance_reusable": False,
            "may_use_for_error_analysis": True,
            "may_use_for_training": False,
            "may_use_for_model_selection": False,
            "may_use_for_threshold_tuning": False,
        },
        "gt_mutation_forbidden": True,
        "defined_before_holdout_v2": True,
        "written_at_utc": utc_now(),
    }
    body["protocol_fingerprint"] = hashlib.sha256(
        json.dumps(
            {k: v for k, v in body.items() if k not in {"written_at_utc", "protocol_fingerprint"}},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return body


@dataclass(frozen=True)
class LabeledBox:
    frame_index: int
    xyxy: tuple[float, float, float, float]
    role: str
    eligibility: str
    visibility: str
    team_appearance: str

    @property
    def height(self) -> float:
        return max(0.0, self.xyxy[3] - self.xyxy[1])

    @property
    def width(self) -> float:
        return max(0.0, self.xyxy[2] - self.xyxy[0])

    @property
    def area(self) -> float:
        return self.width * self.height

    def is_primary(self) -> bool:
        return self.eligibility in ON_PITCH_ELIG and self.role in PRIMARY_ROLES

    def is_ignore(self) -> bool:
        return self.eligibility in IGNORE_ELIG


def boxes_from_frozen_frame(fr: Mapping[str, Any]) -> list[LabeledBox]:
    out: list[LabeledBox] = []
    idx = int(fr["frame_idx"])
    for h in fr.get("humans") or []:
        x1, y1, x2, y2 = (float(v) for v in h["bbox_xyxy"])
        out.append(
            LabeledBox(
                frame_index=idx,
                xyxy=(x1, y1, x2, y2),
                role=str(h.get("role") or "unknown"),
                eligibility=str(h.get("eligibility") or "uncertain"),
                visibility=str(h.get("visibility") or "clear"),
                team_appearance=str(h.get("team_appearance") or "unknown"),
            )
        )
    return out


def height_bin(h: float) -> str:
    for name, lo, hi in HEIGHT_BINS:
        if lo <= h < hi:
            return name
    return "h_ge_64"


def filter_predictions_with_ignore(
    preds: Sequence[BBoxDetection],
    ignore_boxes: Sequence[LabeledBox],
    *,
    iou_thresh: float = 0.5,
) -> tuple[list[BBoxDetection], list[BBoxDetection]]:
    """Return (kept_for_metrics, ignored_predictions)."""
    kept: list[BBoxDetection] = []
    ignored: list[BBoxDetection] = []
    ignore_by_frame: dict[int, list[LabeledBox]] = {}
    for b in ignore_boxes:
        ignore_by_frame.setdefault(b.frame_index, []).append(b)
    for p in preds:
        hit = False
        for ig in ignore_by_frame.get(int(p.frame_index), []):
            if bbox_iou((p.x1, p.y1, p.x2, p.y2), ig.xyxy) >= iou_thresh:
                hit = True
                break
        if hit:
            ignored.append(p)
        else:
            kept.append(p)
    return kept, ignored


def labeled_to_dets(boxes: Sequence[LabeledBox]) -> list[BBoxDetection]:
    return [
        BBoxDetection(
            frame_index=b.frame_index,
            entity_type="human",
            x1=b.xyxy[0],
            y1=b.xyxy[1],
            x2=b.xyxy[2],
            y2=b.xyxy[3],
            score=1.0,
        )
        for b in boxes
    ]


def evaluate_protocol_v2(
    preds: Sequence[BBoxDetection],
    frames: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    all_boxes: list[LabeledBox] = []
    for fr in frames:
        all_boxes.extend(boxes_from_frozen_frame(fr))
    primary = [b for b in all_boxes if b.is_primary()]
    ignore = [b for b in all_boxes if b.is_ignore()]
    off = [b for b in all_boxes if b.eligibility == "off_pitch"]
    kept, ignored_preds = filter_predictions_with_ignore(preds, ignore, iou_thresh=iou_threshold)

    primary_dets = labeled_to_dets(primary)
    all_dets = labeled_to_dets(all_boxes)

    primary_eval = evaluate_human_detections(kept, primary_dets, iou_threshold=iou_threshold)
    all_eval = evaluate_human_detections(list(preds), all_dets, iou_threshold=iou_threshold)

    # height-bin recall on primary
    matches = greedy_match_iou(kept, primary_dets, iou_threshold=iou_threshold)
    matched_gt = {m.gt_index for m in matches}
    bins: dict[str, dict[str, Any]] = {}
    for name, _, _ in HEIGHT_BINS:
        idxs = [i for i, b in enumerate(primary) if height_bin(b.height) == name]
        hit = sum(1 for i in idxs if i in matched_gt)
        bins[name] = {
            "n_gt": len(idxs),
            "tp": hit,
            "recall": (hit / len(idxs)) if idxs else None,
        }
    small_idxs = [i for i, b in enumerate(primary) if b.height < 55.0 or b.visibility == "small"]
    small_hit = sum(1 for i in small_idxs if i in matched_gt)
    small_recall = (small_hit / len(small_idxs)) if small_idxs else None

    # duplicate rate among kept preds
    dup = 0
    by_f: dict[int, list[BBoxDetection]] = {}
    for p in kept:
        by_f.setdefault(int(p.frame_index), []).append(p)
    for boxes in by_f.values():
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if bbox_iou((a.x1, a.y1, a.x2, a.y2), (b.x1, b.y1, b.x2, b.y2)) >= 0.9:
                    dup += 1
    dup_rate = (dup / max(1, len(kept))) if kept else 0.0

    # merged-person diagnostic on preds
    merged = 0
    for p in kept:
        w, h = p.x2 - p.x1, p.y2 - p.y1
        if h > 1 and (w / h) > 0.85 and w > 55:
            merged += 1

    pd = primary_eval.to_dict()
    pd.pop("matches", None)
    pd.pop("notes", None)
    ad = all_eval.to_dict()
    ad.pop("matches", None)
    ad.pop("notes", None)

    n_frames = len({int(f["frame_idx"]) for f in frames})
    return {
        "schema": "protocol_v2_eval_result",
        "protocol_id": PROTOCOL_ID,
        "primary": {
            **pd,
            "duplicate_rate": dup_rate,
            "duplicate_pairs": dup,
            "merged_person_diag": merged,
            "small_recall": small_recall,
            "n_small_gt": len(small_idxs),
            "height_bin_recall": bins,
            "fp_per_frame": (pd.get("false_positives") or 0) / max(1, n_frames),
        },
        "secondary": {
            "all_human": ad,
            "off_pitch_gt_n": len(off),
            "uncertain_gt_n": len(ignore),
            "ignored_predictions": len(ignored_preds),
            "kept_predictions": len(kept),
            "raw_predictions": len(preds),
        },
        "n_frames": n_frames,
        "n_primary_gt": len(primary),
    }


def dev_gate_passed(primary: Mapping[str, Any]) -> dict[str, Any]:
    thresholds = {
        "precision": 0.88,
        "recall": 0.88,
        "f1": 0.88,
        "ap50": 0.88,
        "small_recall": 0.75,
        "duplicate_rate": 0.01,
    }
    values = {
        "precision": primary.get("precision"),
        "recall": primary.get("recall"),
        "f1": primary.get("f1"),
        "ap50": primary.get("ap50"),
        "small_recall": primary.get("small_recall"),
        "duplicate_rate": primary.get("duplicate_rate"),
    }
    checks = {
        "precision": (values["precision"] or 0) >= thresholds["precision"],
        "recall": (values["recall"] or 0) >= thresholds["recall"],
        "f1": (values["f1"] or 0) >= thresholds["f1"],
        "ap50": (values["ap50"] or 0) >= thresholds["ap50"],
        "small_recall": (values["small_recall"] or 0) >= thresholds["small_recall"],
        "duplicate_rate": float(
            values["duplicate_rate"] if values["duplicate_rate"] is not None else 1.0
        )
        <= thresholds["duplicate_rate"],
    }
    return {
        "thresholds": thresholds,
        "values": values,
        "checks": checks,
        "passed": all(checks.values()),
    }


__all__ = [
    "EXPECTED_FROZEN_FP",
    "HEIGHT_BINS",
    "HOLDOUT_V1_STATUS",
    "LabeledBox",
    "PROTOCOL_ID",
    "boxes_from_frozen_frame",
    "dev_gate_passed",
    "evaluate_protocol_v2",
    "filter_predictions_with_ignore",
    "height_bin",
    "protocol_v2_definition",
]
