"""Independent football human GT schema, validation, and freeze gates (R1-F2-A).

Draft annotations are never human_approved / reviewed_gt / frozen without
explicit user approval. Prior agent/YOLO drafts are not acceptance-eligible.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from football_analytics.annotation.coordinates import (
    COORDINATE_SPACE,
    SOURCE_HEIGHT,
    SOURCE_WIDTH,
    CoordinateError,
    validate_source_bbox_xyxy,
)

SCHEMA_DRAFT = "independent_football_human_gt_draft_v1"
SCHEMA_SELECTION = "independent_gt_frame_selection_v1"
SCHEMA_FREEZE_REPORT = "independent_gt_freeze_validator_v1"

CLASS_NAME = "human"

Role = Literal["player", "goalkeeper", "referee", "staff", "unknown"]
TeamAppearance = Literal["yellow", "white", "official", "unknown", "not_applicable"]
Eligibility = Literal["on_pitch", "off_pitch", "uncertain"]
Visibility = Literal["clear", "small", "occluded", "truncated", "blurred"]
Split = Literal["train", "dev", "holdout"]

ROLES: frozenset[str] = frozenset({"player", "goalkeeper", "referee", "staff", "unknown"})
TEAMS: frozenset[str] = frozenset({"yellow", "white", "official", "unknown", "not_applicable"})
ELIGIBILITIES: frozenset[str] = frozenset({"on_pitch", "off_pitch", "uncertain"})
VISIBILITIES: frozenset[str] = frozenset({"clear", "small", "occluded", "truncated", "blurred"})
SPLITS: frozenset[str] = frozenset({"train", "dev", "holdout"})

# Time isolation (seconds) on canonical 30 fps source.
TRAIN_T = (0.0, 12.0)
DEV_T = (12.0, 22.0)
HOLDOUT_T = (22.0, 34.1)

EXPECTED_SOURCE_SHA256 = "97b298e41a82b567a7d68bd2322993bea34492b1cbb58362b0d72ca4a5471160"
DEFAULT_VIDEO = Path(
    "/home/fdoblak/football_data/videos/raw_matches/own_video_97b298e4/original.mp4"
)
DEFAULT_RUNTIME = Path("/home/fdoblak/workspace/independent_gt_review/own_video_97b298e4")

# Soft warnings (not hard rejects on draft save).
DUPLICATE_IOU = 0.90
WIDE_BOX_AREA_FRAC = 0.12
WIDE_BOX_ASPECT_MAX = 1.8
MIN_BOX_AREA = 80.0
MIN_BOX_H = 12.0


class IndependentGTError(ValueError):
    """Hard failure for independent GT policy / freeze gates."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any], *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n"
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)
    with contextlib.suppress(OSError):
        path.chmod(mode)


def append_audit_line(path: Path, event: Mapping[str, Any], *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(event), ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
    with contextlib.suppress(OSError):
        path.chmod(mode)


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


@dataclass
class HumanBox:
    bbox_xyxy: list[float]
    class_name: str = CLASS_NAME
    role: str = "unknown"
    team_appearance: str = "unknown"
    eligibility: str = "uncertain"
    visibility: str = "clear"
    jersey_number_visible: bool = False
    jersey_number: int | None = None
    origin: str = "manual"  # manual | proposal_reviewed | proposal_unreviewed
    proposal_id: str | None = None
    box_id: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def validate_metadata(box: Mapping[str, Any]) -> list[str]:
    errs: list[str] = []
    if box.get("class_name", CLASS_NAME) != CLASS_NAME:
        errs.append("class_must_be_human")
    if box.get("role") not in ROLES:
        errs.append(f"role_not_allowlisted:{box.get('role')}")
    if box.get("team_appearance") not in TEAMS:
        errs.append(f"team_not_allowlisted:{box.get('team_appearance')}")
    if box.get("eligibility") not in ELIGIBILITIES:
        errs.append(f"eligibility_not_allowlisted:{box.get('eligibility')}")
    if box.get("visibility") not in VISIBILITIES:
        errs.append(f"visibility_not_allowlisted:{box.get('visibility')}")
    jv = box.get("jersey_number_visible", False)
    jn = box.get("jersey_number", None)
    if jv is True:
        if jn is None or not isinstance(jn, int) or jn < 0 or jn > 99:
            errs.append("jersey_visible_requires_readable_int_0_99")
    elif jn is not None:
        errs.append("jersey_number_must_be_null_when_not_visible")
    return errs


def validate_box_geometry(
    xyxy: Sequence[float],
    *,
    source_w: int = SOURCE_WIDTH,
    source_h: int = SOURCE_HEIGHT,
) -> None:
    validate_source_bbox_xyxy(xyxy, source_w=source_w, source_h=source_h)
    x1, y1, x2, y2 = map(float, xyxy)
    area = (x2 - x1) * (y2 - y1)
    h = y2 - y1
    if area < MIN_BOX_AREA or h < MIN_BOX_H:
        raise CoordinateError("bbox_too_small_for_human_label")


def soft_box_warnings(
    xyxy: Sequence[float],
    others: Iterable[Sequence[float]],
    *,
    source_w: int = SOURCE_WIDTH,
    source_h: int = SOURCE_HEIGHT,
) -> list[str]:
    warns: list[str] = []
    x1, y1, x2, y2 = map(float, xyxy)
    w, h = x2 - x1, y2 - y1
    area = w * h
    frame_area = float(source_w * source_h)
    if area / frame_area >= WIDE_BOX_AREA_FRAC and (w / max(h, 1e-6)) >= WIDE_BOX_ASPECT_MAX:
        warns.append("possibly_covers_two_people_wide_box")
    for o in others:
        if bbox_iou(xyxy, o) >= DUPLICATE_IOU:
            warns.append("duplicate_bbox_iou_ge_0.9")
            break
    return warns


def assert_no_prediction_leakage(
    frame: Mapping[str, Any],
    *,
    split: str,
) -> None:
    """Dev/holdout must never carry detector proposals or confidence fields."""
    if split not in {"dev", "holdout"}:
        return
    if frame.get("proposals"):
        raise IndependentGTError(f"PREDICTION_LEAKAGE:{split}:proposals_present")
    for hum in frame.get("humans", []) or []:
        if hum.get("origin") in {"proposal_unreviewed", "proposal"}:
            raise IndependentGTError(f"PREDICTION_LEAKAGE:{split}:proposal_origin")
        if "confidence" in hum or "score" in hum:
            raise IndependentGTError(f"PREDICTION_LEAKAGE:{split}:confidence_field")
    if frame.get("predictions") or frame.get("detector_boxes"):
        raise IndependentGTError(f"PREDICTION_LEAKAGE:{split}:predictions_field")


def train_proposals_are_gt(frame: Mapping[str, Any]) -> bool:
    """True if any unreviewed proposal is treated as GT human."""
    return any(hum.get("origin") == "proposal_unreviewed" for hum in frame.get("humans", []) or [])


@dataclass(frozen=True)
class FreezeCriteria:
    require_all_complete: bool = True
    require_user_approval: bool = True
    min_humans_per_split: Mapping[str, int] = field(
        default_factory=lambda: {"train": 40, "dev": 20, "holdout": 15}
    )
    require_holdout_small_or_crowded: bool = True


def validate_freeze_ready(
    draft: Mapping[str, Any],
    *,
    source_sha256: str,
    expected_sha256: str = EXPECTED_SOURCE_SHA256,
    user_approved: bool = False,
    criteria: FreezeCriteria | None = None,
) -> dict[str, Any]:
    """Return freeze report. Never mutates draft. Hard-fails approval flags."""
    crit = criteria or FreezeCriteria()
    errors: list[str] = []
    warnings: list[str] = []

    if (
        draft.get("human_approved") is True or draft.get("reviewed_gt") is True
    ) and not user_approved:
        errors.append("draft_already_marked_approved_without_explicit_user_flag")
    if draft.get("frozen") is True and not user_approved:
        errors.append("draft_already_frozen_without_explicit_user_flag")

    if source_sha256.lower() != expected_sha256.lower():
        errors.append("SOURCE_SHA256_MISMATCH")

    if crit.require_user_approval and not user_approved:
        errors.append("USER_APPROVAL_REQUIRED")

    frames = list(draft.get("frames", []))
    if not frames:
        errors.append("no_frames")

    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "holdout": []}
    for fr in frames:
        split = str(fr.get("split", ""))
        if split not in SPLITS:
            errors.append(f"bad_split:{split}")
            continue
        by_split[split].append(fr)
        try:
            assert_no_prediction_leakage(fr, split=split)
        except IndependentGTError as exc:
            errors.append(str(exc))
        if train_proposals_are_gt(fr):
            errors.append(f"train_unreviewed_proposal_as_gt:frame_{fr.get('frame_idx')}")
        if crit.require_all_complete and not fr.get("completed"):
            errors.append(f"incomplete_frame:{fr.get('frame_idx')}")
        humans = list(fr.get("humans", []) or [])
        xyxys = [h.get("bbox_xyxy") for h in humans]
        for i, hum in enumerate(humans):
            meta_errs = validate_metadata(hum)
            errors.extend(f"frame_{fr.get('frame_idx')}:{e}" for e in meta_errs)
            try:
                validate_box_geometry(hum["bbox_xyxy"])
            except (CoordinateError, KeyError, TypeError) as exc:
                errors.append(f"frame_{fr.get('frame_idx')}:invalid_bbox:{exc}")
            others = [x for j, x in enumerate(xyxys) if j != i and x is not None]
            for w in soft_box_warnings(hum["bbox_xyxy"], others):  # type: ignore[arg-type]
                if w.startswith("duplicate"):
                    errors.append(f"frame_{fr.get('frame_idx')}:duplicate_bbox")
                else:
                    warnings.append(f"frame_{fr.get('frame_idx')}:{w}")
            if split == "train" and hum.get("origin") == "proposal_unreviewed":
                errors.append(f"train_proposal_not_human_reviewed:frame_{fr.get('frame_idx')}")
            if (
                split in {"dev", "holdout"}
                and hum.get("origin")
                not in {
                    "manual",
                    "manual_blind",
                    None,
                    "",
                }
                and hum.get("origin") not in {"manual", "manual_blind"}
            ):
                errors.append(f"{split}_provenance_not_manual_blind:frame_{fr.get('frame_idx')}")

    for split, need in crit.min_humans_per_split.items():
        n = sum(len(fr.get("humans", []) or []) for fr in by_split.get(split, []))
        if n < int(need):
            errors.append(f"insufficient_humans_{split}:{n}<{need}")

    if crit.require_holdout_small_or_crowded:
        hold = by_split.get("holdout", [])
        has_small = any(
            h.get("visibility") == "small" for fr in hold for h in (fr.get("humans") or [])
        )
        crowded = any(len(fr.get("humans") or []) >= 8 for fr in hold)
        cats = {c for fr in hold for c in (fr.get("categories") or [])}
        if not (has_small or crowded or ("small_distant" in cats) or ("crowded" in cats)):
            warnings.append("holdout_small_or_crowded_not_yet_evidenced_in_labels")

    audit_ok = bool(draft.get("audit_log_path")) or bool(draft.get("audit_present"))
    if not audit_ok:
        warnings.append("audit_log_path_missing_in_draft_header")

    ok = len(errors) == 0
    report = {
        "schema": SCHEMA_FREEZE_REPORT,
        "freeze_allowed": ok,
        "human_approved": False if not user_approved else ok,
        "reviewed_gt": False if not user_approved else ok,
        "frozen": False,
        "errors": errors,
        "warnings": warnings,
        "n_frames": len(frames),
        "counts_by_split": {k: len(v) for k, v in by_split.items()},
        "source_sha256": source_sha256,
        "expected_sha256": expected_sha256,
        "user_approved": user_approved,
        "written_at_utc": utc_now(),
        "note": "R1-F2-A prepares validator only; do not freeze without explicit user approval.",
    }
    if user_approved and ok:
        # Still do not set frozen=true here — caller must require a separate freeze command.
        report["human_approved"] = False
        report["reviewed_gt"] = False
        report["errors"] = list(errors) + ["EXPLICIT_FREEZE_COMMAND_REQUIRED_AFTER_APPROVAL"]
        report["freeze_allowed"] = False
    return report


def empty_draft(
    *,
    video: Path,
    source_sha256: str,
    frames: Sequence[Mapping[str, Any]],
    audit_log_path: str | None = None,
) -> dict[str, Any]:
    out_frames = []
    for fr in frames:
        split = str(fr["split"])
        item: dict[str, Any] = {
            "frame_idx": int(fr["frame_idx"]),
            "t_s": float(fr["t_s"]),
            "split": split,
            "categories": list(fr.get("categories") or []),
            "completed": False,
            "review_status": "not_reviewed",
            "humans": [],
            "source_width": SOURCE_WIDTH,
            "source_height": SOURCE_HEIGHT,
            "coordinate_space": COORDINATE_SPACE,
            "provenance": "manual_blind" if split in {"dev", "holdout"} else "train_mixed",
        }
        if split == "train":
            item["proposals"] = list(fr.get("proposals") or [])
        else:
            item["proposals"] = []
            assert_no_prediction_leakage(item, split=split)
        out_frames.append(item)
    return {
        "schema": SCHEMA_DRAFT,
        "dataset_id": "own_video_independent_human_gt_v1",
        "video": str(video),
        "source_sha256": source_sha256,
        "coordinate_space": COORDINATE_SPACE,
        "source_width": SOURCE_WIDTH,
        "source_height": SOURCE_HEIGHT,
        "human_approved": False,
        "reviewed_gt": False,
        "frozen": False,
        "acceptance_eligible": False,
        "audit_log_path": audit_log_path,
        "audit_present": bool(audit_log_path),
        "frames": out_frames,
        "updated_at_utc": utc_now(),
        "rejection_note": ("Prior agent_hallucinated / YOLO bake-off / blind drafts are NOT GT."),
    }


__all__ = [
    "CLASS_NAME",
    "DEFAULT_RUNTIME",
    "DEFAULT_VIDEO",
    "DEV_T",
    "ELIGIBILITIES",
    "EXPECTED_SOURCE_SHA256",
    "HOLDOUT_T",
    "ROLES",
    "SCHEMA_DRAFT",
    "SCHEMA_FREEZE_REPORT",
    "SCHEMA_SELECTION",
    "SPLITS",
    "TEAMS",
    "TRAIN_T",
    "VISIBILITIES",
    "FreezeCriteria",
    "HumanBox",
    "IndependentGTError",
    "append_audit_line",
    "assert_no_prediction_leakage",
    "atomic_write_json",
    "bbox_iou",
    "empty_draft",
    "sha256_file",
    "soft_box_warnings",
    "train_proposals_are_gt",
    "utc_now",
    "validate_box_geometry",
    "validate_freeze_ready",
    "validate_metadata",
]
