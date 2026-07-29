"""Train false-complete repair helpers (R1-F2-A-FIX2).

Protects carefully labeled train frames 0/5/15 and all dev/holdout.
Resets only empty-complete train frames with leftover proposals.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from football_analytics.annotation.independent_gt import (
    IndependentGTError,
    soft_box_warnings,
    utc_now,
    validate_box_geometry,
    validate_metadata,
)

REPAIR_MODE = "train-empty-complete"
REPAIR_REASON = "EMPTY_COMPLETE_WITH_VISIBLE_PROPOSALS"

# Fixed lists from R1-F2-B integrity failure (do not invent).
FAILED_TRAIN_FRAME_INDICES: tuple[int, ...] = (
    24,
    30,
    36,
    45,
    54,
    60,
    70,
    80,
    90,
    100,
    110,
    120,
    131,
    138,
    144,
    150,
    160,
    170,
    180,
    190,
    200,
    210,
    221,
    232,
    240,
    250,
    260,
    270,
    280,
    290,
    300,
    306,
    312,
    320,
    330,
    340,
    350,
)
PROTECTED_TRAIN_FRAME_INDICES: tuple[int, ...] = (0, 5, 15)

SAFE_BULK_META: dict[str, Any] = {
    "role": "unknown",
    "team_appearance": "unknown",
    "eligibility": "uncertain",
    "visibility": "clear",
    "jersey_number_visible": False,
    "jersey_number": None,
}


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _frame_identity(fr: Mapping[str, Any]) -> dict[str, Any]:
    """Stable identity for immutability checks (bbox + metadata, not UI flags)."""
    humans = []
    for h in fr.get("humans") or []:
        humans.append(
            {
                "bbox_xyxy": [float(x) for x in h["bbox_xyxy"]],
                "class_name": h.get("class_name", "human"),
                "role": h.get("role"),
                "team_appearance": h.get("team_appearance"),
                "eligibility": h.get("eligibility"),
                "visibility": h.get("visibility"),
                "jersey_number_visible": bool(h.get("jersey_number_visible", False)),
                "jersey_number": h.get("jersey_number", None),
                "origin": h.get("origin"),
            }
        )
    humans.sort(key=lambda x: tuple(x["bbox_xyxy"]))
    return {
        "frame_idx": int(fr["frame_idx"]),
        "t_s": float(fr["t_s"]),
        "split": fr["split"],
        "humans": humans,
        "provenance": fr.get("provenance"),
    }


def split_fingerprint(frames: Sequence[Mapping[str, Any]], split: str) -> str:
    rows = [_frame_identity(f) for f in frames if f.get("split") == split]
    rows.sort(key=lambda r: r["frame_idx"])
    return sha256_payload({"split": split, "frames": rows})


def protected_train_fingerprint(frames: Sequence[Mapping[str, Any]]) -> str:
    want = set(PROTECTED_TRAIN_FRAME_INDICES)
    rows = [
        _frame_identity(f)
        for f in frames
        if f.get("split") == "train" and int(f["frame_idx"]) in want
    ]
    rows.sort(key=lambda r: r["frame_idx"])
    return sha256_payload({"protected_train": rows})


def assert_immutable_fingerprints(
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> None:
    for key in ("dev", "holdout", "protected_train"):
        if before.get(key) != after.get(key):
            raise IndependentGTError(f"IMMUTABLE_SPLIT_CHANGED:{key}")


def collect_fingerprints(draft: Mapping[str, Any]) -> dict[str, str]:
    frames = list(draft.get("frames") or [])
    return {
        "dev": split_fingerprint(frames, "dev"),
        "holdout": split_fingerprint(frames, "holdout"),
        "protected_train": protected_train_fingerprint(frames),
        "source_sha256": str(draft.get("source_sha256") or ""),
    }


def reset_false_complete_train_frames(draft: dict[str, Any]) -> dict[str, Any]:
    """Mutate draft in-place: reset only FAILED_TRAIN_FRAME_INDICES.

    Keeps proposals. Clears humans (should already be empty). Sets complete=false.
    """
    failed = set(FAILED_TRAIN_FRAME_INDICES)
    protected = set(PROTECTED_TRAIN_FRAME_INDICES)
    reset_n = 0
    for fr in draft["frames"]:
        idx = int(fr["frame_idx"])
        if fr.get("split") != "train":
            continue
        if idx in protected:
            # Ensure not marked repair
            fr.pop("repair_required", None)
            fr.pop("repair_reason", None)
            continue
        if idx not in failed:
            continue
        if fr.get("humans"):
            raise IndependentGTError(f"UNEXPECTED_HUMANS_ON_FAILED_TRAIN:{idx}")
        props = list(fr.get("proposals") or [])
        if not props:
            raise IndependentGTError(f"FAILED_TRAIN_MISSING_PROPOSALS:{idx}")
        fr["completed"] = False
        fr["review_status"] = "repair_required"
        fr["repair_required"] = True
        fr["repair_reason"] = REPAIR_REASON
        fr["no_human_confirmed"] = False
        fr["rejected_proposals"] = list(fr.get("rejected_proposals") or [])
        # keep proposals untouched
        reset_n += 1
    if reset_n != len(FAILED_TRAIN_FRAME_INDICES):
        raise IndependentGTError(
            f"RESET_COUNT_MISMATCH:{reset_n}!={len(FAILED_TRAIN_FRAME_INDICES)}"
        )
    draft["repair_mode"] = REPAIR_MODE
    draft["repair_updated_at_utc"] = utc_now()
    return {
        "reset_n": reset_n,
        "failed_indices": list(FAILED_TRAIN_FRAME_INDICES),
        "protected_indices": list(PROTECTED_TRAIN_FRAME_INDICES),
    }


def pending_proposal_count(fr: Mapping[str, Any]) -> int:
    return len(fr.get("proposals") or [])


def accepted_human_count(fr: Mapping[str, Any]) -> int:
    return len(fr.get("humans") or [])


def validate_train_complete_allowed(fr: Mapping[str, Any]) -> list[str]:
    """Return hard-fail reasons; empty list means complete is allowed."""
    errs: list[str] = []
    if fr.get("split") != "train":
        return errs
    pending = pending_proposal_count(fr)
    humans = accepted_human_count(fr)
    if pending > 0:
        errs.append("PENDING_PROPOSALS_MUST_BE_ACCEPTED_OR_REJECTED")
    if humans == 0 and not fr.get("no_human_confirmed"):
        errs.append("ZERO_HUMANS_REQUIRES_EXPLICIT_NO_HUMAN_CONFIRMATION")
    # geometry on accepted
    xyxys: list[Any] = []
    for h in fr.get("humans") or []:
        meta = validate_metadata(h)
        if meta:
            errs.extend(meta)
        try:
            validate_box_geometry(h["bbox_xyxy"])
        except Exception as exc:  # noqa: BLE001
            errs.append(f"invalid_bbox:{exc}")
        for w in soft_box_warnings(h["bbox_xyxy"], xyxys):
            if w.startswith("duplicate"):
                errs.append("duplicate_bbox")
        xyxys.append(h["bbox_xyxy"])
    return errs


def validate_repair_complete(draft: Mapping[str, Any]) -> dict[str, Any]:
    frames = list(draft.get("frames") or [])
    by_idx = {int(f["frame_idx"]): f for f in frames if f.get("split") == "train"}
    errors: list[str] = []
    complete_n = 0
    for idx in FAILED_TRAIN_FRAME_INDICES:
        fr = by_idx.get(idx)
        if fr is None:
            errors.append(f"missing_frame:{idx}")
            continue
        if not fr.get("completed"):
            errors.append(f"incomplete:{idx}")
        else:
            complete_n += 1
        if pending_proposal_count(fr) > 0:
            errors.append(f"pending_proposals:{idx}")
        if accepted_human_count(fr) == 0 and not fr.get("no_human_confirmed"):
            errors.append(f"empty_without_no_human:{idx}")
        for reason in validate_train_complete_allowed(fr):
            errors.append(f"{idx}:{reason}")
    # protected still complete with humans
    for idx in PROTECTED_TRAIN_FRAME_INDICES:
        fr = by_idx.get(idx)
        if fr is None:
            errors.append(f"missing_protected:{idx}")
            continue
        if not fr.get("completed"):
            errors.append(f"protected_incomplete:{idx}")
        if accepted_human_count(fr) <= 0:
            errors.append(f"protected_empty:{idx}")
    ok = len(errors) == 0 and complete_n == len(FAILED_TRAIN_FRAME_INDICES)
    return {
        "schema": "r1_f2a_fix2_repair_validator_v1",
        "repair_complete": ok,
        "complete_n": complete_n,
        "target_n": len(FAILED_TRAIN_FRAME_INDICES),
        "errors": errors,
        "written_at_utc": utc_now(),
        "note": "Repair completion only — does not freeze GT.",
    }


def bulk_accept_proposals(
    fr: dict[str, Any],
    *,
    box_id_factory: Any,
) -> list[dict[str, Any]]:
    """Convert all pending proposals to humans with safe metadata. Returns new humans."""
    if fr.get("split") != "train":
        raise IndependentGTError("BULK_ACCEPT_TRAIN_ONLY")
    from football_analytics.annotation.independent_gt import bbox_iou

    props = list(fr.get("proposals") or [])
    if not props:
        return []
    existing = [h["bbox_xyxy"] for h in fr.get("humans") or []]
    added: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    for prop in props:
        xyxy = [float(x) for x in prop["bbox_xyxy"]]
        validate_box_geometry(xyxy)
        # skip near-duplicates of already accepted
        dup = False
        for o in existing:
            if bbox_iou(xyxy, o) >= 0.90:
                dup = True
                break
        if dup:
            remaining.append(prop)
            continue
        hum = {
            "box_id": str(box_id_factory()),
            "bbox_xyxy": xyxy,
            "class_name": "human",
            **SAFE_BULK_META,
            "origin": "proposal_reviewed_bulk",
            "proposal_id": prop.get("proposal_id"),
            "warnings": soft_box_warnings(xyxy, existing),
        }
        errs = validate_metadata(hum)
        if errs:
            raise IndependentGTError(";".join(errs))
        fr.setdefault("humans", []).append(hum)
        existing.append(xyxy)
        added.append(hum)
    fr["proposals"] = remaining
    # bulk accept must never auto-complete — user reviews then Completes
    fr["completed"] = False
    if fr.get("repair_required"):
        fr["review_status"] = "repair_required"
    else:
        fr["review_status"] = "incomplete"
    return added


def reject_pending_proposals(fr: dict[str, Any]) -> int:
    """Move all pending proposals into rejected_proposals. Returns count rejected."""
    props = list(fr.get("proposals") or [])
    if not props:
        return 0
    rejected = list(fr.get("rejected_proposals") or [])
    for prop in props:
        row = dict(prop)
        row["rejected_at_utc"] = utc_now()
        rejected.append(row)
    fr["rejected_proposals"] = rejected
    fr["proposals"] = []
    return len(props)


def set_no_human_confirmed(fr: dict[str, Any], confirmed: bool) -> None:
    if fr.get("split") != "train":
        raise IndependentGTError("NO_HUMAN_CONFIRM_TRAIN_ONLY")
    if confirmed and pending_proposal_count(fr) > 0:
        raise IndependentGTError("REJECT_PROPOSALS_BEFORE_NO_HUMAN_CONFIRM")
    fr["no_human_confirmed"] = bool(confirmed)


def frame_gate_counts(fr: Mapping[str, Any]) -> dict[str, int]:
    return {
        "accepted_humans": accepted_human_count(fr),
        "pending_proposals": pending_proposal_count(fr),
        "rejected_proposals": len(fr.get("rejected_proposals") or []),
    }


def soft_complete_warnings(fr: Mapping[str, Any]) -> list[str]:
    """Non-blocking UI warnings shown before Complete."""
    warns: list[str] = []
    counts = frame_gate_counts(fr)
    if counts["pending_proposals"] > 0:
        warns.append(f"pending_proposals:{counts['pending_proposals']}")
    if counts["accepted_humans"] == 0 and not fr.get("no_human_confirmed"):
        warns.append("zero_accepted_human")
    xyxys: list[Any] = []
    for h in fr.get("humans") or []:
        for w in soft_box_warnings(h["bbox_xyxy"], xyxys):
            warns.append(w)
        xyxys.append(h["bbox_xyxy"])
        meta = validate_metadata(h)
        warns.extend(meta)
    return warns


__all__ = [
    "FAILED_TRAIN_FRAME_INDICES",
    "PROTECTED_TRAIN_FRAME_INDICES",
    "REPAIR_MODE",
    "REPAIR_REASON",
    "SAFE_BULK_META",
    "assert_immutable_fingerprints",
    "bulk_accept_proposals",
    "collect_fingerprints",
    "frame_gate_counts",
    "pending_proposal_count",
    "protected_train_fingerprint",
    "reject_pending_proposals",
    "reset_false_complete_train_frames",
    "set_no_human_confirmed",
    "sha256_payload",
    "soft_complete_warnings",
    "split_fingerprint",
    "validate_repair_complete",
    "validate_train_complete_allowed",
]
