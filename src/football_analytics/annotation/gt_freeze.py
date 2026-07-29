"""Freeze repaired independent human GT (R1-F2-B-R1)."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from football_analytics.annotation.independent_gt import (
    DEFAULT_RUNTIME,
    DEV_T,
    EXPECTED_SOURCE_SHA256,
    HOLDOUT_T,
    TRAIN_T,
    IndependentGTError,
    assert_no_prediction_leakage,
    atomic_write_json,
    soft_box_warnings,
    utc_now,
    validate_box_geometry,
    validate_metadata,
)
from football_analytics.annotation.train_repair import (
    FAILED_TRAIN_FRAME_INDICES,
    PROTECTED_TRAIN_FRAME_INDICES,
    collect_fingerprints,
    pending_proposal_count,
    reject_pending_proposals,
    sha256_payload,
    validate_repair_complete,
    validate_train_complete_allowed,
)

REPO = Path(__file__).resolve().parents[3]
DEFAULT_FROZEN_DIR = REPO / "annotations" / "own_video_97b298e4" / "human_detection_v1"

# Fingerprints captured at FIX2 prepare (must remain for dev/holdout).
EXPECTED_DEV_FP = "e2a9020a09bcba7e80c2b5e419cd79bfb05c344f7f836b03888807ea3e234d81"
EXPECTED_HOLDOUT_FP = "29644e511b4f32369f9fb64459eabc92cac9bb7cef5ed894eafa44f1095519ed"
EXPECTED_PROTECTED_TRAIN_FP = "9d4d5033d5c23f11e6c5ffc9c7c607c94ff4fcc6d8dabc7b1b8c221202d5f56f"


def clear_leftover_train_proposals(draft: dict[str, Any]) -> dict[str, Any]:
    """Move leftover pending proposals out of train frames (they are never GT).

    Protected frames 0/5/15 may still carry unused proposal seeds from the original
    review pass; clearing them does not alter accepted human boxes / fingerprints.
    """
    cleared = 0
    frames: list[int] = []
    for fr in draft["frames"]:
        if fr.get("split") != "train":
            continue
        n = pending_proposal_count(fr)
        if n <= 0:
            continue
        reject_pending_proposals(fr)
        cleared += n
        frames.append(int(fr["frame_idx"]))
    return {"cleared_proposals": cleared, "frames": frames}


def validate_repaired_gt_integrity(draft: Mapping[str, Any]) -> dict[str, Any]:
    """Hard integrity gate before freeze. Does not mutate."""
    errors: list[str] = []
    problem_frames: list[int] = []
    if draft.get("source_sha256") != EXPECTED_SOURCE_SHA256:
        errors.append("SOURCE_SHA_MISMATCH")
    fp = collect_fingerprints(draft)
    if fp["dev"] != EXPECTED_DEV_FP:
        errors.append("DEV_FINGERPRINT_CHANGED")
    if fp["holdout"] != EXPECTED_HOLDOUT_FP:
        errors.append("HOLDOUT_FINGERPRINT_CHANGED")
    if fp["protected_train"] != EXPECTED_PROTECTED_TRAIN_FP:
        errors.append("PROTECTED_TRAIN_FINGERPRINT_CHANGED")
    if fp["source_sha256"] != EXPECTED_SOURCE_SHA256:
        errors.append("SOURCE_FP_MISMATCH")

    repair = validate_repair_complete(draft)
    if not repair["repair_complete"]:
        errors.append("REPAIR_INCOMPLETE")
        errors.extend(repair.get("errors") or [])

    by = {"train": {"n": 0, "complete": 0, "boxes": 0}, "dev": {}, "holdout": {}}
    for sp in ("train", "dev", "holdout"):
        by[sp] = {"n": 0, "complete": 0, "boxes": 0}

    pending_total = 0
    empty_bad = 0
    invalid = 0
    duplicate = 0
    for fr in draft.get("frames") or []:
        split = fr.get("split")
        if split not in by:
            errors.append(f"bad_split:{fr.get('frame_idx')}")
            continue
        by[split]["n"] += 1
        if fr.get("completed"):
            by[split]["complete"] += 1
        humans = list(fr.get("humans") or [])
        by[split]["boxes"] += len(humans)
        p = pending_proposal_count(fr)
        pending_total += p
        if p > 0:
            errors.append(f"pending_proposals:frame_{fr.get('frame_idx')}:{p}")
            problem_frames.append(int(fr["frame_idx"]))
        try:
            assert_no_prediction_leakage(fr, split=str(split))
        except IndependentGTError as exc:
            errors.append(str(exc))
            problem_frames.append(int(fr["frame_idx"]))

        t = float(fr["t_s"])
        if split == "train" and not (TRAIN_T[0] <= t < TRAIN_T[1]):
            errors.append(f"train_time_window:frame_{fr.get('frame_idx')}")
        if split == "dev" and not (DEV_T[0] <= t < DEV_T[1]):
            errors.append(f"dev_time_window:frame_{fr.get('frame_idx')}")
        if split == "holdout" and not (HOLDOUT_T[0] <= t <= HOLDOUT_T[1]):
            errors.append(f"holdout_time_window:frame_{fr.get('frame_idx')}")

        if split == "train":
            for e in validate_train_complete_allowed(fr):
                errors.append(f"frame_{fr.get('frame_idx')}:{e}")
                problem_frames.append(int(fr["frame_idx"]))
            if len(humans) == 0 and fr.get("completed") and not fr.get("no_human_confirmed"):
                empty_bad += 1
                errors.append(f"empty_complete:frame_{fr.get('frame_idx')}")
                problem_frames.append(int(fr["frame_idx"]))

        xyxys: list[Any] = []
        for h in humans:
            origin = h.get("origin")
            if split in {"dev", "holdout"} and origin != "manual_blind":
                errors.append(f"{split}_not_manual_blind:frame_{fr.get('frame_idx')}")
                problem_frames.append(int(fr["frame_idx"]))
            if split == "train" and origin not in {
                "manual",
                "proposal_reviewed",
                "proposal_reviewed_bulk",
            }:
                errors.append(f"train_bad_origin:{origin}:frame_{fr.get('frame_idx')}")
            meta = validate_metadata(h)
            if meta:
                errors.extend(f"frame_{fr.get('frame_idx')}:{m}" for m in meta)
            try:
                validate_box_geometry(h["bbox_xyxy"])
            except Exception as exc:  # noqa: BLE001
                invalid += 1
                errors.append(f"invalid_bbox:frame_{fr.get('frame_idx')}:{exc}")
                problem_frames.append(int(fr["frame_idx"]))
            for w in soft_box_warnings(h["bbox_xyxy"], xyxys):
                if w.startswith("duplicate"):
                    duplicate += 1
                    errors.append(f"duplicate_bbox:frame_{fr.get('frame_idx')}")
                    problem_frames.append(int(fr["frame_idx"]))
            xyxys.append(h["bbox_xyxy"])

    if by["train"]["complete"] != 40 or by["train"]["n"] != 40:
        errors.append(f"train_complete:{by['train']}")
    if by["dev"]["complete"] != 20 or by["dev"]["n"] != 20:
        errors.append(f"dev_complete:{by['dev']}")
    if by["holdout"]["complete"] != 20 or by["holdout"]["n"] != 20:
        errors.append(f"holdout_complete:{by['holdout']}")
    if pending_total != 0:
        errors.append(f"pending_proposals_total:{pending_total}")
    if empty_bad != 0:
        errors.append(f"empty_complete_without_no_human:{empty_bad}")
    if invalid != 0:
        errors.append(f"invalid_bbox_count:{invalid}")
    if duplicate != 0:
        errors.append(f"duplicate_bbox_count:{duplicate}")

    # leakage: unique frame indices across splits
    seen: dict[int, str] = {}
    for fr in draft.get("frames") or []:
        idx = int(fr["frame_idx"])
        if idx in seen and seen[idx] != fr["split"]:
            errors.append(f"cross_split_frame:{idx}")
            problem_frames.append(idx)
        seen[idx] = str(fr["split"])

    ok = len(errors) == 0
    return {
        "schema": "r1_f2b_r1_repaired_gt_integrity_v1",
        "ok": ok,
        "gate": ("PASS — REPAIRED GT INTEGRITY" if ok else "NO-GO — REPAIRED GT INTEGRITY FAILURE"),
        "errors": errors,
        "problem_frames": sorted(set(problem_frames)),
        "by_split": by,
        "fingerprints": fp,
        "repair": repair,
        "pending_total": pending_total,
        "empty_bad": empty_bad,
        "invalid": invalid,
        "duplicate": duplicate,
        "written_at_utc": utc_now(),
    }


def _role_team_counts(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    roles: dict[str, int] = {}
    teams: dict[str, int] = {}
    elig: dict[str, int] = {}
    vis: dict[str, int] = {}
    for fr in frames:
        for h in fr.get("humans") or []:
            roles[str(h.get("role"))] = roles.get(str(h.get("role")), 0) + 1
            teams[str(h.get("team_appearance"))] = teams.get(str(h.get("team_appearance")), 0) + 1
            elig[str(h.get("eligibility"))] = elig.get(str(h.get("eligibility")), 0) + 1
            vis[str(h.get("visibility"))] = vis.get(str(h.get("visibility")), 0) + 1
    return {"role": roles, "team_appearance": teams, "eligibility": elig, "visibility": vis}


def write_frozen_gt(
    draft: Mapping[str, Any],
    *,
    out_dir: Path | None = None,
    reviewer: str = "user_manual_review",
    audit_path: Path | None = None,
) -> dict[str, Any]:
    out_dir = out_dir or DEFAULT_FROZEN_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    integrity = validate_repaired_gt_integrity(draft)
    if not integrity["ok"]:
        raise IndependentGTError(integrity["gate"] + ":" + ";".join(integrity["errors"][:20]))

    frames_out = []
    for fr in draft["frames"]:
        humans = []
        for h in fr.get("humans") or []:
            humans.append(
                {
                    "box_id": h.get("box_id"),
                    "bbox_xyxy": [float(x) for x in h["bbox_xyxy"]],
                    "class_name": "human",
                    "role": h.get("role"),
                    "team_appearance": h.get("team_appearance"),
                    "eligibility": h.get("eligibility"),
                    "visibility": h.get("visibility"),
                    "jersey_number_visible": bool(h.get("jersey_number_visible", False)),
                    "jersey_number": h.get("jersey_number"),
                    "origin": h.get("origin"),
                }
            )
        frames_out.append(
            {
                "frame_idx": int(fr["frame_idx"]),
                "t_s": float(fr["t_s"]),
                "split": fr["split"],
                "categories": list(fr.get("categories") or []),
                "completed": True,
                "humans": humans,
                "provenance": (
                    "manual_blind" if fr["split"] in {"dev", "holdout"} else "train_reviewed"
                ),
                "no_human_confirmed": bool(fr.get("no_human_confirmed", False)),
            }
        )

    freeze_ts = utc_now()
    annotations = {
        "schema": "own_video_human_detection_v1",
        "dataset_id": "own_video_97b298e4_human_detection_v1",
        "source_id": "own_video_97b298e4",
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "source_width": 1336,
        "source_height": 744,
        "coordinate_space": "source_xyxy_px_v1",
        "class_name": "human",
        "frozen": True,
        "freeze_timestamp_utc": freeze_ts,
        "reviewer": reviewer,
        "acceptance_eligible": False,
        "production_approved": False,
        "own_video_clip_specific": True,
        "frames": frames_out,
        "split_windows_s": {
            "train": list(TRAIN_T),
            "dev": list(DEV_T),
            "holdout": list(HOLDOUT_T),
        },
        "counts": integrity["by_split"],
        "fingerprints": integrity["fingerprints"],
        "canonical_fingerprint": sha256_payload(
            {
                "source_sha256": EXPECTED_SOURCE_SHA256,
                "frames": frames_out,
            }
        ),
        "metadata_distribution": _role_team_counts(frames_out),
    }

    split_manifest = {
        "schema": "human_detection_split_manifest_v1",
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "splits": {
            sp: [int(f["frame_idx"]) for f in frames_out if f["split"] == sp]
            for sp in ("train", "dev", "holdout")
        },
        "windows_s": annotations["split_windows_s"],
        "leakage_guards": {
            "unique_frames": True,
            "time_separated": True,
            "holdout_not_used_for_selection": True,
        },
    }

    class_policy = {
        "detection_class": {"id": 0, "name": "human"},
        "role_team_not_in_detector_labels": True,
        "note": "Role/team/eligibility retained in frozen annotations for R2 only.",
    }

    provenance = {
        "schema": "review_provenance_v1",
        "reviewer": reviewer,
        "runtime_draft": str(DEFAULT_RUNTIME / "draft_annotations.json"),
        "audit_log": str(audit_path or (DEFAULT_RUNTIME / "review_audit.jsonl")),
        "train_origins_allowed": [
            "manual",
            "proposal_reviewed",
            "proposal_reviewed_bulk",
        ],
        "dev_holdout_origin": "manual_blind",
        "protected_train_frames": list(PROTECTED_TRAIN_FRAME_INDICES),
        "repaired_train_frames": list(FAILED_TRAIN_FRAME_INDICES),
        "fingerprints_at_freeze": integrity["fingerprints"],
    }

    freeze_receipt = {
        "schema": "freeze_receipt_v1",
        "frozen": True,
        "freeze_timestamp_utc": freeze_ts,
        "canonical_fingerprint": annotations["canonical_fingerprint"],
        "integrity": integrity,
        "reviewer": reviewer,
        "explicit_freeze_command": "R1-F2-B-R1",
    }

    atomic_write_json(out_dir / "annotations.json", annotations, mode=0o644)
    atomic_write_json(out_dir / "split_manifest.json", split_manifest, mode=0o644)
    (out_dir / "class_policy.yaml").write_text(
        yaml.safe_dump(class_policy, sort_keys=False), encoding="utf-8"
    )
    atomic_write_json(out_dir / "review_provenance.json", provenance, mode=0o644)
    atomic_write_json(out_dir / "freeze_receipt.json", freeze_receipt, mode=0o644)

    checksums: dict[str, str] = {}
    for name in (
        "annotations.json",
        "split_manifest.json",
        "class_policy.yaml",
        "review_provenance.json",
        "freeze_receipt.json",
    ):
        data = (out_dir / name).read_bytes()
        checksums[name] = hashlib.sha256(data).hexdigest()
    (out_dir / "checksums.sha256").write_text(
        "\n".join(f"{h}  {n}" for n, h in sorted(checksums.items())) + "\n",
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(
        "# Own-video human detection GT v1\n\n"
        "Frozen independent human boxes for `own_video_97b298e4`.\n"
        "Detection class only: human. Role/team kept for R2, not detector labels.\n"
        f"Canonical fingerprint: `{annotations['canonical_fingerprint']}`\n"
        "Source video / frames are NOT in Git.\n",
        encoding="utf-8",
    )
    return {
        "out_dir": str(out_dir),
        "canonical_fingerprint": annotations["canonical_fingerprint"],
        "counts": integrity["by_split"],
        "metadata_distribution": annotations["metadata_distribution"],
        "checksums": checksums,
        "freeze_timestamp_utc": freeze_ts,
    }


def assert_audit_append_only(before: str, after: str) -> None:
    if not after.startswith(before):
        raise IndependentGTError("AUDIT_NOT_APPEND_ONLY")


__all__ = [
    "DEFAULT_FROZEN_DIR",
    "EXPECTED_DEV_FP",
    "EXPECTED_HOLDOUT_FP",
    "EXPECTED_PROTECTED_TRAIN_FP",
    "assert_audit_append_only",
    "clear_leftover_train_proposals",
    "validate_repaired_gt_integrity",
    "write_frozen_gt",
]
