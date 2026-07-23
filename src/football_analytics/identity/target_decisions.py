"""Schema-validated manual target decisions + CAS / audit (Stage 7E)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from football_analytics.core.hashing import hash_canonical_json
from football_analytics.core.records import write_json_record
from football_analytics.identity.contracts import (
    load_identity_json_schema,
    validate_against_json_schema,
)
from football_analytics.identity.review_audit import append_audit_log, read_audit_log
from football_analytics.identity.types import AssignmentStatus, IdentityContractError


class TargetDecisionError(IdentityContractError):
    """Manual decision validation / CAS failure."""


DECISION_TO_STATUS = {
    "confirm": AssignmentStatus.CONFIRMED.value,
    "reject": AssignmentStatus.REJECTED.value,
    "keep_provisional": AssignmentStatus.PROVISIONAL.value,
    "revoke": AssignmentStatus.REVOKED.value,
    "unknown": AssignmentStatus.UNKNOWN.value,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def compute_decision_record_hash(decision_without_hash: Mapping[str, Any]) -> str:
    payload = dict(decision_without_hash)
    payload.pop("record_hash", None)
    return hash_canonical_json(payload)


def validate_target_decision(payload: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(payload)
    schema = load_identity_json_schema("target_decision")
    validate_against_json_schema(data, schema)
    if int(data["end_frame_index"]) < int(data["start_frame_index"]):
        raise TargetDecisionError("decision interval invalid")
    expected_status = DECISION_TO_STATUS[str(data["decision"])]
    if str(data["new_status"]) != expected_status:
        raise TargetDecisionError("new_status must match decision")
    expected_hash = compute_decision_record_hash(data)
    if data["record_hash"] != expected_hash:
        raise TargetDecisionError("HASH_VERSION_MISMATCH")
    if data["provenance"].get("append_only") is not True:
        raise TargetDecisionError("APPEND_ONLY_VIOLATION")
    if data["provenance"].get("runtime_only") is not True:
        raise TargetDecisionError("decisions must be runtime_only")
    if data["provenance"].get("face_biometric_forbidden") is not True:
        raise TargetDecisionError("FACE_BIOMETRIC_FORBIDDEN")
    return data


def build_target_decision(
    *,
    decision_id: str,
    manifest: Mapping[str, Any],
    track_id: int,
    start_frame_index: int,
    end_frame_index: int,
    decision: str,
    reviewer_id: str,
    reason: str,
    expected_assignment_version: int,
    expected_previous_status: str | None,
    evidence_fingerprints: Sequence[str],
    linked_evidence_ids: Sequence[str] | None = None,
    previous_audit_hash: str | None,
    candidate_id: str | None = None,
    supersedes_decision_id: str | None = None,
    synthetic_fixture: bool = True,
    decided_at_utc: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    if decision not in DECISION_TO_STATUS:
        raise TargetDecisionError(f"invalid decision: {decision}")
    if decision not in set(manifest.get("allowed_decisions") or []):
        raise TargetDecisionError("decision not allowed by manifest")
    if str(manifest["manifest_hash"]) == "":
        raise TargetDecisionError("manifest_hash required")
    body: dict[str, Any] = {
        "schema_version": 1,
        "decision_id": decision_id,
        "manifest_id": manifest["manifest_id"],
        "manifest_hash": manifest["manifest_hash"],
        "run_id": manifest["run_id"],
        "video_id": manifest["video_id"],
        "target_player_id": manifest["target_player_id"],
        "track_id": int(track_id),
        "candidate_id": candidate_id,
        "start_frame_index": int(start_frame_index),
        "end_frame_index": int(end_frame_index),
        "decision": decision,
        "reviewer_id": reviewer_id,
        "decided_at_utc": decided_at_utc or _utc_now(),
        "expected_assignment_version": int(expected_assignment_version),
        "expected_previous_status": expected_previous_status,
        "new_status": DECISION_TO_STATUS[decision],
        "reason": reason,
        "evidence_fingerprints": list(evidence_fingerprints),
        "linked_evidence_ids": list(linked_evidence_ids or []),
        "previous_audit_hash": previous_audit_hash,
        "supersedes_decision_id": supersedes_decision_id,
        "provenance": {
            "stage": "7E",
            "append_only": True,
            "runtime_only": True,
            "synthetic_fixture": bool(synthetic_fixture),
            "face_biometric_forbidden": True,
            "notes": notes,
        },
    }
    body["record_hash"] = compute_decision_record_hash(body)
    return validate_target_decision(body)


def assert_manifest_cas(
    decision: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    current_assignment_version: int,
) -> None:
    if decision["manifest_id"] != manifest["manifest_id"]:
        raise TargetDecisionError("stale or mismatched manifest_id")
    if decision["manifest_hash"] != manifest["manifest_hash"]:
        raise TargetDecisionError("stale decision rejection: manifest_hash mismatch")
    if int(decision["expected_assignment_version"]) != int(current_assignment_version):
        raise TargetDecisionError("stale decision rejection: assignment version CAS failed")
    if int(decision["expected_assignment_version"]) != int(manifest["expected_assignment_version"]):
        raise TargetDecisionError("stale decision rejection: manifest version mismatch")


def assert_no_duplicate_decision(
    decision: Mapping[str, Any],
    *,
    existing_decision_ids: Sequence[str],
    existing_record_hashes: Sequence[str],
) -> None:
    if decision["decision_id"] in set(existing_decision_ids):
        raise TargetDecisionError("duplicate decision rejection")
    if decision["record_hash"] in set(existing_record_hashes):
        raise TargetDecisionError("duplicate/replay decision rejection")


def write_decision_file(
    path: Path,
    decision: Mapping[str, Any],
    *,
    contain_root: Path | None = None,
) -> Path:
    validated = validate_target_decision(decision)
    return write_json_record(
        path,
        validated,
        contain_root=contain_root,
        overwrite=False,
        mode=0o600,
    )


def append_decision_audit(
    audit_path: Path,
    decision: Mapping[str, Any],
    *,
    contain_root: Path | None = None,
) -> str:
    """Append Stage 7A-compatible audit entry derived from a 7E decision."""
    action = str(decision["decision"])
    if action == "keep_provisional":
        action = "annotate"
    entry = {
        "schema_version": 1,
        "audit_id": f"aud_{decision['decision_id']}"[:64],
        "run_id": decision["run_id"],
        "video_id": decision["video_id"],
        "target_player_id": decision["target_player_id"],
        "assignment_id": None,
        "track_id": int(decision["track_id"]),
        "actor_id": decision["reviewer_id"],
        "acted_at_utc": decision["decided_at_utc"],
        "action": (
            action
            if action
            in {
                "confirm",
                "reject",
                "revoke",
                "supersede",
                "request_review",
                "annotate",
                "unknown",
            }
            else "unknown"
        ),
        "previous_decision": decision.get("expected_previous_status"),
        "new_decision": decision["new_status"],
        "reason": decision["reason"],
        "artifact_hashes": list(decision.get("evidence_fingerprints") or [])
        + [decision["record_hash"], decision["manifest_hash"]],
        "linked_evidence_ids": list(decision.get("linked_evidence_ids") or []),
        "provenance": {
            "stage": "7A",
            "append_only": True,
            "notes": json.dumps(
                {
                    "stage7e_decision_id": decision["decision_id"],
                    "previous_audit_hash": decision.get("previous_audit_hash"),
                    "synthetic_fixture": bool(
                        (decision.get("provenance") or {}).get("synthetic_fixture", True)
                    ),
                },
                sort_keys=True,
            ),
        },
    }
    # audit_id must match pattern ^[a-z][a-z0-9_]{1,63}$
    aid = str(entry["audit_id"]).replace("-", "_").lower()
    if not aid[0].isalpha():
        aid = "a" + aid
    entry["audit_id"] = aid[:64]
    return append_audit_log(audit_path, entry, contain_root=contain_root)


def load_decision_ids(decision_dir: Path) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    hashes: list[str] = []
    if not decision_dir.is_dir():
        return ids, hashes
    for path in sorted(decision_dir.glob("*.json")):
        if path.is_symlink():
            raise TargetDecisionError(f"symlink rejected: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        validated = validate_target_decision(data)
        ids.append(validated["decision_id"])
        hashes.append(validated["record_hash"])
    return ids, hashes


def latest_audit_hash(audit_path: Path) -> str | None:
    if not audit_path.is_file():
        return None
    entries = read_audit_log(audit_path)
    if not entries:
        return None
    return hash_canonical_json(entries[-1])


__all__ = [
    "TargetDecisionError",
    "DECISION_TO_STATUS",
    "compute_decision_record_hash",
    "validate_target_decision",
    "build_target_decision",
    "assert_manifest_cas",
    "assert_no_duplicate_decision",
    "write_decision_file",
    "append_decision_audit",
    "load_decision_ids",
    "latest_audit_hash",
]
