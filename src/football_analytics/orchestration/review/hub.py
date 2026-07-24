"""Unified manual review hub (Stage 14B)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from football_analytics.core.hashing import hash_canonical_json
from football_analytics.core.records import write_json_record
from football_analytics.orchestration.config import load_review_config, review_config_fingerprint
from football_analytics.orchestration.contracts import (
    REVIEW_DOMAINS,
    load_review_json_schema,
    validate_against_json_schema,
)
from football_analytics.orchestration.types import OrchestrationError, StaleArtifactError

DECISION_TO_STATUS = {
    "confirm": "confirmed",
    "reject": "rejected",
    "keep_provisional": "provisional",
    "revoke": "revoked",
    "unknown": "unknown",
    "request_review": "review_requested",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _package_hash(body: Mapping[str, Any]) -> str:
    payload = {k: v for k, v in body.items() if k != "package_hash"}
    return hash_canonical_json(payload)


def _decision_hash(body: Mapping[str, Any]) -> str:
    payload = {k: v for k, v in body.items() if k != "record_hash"}
    return hash_canonical_json(payload)


def _synthetic_domain_items(domain: str) -> list[dict[str, Any]]:
    return [
        {
            "item_id": f"{domain}_item_001",
            "proposed_status": "provisional",
            "manual_review_required": True,
            "summary": f"synthetic {domain} ambiguity",
            "artifact_refs": {},
        }
    ]


def prepare_review_package(
    *,
    package_id: str,
    run_id: str,
    video_id: str,
    target_player_id: str,
    output_dir: Path,
    domains: Sequence[str] | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    cfg = load_review_config(project_root=project_root)
    cfg_fp = review_config_fingerprint(cfg)
    domain_list = list(domains) if domains is not None else list(REVIEW_DOMAINS)
    for d in domain_list:
        if d not in REVIEW_DOMAINS:
            raise OrchestrationError(f"unknown review domain: {d}")

    domain_rows = []
    for d in domain_list:
        domain_rows.append(
            {
                "domain": d,
                "items": _synthetic_domain_items(d),
                "review_required_for_confirmed": True,
            }
        )

    body: dict[str, Any] = {
        "schema_version": 1,
        "package_id": package_id,
        "run_id": run_id,
        "video_id": video_id,
        "target_player_id": target_player_id,
        "config_fingerprint": cfg_fp,
        "created_at_utc": _utc_now(),
        "domains": domain_rows,
        "allowed_decisions": list(cfg.get("allowed_decisions") or list(DECISION_TO_STATUS)),
        "provenance": {
            "append_only": True,
            "auto_confirm_forbidden": True,
            "confirmed_requires_review": True,
            "synthetic_fixture": True,
            "notes": "stage_14b_unified_review",
        },
        "package_hash": "",
    }
    body["package_hash"] = _package_hash(body)
    schema = load_review_json_schema("unified_review_package")
    validate_against_json_schema(body, schema)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pkg_path = output_dir / "unified_review_package.json"
    write_json_record(pkg_path, body, overwrite=False)
    audit_path = output_dir / "unified_review_audit.jsonl"
    append_audit(
        audit_path,
        {
            "schema_version": 1,
            "audit_id": f"aud_prepare_{package_id}"[:64],
            "run_id": run_id,
            "package_id": package_id,
            "domain": "identity",
            "item_id": "package",
            "decision_id": None,
            "actor_id": "system",
            "acted_at_utc": _utc_now(),
            "action": "prepare",
            "previous_decision": None,
            "new_decision": None,
            "reason": "prepare_unified_review_package",
            "artifact_hashes": {"package_hash": body["package_hash"]},
            "provenance": {"append_only": True, "notes": None},
        },
    )
    return body


def append_audit(path: Path, entry: Mapping[str, Any]) -> str:
    schema = load_review_json_schema("unified_review_audit")
    data = dict(entry)
    validate_against_json_schema(data, schema)
    if path.is_symlink():
        raise OrchestrationError(f"symlink rejected: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(data, sort_keys=True, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return hash_canonical_json(data)


def build_decision(
    *,
    decision_id: str,
    package: Mapping[str, Any],
    domain: str,
    item_id: str,
    decision: str,
    reviewer_id: str,
    reason: str,
    previous_audit_hash: str | None = None,
    supersedes_decision_id: str | None = None,
) -> dict[str, Any]:
    if decision not in DECISION_TO_STATUS:
        raise OrchestrationError(f"invalid decision: {decision}")
    if decision not in set(package.get("allowed_decisions") or []):
        raise OrchestrationError("decision not allowed by package")
    # No confirmed without review when required.
    if decision == "confirm":
        domain_row = next((d for d in package["domains"] if d["domain"] == domain), None)
        if domain_row is None:
            raise OrchestrationError("domain missing from package")
        item = next((i for i in domain_row["items"] if i["item_id"] == item_id), None)
        if item is None:
            raise OrchestrationError("item missing from package")
        if item.get("manual_review_required") and reviewer_id in {"", "system", "auto"}:
            raise OrchestrationError("confirmed requires scoped manual review")
        if domain_row.get("review_required_for_confirmed") is not True:
            raise OrchestrationError("confirmed requires review flag")

    body: dict[str, Any] = {
        "schema_version": 1,
        "decision_id": decision_id,
        "package_id": package["package_id"],
        "package_hash": package["package_hash"],
        "expected_package_hash": package["package_hash"],
        "run_id": package["run_id"],
        "domain": domain,
        "item_id": item_id,
        "decision": decision,
        "reviewer_id": reviewer_id,
        "decided_at_utc": _utc_now(),
        "new_status": DECISION_TO_STATUS[decision],
        "reason": reason,
        "supersedes_decision_id": supersedes_decision_id,
        "previous_audit_hash": previous_audit_hash,
        "record_hash": "",
        "provenance": {
            "append_only": True,
            "runtime_only": True,
            "cas": True,
            "synthetic_fixture": True,
            "notes": None,
        },
    }
    body["record_hash"] = _decision_hash(body)
    schema = load_review_json_schema("unified_review_decision")
    validate_against_json_schema(body, schema)
    return body


def assert_package_cas(decision: Mapping[str, Any], *, package: Mapping[str, Any]) -> None:
    if decision["package_id"] != package["package_id"]:
        raise StaleArtifactError("stale decision: package_id mismatch")
    if decision["expected_package_hash"] != package["package_hash"]:
        raise StaleArtifactError("stale decision: package_hash CAS failed")
    if decision["package_hash"] != package["package_hash"]:
        raise StaleArtifactError("stale decision: embedded package_hash mismatch")


def apply_decision(
    *,
    decision: Mapping[str, Any],
    package: Mapping[str, Any],
    output_dir: Path,
    existing_decision_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    assert_package_cas(decision, package=package)
    existing = set(existing_decision_ids or [])
    if decision["decision_id"] in existing:
        raise OrchestrationError("duplicate decision_id (append-only CAS)")

    output_dir = Path(output_dir)
    decisions_dir = output_dir / "decisions"
    decisions_dir.mkdir(parents=True, exist_ok=True)
    out_path = decisions_dir / f"{decision['decision_id']}.json"
    if out_path.exists():
        raise OrchestrationError("decision file exists (no-overwrite / append-only)")
    validated = dict(decision)
    schema = load_review_json_schema("unified_review_decision")
    validate_against_json_schema(validated, schema)
    write_json_record(out_path, validated, overwrite=False)

    audit_path = output_dir / "unified_review_audit.jsonl"
    append_audit(
        audit_path,
        {
            "schema_version": 1,
            "audit_id": f"aud_{decision['decision_id']}"[:64],
            "run_id": decision["run_id"],
            "package_id": decision["package_id"],
            "domain": decision["domain"],
            "item_id": decision["item_id"],
            "decision_id": decision["decision_id"],
            "actor_id": decision["reviewer_id"],
            "acted_at_utc": decision["decided_at_utc"],
            "action": decision["decision"],
            "previous_decision": None,
            "new_decision": decision["decision"],
            "reason": decision["reason"],
            "artifact_hashes": {"record_hash": decision["record_hash"]},
            "provenance": {"append_only": True, "notes": None},
        },
    )
    return validated


def revoke_decision(
    *,
    previous: Mapping[str, Any],
    package: Mapping[str, Any],
    output_dir: Path,
    reviewer_id: str,
    reason: str,
    new_decision_id: str,
) -> dict[str, Any]:
    dec = build_decision(
        decision_id=new_decision_id,
        package=package,
        domain=str(previous["domain"]),
        item_id=str(previous["item_id"]),
        decision="revoke",
        reviewer_id=reviewer_id,
        reason=reason,
        supersedes_decision_id=str(previous["decision_id"]),
    )
    return apply_decision(decision=dec, package=package, output_dir=output_dir)


__all__ = [
    "prepare_review_package",
    "build_decision",
    "apply_decision",
    "assert_package_cas",
    "revoke_decision",
    "append_audit",
]
