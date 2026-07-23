"""Manual review sampling + append-only audit (Stage 7A)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from football_analytics.core.hashing import hash_canonical_json
from football_analytics.identity.contracts import (
    load_identity_json_schema,
    validate_against_json_schema,
)
from football_analytics.identity.types import IdentityContractError


def should_enqueue_review(
    *,
    nearby_target_candidates: bool = False,
    evidence_conflict: bool = False,
    long_gap_or_cut_link: bool = False,
    multiple_confirmed_same_target: bool = False,
    track_multi_identity: bool = False,
    assignment_revoked: bool = False,
    low_coverage: bool = False,
) -> bool:
    return any(
        (
            nearby_target_candidates,
            evidence_conflict,
            long_gap_or_cut_link,
            multiple_confirmed_same_target,
            track_multi_identity,
            assignment_revoked,
            low_coverage,
        )
    )


def sample_review_items(
    candidates: Sequence[Mapping[str, Any]],
    *,
    max_items: int = 32,
) -> list[Mapping[str, Any]]:
    """Deterministic cap — do not spam every unknown."""
    if max_items < 0:
        raise IdentityContractError("max_items must be >= 0")
    ordered = sorted(
        candidates,
        key=lambda c: (
            str(c.get("priority", "z")),
            str(c.get("run_id", "")),
            str(c.get("video_id", "")),
            str(c.get("item_id", "")),
        ),
    )
    return list(ordered[:max_items])


def validate_audit_entry(payload: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(payload)
    schema = load_identity_json_schema("identity_manual_audit")
    validate_against_json_schema(data, schema)
    if data["provenance"].get("append_only") is not True:
        raise IdentityContractError("audit must be append_only")
    return data


def append_audit_log(
    path: Path,
    entry: Mapping[str, Any],
    *,
    contain_root: Path | None = None,
) -> str:
    """Append one validated audit entry as a JSONL line. Returns entry fingerprint."""
    validated = validate_audit_entry(entry)
    if path.is_symlink():
        raise IdentityContractError(f"symlink rejected: {path}")
    if contain_root is not None:
        resolved = path.resolve()
        root = contain_root.resolve()
        if not str(resolved).startswith(str(root)):
            raise IdentityContractError("audit path escapes contain_root")
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(validated, sort_keys=True, separators=(",", ":"))
    # Append-only: never rewrite prior lines.
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return hash_canonical_json(validated)


def read_audit_log(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise IdentityContractError(f"audit log missing or symlink: {path}")
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise IdentityContractError("audit line must be object")
        out.append(validate_audit_entry(obj))
    return out


__all__ = [
    "should_enqueue_review",
    "sample_review_items",
    "validate_audit_entry",
    "append_audit_log",
    "read_audit_log",
]
