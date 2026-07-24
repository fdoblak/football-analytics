"""Atomic output helpers and failure receipts for resumable runs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from football_analytics.core.hashing import hash_canonical_json
from football_analytics.core.records import write_json_record
from football_analytics.core.redaction import is_sensitive_key, redact_text, redact_value
from football_analytics.hardening.policy import HardeningPolicy, load_hardening_policy


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def sanitize_record_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Redact values and strip sensitive keys (records forbid secret-bearing keys)."""
    red = redact_value(payload)
    if not isinstance(red, dict):
        raise TypeError("payload must redact to a mapping")
    out: dict[str, Any] = {}
    stripped: list[str] = []
    for key, value in red.items():
        if is_sensitive_key(key):
            stripped.append(str(key))
            continue
        if isinstance(value, dict):
            out[key] = sanitize_record_payload(value)
        else:
            out[key] = value
    if stripped:
        out["sensitive_keys_stripped"] = sorted(stripped)
    return out


def write_atomic_json(
    path: Path,
    payload: dict[str, Any],
    *,
    contain_root: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Write JSON via the shared atomic record writer (temp + rename)."""
    safe = sanitize_record_payload(payload)
    return write_json_record(
        path,
        safe,
        contain_root=contain_root,
        overwrite=overwrite,
    )


def write_failure_receipt(
    output_dir: Path,
    *,
    run_id: str,
    stage_name: str,
    error: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a redacted failure receipt for interrupted / failed stages."""
    out = Path(output_dir)
    out.mkdir(parents=True, mode=0o700, exist_ok=True)
    base: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "stage_name": stage_name,
        "error": redact_text(str(error)),
        "failed_at_utc": _utc_now(),
        "permanent_delete_performed": False,
    }
    if extra:
        base["extra"] = sanitize_record_payload(dict(extra))
    base["receipt_fingerprint"] = hash_canonical_json(
        {k: v for k, v in base.items() if k != "receipt_fingerprint"}
    )
    path = out / f"failure_receipt_{stage_name}.json"
    write_json_record(path, base, contain_root=out, overwrite=True)
    base["receipt_path"] = str(path)
    return base


def mark_interrupted(run_dir: Path, *, policy: HardeningPolicy | None = None) -> Path:
    pol = policy or load_hardening_policy()
    marker_name = str(pol.raw["recovery"]["interrupted_marker_name"])
    run = Path(run_dir)
    run.mkdir(parents=True, mode=0o700, exist_ok=True)
    marker = run / marker_name
    write_json_record(
        marker,
        {
            "schema_version": 1,
            "status": "interrupted",
            "marked_at_utc": _utc_now(),
        },
        contain_root=run,
        overwrite=True,
    )
    return marker


def clear_interrupted(run_dir: Path, *, policy: HardeningPolicy | None = None) -> bool:
    pol = policy or load_hardening_policy()
    marker = Path(run_dir) / str(pol.raw["recovery"]["interrupted_marker_name"])
    if marker.exists():
        marker.unlink()
        return True
    return False


def is_interrupted(run_dir: Path, *, policy: HardeningPolicy | None = None) -> bool:
    pol = policy or load_hardening_policy()
    marker = Path(run_dir) / str(pol.raw["recovery"]["interrupted_marker_name"])
    return marker.is_file() and not marker.is_symlink()


def recover_interrupted_run(
    run_dir: Path,
    *,
    policy: HardeningPolicy | None = None,
) -> dict[str, Any]:
    """Describe recovery action for an interrupted run (resume-safe; no delete)."""
    interrupted = is_interrupted(run_dir, policy=policy)
    return {
        "run_dir": str(run_dir),
        "interrupted": interrupted,
        "action": "resume_or_restart_required" if interrupted else "none",
        "permanent_delete_performed": False,
        "recovered_at_utc": _utc_now(),
    }
