"""RISK-041 safe cache GC: dry-run and quarantine only (no permanent delete by default)."""

from __future__ import annotations

import contextlib
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from football_analytics.core.hashing import hash_canonical_json
from football_analytics.core.records import write_json_record
from football_analytics.core.redaction import redact_text
from football_analytics.hardening.policy import HardeningPolicy, load_hardening_policy
from football_analytics.pipeline.cache import acquire_key_lock, entry_dir

GcMode = Literal["dry_run", "quarantine"]


class CacheGcError(ValueError):
    """Cache GC failure."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _entry_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            fp = Path(root) / name
            if fp.is_symlink():
                continue
            with contextlib.suppress(OSError):
                total += fp.stat().st_size
    return total


def list_cache_entries(cache_root: Path) -> list[dict[str, Any]]:
    """List v1/sha256 cache entries with age/size metadata."""
    root = Path(cache_root) / "v1" / "sha256"
    if not root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    now = time.time()
    for ab_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.is_symlink()):
        for entry in sorted(p for p in ab_dir.iterdir() if p.is_dir() and not p.is_symlink()):
            key = f"{ab_dir.name}{entry.name}"
            if len(key) != 64:
                continue
            mtime = entry.stat().st_mtime
            out.append(
                {
                    "cache_key": key,
                    "path": str(entry),
                    "age_seconds": max(0.0, now - mtime),
                    "size_bytes": _entry_size_bytes(entry),
                    "mtime_utc": datetime.fromtimestamp(mtime, tz=timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                }
            )
    return out


def plan_cache_gc(
    cache_root: Path,
    *,
    policy: HardeningPolicy | None = None,
    max_age_days: float | None = None,
    max_entry_bytes: int | None = None,
) -> dict[str, Any]:
    """Dry-run plan of GC candidates. Never deletes."""
    pol = policy or load_hardening_policy()
    if pol.automatic_purge or pol.permanent_delete_by_default:
        raise CacheGcError("unsafe GC policy flags must remain false")
    age_days = (
        float(max_age_days)
        if max_age_days is not None
        else float(pol.raw["cache_gc"]["max_age_days"])
    )
    size_cap = (
        int(max_entry_bytes)
        if max_entry_bytes is not None
        else int(pol.raw["cache_gc"]["max_entry_bytes"])
    )
    age_sec = age_days * 86400.0
    entries = list_cache_entries(cache_root)
    candidates: list[dict[str, Any]] = []
    for item in entries:
        reasons: list[str] = []
        if item["age_seconds"] >= age_sec:
            reasons.append("stale_age")
        if int(item["size_bytes"]) > size_cap:
            reasons.append("oversize")
        if reasons:
            candidates.append({**item, "reasons": reasons})
    plan = {
        "schema_version": 1,
        "mode": "dry_run",
        "cache_root": str(cache_root),
        "scanned_entries": len(entries),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "permanent_delete_performed": False,
        "automatic_purge": False,
        "planned_at_utc": _utc_now(),
        "risk_id": "RISK-041",
    }
    plan["plan_fingerprint"] = hash_canonical_json(
        {k: v for k, v in plan.items() if k != "plan_fingerprint"}
    )
    return plan


def execute_cache_gc(
    cache_root: Path,
    *,
    quarantine_root: Path,
    mode: GcMode = "dry_run",
    policy: HardeningPolicy | None = None,
    max_age_days: float | None = None,
    max_entry_bytes: int | None = None,
) -> dict[str, Any]:
    """Execute GC as dry-run or quarantine moves. Never permanent-deletes by default."""
    pol = policy or load_hardening_policy()
    if mode not in {"dry_run", "quarantine"}:
        raise CacheGcError(f"unsupported GC mode: {mode}")
    plan = plan_cache_gc(
        cache_root,
        policy=pol,
        max_age_days=max_age_days,
        max_entry_bytes=max_entry_bytes,
    )
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "mode": mode,
        "cache_root": str(cache_root),
        "quarantine_root": str(quarantine_root),
        "planned_candidates": plan["candidate_count"],
        "quarantined": [],
        "permanent_delete_performed": False,
        "automatic_purge": False,
        "executed_at_utc": _utc_now(),
        "plan_fingerprint": plan["plan_fingerprint"],
        "risk_id": "RISK-041",
    }
    if mode == "dry_run":
        receipt["receipt_fingerprint"] = hash_canonical_json(
            {k: v for k, v in receipt.items() if k != "receipt_fingerprint"}
        )
        return receipt

    qroot = Path(quarantine_root)
    qroot.mkdir(parents=True, mode=0o700, exist_ok=True)
    for cand in plan["candidates"]:
        key = str(cand["cache_key"])
        entry = entry_dir(cache_root, key)
        if not entry.exists():
            continue
        dest = qroot / key
        if dest.exists():
            dest = qroot / f"{key}_{secrets.token_hex(4)}"
        with acquire_key_lock(
            cache_root,
            key,
            timeout_seconds=float(pol.raw["concurrency"]["lock_timeout_seconds"]),
        ):
            if not entry.exists():
                continue
            dest.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
            os.rename(str(entry), str(dest))
        item_receipt = {
            "schema_version": 1,
            "original_cache_key": key,
            "quarantine_path": str(dest),
            "reasons": list(cand.get("reasons") or []),
            "permanent_delete_performed": False,
            "reason": redact_text("cache_gc_quarantine"),
            "detected_at_utc": _utc_now(),
        }
        item_receipt["receipt_fingerprint"] = hash_canonical_json(
            {k: v for k, v in item_receipt.items() if k != "receipt_fingerprint"}
        )
        write_json_record(
            dest / "cache_gc_quarantine_receipt.json",
            item_receipt,
            contain_root=dest,
            overwrite=False,
        )
        receipt["quarantined"].append(item_receipt)
    receipt["receipt_fingerprint"] = hash_canonical_json(
        {k: v for k, v in receipt.items() if k != "receipt_fingerprint"}
    )
    return receipt
