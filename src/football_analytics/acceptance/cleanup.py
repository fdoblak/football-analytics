"""Allowlisted exact-path cleanup for Stage 16 temporary media."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from football_analytics.acceptance.download_manifest import sha256_file


class CleanupError(RuntimeError):
    pass


def safe_delete_exact_paths(
    *,
    allowlist: list[Path],
    receipt_path: Path,
    require_existing: bool = False,
) -> dict[str, Any]:
    """Delete only exact validated paths (no globs, no home roots)."""
    deleted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for raw in allowlist:
        path = Path(raw)
        text = str(path)
        if text in {"/", "/home", "/home/fdoblak", "~"} or text.startswith("~"):
            raise CleanupError(f"refusing broad path: {text}")
        if "*" in text or "?" in text:
            raise CleanupError(f"refusing glob path: {text}")
        if not path.exists():
            skipped.append({"path": text, "reason": "missing"})
            if require_existing:
                raise CleanupError(f"missing required path: {text}")
            continue
        if not path.is_file():
            raise CleanupError(f"only files allowed in Stage16 cleanup allowlist: {text}")
        rec = {
            "path": text,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        path.unlink()
        deleted.append(rec)
    receipt = {
        "permanent_delete_performed": bool(deleted),
        "deleted": deleted,
        "skipped": skipped,
        "bytes_deleted": sum(d["size_bytes"] for d in deleted),
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt
