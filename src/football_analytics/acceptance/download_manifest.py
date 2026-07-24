"""Download provenance + SHA-256 manifest helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path, *, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def file_record(
    *,
    path: Path,
    root: Path,
    source_url: str,
    license_id: str,
    match_id: str,
    half: int | None,
    media_type: str,
    normalization_status: str = "source_immutable",
) -> dict[str, Any]:
    path = Path(path)
    rel = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
    return {
        "relative_path": rel,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "source_url": source_url,
        "license": license_id,
        "match_id": match_id,
        "half": half,
        "media_type": media_type,
        "modification_or_normalization": normalization_status,
    }


def write_source_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **payload,
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
