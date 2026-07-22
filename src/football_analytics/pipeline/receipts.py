"""Stage execution receipt writer (Stage 2D)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from football_analytics.core.records import write_json_record
from football_analytics.pipeline.exceptions import StageError
from football_analytics.pipeline.types import StageResult

RECEIPT_SCHEMA_VERSION = 1


def build_stage_execution_receipt(result: StageResult) -> dict[str, Any]:
    """Build a schema_version=1 stage execution receipt payload."""
    if not isinstance(result, StageResult):
        raise StageError("result must be StageResult")
    payload = result.to_dict()
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "run_id": payload["run_id"],
        "stage_name": payload["stage_name"],
        "stage_version": payload["stage_version"],
        "status": payload["status"],
        "cache_key": payload["cache_key"],
        "cache_hit": payload["cache_hit"],
        "started_at_utc": payload["started_at_utc"],
        "finished_at_utc": payload["finished_at_utc"],
        "duration_ms": payload["duration_ms"],
        "execution_fingerprint": payload["execution_fingerprint"],
        "inputs": payload["inputs"],
        "outputs": payload["outputs"],
        "metrics": payload["metrics"],
        "warnings": payload["warnings"],
        "error": payload["error"],
    }


def write_stage_execution_receipt(
    path: Path | str,
    result: StageResult,
    *,
    contain_root: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Atomically write ``stage_execution_receipt.json`` via ``write_json_record``."""
    payload = build_stage_execution_receipt(result)
    return write_json_record(
        path,
        payload,
        contain_root=contain_root,
        overwrite=overwrite,
    )
