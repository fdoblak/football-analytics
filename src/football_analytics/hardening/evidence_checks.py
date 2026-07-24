"""Evidence index consistency and report data consistency checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_analytics.core.hashing import hash_canonical_json
from football_analytics.evidence.collector import load_index


class EvidenceConsistencyError(ValueError):
    """Evidence index / report consistency failure."""


def check_evidence_index(
    index_path: Path,
    *,
    require_stage: str | None = None,
) -> dict[str, Any]:
    """Validate evidence index JSON structure and optional stage presence."""
    if not index_path.is_file():
        raise EvidenceConsistencyError(f"evidence index missing: {index_path}")
    raw = load_index(index_path)
    if not isinstance(raw, dict):
        raise EvidenceConsistencyError("evidence index must be a mapping")
    entries = raw.get("entries") or raw.get("items") or raw.get("files") or []
    if entries is None:
        entries = []
    if not isinstance(entries, list):
        # Some indexes nest under stages
        stages = raw.get("stages")
        if isinstance(stages, dict):
            entries = []
            for stage_entries in stages.values():
                if isinstance(stage_entries, list):
                    entries.extend(stage_entries)
        else:
            raise EvidenceConsistencyError("evidence index entries not list-like")
    stage_ids: set[str] = set()
    for item in entries:
        if isinstance(item, dict):
            sid = item.get("stage_id") or item.get("stage") or item.get("stage_dir")
            if sid:
                stage_ids.add(str(sid))
    # Also accept top-level stage directories via filesystem sibling check
    evidence_root = index_path.parent
    for child in evidence_root.iterdir():
        if child.is_dir() and child.name.startswith("stage_"):
            stage_ids.add(child.name)
    if (
        require_stage
        and require_stage not in stage_ids
        and not (evidence_root / require_stage).is_dir()
    ):
        raise EvidenceConsistencyError(f"required stage missing from index/fs: {require_stage}")
    return {
        "index_path": str(index_path),
        "entry_count": len(entries) if isinstance(entries, list) else 0,
        "stage_ids_sample": sorted(stage_ids)[-10:],
        "has_required_stage": (
            require_stage is None
            or require_stage in stage_ids
            or (evidence_root / require_stage).is_dir()
        ),
        "status": "PASS",
    }


def check_report_data_consistency(report: dict[str, Any]) -> dict[str, Any]:
    """Ensure single-player report invariants for Stage 15 acceptance."""
    errors: list[str] = []
    if report.get("team_summary_forbidden") is not True and "team_summary" in report:
        errors.append("team_summary present")
    if "team_summary" in report:
        errors.append("team_summary key forbidden")
    fp = report.get("reproducibility_fingerprint")
    if not fp or len(str(fp)) != 64:
        errors.append("missing or invalid reproducibility_fingerprint")
    # Recompute when builder payload fields available
    if "report_fingerprint_payload" in report:
        again = hash_canonical_json(report["report_fingerprint_payload"])
        if again != report.get("reproducibility_fingerprint"):
            errors.append("fingerprint mismatch vs payload")
    metrics = report.get("metrics") or report.get("metric_rows") or []
    not_evaluable = report.get("not_evaluable_metric_ids") or []
    if errors:
        raise EvidenceConsistencyError("; ".join(errors))
    return {
        "status": "PASS",
        "metric_count": len(metrics) if isinstance(metrics, list) else 0,
        "not_evaluable_count": len(not_evaluable) if isinstance(not_evaluable, list) else 0,
        "fingerprint": fp,
    }


def stage_evidence_json_only(stage_dir: Path) -> dict[str, Any]:
    """Assert stage evidence directory contains JSON files only (plus dirs)."""
    if not stage_dir.is_dir():
        raise EvidenceConsistencyError(f"stage evidence dir missing: {stage_dir}")
    bad: list[str] = []
    json_files: list[str] = []
    for path in sorted(stage_dir.iterdir()):
        if path.is_dir():
            continue
        if path.suffix.lower() != ".json":
            bad.append(path.name)
        else:
            # ensure parseable
            json.loads(path.read_text(encoding="utf-8"))
            json_files.append(path.name)
    if bad:
        raise EvidenceConsistencyError(f"non-json evidence files: {bad}")
    return {"json_files": json_files, "status": "PASS"}
