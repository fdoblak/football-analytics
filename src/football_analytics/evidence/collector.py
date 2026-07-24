"""Evidence retention helpers — small safe artifacts for GitHub history."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from football_analytics.core.hashing import sha256_file
from football_analytics.data.registry import default_project_root

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_BACKFILL_TOTAL_BYTES = 100 * 1024 * 1024

SAFE_SUFFIXES = frozenset({".json", ".jsonl", ".md", ".svg", ".png", ".txt", ".yaml", ".yml"})
UNSAFE_SUFFIXES = frozenset(
    {".pt", ".pth", ".onnx", ".mp4", ".mkv", ".avi", ".npy", ".npz", ".bin", ".ckpt"}
)

# Workspace check dirs mapped to stage ids (best-effort).
WORKSPACE_STAGE_MAP: dict[str, str] = {
    "foundation_checks": "stage_02",
    "storage_checks": "stage_01",
    "data_contract_checks": "stage_02c",
    "video_contract_checks": "stage_03a",
    "video_probe_checks": "stage_03b",
    "video_normalization_checks": "stage_03c",
    "frame_timeline_checks": "stage_03d",
    "broadcast_contract_checks": "stage_04a",
    "shot_boundary_checks": "stage_04b",
    "camera_view_checks": "stage_04c",
    "broadcast_pipeline_checks": "stage_04d",
    "detection_contract_checks": "stage_05a",
    "human_detection_checks": "stage_05b",
    "ball_detection_checks": "stage_05c",
    "human_role_checks": "stage_05d",
    "detection_pipeline_checks": "stage_05e",
    "tracking_contract_checks": "stage_06a",
    "human_tracking_checks": "stage_06b",
    "ball_tracking_checks": "stage_06c",
    "tracking_pipeline_checks": "stage_06d",
    "identity_contract_checks": "stage_07a",
    "appearance_reid_checks": "stage_07b",
    "team_assignment_checks": "stage_07c",
    "jersey_ocr_checks": "stage_07d",
    "target_identity_checks": "stage_07e",
    "calibration_contract_checks": "stage_08a",
    "pitch_feature_checks": "stage_08b",
    "homography_checks": "stage_08c",
    "pitch_projection_checks": "stage_08d",
    "physical_metric_contract_checks": "stage_09a",
    "target_trajectory_checks": "stage_09b",
}


def evidence_root(*, project_root: Path | None = None) -> Path:
    return (project_root or default_project_root()) / "artifacts" / "evidence"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def is_safe_evidence_file(path: Path) -> tuple[bool, str]:
    if not path.is_file() or path.is_symlink():
        return False, "not_regular_file"
    suffix = path.suffix.lower()
    if suffix in UNSAFE_SUFFIXES:
        return False, "unsafe_suffix"
    if suffix == ".parquet":
        # Allow only tiny synthetic parquet under evidence staging.
        size = path.stat().st_size
        if size > 512 * 1024:
            return False, "parquet_too_large"
        return True, "ok"
    if suffix not in SAFE_SUFFIXES:
        return False, "suffix_not_allowlisted"
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        return False, "file_too_large"
    if size == 0:
        return False, "empty"
    return True, "ok"


def load_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "updated_at_utc": _utc_now(), "entries": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("index root must be object")
    return data


def save_index(path: Path, index: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(index)
    payload["updated_at_utc"] = _utc_now()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _known_hashes(entries: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(e["sha256"]) for e in entries if e.get("sha256")}


def register_or_copy(
    *,
    stage_id: str,
    source: Path,
    dest_dir: Path,
    index: dict[str, Any],
    artifact_type: str,
    git_tracked: bool,
    source_path: str | None = None,
    notes: str = "",
    commit_sha: str | None = None,
) -> dict[str, Any] | None:
    ok, reason = is_safe_evidence_file(source)
    entries = list(index.get("entries") or [])
    hashes = _known_hashes(entries)
    if not ok:
        entry = {
            "artifact_id": f"{stage_id}_{source.name}_skipped",
            "stage_id": stage_id,
            "relative_path": None,
            "source_path": source_path or str(source),
            "sha256": None,
            "size_bytes": source.stat().st_size if source.exists() else None,
            "artifact_type": artifact_type,
            "license_class": "unknown",
            "data_classification": "restricted_or_large",
            "status": "skipped_unsafe",
            "git_tracked": False,
            "reason_not_git": reason,
            "commit_sha": commit_sha,
            "notes": notes,
        }
        entries.append(entry)
        index["entries"] = entries
        return entry

    digest = sha256_file(source)
    if digest in hashes:
        return None  # dedupe
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / source.name
    if dest.exists():
        dest = dest_dir / f"{digest[:12]}_{source.name}"
    shutil.copy2(source, dest)
    rel = str(dest.relative_to(evidence_root().parent.parent)) if False else None
    # Prefer path relative to project root
    try:
        rel = str(dest.resolve().relative_to(default_project_root().resolve()))
    except ValueError:
        rel = str(dest)
    entry = {
        "artifact_id": f"{stage_id}_{digest[:16]}",
        "stage_id": stage_id,
        "relative_path": rel,
        "source_path": source_path or str(source),
        "sha256": digest,
        "size_bytes": dest.stat().st_size,
        "artifact_type": artifact_type,
        "license_class": "synthetic_or_project",
        "data_classification": "safe_small_evidence",
        "status": "present",
        "git_tracked": git_tracked,
        "reason_not_git": None,
        "commit_sha": commit_sha,
        "notes": notes,
    }
    entries.append(entry)
    index["entries"] = entries
    return entry


def backfill_from_workspace(
    *,
    workspace_root: Path,
    project_root: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Copy small safe JSON summaries from workspace check dirs into artifacts/evidence."""
    root = project_root or default_project_root()
    ev = evidence_root(project_root=root)
    index_path = ev / "index.json"
    index = load_index(index_path)
    copied = 0
    skipped = 0
    not_available = 0
    total_bytes = sum(int(e.get("size_bytes") or 0) for e in index.get("entries") or [])

    for dirname, stage_id in sorted(WORKSPACE_STAGE_MAP.items()):
        src_dir = workspace_root / dirname
        stage_dir = ev / stage_id
        if not src_dir.is_dir():
            entry = {
                "artifact_id": f"{stage_id}_workspace_missing",
                "stage_id": stage_id,
                "relative_path": None,
                "source_path": str(src_dir),
                "sha256": None,
                "size_bytes": None,
                "artifact_type": "workspace_check_dir",
                "license_class": "n/a",
                "data_classification": "missing",
                "status": "not_available_cleaned",
                "git_tracked": False,
                "reason_not_git": "workspace_dir_absent",
                "commit_sha": None,
                "notes": "cleaned_or_never_present",
            }
            # avoid duplicate missing markers
            if not any(
                e.get("artifact_id") == entry["artifact_id"] for e in index.get("entries") or []
            ):
                index.setdefault("entries", []).append(entry)
                not_available += 1
            continue

        candidates: list[Path] = []
        for pattern in ("*.json", "*receipt*.json", "*quality*.json", "*summary*.json"):
            candidates.extend(src_dir.glob(pattern))
            candidates.extend(src_dir.glob(f"**/{pattern}"))
        # Deduplicate paths
        uniq = sorted({p.resolve() for p in candidates if p.is_file()})
        for path in uniq:
            if total_bytes >= MAX_BACKFILL_TOTAL_BYTES:
                skipped += 1
                continue
            ok, _reason = is_safe_evidence_file(path)
            if not ok:
                skipped += 1
                if not dry_run:
                    register_or_copy(
                        stage_id=stage_id,
                        source=path,
                        dest_dir=stage_dir,
                        index=index,
                        artifact_type="manifest_candidate",
                        git_tracked=False,
                        notes="backfill_skip",
                    )
                continue
            if dry_run:
                copied += 1
                continue
            before = len(index.get("entries") or [])
            copied_entry = register_or_copy(
                stage_id=stage_id,
                source=path,
                dest_dir=stage_dir,
                index=index,
                artifact_type="validator_or_receipt_summary",
                git_tracked=True,
                notes="workspace_backfill",
            )
            if copied_entry and copied_entry.get("status") == "present":
                copied += 1
                total_bytes += int(copied_entry.get("size_bytes") or 0)
            elif copied_entry is None:
                skipped += 1  # deduped
            else:
                skipped += 1
            _ = before

    if not dry_run:
        save_index(index_path, index)
    return {
        "copied": copied,
        "skipped": skipped,
        "not_available_cleaned": not_available,
        "total_bytes": total_bytes,
        "index_path": str(index_path),
    }


def mark_missing_stages(stage_ids: Iterable[str], *, project_root: Path | None = None) -> int:
    root = project_root or default_project_root()
    index_path = evidence_root(project_root=root) / "index.json"
    index = load_index(index_path)
    added = 0
    existing = {e.get("artifact_id") for e in index.get("entries") or []}
    for stage_id in stage_ids:
        aid = f"{stage_id}_historical_not_available"
        if aid in existing:
            continue
        index.setdefault("entries", []).append(
            {
                "artifact_id": aid,
                "stage_id": stage_id,
                "relative_path": None,
                "source_path": None,
                "sha256": None,
                "size_bytes": None,
                "artifact_type": "historical_evidence",
                "license_class": "n/a",
                "data_classification": "missing",
                "status": "not_available_cleaned",
                "git_tracked": False,
                "reason_not_git": "cleaned_before_retention_policy",
                "commit_sha": None,
                "notes": "Prior successful runtime cleaned; not reconstructed.",
            }
        )
        added += 1
    save_index(index_path, index)
    return added


__all__ = [
    "MAX_FILE_BYTES",
    "MAX_BACKFILL_TOTAL_BYTES",
    "WORKSPACE_STAGE_MAP",
    "evidence_root",
    "is_safe_evidence_file",
    "load_index",
    "save_index",
    "register_or_copy",
    "backfill_from_workspace",
    "mark_missing_stages",
]
