"""Stage-owned temp cleanup (never mutate user video)."""

from __future__ import annotations

import shutil
from pathlib import Path

from football_analytics.core.records import write_json_record
from football_analytics.orchestration.contracts import RESERVED_FINAL_VISUAL_PATHS


def cleanup_stage_owned_temp(
    run_root: Path,
    *,
    patterns: tuple[str, ...] = ("_tmp", "_chain_inputs", "render_tmp"),
) -> list[str]:
    """Remove only automation-owned temp dirs under the run root."""
    removed: list[str] = []
    if not run_root.is_dir():
        return removed
    for child in list(run_root.iterdir()):
        if not child.is_dir() or child.is_symlink():
            continue
        name = child.name
        if (
            name in patterns
            or name.startswith("_tmp")
            or name.startswith("render_tmp")
            or name == "_chain_inputs"
        ):
            shutil.rmtree(child, ignore_errors=True)
            removed.append(str(child))
    return removed


def assert_not_reserved_final_visual(path: Path) -> None:
    resolved = str(path.resolve()) if path.exists() else str(path)
    for reserved in RESERVED_FINAL_VISUAL_PATHS:
        if resolved.endswith("single_player_analysis_summary.png") and (
            "rendered_outputs/final" in resolved or "/artifacts/final/" in resolved
        ):
            raise RuntimeError(f"refusing to write Stage 16 reserved final visual: {reserved}")


def write_cleanup_receipt(path: Path, *, removed: list[str], run_id: str) -> None:
    write_json_record(
        path,
        {
            "schema_version": 1,
            "run_id": run_id,
            "removed": removed,
            "user_video_mutated": False,
            "reserved_final_visuals_untouched": True,
        },
        overwrite=True,
    )


__all__ = [
    "cleanup_stage_owned_temp",
    "assert_not_reserved_final_visual",
    "write_cleanup_receipt",
]
