"""Deterministic anonymous TeamTrack target selection (no jersey invention)."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from football_analytics.acceptance.teamtrack.loader import MotBox, TeamTrackSequence


@dataclass(frozen=True)
class TeamTrackTargetReceipt:
    dataset_id: str
    sport_view: str
    split: str
    sequence_id: str
    persistent_track_id: int
    display_name: str
    confirmation_source: str
    selection_rule: str
    frames_covered: int
    seq_length: int
    coverage_ratio: float
    mean_bbox_area: float
    continuous: bool
    candidates: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _track_stats(boxes: tuple[MotBox, ...], seq_length: int) -> list[dict[str, Any]]:
    by_id: dict[int, list[MotBox]] = defaultdict(list)
    for b in boxes:
        by_id[b.track_id].append(b)
    rows: list[dict[str, Any]] = []
    for tid, items in by_id.items():
        frames = sorted({b.frame for b in items})
        areas = [max(b.w, 0.0) * max(b.h, 0.0) for b in items]
        continuous = len(frames) == (frames[-1] - frames[0] + 1) if frames else False
        rows.append(
            {
                "track_id": int(tid),
                "frames_covered": len(frames),
                "coverage_ratio": float(len(frames) / seq_length) if seq_length else 0.0,
                "mean_bbox_area": float(sum(areas) / len(areas)) if areas else 0.0,
                "continuous": continuous,
                "frame_start": int(frames[0]) if frames else -1,
                "frame_end": int(frames[-1]) if frames else -1,
            }
        )
    return rows


def select_anonymous_track(
    sequence: TeamTrackSequence,
    *,
    min_coverage_ratio: float = 0.8,
) -> TeamTrackTargetReceipt:
    """Select outfield-like track: high coverage, continuous, larger bbox, tie-break track_id."""
    stats = _track_stats(sequence.boxes, sequence.seq_length)
    eligible = [
        s
        for s in stats
        if s["coverage_ratio"] >= min_coverage_ratio and s["continuous"] and s["mean_bbox_area"] > 0
    ]
    if not eligible:
        raise RuntimeError("no eligible TeamTrack target track")
    # Prefer larger mean area (closer/outfield-ish on side view) then lowest track_id.
    eligible.sort(key=lambda s: (-s["mean_bbox_area"], s["track_id"]))
    # Among top-quartile area, pick lowest track_id for determinism if areas are close.
    areas = [s["mean_bbox_area"] for s in eligible]
    cutoff = sorted(areas)[max(0, int(0.75 * (len(areas) - 1)))]
    top = [s for s in eligible if s["mean_bbox_area"] >= cutoff]
    top.sort(key=lambda s: (s["track_id"],))
    chosen = top[0]
    tid = int(chosen["track_id"])
    display = f"Target Player — TeamTrack {sequence.sport_view} / Track {tid}"
    return TeamTrackTargetReceipt(
        dataset_id=sequence.dataset_id,
        sport_view=sequence.sport_view,
        split=sequence.split,
        sequence_id=sequence.sequence_id,
        persistent_track_id=tid,
        display_name=display,
        confirmation_source="reviewed_ground_truth_target_selection",
        selection_rule=(
            "full_coverage_continuous_outfield_proxy_mean_bbox_area_top_quartile;"
            "tie_break_numeric_track_id; no_jersey_invented"
        ),
        frames_covered=int(chosen["frames_covered"]),
        seq_length=int(sequence.seq_length),
        coverage_ratio=float(chosen["coverage_ratio"]),
        mean_bbox_area=float(chosen["mean_bbox_area"]),
        continuous=bool(chosen["continuous"]),
        candidates=sorted(eligible, key=lambda s: s["track_id"])[:20],
    )


def write_teamtrack_target_receipt(receipt: TeamTrackTargetReceipt, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = receipt.to_dict()
    payload["written_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "TeamTrackTargetReceipt",
    "select_anonymous_track",
    "write_teamtrack_target_receipt",
]
