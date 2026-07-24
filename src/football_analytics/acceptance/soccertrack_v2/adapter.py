"""Convert SoccerTrack v2 source into namespaced reference_ground_truth artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_analytics.acceptance.contracts import (
    EXTERNAL_CC_BY_REFERENCE_GT,
    NAMESPACE_PREDICTIONS,
    NAMESPACE_REFERENCE_GT,
)
from football_analytics.acceptance.leakage import LeakageError
from football_analytics.acceptance.soccertrack_v2.loader import (
    bas_path,
    gsr_path,
    iter_gsr_player_observations,
    load_bas_events,
    read_gsr_info,
)
from football_analytics.acceptance.soccertrack_v2.target_selection import (
    TargetSelectionReceipt,
)


def _refuse_predictions_path(path: Path) -> None:
    parts = {p.lower() for p in path.parts}
    if NAMESPACE_PREDICTIONS in parts:
        raise LeakageError(f"refusing to write reference GT under predictions/: {path}")


def write_bas_reference(root: Path, match_id: str, out_dir: Path) -> Path:
    _refuse_predictions_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    events = load_bas_events(bas_path(root, match_id))
    payload = {
        "annotation_provenance": EXTERNAL_CC_BY_REFERENCE_GT,
        "match_id": str(match_id),
        "n_events": len(events),
        "events": [
            {
                "half": e.half,
                "clock": e.clock,
                "t_ms": e.t_ms,
                "label": e.label,
                "team": e.team,
                "player_id": e.player_id,
                "visibility": e.visibility,
                "source": EXTERNAL_CC_BY_REFERENCE_GT,
            }
            for e in events
        ],
    }
    path = out_dir / "bas_reference_events.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def write_target_trajectory_reference(
    *,
    root: Path,
    match_id: str,
    player_id: str,
    out_dir: Path,
    stride: int = 5,
) -> Path:
    """Export a compact target trajectory (strided) for evaluation — not predictions."""
    _refuse_predictions_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    points: list[dict[str, Any]] = []
    for half in (1, 2):
        path = gsr_path(root, match_id, half)
        if not path.is_file():
            continue
        for obs in iter_gsr_player_observations(path, half=half):
            if obs.player_id != str(player_id):
                continue
            if obs.frame_index % stride != 0:
                continue
            points.append(
                {
                    "half": half,
                    "frame_index": obs.frame_index,
                    "t_ms": int(round(obs.frame_index * 1000 / 25)),
                    "x_m": obs.x_m,
                    "y_m": obs.y_m,
                    "jersey_number": obs.jersey_number,
                    "team_side": obs.team_side,
                    "role": obs.role,
                    "track_id": obs.track_id,
                }
            )
    payload = {
        "annotation_provenance": EXTERNAL_CC_BY_REFERENCE_GT,
        "match_id": str(match_id),
        "player_id": str(player_id),
        "stride": stride,
        "n_points": len(points),
        "points": points,
    }
    out = out_dir / "target_trajectory_reference.json"
    out.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return out


def write_gsr_half_manifest(root: Path, match_id: str, out_dir: Path) -> Path:
    _refuse_predictions_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    halves = []
    for half in (1, 2):
        path = gsr_path(root, match_id, half)
        info = read_gsr_info(path) if path.is_file() else {}
        halves.append(
            {
                "half": half,
                "path": str(path),
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "info": {
                    "seq_length": info.get("seq_length"),
                    "frame_rate": info.get("frame_rate"),
                    "name": info.get("name"),
                },
                "annotation_provenance": EXTERNAL_CC_BY_REFERENCE_GT,
            }
        )
    out = out_dir / "gsr_half_manifest.json"
    out.write_text(
        json.dumps({"match_id": match_id, "halves": halves}, indent=2) + "\n",
        encoding="utf-8",
    )
    return out


def materialize_reference_bundle(
    *,
    root: Path,
    match_id: str,
    run_dir: Path,
    receipt: TargetSelectionReceipt,
    stride: int = 5,
) -> dict[str, str]:
    ref = Path(run_dir) / NAMESPACE_REFERENCE_GT
    if (Path(run_dir) / NAMESPACE_PREDICTIONS) == ref:
        raise LeakageError("predictions and reference_ground_truth must differ")
    paths = {
        "bas": str(write_bas_reference(root, match_id, ref)),
        "trajectory": str(
            write_target_trajectory_reference(
                root=root,
                match_id=match_id,
                player_id=receipt.selected_player_id,
                out_dir=ref,
                stride=stride,
            )
        ),
        "gsr_manifest": str(write_gsr_half_manifest(root, match_id, ref)),
        "target_receipt": str(ref / "target_selection_receipt.json"),
    }
    (ref / "target_selection_receipt.json").write_text(
        json.dumps(
            {
                **receipt.__dict__,
                "annotation_provenance": EXTERNAL_CC_BY_REFERENCE_GT,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths
