"""SoccerTrack v2 annotation-derived reference analysis (not video prediction)."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from football_analytics.acceptance.namespaces import (
    AUTHORITATIVE_SOCCERTRACK_TARGET,
    DEPRECATED_INVALID_TARGET,
    NAMESPACE_SOCCERTRACK_REFERENCE,
)

BAS_SUPPORTED = {
    "Pass",
    "Drive",
    "Header",
    "High Pass",
    "Out",
    "Cross",
    "Throw In",
    "Shot",
    "Ball Player Block",
    "Player Successful Tackle",
    "Free Kick",
    "Goal",
}


def _metric(
    value: Any,
    *,
    status: str,
    source: str,
    definition: str,
    coverage: Any = None,
    limitations: str | None = None,
) -> dict[str, Any]:
    return {
        "value": value,
        "status": status,
        "source": source,
        "definition": definition,
        "coverage": coverage,
        "limitations": limitations,
        "provenance": "reference_annotation_derived",
    }


def refuse_deprecated_target(player_id: str, jersey_number: int) -> None:
    deprecated_pid = str(DEPRECATED_INVALID_TARGET["player_id"])
    deprecated_jersey = int(str(DEPRECATED_INVALID_TARGET["jersey_number"]))
    if str(player_id) == deprecated_pid or int(jersey_number) == deprecated_jersey:
        raise ValueError("deprecated_invalid_delayed_result target refused")


def analyze_soccertrack_v2_reference(
    *,
    trajectory_path: Path,
    bas_path: Path,
    target: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build reference metrics from existing GSR trajectory + BAS exports."""
    tgt = dict(target or AUTHORITATIVE_SOCCERTRACK_TARGET)
    refuse_deprecated_target(str(tgt["player_id"]), int(tgt["jersey_number"]))
    if str(tgt["player_id"]) != AUTHORITATIVE_SOCCERTRACK_TARGET["player_id"]:
        raise ValueError("only authoritative SoccerTrack target is allowed")

    traj = json.loads(Path(trajectory_path).read_text(encoding="utf-8"))
    bas = json.loads(Path(bas_path).read_text(encoding="utf-8"))
    points = traj.get("points") or []
    pid = str(tgt["player_id"])

    # physical from pitch coords
    distance = 0.0
    speeds: list[float] = []
    segment_distances: list[float] = []
    xs: list[float] = []
    ys: list[float] = []
    for a, b in zip(points, points[1:], strict=False):
        if str(a.get("player_id")) != pid or str(b.get("player_id")) != pid:
            continue
        if a.get("half") != b.get("half"):
            continue
        dx = float(b["x_m"]) - float(a["x_m"])
        dy = float(b["y_m"]) - float(a["y_m"])
        d = math.hypot(dx, dy)
        distance += d
        dt = max(1e-6, (float(b["t_ms"]) - float(a["t_ms"])) / 1000.0)
        speeds.append(d / dt)
        segment_distances.append(d)
        xs.append(float(a["x_m"]))
        ys.append(float(a["y_m"]))
    if points and str(points[-1].get("player_id")) == pid:
        xs.append(float(points[-1]["x_m"]))
        ys.append(float(points[-1]["y_m"]))

    sprint_count = sum(1 for s in speeds if s >= 7.0)
    sprint_distance = sum(d for d, s in zip(segment_distances, speeds, strict=True) if s >= 7.0)

    halves = {int(p.get("half", 0)) for p in points}
    coverage = {
        "n_points": len(points),
        "halves_present": sorted(halves),
        "stride_note": traj.get("stride_note"),
    }

    # BAS for target only
    labels: Counter[str] = Counter()
    for e in bas.get("events") or []:
        if str(e.get("player_id")) != pid:
            continue
        lab = e.get("label")
        if lab in BAS_SUPPORTED:
            labels[lab] += 1

    metrics = {
        "visibility_coverage": _metric(
            None,
            status="REFERENCE_ANNOTATION_DERIVED",
            source="GSR_strided_trajectory",
            definition="Point count present; full-frame visibility ratio not claimed",
            coverage=coverage,
            limitations="Strided export; not per-frame visibility ratio",
        ),
        "heatmap": _metric(
            {"n_points": len(xs), "x_mean": (sum(xs) / len(xs)) if xs else None},
            status="REFERENCE_ANNOTATION_DERIVED",
            source="GSR",
            definition="Reference pitch occupancy summary from annotated trajectory",
            coverage=coverage,
        ),
        "measured_distance_m": _metric(
            round(distance, 3),
            status="REFERENCE_ANNOTATION_DERIVED",
            source="GSR",
            definition="Polyline length of strided pitch coordinates",
            coverage=coverage,
            limitations="Stride undersamples path length",
        ),
        "mean_speed_m_s": _metric(
            round(sum(speeds) / len(speeds), 4) if speeds else None,
            status="REFERENCE_ANNOTATION_DERIVED",
            source="GSR",
            definition="Mean segment speed from annotated trajectory",
            coverage=coverage,
        ),
        "peak_speed_m_s": _metric(
            round(max(speeds), 4) if speeds else None,
            status="REFERENCE_ANNOTATION_DERIVED",
            source="GSR",
            definition="Peak segment speed from annotated trajectory",
            coverage=coverage,
        ),
        "sprint_count": _metric(
            sprint_count,
            status="REFERENCE_ANNOTATION_DERIVED",
            source="GSR",
            definition="Segments with speed >= 7 m/s",
            coverage=coverage,
            limitations="Threshold heuristic; not Opta sprint definition",
        ),
        "sprint_distance_m": _metric(
            round(sprint_distance, 3),
            status="REFERENCE_ANNOTATION_DERIVED",
            source="GSR",
            definition="Distance while segment speed >= 7 m/s",
            coverage=coverage,
        ),
        "activity_index": _metric(
            sum(labels.values()) / max(1, len(halves)),
            status="REFERENCE_ANNOTATION_DERIVED",
            source="BAS",
            definition="Target BAS events per half present",
            coverage={"target_bas_events": dict(labels)},
        ),
        "penalty_area_presence_points": _metric(
            sum(1 for x in xs if x >= 36.0),
            status="REFERENCE_ANNOTATION_DERIVED",
            source="GSR",
            definition="Trajectory points with x_m >= 36 (attacking third/box proxy)",
            coverage=coverage,
            limitations="Proxy threshold; not exact penalty-area polygon",
        ),
        "bas_pass_attempts": _metric(
            labels.get("Pass", 0),
            status="REFERENCE_ANNOTATION_DERIVED",
            source="BAS",
            definition="Pass action count for target",
        ),
        "bas_high_pass_attempts": _metric(
            labels.get("High Pass", 0),
            status="REFERENCE_ANNOTATION_DERIVED",
            source="BAS",
            definition="High Pass action count (long-pass candidate)",
        ),
        "bas_drive_actions": _metric(
            labels.get("Drive", 0),
            status="REFERENCE_ANNOTATION_DERIVED",
            source="BAS",
            definition="Drive actions; not equated to completed dribbles",
            limitations="Drive ≠ successful dribble/take-on",
        ),
        "bas_header_actions": _metric(
            labels.get("Header", 0),
            status="REFERENCE_ANNOTATION_DERIVED",
            source="BAS",
            definition="Header / aerial action candidate",
        ),
        "bas_successful_tackles": _metric(
            labels.get("Player Successful Tackle", 0),
            status="REFERENCE_ANNOTATION_DERIVED",
            source="BAS",
            definition="Player Successful Tackle count",
        ),
        "pass_accuracy": _metric(
            None,
            status="NOT_EVALUABLE",
            source="BAS",
            definition="Pass completion not in SoccerTrack BAS",
            limitations="Outcome labels unavailable",
        ),
        "duel_win_rate": _metric(
            None,
            status="NOT_EVALUABLE",
            source="BAS",
            definition="Duel win rate not supported",
        ),
        "failed_dribbles": _metric(
            None,
            status="NOT_EVALUABLE",
            source="BAS",
            definition="Failed dribble outcomes not labeled",
        ),
        "clearances": _metric(
            None,
            status="NOT_EVALUABLE",
            source="BAS",
            definition="Clearance class not in 12 BAS labels",
        ),
        "possession": _metric(
            None,
            status="NOT_EVALUABLE",
            source="BAS/GSR",
            definition="Possession intervals not provided",
        ),
        "box_touches": _metric(
            None,
            status="NOT_EVALUABLE",
            source="BAS",
            definition="Explicit box-touch events not labeled",
        ),
    }

    return {
        "schema": "soccertrack_v2_reference_analysis_v1",
        "namespace": NAMESPACE_SOCCERTRACK_REFERENCE,
        "evidence_level": "REFERENCE_ANNOTATION_DERIVED",
        "disclaimer": (
            "TECHNICAL PREVIEW — REFERENCE-ANNOTATION-DERIVED — "
            "VIDEO EVENT-INFERENCE ACCURACY NOT VALIDATED — NOT OFFICIAL OPTA DATA"
        ),
        "target": tgt,
        "inputs": {
            "trajectory_path": str(trajectory_path),
            "bas_path": str(bas_path),
            "trajectory_n_points": traj.get("n_points"),
            "bas_n_events": bas.get("n_events"),
        },
        "bas_target_label_counts": dict(labels),
        "metrics": metrics,
        "not_video_prediction": True,
        "not_model_accuracy": True,
    }


__all__ = [
    "BAS_SUPPORTED",
    "analyze_soccertrack_v2_reference",
    "refuse_deprecated_target",
]
