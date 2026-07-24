"""GSR / identity / position evaluation helpers (no fake detection mAP)."""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional


def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def compare_trajectories(
    *,
    predicted: Iterable[dict[str, Any]],
    reference: Iterable[dict[str, Any]],
    max_match_m: float = 5.0,
) -> dict[str, Any]:
    """Match predicted points to reference by (half, nearest time) and report errors."""
    refs = list(reference)
    preds = list(predicted)
    if not refs:
        return {
            "status": "reference_gt_not_available",
            "matched": 0,
            "coverage": 0.0,
        }
    # index refs by half
    by_half: dict[int, list[dict[str, Any]]] = {}
    for r in refs:
        by_half.setdefault(int(r.get("half") or 0), []).append(r)
    for h in by_half:
        by_half[h].sort(key=lambda x: int(x.get("t_ms") or x.get("frame_index") or 0))

    errors: list[float] = []
    matched = 0
    for p in preds:
        h = int(p.get("half") or 0)
        cand = by_half.get(h) or []
        if not cand:
            continue
        pt = int(p.get("t_ms") or 0)
        # binary-ish nearest by linear scan (compact trajectories)
        best = min(cand, key=lambda r: abs(int(r.get("t_ms") or 0) - pt))
        if abs(int(best.get("t_ms") or 0) - pt) > 2000:
            continue
        d = _dist(float(p["x_m"]), float(p["y_m"]), float(best["x_m"]), float(best["y_m"]))
        if d <= max_match_m:
            matched += 1
            errors.append(d)

    coverage = matched / len(refs) if refs else 0.0
    mean_err = sum(errors) / len(errors) if errors else None
    return {
        "status": "evaluated_against_external_gt",
        "n_reference": len(refs),
        "n_predicted": len(preds),
        "matched": matched,
        "matched_observation_coverage": coverage,
        "mean_pitch_error_m": mean_err,
        "detection_map": "not_evaluable",
        "detection_map_reason": "MOT bbox pack not present for selected Drive mirror match",
    }


def jersey_team_agreement(
    *,
    predicted_jersey: Optional[int],
    predicted_team: Optional[str],
    reference_jersey: Optional[int],
    reference_team: Optional[str],
) -> dict[str, Any]:
    return {
        "jersey_agree": (
            predicted_jersey is not None
            and reference_jersey is not None
            and int(predicted_jersey) == int(reference_jersey)
        ),
        "team_agree": (
            predicted_team is not None
            and reference_team is not None
            and str(predicted_team) == str(reference_team)
        ),
        "status": "evaluated_against_external_gt",
    }
