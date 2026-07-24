"""Consolidated single-player report PNG renderer (Stage 14D).

Synthetic / workspace test renders only. Stage 16 reserved final paths must not
be written as customer finals from this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from football_analytics.orchestration.cleanup import assert_not_reserved_final_visual
from football_analytics.orchestration.types import OrchestrationError


def render_single_player_summary_png(
    report: Mapping[str, Any],
    output_path: Path,
    *,
    dpi: int = 120,
    allow_reserved_final: bool = False,
) -> Path:
    """Render a consolidated summary figure from report JSON.

    ``allow_reserved_final`` is intentionally default False — Stage 16 only.
    """
    output_path = Path(output_path)
    if not allow_reserved_final:
        assert_not_reserved_final_visual(output_path)
        # Extra guard on path string
        s = str(output_path)
        if "rendered_outputs/final/single_player_analysis_summary.png" in s:
            raise OrchestrationError("Stage 16 reserved final visual path forbidden")
        if s.endswith("artifacts/final/single_player_analysis_summary.png"):
            raise OrchestrationError("Stage 16 reserved final visual path forbidden")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise OrchestrationError("matplotlib required for report render") from exc

    metrics = list(report.get("metrics") or [])
    by_id = {str(m["metric_id"]): m for m in metrics}
    target = report.get("target_player") or {}
    coverage = report.get("coverage") or {}
    warnings = list(report.get("warnings") or [])
    not_eval = list(report.get("not_evaluable_metric_ids") or [])

    fig = plt.figure(figsize=(14, 10), facecolor="#0f1714")
    fig.suptitle(
        f"Single-Player Analysis — {target.get('display_name', 'target')}",
        color="#e8f0ea",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Layout: summary | heatmap placeholder
    ax_sum = fig.add_axes((0.05, 0.72, 0.55, 0.2))
    ax_hm = fig.add_axes((0.65, 0.72, 0.3, 0.2))
    ax_phys = fig.add_axes((0.05, 0.42, 0.4, 0.25))
    ax_pass = fig.add_axes((0.5, 0.42, 0.45, 0.25))
    ax_duel = fig.add_axes((0.05, 0.08, 0.55, 0.28))
    ax_warn = fig.add_axes((0.65, 0.08, 0.3, 0.28))

    for ax in (ax_sum, ax_hm, ax_phys, ax_pass, ax_duel, ax_warn):
        ax.set_facecolor("#15201b")
        ax.tick_params(colors="#c5d5cb")
        for spine in ax.spines.values():
            spine.set_color("#2a3d34")

    # Coverage summary
    ax_sum.set_title("Coverage / identity", color="#c5d5cb", fontsize=11)
    ax_sum.axis("off")
    lines = [
        f"run_id: {report.get('run_id')}",
        f"match: {target.get('match_id')}",
        f"identity: {target.get('manual_identity_status')} "
        f"(conf={report.get('identity_confidence')})",
        f"track={coverage.get('track_coverage')}  "
        f"cal={coverage.get('calibration_coverage')}  "
        f"ball={coverage.get('ball_tracking_coverage')}",
        f"duration_s: {(report.get('match') or {}).get('analysis_duration_s')}",
    ]
    ax_sum.text(
        0.02, 0.95, "\n".join(lines), va="top", color="#e8f0ea", fontsize=9, family="monospace"
    )

    # Heatmap placeholder (zone bars)
    ax_hm.set_title("Pitch zones (synthetic)", color="#c5d5cb", fontsize=11)
    zones = ["def", "mid", "att"]
    vals = [0.25, 0.45, 0.3]
    act = by_id.get("activation", {}).get("value")
    if isinstance(act, dict):
        vals = [
            float(act.get("low", 0.2)),
            float(act.get("medium", 0.5)),
            float(act.get("high", 0.3)),
        ]
    ax_hm.bar(zones, vals, color=["#3d6b54", "#5a9e78", "#8fd4a8"])
    ax_hm.set_ylim(0, 1)

    def _num(mid: str) -> float:
        m = by_id.get(mid) or {}
        v = m.get("value")
        if v is None:
            return 0.0
        if isinstance(v, (int, float)):
            return float(v)
        return 0.0

    # Physical
    ax_phys.set_title("Distance / speed / sprint", color="#c5d5cb", fontsize=11)
    labels = ["dist_m/100", "avg_mps", "sprints", "max_mps"]
    pvals = [
        _num("distance_covered_m") / 100.0,
        _num("speed_avg_mps"),
        _num("sprint_count"),
        _num("sprint_max_speed_mps"),
    ]
    ax_phys.barh(labels, pvals, color="#6cb3ff")
    ax_phys.invert_yaxis()

    # Passes / dribbles
    ax_pass.set_title("Passes / dribbles", color="#c5d5cb", fontsize=11)
    plabels = ["pass_att", "pass_rate", "drib_ok", "drib_fail", "take_on"]
    pvals2 = [
        _num("pass_attempts"),
        _num("pass_completion_rate") * 10,
        _num("dribbles_successful"),
        _num("dribbles_failed"),
        _num("take_on_success_rate") * 10,
    ]
    ax_pass.bar(plabels, pvals2, color="#e0b35a")
    ax_pass.tick_params(axis="x", rotation=20)

    # Duels / defensive
    ax_duel.set_title("Duels / defensive / box", color="#c5d5cb", fontsize=11)
    tir = by_id.get("tackles_interceptions_recoveries", {}).get("value") or {}
    dlabels = ["duels_won", "aerial", "clear", "losses", "box", "tackles", "rec"]
    dvals = [
        _num("duels_won"),
        _num("aerial_duels"),
        _num("clearances"),
        _num("ball_losses"),
        _num("penalty_area_ball_touches"),
        float(tir.get("tackles", 0)) if isinstance(tir, dict) else 0.0,
        float(tir.get("recoveries", 0)) if isinstance(tir, dict) else 0.0,
    ]
    ax_duel.bar(dlabels, dvals, color="#c57b8a")
    ax_duel.tick_params(axis="x", rotation=25)

    # Warnings / not evaluable
    ax_warn.set_title("Confidence / not_evaluable", color="#c5d5cb", fontsize=11)
    ax_warn.axis("off")
    warn_text = "Warnings:\n" + "\n".join(f"- {w}" for w in warnings[:6])
    warn_text += "\n\nNot evaluable:\n" + "\n".join(f"- {m}" for m in not_eval[:8])
    ax_warn.text(0.02, 0.98, warn_text, va="top", color="#f0d9a0", fontsize=8, family="monospace")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return output_path


__all__ = ["render_single_player_summary_png"]
