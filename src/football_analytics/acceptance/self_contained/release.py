"""Assemble Stage 16-R4 technical-preview final report + dual PNG."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from football_analytics.acceptance.namespaces import (
    AUTHORITATIVE_SOCCERTRACK_TARGET,
    GATE_TECHNICAL_PREVIEW,
    NAMESPACE_SELF_CONTAINED,
    NAMESPACE_SOCCERTRACK_REFERENCE,
    NAMESPACE_TEAMTRACK_REAL_VIDEO,
    TEAMTRACK_PILOT_TARGET,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def build_technical_preview_report(
    *,
    reference: dict[str, Any],
    self_contained_receipt: dict[str, Any],
    teamtrack_summary: dict[str, Any],
    deterministic_ts: str | None = None,
) -> dict[str, Any]:
    tgt = AUTHORITATIVE_SOCCERTRACK_TARGET
    return {
        "schema": "single_player_technical_preview_report_v1",
        "generated_at_utc": deterministic_ts
        or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gate": GATE_TECHNICAL_PREVIEW,
        "title": "TECHNICAL PREVIEW — REFERENCE-ANNOTATION-DERIVED",
        "disclaimers": [
            "REFERENCE-ANNOTATION-DERIVED",
            "VIDEO EVENT-INFERENCE ACCURACY NOT VALIDATED",
            "NOT OFFICIAL OPTA DATA",
            "SoccerTrack v2 panoramic video is an optional external validation source, "
            "not a release dependency.",
        ],
        "target": {
            "match_id": tgt["match_id"],
            "team_side": tgt["team_side"],
            "jersey_number": tgt["jersey_number"],
            "player_id": tgt["player_id"],
            "display_name": (
                f"SoccerTrack v2 Match {tgt['match_id']} / "
                f"Team {tgt['team_side']} / Jersey {tgt['jersey_number']}"
            ),
        },
        "evidence_tracks": {
            NAMESPACE_TEAMTRACK_REAL_VIDEO: {
                "level": "REAL_VIDEO_VALIDATED",
                "summary": teamtrack_summary,
                "proves": [
                    "real_video_ingest",
                    "gpu_inference",
                    "human_detection",
                    "target_tracking",
                ],
                "does_not_prove": ["pass_accuracy", "duels", "pitch_meters_events"],
            },
            NAMESPACE_SOCCERTRACK_REFERENCE: {
                "level": "REFERENCE_ANNOTATION_DERIVED",
                "summary": {
                    "bas_target_label_counts": reference.get("bas_target_label_counts"),
                    "metric_keys": sorted((reference.get("metrics") or {}).keys()),
                },
                "not_video_prediction": True,
            },
            NAMESPACE_SELF_CONTAINED: {
                "level": "SELF_CONTAINED_TESTED",
                "receipt": {
                    "config_fingerprint": self_contained_receipt.get("config_fingerprint"),
                    "metrics_fingerprint": self_contained_receipt.get("metrics_fingerprint"),
                    "status": self_contained_receipt.get("status"),
                },
            },
        },
        "metrics": reference.get("metrics"),
        "attribution": (
            "SoccerTrack v2 annotations — CC BY 4.0 (Atom Scott et al.). "
            "TeamTrack pilot — MIT. Project-generated technical preview; not official Opta data."
        ),
        "teamtrack_pilot_target_isolated": TEAMTRACK_PILOT_TARGET,
        "hf_required": False,
        "drive_required": False,
        "network_required": False,
    }


def render_technical_preview_png(
    *,
    report: dict[str, Any],
    output_local: Path,
    output_github: Path,
    heatmap_points: list[tuple[float, float]] | None = None,
) -> dict[str, str]:
    """Deterministic dual-path final customer PNG (only customer visual)."""
    metrics = report.get("metrics") or {}
    tgt = report["target"]
    tt = report["evidence_tracks"][NAMESPACE_TEAMTRACK_REAL_VIDEO]["summary"]
    bas_counts = (
        report["evidence_tracks"][NAMESPACE_SOCCERTRACK_REFERENCE]["summary"].get(
            "bas_target_label_counts"
        )
        or {}
    )

    # Fixed RNG-free layout; Agg backend + fixed DPI for pixel stability.
    fig = plt.figure(figsize=(14.0, 10.0), dpi=120)
    fig.patch.set_facecolor("#0e1512")
    fig.text(
        0.03,
        0.955,
        "TECHNICAL PREVIEW",
        color="#ffe082",
        fontsize=20,
        fontweight="bold",
        fontfamily="DejaVu Sans",
    )
    fig.text(
        0.03,
        0.915,
        "REFERENCE-ANNOTATION-DERIVED · VIDEO-EVENT NOT VALIDATED · NOT OFFICIAL OPTA DATA",
        color="#ffab91",
        fontsize=10,
        fontfamily="DejaVu Sans",
    )
    fig.text(
        0.03,
        0.875,
        (
            f"SoccerTrack v2  Match {tgt['match_id']}  ·  Team {tgt['team_side']}  ·  "
            f"Jersey {tgt['jersey_number']}  ·  Player {tgt['player_id']}"
        ),
        color="#c8e6c9",
        fontsize=12,
        fontfamily="DejaVu Sans",
    )

    # Heatmap / pitch occupancy (reference trajectory subsample if provided)
    ax_hm = fig.add_axes((0.06, 0.48, 0.38, 0.34))
    ax_hm.set_facecolor("#1b2a22")
    ax_hm.set_xlim(-52.5, 52.5)
    ax_hm.set_ylim(-34.0, 34.0)
    ax_hm.set_aspect("equal")
    ax_hm.tick_params(colors="#90a4ae", labelsize=7)
    ax_hm.set_title("GSR reference occupancy (not video prediction)", color="#e8f5e9", fontsize=9)
    if heatmap_points:
        xs = [p[0] for p in heatmap_points]
        ys = [p[1] for p in heatmap_points]
        ax_hm.scatter(xs, ys, s=2.0, c="#81c784", alpha=0.35, linewidths=0)
    else:
        hm = (metrics.get("heatmap") or {}).get("value") or {}
        ax_hm.text(
            0.5,
            0.5,
            f"n_points={hm.get('n_points', 'n/a')}",
            ha="center",
            va="center",
            color="#a5d6a7",
            transform=ax_hm.transAxes,
            fontsize=11,
        )

    # Metric bars
    labels: list[str] = []
    values: list[float] = []
    for key in (
        "measured_distance_m",
        "mean_speed_m_s",
        "peak_speed_m_s",
        "sprint_count",
        "bas_pass_attempts",
        "bas_successful_tackles",
        "activity_index",
        "penalty_area_presence_points",
    ):
        m = metrics.get(key) or {}
        if m.get("status") == "NOT_EVALUABLE" or m.get("value") is None:
            continue
        try:
            values.append(float(m["value"]))
        except (TypeError, ValueError):
            continue
        labels.append(key.replace("_", "\n"))
    ax = fig.add_axes((0.50, 0.48, 0.46, 0.34))
    ax.set_facecolor("#1b2a22")
    if values:
        ax.bar(range(len(values)), values, color="#66bb6a")
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, fontsize=6, color="#e8f5e9")
    ax.tick_params(colors="#cfd8dc")
    ax.set_title("Reference-derived metrics", color="#e8f5e9", fontsize=9)

    # Evidence + BAS + real-video box
    ax2 = fig.add_axes((0.06, 0.18, 0.42, 0.26))
    ax2.set_facecolor("#1b2a22")
    ax2.axis("off")
    det = (tt.get("detection") or {}) if isinstance(tt, dict) else {}
    track = (tt.get("target_tracking") or {}) if isinstance(tt, dict) else {}
    f1 = det.get("f1", det.get("f1_score", "n/a"))
    box = (
        "EVIDENCE LEVEL LEGEND\n"
        "• REAL_VIDEO_VALIDATED — TeamTrack pilot (separate person)\n"
        f"  detections F1≈{f1}  coverage={track.get('target_coverage_ratio', 'n/a')}\n"
        f"  mean IoU≈{track.get('mean_iou', track.get('mean_target_iou', 'n/a'))}\n"
        "• REFERENCE_ANNOTATION_DERIVED — SoccerTrack v2 GSR/BAS\n"
        "• SELF_CONTAINED_TESTED — offline deterministic contracts\n"
        "• NOT_EVALUABLE — unsupported without richer labels\n"
    )
    ax2.text(0.04, 0.96, box, va="top", color="#e8f5e9", fontsize=8, family="DejaVu Sans")

    ax3 = fig.add_axes((0.52, 0.18, 0.44, 0.26))
    ax3.set_facecolor("#1b2a22")
    ax3.axis("off")
    bas_lines = [
        "BAS reference events (target)",
        *[f"  {k}: {v}" for k, v in sorted(bas_counts.items())],
    ]
    ax3.text(
        0.04,
        0.96,
        "\n".join(bas_lines[:14]),
        va="top",
        color="#e8f5e9",
        fontsize=8,
        family="DejaVu Sans",
    )

    ne = [k for k, m in metrics.items() if (m or {}).get("status") == "NOT_EVALUABLE"]
    fig.text(
        0.03,
        0.12,
        "not_evaluable: " + (", ".join(ne) if ne else "(none)"),
        color="#ffcc80",
        fontsize=8,
    )
    fig.text(
        0.03,
        0.07,
        report.get("attribution", ""),
        color="#90a4ae",
        fontsize=7,
        fontfamily="DejaVu Sans",
    )
    fig.text(
        0.03,
        0.03,
        report.get("gate", ""),
        color="#ffe082",
        fontsize=7,
        fontfamily="DejaVu Sans",
    )

    output_local.parent.mkdir(parents=True, exist_ok=True)
    output_github.parent.mkdir(parents=True, exist_ok=True)
    # Fixed canvas (no bbox_inches='tight') for deterministic pixels across runs.
    fig.savefig(output_local, facecolor=fig.get_facecolor(), dpi=120)
    plt.close(fig)
    output_github.write_bytes(output_local.read_bytes())
    h1, h2 = _sha256(output_local), _sha256(output_github)
    if h1 != h2:
        raise RuntimeError("final PNG hash mismatch")
    return {"local": str(output_local), "github": str(output_github), "sha256": h1}


def write_report(path: Path, report: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "build_technical_preview_report",
    "render_technical_preview_png",
    "write_report",
]
