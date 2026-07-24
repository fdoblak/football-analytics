"""Render Stage 16-R2 real-video pilot summary PNG (not the final customer visual)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def render_real_video_pilot_png(
    *,
    report: dict[str, Any],
    output_local: Path,
    output_github: Path,
) -> dict[str, str]:
    target = report["target"]
    ev = report["evaluation"]
    phys = report["physical_metrics"]
    events = report["product_event_metrics"]

    fig = plt.figure(figsize=(14.0, 9.0), dpi=140)
    fig.patch.set_facecolor("#101820")
    fig.text(
        0.03,
        0.94,
        "REAL-VIDEO PILOT",
        color="#ffe082",
        fontsize=22,
        fontweight="bold",
        fontfamily="DejaVu Sans",
    )
    fig.text(
        0.03,
        0.90,
        "NOT FULL STAGE-16 ACCEPTANCE",
        color="#ff8a65",
        fontsize=14,
        fontweight="bold",
        fontfamily="DejaVu Sans",
    )
    fig.text(
        0.03,
        0.86,
        target.get("display_name", ""),
        color="#b3e5fc",
        fontsize=12,
        fontfamily="DejaVu Sans",
    )
    ds = report["dataset"]
    fig.text(
        0.03,
        0.825,
        f"{ds['id']} / {ds['sport_view']} / {ds['sequence_id']}  |  license {ds['license']}",
        color="#90a4ae",
        fontsize=10,
        fontfamily="DejaVu Sans",
    )

    ax1 = fig.add_axes((0.05, 0.42, 0.42, 0.35))
    ax1.set_facecolor("#1b2838")
    det = ev["detection"]
    ax1.bar(
        ["precision", "recall", "f1"],
        [det["precision"], det["recall"], det["f1"]],
        color=["#4fc3f7", "#81c784", "#ffb74d"],
    )
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Detection (greedy IoU)", color="#eceff1", fontsize=11)
    ax1.tick_params(colors="#cfd8dc")
    for spine in ax1.spines.values():
        spine.set_color("#455a64")

    ax2 = fig.add_axes((0.55, 0.42, 0.40, 0.35))
    ax2.set_facecolor("#1b2838")
    tt = ev["target_tracking"]
    ax2.bar(
        ["coverage", "mean IoU"],
        [tt["target_coverage_ratio"], tt["mean_iou"]],
        color=["#ce93d8", "#80cbc4"],
    )
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Target track agreement", color="#eceff1", fontsize=11)
    ax2.tick_params(colors="#cfd8dc")
    for spine in ax2.spines.values():
        spine.set_color("#455a64")

    evaluable = [
        f"detections={report['pilot']['human_detections']}",
        f"det_f1={det['f1']:.3f}",
        f"target_coverage={tt['target_coverage_ratio']:.3f}",
        f"distance_px={phys['measured_distance_px']:.1f}",
        f"mean_speed_px_s={phys['mean_speed_px_s']}",
    ]
    not_eval = [
        "HOTA/MOTA/IDF1",
        "pitch m distance/speed",
        "sprints",
        *[k for k, v in events.items() if v == "not_evaluable"],
    ]
    fig.text(0.05, 0.34, "Evaluable", color="#a5d6a7", fontsize=11, fontweight="bold")
    fig.text(0.05, 0.18, "\n".join(evaluable[:8]), color="#eceff1", fontsize=9, va="top")
    fig.text(0.55, 0.34, "not_evaluable", color="#ffab91", fontsize=11, fontweight="bold")
    fig.text(0.55, 0.18, "\n".join(not_eval[:12]), color="#eceff1", fontsize=9, va="top")

    fig.text(
        0.03,
        0.06,
        report.get("attribution", ""),
        color="#90a4ae",
        fontsize=8,
        fontfamily="DejaVu Sans",
    )
    fig.text(
        0.03,
        0.035,
        report.get("gate_candidate", ""),
        color="#ffe082",
        fontsize=8,
        fontfamily="DejaVu Sans",
    )

    output_local.parent.mkdir(parents=True, exist_ok=True)
    output_github.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_local, facecolor=fig.get_facecolor())
    plt.close(fig)
    data = output_local.read_bytes()
    output_github.write_bytes(data)
    h1, h2 = _sha256(output_local), _sha256(output_github)
    if h1 != h2:
        raise RuntimeError("pilot PNG hash mismatch")
    if output_local.stat().st_size >= 10 * 1024 * 1024:
        raise RuntimeError("pilot PNG exceeds 10 MiB")
    return {"local": str(output_local), "github": str(output_github), "sha256": h1}


__all__ = ["render_real_video_pilot_png"]
