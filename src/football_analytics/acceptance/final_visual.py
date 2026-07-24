"""Render Stage 16 final single-player summary PNG (no match frames required)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

from football_analytics.acceptance.final_report import ATTRIBUTION_FOOTER


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def render_acceptance_summary_png(
    *,
    report: dict[str, Any],
    output_local: Path,
    output_github: Path,
) -> dict[str, str]:
    """Produce one professional Turkish-labeled summary PNG; copy to both reserved paths."""
    target = report["target"]
    metrics = report["metrics"]
    gsr = report.get("external_gt_comparison", {}).get("gsr", {})
    bas = report.get("external_gt_comparison", {}).get("bas", {})
    coverage = report.get("coverage", {})

    fig = plt.figure(figsize=(14.0, 9.0), dpi=160)
    fig.patch.set_facecolor("#0f1a14")
    # atmospheric background gradient via imshow
    ax_bg = fig.add_axes([0, 0, 1, 1])
    ax_bg.axis("off")
    gradient = [[i / 255.0] * 20 for i in range(40, 90)]
    ax_bg.imshow(gradient, aspect="auto", cmap="Greens", extent=[0, 1, 0, 1], alpha=0.35)

    # Title
    fig.text(
        0.03,
        0.94,
        "Tek Futbolcu Analiz Özeti",
        color="#e8f5e9",
        fontsize=22,
        fontweight="bold",
        fontfamily="DejaVu Sans",
    )
    fig.text(
        0.03,
        0.90,
        target.get("display_name", ""),
        color="#a5d6a7",
        fontsize=12,
        fontfamily="DejaVu Sans",
    )

    # Pitch / heatmap panel
    ax_pitch = fig.add_axes([0.05, 0.28, 0.42, 0.55])
    ax_pitch.set_facecolor("#1b5e20")
    ax_pitch.set_xlim(-52.5, 52.5)
    ax_pitch.set_ylim(-34, 34)
    ax_pitch.set_aspect("equal")
    ax_pitch.add_patch(Rectangle((-52.5, -34), 105, 68, fill=False, edgecolor="#c8e6c9", lw=1.5))
    ax_pitch.plot([0, 0], [-34, 34], color="#c8e6c9", lw=1)
    ax_pitch.add_patch(Ellipse((0, 0), 18.3, 18.3, fill=False, edgecolor="#c8e6c9", lw=1))
    ax_pitch.set_xticks([])
    ax_pitch.set_yticks([])
    ax_pitch.set_title("Saha / Isı Haritası (projeksiyon)", color="#e8f5e9", fontsize=11)

    heat_pts = report.get("heatmap_points") or []
    if heat_pts:
        xs = [p["x_m"] for p in heat_pts]
        ys = [p["y_m"] for p in heat_pts]
        ax_pitch.hexbin(xs, ys, gridsize=18, cmap="YlOrRd", mincnt=1, alpha=0.85)
    else:
        ax_pitch.text(
            0,
            0,
            "Isı haritası: yetersiz pipeline kapsaması\nveya video henüz işlenmedi",
            ha="center",
            va="center",
            color="#ffe082",
            fontsize=10,
        )

    # Metrics panel
    ax_m = fig.add_axes([0.52, 0.28, 0.44, 0.55])
    ax_m.set_facecolor("#102018")
    ax_m.axis("off")
    ax_m.set_title("Metrikler / Değerlendirilebilirlik", color="#e8f5e9", fontsize=11, loc="left")

    def _fmt(m: dict[str, Any] | None) -> str:
        if not isinstance(m, dict):
            return "—"
        val = m.get("value")
        ev = m.get("evaluability", "")
        if val is None:
            return f"not_evaluable ({ev})" if ev else "not_evaluable"
        if isinstance(val, float):
            return f"{val:.2f} [{ev}]"
        if isinstance(val, dict):
            f1 = val.get("f1")
            return f"F1={f1:.2f}" if isinstance(f1, float) else str(val)[:40]
        return f"{val} [{ev}]"

    lines = [
        f"Kapsama: {coverage.get('visibility_coverage', '—')}",
        f"Kimlik kaynağı: {target.get('identity_source', '—')}",
        f"Mesafe (m): {_fmt(metrics.get('measured_distance_m'))}",
        f"Ort. hız: {_fmt(metrics.get('robust_mean_speed_mps'))}",
        f"Zirve hız: {_fmt(metrics.get('peak_speed_mps'))}",
        f"Sprint: {_fmt(metrics.get('sprint_count'))}",
        f"Pas (BAS): {_fmt(metrics.get('pass_events'))}",
        f"Pas isabeti: {_fmt(metrics.get('pass_accuracy'))}",
        f"Drive: {_fmt(metrics.get('drive_events'))}",
        f"İkili mücadele kazanma: {_fmt(metrics.get('duel_win_rate'))}",
        f"GSR mean err (m): {gsr.get('mean_pitch_error_m', '—')}",
        (
            "BAS held-out F1: "
            + str(
                (bas.get("precision") and bas.get("recall") and "see report")
                or bas.get("note", "—")
            )
        ),
        f"Kamera: {report.get('camera_domain', {}).get('camera_domain', '—')}",
    ]
    ax_m.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        color="#e8f5e9",
        fontsize=10,
        fontfamily="DejaVu Sans",
        transform=ax_m.transAxes,
        linespacing=1.45,
    )

    findings = report.get("warnings_findings") or []
    fig.text(
        0.03,
        0.18,
        "Bulgular: " + (" | ".join(findings[:4]) if findings else "—"),
        color="#ffcc80",
        fontsize=9,
        wrap=True,
    )
    fig.text(
        0.03,
        0.08,
        ATTRIBUTION_FOOTER,
        color="#b0bec5",
        fontsize=9,
        fontfamily="DejaVu Sans",
    )

    output_local = Path(output_local)
    output_github = Path(output_github)
    output_local.parent.mkdir(parents=True, exist_ok=True)
    output_github.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_local, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    data = output_local.read_bytes()
    output_github.write_bytes(data)
    h1 = _sha256(output_local)
    h2 = _sha256(output_github)
    if h1 != h2:
        raise RuntimeError("PNG hash mismatch between local and github paths")
    if output_local.stat().st_size > 10 * 1024 * 1024:
        raise RuntimeError("Final PNG exceeds 10 MiB")
    return {
        "local": str(output_local),
        "github": str(output_github),
        "sha256": h1,
        "bytes": str(output_local.stat().st_size),
    }
