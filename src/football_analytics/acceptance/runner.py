"""Stage 16 panoramic acceptance runner (CPU-safe, resume-friendly)."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from football_analytics.acceptance.contracts import (
    EXTERNAL_REFERENCE_CONFIRMATION,
    NAMESPACE_EVALUATION,
    NAMESPACE_PREDICTIONS,
    NAMESPACE_REFERENCE_GT,
)
from football_analytics.acceptance.download_manifest import sha256_file
from football_analytics.acceptance.evaluation.bas_metrics import match_events
from football_analytics.acceptance.evaluation.gsr_metrics import compare_trajectories
from football_analytics.acceptance.evaluation.metric_taxonomy import taxonomy_table
from football_analytics.acceptance.final_report import build_acceptance_report, write_report
from football_analytics.acceptance.final_visual import render_acceptance_summary_png
from football_analytics.acceptance.leakage import validate_run_dir
from football_analytics.acceptance.panorama_domain import classify_camera_domain


def _ffprobe(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=width,height,r_frame_rate,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    return json.loads(subprocess.check_output(cmd, text=True))


def normalize_panorama(
    *,
    source: Path,
    output: Path,
    max_width: int = 1280,
    fps: int = 5,
) -> dict[str, Any]:
    """Downscale panoramic video for CPU acceptance (deterministic flags)."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 1_000_000:
        return {"path": str(output), "cached": True, "sha256": sha256_file(output)}
    scale = f"scale='min({max_width},iw)':-2"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-vf",
        f"{scale},fps={fps}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "28",
        "-an",
        str(output),
    ]
    t0 = time.time()
    subprocess.run(cmd, check=True, capture_output=True)
    return {
        "path": str(output),
        "cached": False,
        "sha256": sha256_file(output),
        "elapsed_s": time.time() - t0,
        "max_width": max_width,
        "fps": fps,
    }


def derive_trajectory_metrics(points: list[dict[str, Any]]) -> dict[str, Any]:
    """Pipeline-style physical metrics from predicted trajectory points only."""
    if len(points) < 2:
        return {
            "measured_distance_m": None,
            "robust_mean_speed_mps": None,
            "peak_speed_mps": None,
            "sprint_count": None,
            "analyzed_duration_s": None,
        }
    ordered = sorted(points, key=lambda p: (int(p["half"]), int(p["t_ms"])))
    dist = 0.0
    speeds: list[float] = []
    for a, b in zip(ordered, ordered[1:], strict=False):
        if int(a["half"]) != int(b["half"]):
            continue
        dt = (int(b["t_ms"]) - int(a["t_ms"])) / 1000.0
        if dt <= 0:
            continue
        dx = float(b["x_m"]) - float(a["x_m"])
        dy = float(b["y_m"]) - float(a["y_m"])
        step = (dx * dx + dy * dy) ** 0.5
        dist += step
        speeds.append(step / dt)
    speeds_sorted = sorted(speeds)
    mean = None
    if speeds_sorted:
        mid = len(speeds_sorted) // 2
        mean = speeds_sorted[mid]
    peak = max(speeds_sorted) if speeds_sorted else None
    sprints = sum(1 for s in speeds_sorted if s >= 7.0)
    dur = (int(ordered[-1]["t_ms"]) - int(ordered[0]["t_ms"])) / 1000.0
    return {
        "measured_distance_m": dist,
        "robust_mean_speed_mps": mean,
        "peak_speed_mps": peak,
        "sprint_count": sprints,
        "analyzed_duration_s": dur,
    }


def build_predictions_from_sparse_track(
    *,
    run_dir: Path,
    predicted_points: list[dict[str, Any]],
    predicted_events: list[dict[str, Any]],
) -> None:
    pred = Path(run_dir) / NAMESPACE_PREDICTIONS
    pred.mkdir(parents=True, exist_ok=True)
    (pred / "target_trajectory_predicted.json").write_text(
        json.dumps(
            {
                "provenance": "pipeline_prediction",
                "n_points": len(predicted_points),
                "points": predicted_points,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (pred / "bas_events_predicted.json").write_text(
        json.dumps(
            {
                "provenance": "pipeline_prediction",
                "n_events": len(predicted_events),
                "events": predicted_events,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def finalize_acceptance_artifacts(
    *,
    run_dir: Path,
    evidence_dir: Path,
    report_path: Path,
    local_png: Path,
    github_png: Path,
    receipt: dict[str, Any],
    predicted_points: list[dict[str, Any]],
    predicted_events: list[dict[str, Any]],
    findings: list[str],
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    validate_run_dir(run_dir)
    ref_traj = json.loads(
        (run_dir / NAMESPACE_REFERENCE_GT / "target_trajectory_reference.json").read_text()
    )
    bas_path = run_dir / NAMESPACE_REFERENCE_GT / "bas_reference_events.json"
    ref_bas = json.loads(bas_path.read_text())
    pid = str(receipt["selected_player_id"])
    held_ref = [e for e in ref_bas["events"] if e.get("player_id") == pid and e.get("half") == 2]
    held_pred = [e for e in predicted_events if int(e.get("half") or 0) == 2]
    gsr_eval = compare_trajectories(
        predicted=predicted_points, reference=ref_traj.get("points") or []
    )
    bas_eval = match_events(predicted=held_pred, reference=held_ref, tolerance_ms=1000)
    bas_eval["partition"] = "held_out_half2"
    pipeline_metrics = derive_trajectory_metrics(predicted_points)
    pipeline_metrics["identity_confidence"] = EXTERNAL_REFERENCE_CONFIRMATION
    pipeline_metrics["heatmap_summary"] = {
        "n_points": len(predicted_points),
        "status": "pipeline_derived" if predicted_points else "insufficient_coverage",
    }
    coverage = {
        "visibility_coverage": gsr_eval.get("matched_observation_coverage"),
        "n_predicted_points": len(predicted_points),
        "n_reference_points": ref_traj.get("n_points"),
    }
    report = build_acceptance_report(
        match_id=str(receipt["match_id"]),
        target_display=str(receipt["display_name"]),
        target_player_id=pid,
        team_side=str(receipt["team_side"]),
        jersey_number=int(receipt["jersey_number"]),
        camera_domain=classify_camera_domain(),
        evaluability_table=taxonomy_table(),
        gsr_eval=gsr_eval,
        bas_eval=bas_eval,
        pipeline_metrics=pipeline_metrics,
        coverage=coverage,
        findings=findings,
        identity_source=EXTERNAL_REFERENCE_CONFIRMATION,
    )
    if predicted_points:
        heat_step = max(1, len(predicted_points) // 400)
        report["heatmap_points"] = predicted_points[::heat_step]
    else:
        report["heatmap_points"] = []
    write_report(report, report_path)
    png = render_acceptance_summary_png(
        report=report, output_local=local_png, output_github=github_png
    )
    (Path(evidence_dir) / "final_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (Path(evidence_dir) / "final_png_hashes.json").write_text(
        json.dumps(png, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / NAMESPACE_EVALUATION / "bas_held_out_eval.json").write_text(
        json.dumps(bas_eval, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / NAMESPACE_EVALUATION / "gsr_eval.json").write_text(
        json.dumps(gsr_eval, indent=2) + "\n", encoding="utf-8"
    )
    return {"report": str(report_path), "png": png}
