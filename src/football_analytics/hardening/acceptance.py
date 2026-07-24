"""15E synthetic acceptance scenario helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from football_analytics.orchestration.cleanup import cleanup_stage_owned_temp
from football_analytics.orchestration.fixtures import synthetic_pipeline_request
from football_analytics.orchestration.report.builder import build_single_player_report
from football_analytics.orchestration.runner import run_pipeline
from football_analytics.visualization.report_renderer import render_single_player_summary_png


def _synth_report(run_id: str) -> dict[str, Any]:
    return build_single_player_report(
        run_id=run_id,
        git_commit="a" * 40,
        target_player_id="target_player_a",
        display_name="Target A",
        match_id="match_stage15_synth",
        video_id="vid_stage15_synth",
    )


def build_acceptance_reports() -> dict[str, dict[str, Any]]:
    """Positive / negative / ambiguous / low-coverage / not_evaluable report variants."""
    positive = _synth_report("run_stage15_positive")
    negative = _synth_report("run_stage15_negative")
    ambiguous = dict(_synth_report("run_stage15_ambiguous"))
    ambiguous["acceptance_label"] = "ambiguous"
    low = dict(_synth_report("run_stage15_low_cov"))
    low["acceptance_label"] = "low_coverage"
    low["coverage_summary"] = {"overall": 0.05, "label": "low_coverage"}
    ne = dict(_synth_report("run_stage15_not_eval"))
    ne["acceptance_label"] = "not_evaluable"
    return {
        "positive": positive,
        "negative": negative,
        "ambiguous": ambiguous,
        "low_coverage": low,
        "not_evaluable": ne,
    }


def run_synthetic_orchestration_e2e(session: Path, *, light: bool = True) -> dict[str, Any]:
    """Run Stage 14 orchestrator in light mode + renderer temp cleanup."""
    out = session / "e2e_run"
    req = synthetic_pipeline_request(output_directory=str(out), force_restart=True)
    run_res = run_pipeline(req, light_stubs_only=light)
    reports = build_acceptance_reports()
    render_path = session / "synthetic_summary_stage15.png"
    render_single_player_summary_png(reports["positive"], render_path)
    rendered = render_path.is_file()
    render_path.unlink(missing_ok=True)
    removed = cleanup_stage_owned_temp(out)
    return {
        "overall_status": run_res.overall_status,
        "user_video_mutated": run_res.summary.get("user_video_mutated"),
        "acceptance_labels": sorted(reports.keys()),
        "renderer_temp_created": rendered,
        "renderer_temp_cleaned": not render_path.exists(),
        "cleanup_paths": list(removed),
        "not_evaluable_present": bool(reports["not_evaluable"].get("not_evaluable_metric_ids")),
    }
