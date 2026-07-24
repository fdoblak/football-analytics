"""Stage 14 synthetic fixtures."""

from __future__ import annotations

from typing import Any

from football_analytics.orchestration.config import (
    load_pipeline_config,
    pipeline_config_fingerprint,
)
from football_analytics.orchestration.planner import build_pipeline_request


def synthetic_pipeline_request(
    *,
    output_directory: str,
    request_id: str = "req_stage14_synth",
    run_id: str = "run_stage14_synth01",
    video_id: str = "vid_stage14_synth",
    target_player_id: str = "target_player_a",
    force_restart: bool = True,
    cancel_requested: bool = False,
    resume_from_stage: str | None = None,
) -> dict[str, Any]:
    cfg = load_pipeline_config()
    fp = pipeline_config_fingerprint(cfg)
    return build_pipeline_request(
        request_id=request_id,
        run_id=run_id,
        video_id=video_id,
        target_player_id=target_player_id,
        output_directory=output_directory,
        config_fingerprint=fp,
        mode="synthetic_fixture",
        match_id="match_stage14_synth",
        force_restart=force_restart,
        cancel_requested=cancel_requested,
        resume_from_stage=resume_from_stage,
        notes="stage_14_synthetic_e2e",
    )


__all__ = ["synthetic_pipeline_request"]
