"""Per-stage synthetic handlers for Stage 14A E2E orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from football_analytics.core.hashing import hash_canonical_json
from football_analytics.core.records import write_json_record


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def write_stage_receipt(
    stage_dir: Path,
    *,
    stage_name: str,
    run_id: str,
    cache_key: str,
    status: str = "succeeded",
    mode: str = "synthetic_stub",
    extras: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": stage_name,
        "run_id": run_id,
        "cache_key": cache_key,
        "status": status,
        "mode": mode,
        "created_at_utc": _utc_now(),
        "user_video_mutated": False,
        "bounded_memory": True,
        "extras": dict(extras or {}),
    }
    receipt["receipt_fingerprint"] = hash_canonical_json(
        {k: v for k, v in receipt.items() if k != "receipt_fingerprint"}
    )
    path = stage_dir / "stage_receipt.json"
    write_json_record(path, receipt, overwrite=overwrite)
    return receipt


def run_synthetic_stub(
    *,
    stage_name: str,
    stage_dir: Path,
    run_id: str,
    cache_key: str,
    overwrite: bool = False,
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return write_stage_receipt(
        stage_dir,
        stage_name=stage_name,
        run_id=run_id,
        cache_key=cache_key,
        mode="synthetic_stub",
        extras={"contract_valid_stub": True, **dict(extras or {})},
        overwrite=overwrite,
    )


def _fallback(
    *,
    stage_name: str,
    stage_dir: Path,
    run_id: str,
    cache_key: str,
    exc: BaseException,
) -> dict[str, Any]:
    return run_synthetic_stub(
        stage_name=stage_name,
        stage_dir=stage_dir,
        run_id=run_id,
        cache_key=cache_key,
        overwrite=True,
        extras={"fixture_fallback": True, "error": str(exc)[:240]},
    )


def run_physical_fixture(*, stage_dir: Path, run_id: str, cache_key: str) -> dict[str, Any]:
    from football_analytics.physical.pipeline_config import load_pipeline_config
    from football_analytics.physical.pipeline_fixtures import run_consistent_chain
    from football_analytics.physical.pipeline_service import integrate_physical_metrics

    try:
        cfg = load_pipeline_config()
        chain_root = stage_dir / "_chain_inputs"
        chain_root.mkdir(parents=True, exist_ok=True)
        chain = run_consistent_chain(chain_root)
        import json

        motion_sum = json.loads(Path(str(chain["motion"].summary_json)).read_text(encoding="utf-8"))
        spatial_sum = json.loads(
            Path(str(chain["spatial"].summary_json)).read_text(encoding="utf-8")
        )
        traj_receipt = json.loads(
            Path(str(chain["trajectory"].receipt_json)).read_text(encoding="utf-8")
        )
        motion_receipt = json.loads(
            Path(str(chain["motion"].receipt_json)).read_text(encoding="utf-8")
        )
        spatial_receipt = json.loads(
            Path(str(chain["spatial"].receipt_json)).read_text(encoding="utf-8")
        )
        hm = json.loads(Path(str(chain["spatial"].heatmap_json)).read_text(encoding="utf-8"))
        zones = json.loads(Path(str(chain["spatial"].zones_json)).read_text(encoding="utf-8"))
        activity = json.loads(Path(str(chain["spatial"].activity_json)).read_text(encoding="utf-8"))
        traj_sum = {
            "run_id": chain["identity"]["run_id"],
            "video_id": chain["identity"]["video_id"],
            "target_player_id": chain["identity"]["target_player_id"],
            **dict(chain["trajectory"].summary),
        }
        fuse_dir = stage_dir / "fixture"
        result = integrate_physical_metrics(
            output_dir=fuse_dir,
            identity=chain["identity"],
            trajectory_summary=traj_sum,
            trajectory_receipt=traj_receipt,
            motion_summary=motion_sum,
            motion_receipt=motion_receipt,
            spatial_summary=spatial_sum,
            spatial_receipt=spatial_receipt,
            heatmap_ref=hm,
            zone_ref=zones,
            activity_ref=activity,
            recounted_distance_m=motion_sum.get("measured_distance_m"),
            config=cfg,
        )
        if not result.accepted:
            return _fallback(
                stage_name="physical",
                stage_dir=stage_dir,
                run_id=run_id,
                cache_key=cache_key,
                exc=RuntimeError(str(result.error_code)),
            )
        return write_stage_receipt(
            stage_dir,
            stage_name="physical",
            run_id=run_id,
            cache_key=cache_key,
            mode="fixture_entrypoint",
            extras={"summary_json": result.summary_json, "accepted": True},
            overwrite=True,
        )
    except Exception as exc:  # noqa: BLE001
        return _fallback(
            stage_name="physical",
            stage_dir=stage_dir,
            run_id=run_id,
            cache_key=cache_key,
            exc=exc,
        )


def run_interaction_fixture(*, stage_dir: Path, run_id: str, cache_key: str) -> dict[str, Any]:
    from football_analytics.interaction.pipeline_fixtures import load_pipeline_fixture
    from football_analytics.interaction.pipeline_service import integrate_human_ball_interaction

    try:
        fx = load_pipeline_fixture("controlled_carry", run_id=run_id)
        out = stage_dir / "fixture"
        out.mkdir(parents=True, exist_ok=True)
        result = integrate_human_ball_interaction(output_dir=out, points=fx["points"])
        if not result.accepted:
            return _fallback(
                stage_name="interaction",
                stage_dir=stage_dir,
                run_id=run_id,
                cache_key=cache_key,
                exc=RuntimeError(str(result.error_code)),
            )
        return write_stage_receipt(
            stage_dir,
            stage_name="interaction",
            run_id=run_id,
            cache_key=cache_key,
            mode="fixture_entrypoint",
            extras={"summary_json": result.summary_json, "accepted": True},
            overwrite=True,
        )
    except Exception as exc:  # noqa: BLE001
        return _fallback(
            stage_name="interaction",
            stage_dir=stage_dir,
            run_id=run_id,
            cache_key=cache_key,
            exc=exc,
        )


def run_passing_fixture(*, stage_dir: Path, run_id: str, cache_key: str) -> dict[str, Any]:
    from football_analytics.passing.pipeline_fixtures import load_pipeline_fixture
    from football_analytics.passing.pipeline_service import integrate_passing

    try:
        fx = load_pipeline_fixture("completed_with_box", run_id=run_id)
        out = stage_dir / "fixture"
        out.mkdir(parents=True, exist_ok=True)
        result = integrate_passing(
            output_dir=out,
            transitions=fx["transitions"],
            touch_inputs=fx.get("touch_inputs"),
            run_id=str(fx.get("run_id") or run_id),
            video_id=fx.get("video_id"),
        )
        if not result.accepted:
            return _fallback(
                stage_name="passing",
                stage_dir=stage_dir,
                run_id=run_id,
                cache_key=cache_key,
                exc=RuntimeError(str(result.error_code)),
            )
        return write_stage_receipt(
            stage_dir,
            stage_name="passing",
            run_id=run_id,
            cache_key=cache_key,
            mode="fixture_entrypoint",
            extras={"summary_json": result.summary_json, "accepted": True},
            overwrite=True,
        )
    except Exception as exc:  # noqa: BLE001
        return _fallback(
            stage_name="passing",
            stage_dir=stage_dir,
            run_id=run_id,
            cache_key=cache_key,
            exc=exc,
        )


def run_events_fixture(*, stage_dir: Path, run_id: str, cache_key: str) -> dict[str, Any]:
    from football_analytics.events.fixtures import pipeline_fixture
    from football_analytics.events.pipeline_service import integrate_target_events

    try:
        fx = pipeline_fixture("full_package")
        out = stage_dir / "fixture"
        out.mkdir(parents=True, exist_ok=True)
        result = integrate_target_events(
            output_dir=out,
            sources=fx["sources"],
            replay_contexts=fx["replay_contexts"],
            attack_periods=fx["attack_periods"],
            run_id=run_id,
            video_id=fx["video_id"],
            interaction_coverage=fx["interaction_coverage"],
        )
        if not result.accepted:
            return _fallback(
                stage_name="events",
                stage_dir=stage_dir,
                run_id=run_id,
                cache_key=cache_key,
                exc=RuntimeError(str(result.error_code)),
            )
        return write_stage_receipt(
            stage_dir,
            stage_name="events",
            run_id=run_id,
            cache_key=cache_key,
            mode="fixture_entrypoint",
            extras={"summary_json": result.summary_json, "accepted": True},
            overwrite=True,
        )
    except Exception as exc:  # noqa: BLE001
        return _fallback(
            stage_name="events",
            stage_dir=stage_dir,
            run_id=run_id,
            cache_key=cache_key,
            exc=exc,
        )


def execute_stage_handler(
    *,
    stage_name: str,
    stage_dir: Path,
    run_id: str,
    cache_key: str,
    mode: str,
    report_builder: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if mode == "report_builder":
        if report_builder is None:
            raise RuntimeError("report_builder required for report stage")
        return report_builder(stage_dir=stage_dir, run_id=run_id, cache_key=cache_key)
    if mode == "fixture_entrypoint":
        dispatch = {
            "physical": run_physical_fixture,
            "interaction": run_interaction_fixture,
            "passing": run_passing_fixture,
            "events": run_events_fixture,
        }
        fn = dispatch.get(stage_name)
        if fn is None:
            return run_synthetic_stub(
                stage_name=stage_name,
                stage_dir=stage_dir,
                run_id=run_id,
                cache_key=cache_key,
                overwrite=True,
            )
        return fn(stage_dir=stage_dir, run_id=run_id, cache_key=cache_key)
    return run_synthetic_stub(
        stage_name=stage_name,
        stage_dir=stage_dir,
        run_id=run_id,
        cache_key=cache_key,
        overwrite=True,
    )


__all__ = [
    "write_stage_receipt",
    "run_synthetic_stub",
    "execute_stage_handler",
]
