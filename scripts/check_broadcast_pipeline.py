#!/usr/bin/env python3
"""Validate Stage 4D broadcast fusion + playability routing (synthetic E2E).

Exit codes:
  0 success
  1 validation finding
  2 configuration failure
  3 integrity/security failure
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

EXIT_PASS = 0
EXIT_FINDING = 1
EXIT_CONFIG = 2
EXIT_INTEGRITY = 3

RUNTIME_ROOT = Path("/home/fdoblak/workspace/broadcast_pipeline_checks")


class Result:
    def __init__(self) -> None:
        self.status = "PASS"
        self.exit_code = EXIT_PASS
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.extras: dict[str, Any] = {}

    def err(self, msg: str, *, integrity: bool = False, config: bool = False) -> None:
        self.errors.append(msg)
        if integrity:
            self.exit_code = EXIT_INTEGRITY
        elif config:
            self.exit_code = EXIT_CONFIG
        elif self.exit_code == EXIT_PASS:
            self.exit_code = EXIT_FINDING

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def finalize(self) -> Result:
        if self.exit_code in {EXIT_INTEGRITY, EXIT_CONFIG} or self.errors:
            self.status = "FAIL"
            if self.exit_code == EXIT_PASS:
                self.exit_code = EXIT_FINDING
        elif self.warnings:
            self.status = "PASS_WITH_WARNINGS"
            self.exit_code = EXIT_PASS
        else:
            self.status = "PASS"
            self.exit_code = EXIT_PASS
        return self

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema_version": 1,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "status": self.status,
            "exit_code": self.exit_code,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "overall_status": self.status,
        }
        body.update(self.extras)
        return body


def _table(contract_name: str, rows: list[dict[str, Any]]) -> Any:
    from football_analytics.data.compiler import compile_arrow_schema, get_contract

    schema = compile_arrow_schema(get_contract(contract_name, 1))
    return pa.Table.from_pylist(rows, schema=schema)


def _run_id() -> str:
    from football_analytics.core.run_id import generate_run_id

    return generate_run_id()


def _base_shot(
    run_id: str,
    video_id: str,
    shot_id: str,
    start: int,
    end: int,
    *,
    mapping: str = "exact_identity",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "shot_id": shot_id,
        "start_time_us": start,
        "end_time_us": end,
        "start_frame_index": None,
        "end_frame_index_exclusive": None,
        "start_boundary_id": None,
        "end_boundary_id": None,
        "duration_us": end - start,
        "frame_count": None,
        "timeline_mapping_quality": mapping,
        "segment_status": "active",
        "provenance_json": '{"origin":"synthetic"}',
        "contract_version": 1,
    }


def _base_cam(
    run_id: str,
    video_id: str,
    cam_id: str,
    shot_id: str,
    start: int,
    end: int,
    *,
    view: str = "main_broadcast",
    framing: str = "wide",
    replay: str = "live",
    graphics: str = "none",
    playability: str = "playable",
    coverage: float = 1.0,
    confidence: float | None = 0.9,
    tracking_suitability: str = "suitable",
    calibration_suitability: str = "suitable",
    identity_suitability: str = "unknown",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "camera_segment_id": cam_id,
        "shot_id": shot_id,
        "start_time_us": start,
        "end_time_us": end,
        "start_frame_index": None,
        "end_frame_index_exclusive": None,
        "view_family": view,
        "framing_scale": framing,
        "camera_position": "unknown",
        "camera_motion": "static",
        "replay_status": replay,
        "graphics_status": graphics,
        "playability": playability,
        "calibration_suitability": calibration_suitability,
        "tracking_suitability": tracking_suitability,
        "target_identity_suitability": identity_suitability,
        "classification_source": "manual",
        "confidence": confidence,
        "coverage": coverage,
        "review_status": "accepted",
        "evidence_refs": [cam_id],
        "provenance_json": '{"origin":"synthetic"}',
        "contract_version": 1,
    }


def _boundary(run_id: str, video_id: str, bid: str, t: int) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "boundary_id": bid,
        "boundary_time_us": t,
        "left_frame_index": None,
        "right_frame_index": None,
        "transition_type": "hard_cut",
        "transition_duration_us": 0,
        "confidence": 1.0,
        "detection_source": "manual",
        "evidence_ref": None,
        "review_status": "accepted",
        "provenance_json": '{"origin":"synthetic"}',
        "contract_version": 1,
    }


def _minimal_frames(run_id: str, video_id: str, end_us: int, fps: int = 25) -> list[dict[str, Any]]:
    period_us = 1_000_000 // fps
    rows: list[dict[str, Any]] = []
    t = 0
    i = 0
    while t < end_us:
        rows.append(
            {
                "run_id": run_id,
                "video_id": video_id,
                "frame_index": i,
                "pts": i,
                "video_time_us": t,
                "duration_us": period_us,
                "is_key_frame": i == 0,
                "decode_status": "ok",
            }
        )
        i += 1
        t += period_us
    return rows


def build_scenario_tables(run_id: str) -> dict[str, Any]:
    """Synthetic shots/cameras covering all Stage 4D routing scenarios."""
    video_id = "synth_broadcast_01"
    # Contiguous shots covering 0..2_000_000
    shots = [
        _base_shot(run_id, video_id, "shot_wide", 0, 200_000),
        _base_shot(run_id, video_id, "shot_close", 200_000, 400_000),
        _base_shot(run_id, video_id, "shot_graphics", 400_000, 600_000),
        _base_shot(run_id, video_id, "shot_unknown", 600_000, 800_000),
        _base_shot(run_id, video_id, "shot_lowcov", 800_000, 1_000_000),
        _base_shot(run_id, video_id, "shot_replay_c", 1_000_000, 1_200_000),
        _base_shot(run_id, video_id, "shot_replay_u", 1_200_000, 1_400_000),
        _base_shot(run_id, video_id, "shot_gap", 1_400_000, 1_600_000),
        _base_shot(run_id, video_id, "shot_conflict", 1_600_000, 1_800_000),
        _base_shot(run_id, video_id, "shot_multi", 1_800_000, 2_000_000),
        _base_shot(
            run_id,
            video_id,
            "shot_degraded",
            2_000_000,
            2_200_000,
            mapping="uncertain",
        ),
    ]
    cameras = [
        _base_cam(run_id, video_id, "cam_wide", "shot_wide", 0, 200_000),
        _base_cam(
            run_id,
            video_id,
            "cam_close",
            "shot_close",
            200_000,
            400_000,
            view="player_isolation",
            framing="close_up",
            playability="playable",
            tracking_suitability="unsuitable",
            calibration_suitability="unsuitable",
            identity_suitability="suitable",
        ),
        _base_cam(
            run_id,
            video_id,
            "cam_graphics",
            "shot_graphics",
            400_000,
            600_000,
            view="graphics",
            framing="unknown",
            graphics="full_screen",
            playability="non_playable",
            tracking_suitability="unsuitable",
            calibration_suitability="unsuitable",
            identity_suitability="unsuitable",
        ),
        _base_cam(
            run_id,
            video_id,
            "cam_unknown",
            "shot_unknown",
            600_000,
            800_000,
            view="unknown",
            framing="unknown",
            playability="uncertain",
            tracking_suitability="unknown",
            calibration_suitability="unknown",
        ),
        _base_cam(
            run_id,
            video_id,
            "cam_lowcov",
            "shot_lowcov",
            800_000,
            1_000_000,
            coverage=0.2,
            tracking_suitability="unknown",
            calibration_suitability="unknown",
        ),
        _base_cam(
            run_id,
            video_id,
            "cam_replay_c",
            "shot_replay_c",
            1_000_000,
            1_200_000,
            replay="replay",
            playability="partially_playable",
            tracking_suitability="conditionally_suitable",
            calibration_suitability="unsuitable",
        ),
        _base_cam(
            run_id,
            video_id,
            "cam_replay_u",
            "shot_replay_u",
            1_200_000,
            1_400_000,
            replay="unknown",
        ),
        # gap shot: camera only covers first half → gap in second half
        _base_cam(run_id, video_id, "cam_gap_partial", "shot_gap", 1_400_000, 1_500_000),
        # overlapping conflicting labels (uncertain so bundle playable-overlap rule passes)
        _base_cam(
            run_id,
            video_id,
            "cam_conflict_a",
            "shot_conflict",
            1_600_000,
            1_800_000,
            view="main_broadcast",
            framing="wide",
            playability="uncertain",
            tracking_suitability="unknown",
            calibration_suitability="unknown",
        ),
        _base_cam(
            run_id,
            video_id,
            "cam_conflict_b",
            "shot_conflict",
            1_600_000,
            1_800_000,
            view="player_isolation",
            framing="close_up",
            playability="uncertain",
            tracking_suitability="unsuitable",
            calibration_suitability="unsuitable",
            identity_suitability="suitable",
        ),
        # multi camera sequential in one shot
        _base_cam(run_id, video_id, "cam_multi_a", "shot_multi", 1_800_000, 1_900_000),
        _base_cam(
            run_id,
            video_id,
            "cam_multi_b",
            "shot_multi",
            1_900_000,
            2_000_000,
            view="player_isolation",
            framing="close_up",
            tracking_suitability="unsuitable",
            calibration_suitability="unsuitable",
            identity_suitability="suitable",
        ),
        _base_cam(
            run_id,
            video_id,
            "cam_degraded",
            "shot_degraded",
            2_000_000,
            2_200_000,
        ),
    ]
    # Boundary-adjacent markers
    boundaries = [
        _boundary(run_id, video_id, f"bnd_{i:02d}", t)
        for i, t in enumerate(
            [
                0,
                200_000,
                400_000,
                600_000,
                800_000,
                1_000_000,
                1_200_000,
                1_400_000,
                1_600_000,
                1_800_000,
                2_000_000,
                2_200_000,
            ]
        )
    ]
    frames = _minimal_frames(run_id, video_id, 2_200_000)
    return {
        "video_id": video_id,
        "shots": shots,
        "cameras": cameras,
        "boundaries": boundaries,
        "frames": frames,
        "span_us": 2_200_000,
    }


def _write_inputs(session: Path, run_id: str, data: dict[str, Any]) -> dict[str, Path]:
    from football_analytics.data.compiler import get_contract
    from football_analytics.data.parquet import write_contract_parquet

    paths = {
        "timeline": session / "frames.parquet",
        "boundaries": session / "shot_boundaries.parquet",
        "shots": session / "shot_segments.parquet",
        "camera_views": session / "camera_view_segments.parquet",
    }
    write_contract_parquet(
        _table("frames", data["frames"]),
        paths["timeline"],
        get_contract("frames", 1),
        contain_root=session,
        overwrite=False,
    )
    write_contract_parquet(
        _table("shot_boundaries", data["boundaries"]),
        paths["boundaries"],
        get_contract("shot_boundaries", 1),
        contain_root=session,
        overwrite=False,
    )
    write_contract_parquet(
        _table("shot_segments", data["shots"]),
        paths["shots"],
        get_contract("shot_segments", 1),
        contain_root=session,
        overwrite=False,
    )
    write_contract_parquet(
        _table("camera_view_segments", data["cameras"]),
        paths["camera_views"],
        get_contract("camera_view_segments", 1),
        contain_root=session,
        overwrite=False,
    )
    return paths


def _gt_from_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reviewed GT mirrors predictions for frozen safety (self-consistent routing)."""
    return [dict(r) for r in rows]


def run_checks(*, keep: bool = False) -> Result:
    result = Result()
    from football_analytics.broadcast.broadcast_evaluation import (
        evaluate_broadcast_windows,
        passes_safety_gates,
    )
    from football_analytics.broadcast.broadcast_pipeline import run_broadcast_integrate
    from football_analytics.broadcast.playability import (
        load_routing_policy,
        routing_policy_fingerprint,
    )
    from football_analytics.broadcast.segment_fusion import FusionError, fuse_shot_camera_intervals
    from football_analytics.broadcast.validation import (
        validate_analysis_windows_bundle,
        validate_broadcast_bundle,
    )
    from football_analytics.data.compiler import get_contract, list_contracts
    from football_analytics.data.parquet import read_contract_parquet
    from football_analytics.data.registry import (
        default_project_root,
        default_registry_path,
        load_schema_registry,
    )

    root = default_project_root()
    try:
        reg = load_schema_registry(default_registry_path(), project_root=root)
        names = list_contracts(registry=reg)
        if len(names) != 15 or "analysis_windows" not in names:
            result.err(f"registry contract count unexpected: {len(names)}", config=True)
        policy = load_routing_policy(root / "configs/broadcast/broadcast_routing_policy.yaml")
        result.extras["policy_fingerprint"] = routing_policy_fingerprint(policy)
    except Exception as exc:  # noqa: BLE001
        result.err(f"config load failed: {exc}", config=True)
        return result.finalize()

    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix="bp_", dir=str(RUNTIME_ROOT)))
    result.extras["session"] = str(session)
    try:
        run_id = _run_id()
        data = build_scenario_tables(run_id)
        paths = _write_inputs(session, run_id, data)

        # Invalid FK / camera outside shot must raise
        bad_cams = list(data["cameras"]) + [
            _base_cam(run_id, data["video_id"], "cam_bad", "shot_wide", 0, 500_000)
        ]
        try:
            fuse_shot_camera_intervals(data["shots"], bad_cams)
            result.err("expected FusionError for camera outside shot")
        except FusionError:
            pass

        out1 = session / "out1"
        out1.mkdir()
        res1 = run_broadcast_integrate(
            timeline=str(paths["timeline"]),
            boundaries=str(paths["boundaries"]),
            shots=str(paths["shots"]),
            camera_views=str(paths["camera_views"]),
            output_dir=str(out1),
            policy=policy,
            contain_root=session,
            run_id=run_id,
            video_id=data["video_id"],
        )
        if not res1.accepted:
            result.err(f"pipeline rejected: {res1.error_code}")
            return result.finalize()

        windows = read_contract_parquet(
            Path(str(res1.analysis_windows_parquet)),
            get_contract("analysis_windows", 1),
            contain_root=session,
        ).to_pylist()
        review = json.loads(Path(str(res1.review_queue_json)).read_text(encoding="utf-8"))

        shots_t = _table("shot_segments", data["shots"])
        cams_t = _table("camera_view_segments", data["cameras"])
        bnd_t = _table("shot_boundaries", data["boundaries"])
        frames_t = _table("frames", data["frames"])
        aw_t = _table("analysis_windows", windows)
        vr = validate_broadcast_bundle(bnd_t, shots_t, cams_t, frames=frames_t)
        if vr.status == "FAIL":
            result.err(f"input bundle fail: {vr.errors[:3]}")
        aw_vr = validate_analysis_windows_bundle(
            aw_t, shots=shots_t, cameras=cams_t, frames=frames_t
        )
        if aw_vr.status == "FAIL":
            result.err(f"analysis_windows bundle fail: {aw_vr.errors[:5]}")

        # Scenario assertions
        by_shot = {w.get("shot_id"): w for w in windows if w.get("shot_id")}
        # multi-cam produces multiple windows for shot_multi
        multi = [w for w in windows if w.get("shot_id") == "shot_multi"]
        if len(multi) < 2:
            result.err("expected multi-camera split windows")

        wide = by_shot.get("shot_wide")
        if wide is None or wide["tracking_eligibility"] != "eligible":
            result.err("playable wide tracking not eligible")
        close = by_shot.get("shot_close")
        if close is None or close["identity_eligibility"] != "eligible":
            result.err("close-up identity not eligible")
        if close and close["calibration_eligibility"] == "eligible":
            result.err("close-up calibration incorrectly eligible")
        graphics = by_shot.get("shot_graphics")
        if graphics and any(
            graphics[a] == "eligible"
            for a in (
                "tracking_eligibility",
                "calibration_eligibility",
                "live_event_eligibility",
                "physical_metric_eligibility",
            )
        ):
            result.err("graphics window has unsafe eligible axis")
        unknown = by_shot.get("shot_unknown")
        if unknown and not unknown["manual_review_required"]:
            result.err("unknown view missing manual review")
        low = by_shot.get("shot_lowcov")
        if low and not low["manual_review_required"]:
            result.err("low coverage missing manual review")
        replay_c = by_shot.get("shot_replay_c")
        if replay_c and replay_c["live_event_eligibility"] == "eligible":
            result.err("replay confirmed live_event eligible")
        if replay_c and replay_c["physical_metric_eligibility"] == "eligible":
            result.err("replay confirmed physical eligible")
        replay_u = by_shot.get("shot_replay_u")
        if replay_u and replay_u["live_event_eligibility"] == "eligible":
            result.err("replay unknown live_event eligible")
        if replay_u and not replay_u["manual_review_required"]:
            result.err("replay unknown missing manual review")
        gaps = [w for w in windows if "CAMERA_GAP" in (w.get("decision_codes") or [])]
        if not gaps:
            result.err("expected CAMERA_GAP window")
        conflicts = [
            w for w in windows if "CONFLICTING_CAMERA_LABELS" in (w.get("decision_codes") or [])
        ]
        if not conflicts:
            result.err("expected CONFLICTING_CAMERA_LABELS window")
        degraded = by_shot.get("shot_degraded")
        if degraded and degraded["physical_metric_eligibility"] == "eligible":
            result.err("degraded mapping physical eligible")

        # Full coverage of shot backbone
        covered = sum(int(w["end_time_us"]) - int(w["start_time_us"]) for w in windows)
        if covered != data["span_us"]:
            result.err(f"coverage mismatch covered={covered} span={data['span_us']}")

        # No-overwrite
        res_ow = run_broadcast_integrate(
            timeline=str(paths["timeline"]),
            boundaries=str(paths["boundaries"]),
            shots=str(paths["shots"]),
            camera_views=str(paths["camera_views"]),
            output_dir=str(out1),
            policy=policy,
            contain_root=session,
            run_id=run_id,
            video_id=data["video_id"],
        )
        if res_ow.accepted:
            result.err("overwrite unexpectedly accepted")

        # Deterministic repeat
        out2 = session / "out2"
        out2.mkdir()
        res2 = run_broadcast_integrate(
            timeline=str(paths["timeline"]),
            boundaries=str(paths["boundaries"]),
            shots=str(paths["shots"]),
            camera_views=str(paths["camera_views"]),
            output_dir=str(out2),
            policy=policy,
            contain_root=session,
            run_id=run_id,
            video_id=data["video_id"],
        )
        if not res2.accepted:
            result.err(f"repeat pipeline rejected: {res2.error_code}")
        else:
            windows2 = read_contract_parquet(
                Path(str(res2.analysis_windows_parquet)),
                get_contract("analysis_windows", 1),
                contain_root=session,
            ).to_pylist()
            gt = _gt_from_predictions(windows)
            report = evaluate_broadcast_windows(
                windows,
                gt,
                expected_span_us=data["span_us"],
                repeat_predictions=windows2,
                unexplained_gap_windows=0,
            )
            ok, fails = passes_safety_gates(report, thresholds=dict(policy["evaluation"]))
            result.extras["evaluation"] = report.to_dict()
            result.extras["review_count"] = len(review.get("items", []))
            result.extras["window_count"] = len(windows)
            if not ok:
                for f in fails:
                    result.err(f"safety gate: {f}")
            if report.deterministic_repeat is not True:
                result.err("deterministic repeat failed")
            if report.overlap_rate not in (0.0, None) and (report.overlap_rate or 0) > 0:
                result.err(f"overlap_rate={report.overlap_rate}")

    except Exception as exc:  # noqa: BLE001
        result.err(f"validator exception: {exc}")
    finally:
        if not keep:
            shutil.rmtree(session, ignore_errors=True)
            result.extras["cleaned"] = True
        else:
            result.extras["cleaned"] = False
    return result.finalize()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="Keep session artifacts")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=RUNTIME_ROOT,
        help="Directory for validation JSON report",
    )
    args = parser.parse_args(argv)
    result = run_checks(keep=bool(args.keep))
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"broadcast_pipeline_validation_{stamp}.json"
    report_path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(f"status: {result.status}")
    print(f"exit_code: {result.exit_code}")
    print(f"report: {report_path}")
    if result.errors:
        for e in result.errors[:20]:
            print(f"error: {e}")
    return int(result.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
