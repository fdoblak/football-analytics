#!/usr/bin/env python3
"""Validate Stage 8D pitch projection pipeline + Stage 8 close readiness.

Exit codes:
  0 PASS / PASS_WITH_FINDINGS
  1 validation finding / NO-GO content
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

EXIT_PASS = 0
EXIT_FINDING = 1
EXIT_CONFIG = 2
EXIT_INTEGRITY = 3

RUNTIME_ROOT = Path("/home/fdoblak/workspace/pitch_projection_checks")
GATE = (
    "PASS_WITH_FINDINGS — PITCH PROJECTION PIPELINE ACTIVE; STAGE 8 CLOSED; "
    "REAL FOOTBALL ACCURACY NOT YET VALIDATED"
)


class Result:
    def __init__(self) -> None:
        self.status = "PASS"
        self.exit_code = EXIT_PASS
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.findings: list[str] = []
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

    def finding(self, msg: str) -> None:
        self.findings.append(msg)

    def finalize(self) -> Result:
        if self.exit_code in {EXIT_INTEGRITY, EXIT_CONFIG} or self.errors:
            self.status = "NO-GO" if self.exit_code == EXIT_INTEGRITY else "FAIL"
            if self.exit_code == EXIT_PASS:
                self.exit_code = EXIT_FINDING
        elif self.findings or self.warnings:
            self.status = "PASS_WITH_FINDINGS"
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
            "findings": list(self.findings),
            "overall_status": self.status,
            "gate": GATE if self.status in {"PASS", "PASS_WITH_FINDINGS"} else self.status,
        }
        body.update(self.extras)
        return body


def run_checks(args: argparse.Namespace) -> Result:
    from football_analytics.calibration.contracts import (
        EXPECTED_CALIBRATIONS_FP,
        assert_calibrations_fingerprint_frozen,
        calibration_schema_fingerprints,
    )
    from football_analytics.calibration.pitch_projection import (
        apply_image_to_pitch_projection,
        build_projection_for_observation,
        select_segment_for_time,
    )
    from football_analytics.calibration.pitch_projection_config import (
        load_pitch_projection_config,
        pitch_projection_config_fingerprint,
    )
    from football_analytics.calibration.pitch_projection_evaluation import (
        NOT_EVALUATED_PROJECTED_POS,
        evaluate_pitch_projection,
    )
    from football_analytics.calibration.pitch_projection_fixtures import (
        assert_runtime_root,
        base_bundle,
        coverage_hull_for_H,
        human_bbox_for_footpoint,
        identity_H,
        make_segment,
        obs_row,
        perspective_H,
        singular_w_H,
    )
    from football_analytics.calibration.pitch_projection_service import run_pitch_projection
    from football_analytics.calibration.pitch_template import (
        build_pitch_template,
        pitch_template_fingerprint,
    )
    from football_analytics.core.run_id import generate_run_id

    result = Result()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    assert_runtime_root()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    try:
        config = load_pitch_projection_config(cfg_path)
        result.extras["config_fingerprint"] = pitch_projection_config_fingerprint(config)
    except Exception as exc:  # noqa: BLE001
        result.err(f"config load failed: {exc}", config=True)
        return result.finalize()

    if config["attack_direction"] != "unknown":
        result.err("attack_direction must be unknown", integrity=True)
    if config["compute_physical_metrics"] is not False:
        result.err("compute_physical_metrics must be false", integrity=True)
    if config["ball_source"]["physical_metric_eligible"] is not False:
        result.err("ball physical eligible must be false", integrity=True)
    if config["ball_source"]["event_metric_eligible"] is not False:
        result.err("ball event eligible must be false", integrity=True)
    if config["network_sources_allowed"] is not False:
        result.err("network_sources_allowed must be false", integrity=True)

    try:
        assert_calibrations_fingerprint_frozen()
        fps = calibration_schema_fingerprints()
        result.extras["calibrations_fp"] = fps["calibrations"]
        result.extras["projected_positions_fp"] = fps["projected_positions"]
        if fps["calibrations"] != EXPECTED_CALIBRATIONS_FP:
            result.err("calibrations fingerprint drift", integrity=True)
    except Exception as exc:  # noqa: BLE001
        result.err(f"fingerprint check failed: {exc}", integrity=True)

    # Lazy: importing calibration must not load HRNet
    import sys as _sys

    before = {k for k in _sys.modules if "cls_hrnet" in k.lower() or k.startswith("fa_nbjw_")}
    import football_analytics.calibration as cal  # noqa: F401

    after = {k for k in _sys.modules if "cls_hrnet" in k.lower() or k.startswith("fa_nbjw_")}
    if after - before:
        result.err("import football_analytics.calibration loaded HRNet", integrity=True)
    else:
        result.extras["lazy_import_ok"] = True

    eval_report = evaluate_pitch_projection(has_reviewed_ground_truth=False)
    if eval_report.ground_truth_evaluation_status != NOT_EVALUATED_PROJECTED_POS:
        result.err("expected NOT_EVALUATED_NO_REVIEWED_PROJECTED_POSITION_GROUND_TRUTH")
    result.extras["evaluation_status"] = eval_report.ground_truth_evaluation_status

    session = Path(tempfile.mkdtemp(prefix="proj_val_", dir=str(RUNTIME_ROOT)))
    try:
        bundle = base_bundle(H=perspective_H())
        out = session / "project"
        svc = run_pitch_projection(
            output_dir=out,
            config=config,
            contain_root=RUNTIME_ROOT,
            observations_rows=bundle["observations"],
            segments_rows=bundle["segments"],
            frame_times=bundle["frame_times"],
            coverage_hulls=bundle["coverage_hulls"],
            eligibility_timeline=bundle["eligibility_timeline"],
            analysis_windows=bundle["analysis_windows"],
            fingerprints=bundle["fingerprints"],
            frame_width=bundle["frame_width"],
            frame_height=bundle["frame_height"],
        )
        if not svc.accepted:
            result.err(f"projection failed: {svc.error_code}")
        else:
            result.extras["project_ok"] = True
            receipt = json.loads(Path(str(svc.receipt_json)).read_text(encoding="utf-8"))
            result.extras["ball_physical_eligible"] = receipt["ball_physical_metric_eligible_count"]
            result.extras["ball_event_eligible"] = receipt["ball_event_metric_eligible_count"]
            if receipt["ball_physical_metric_eligible_count"] != 0:
                result.err("ball physical eligible must be 0", integrity=True)
            if receipt["ball_event_metric_eligible_count"] != 0:
                result.err("ball event eligible must be 0", integrity=True)
            human = next(p for p in svc.summary["projections"] if p["entity_type"] == "human")
            ball = next(p for p in svc.summary["projections"] if p["entity_type"] == "ball")
            result.extras["human_source_point"] = human["source_point_type"]
            result.extras["ball_source_point"] = ball["source_point_type"]
            if human["source_point_type"] != "bbox_bottom_centre":
                result.err("human source must be bbox_bottom_centre")
            if ball["source_point_type"] != "bbox_centre":
                result.err("ball source must be bbox_centre")
            if ball["physical_metric_eligibility"] == "eligible":
                result.err("ball must not be physical eligible", integrity=True)

            # Deterministic repeat
            out2 = session / "project_repeat"
            svc2 = run_pitch_projection(
                output_dir=out2,
                config=config,
                contain_root=RUNTIME_ROOT,
                observations_rows=bundle["observations"],
                segments_rows=bundle["segments"],
                frame_times=bundle["frame_times"],
                coverage_hulls=bundle["coverage_hulls"],
                eligibility_timeline=bundle["eligibility_timeline"],
                analysis_windows=bundle["analysis_windows"],
                fingerprints=bundle["fingerprints"],
                frame_width=bundle["frame_width"],
                frame_height=bundle["frame_height"],
            )
            if not svc2.accepted:
                result.err(f"repeat projection failed: {svc2.error_code}")
            else:
                a = [
                    (p["pitch_x_m"], p["pitch_y_m"], p["mapping_status"])
                    for p in svc.summary["projections"]
                ]
                b = [
                    (p["pitch_x_m"], p["pitch_y_m"], p["mapping_status"])
                    for p in svc2.summary["projections"]
                ]
                if a != b:
                    result.err("deterministic repeat mismatch")
                else:
                    result.extras["deterministic_ok"] = True

            # No-overwrite
            again = run_pitch_projection(
                output_dir=out,
                config=config,
                contain_root=RUNTIME_ROOT,
                observations_rows=bundle["observations"],
                segments_rows=bundle["segments"],
            )
            if again.accepted or again.error_code != "NO_OVERWRITE":
                result.err("no-overwrite gate failed")

        # Failure cleanup: fingerprint mismatch
        bad_out = session / "bad_fp"
        bad_fps = dict(bundle["fingerprints"])
        bad_fps["run_id"] = generate_run_id()
        bad = run_pitch_projection(
            output_dir=bad_out,
            config=config,
            contain_root=RUNTIME_ROOT,
            observations_rows=bundle["observations"],
            segments_rows=bundle["segments"],
            fingerprints=bad_fps,
        )
        if bad.accepted:
            result.err("fingerprint mismatch should fail")
        elif bad_out.exists() and any(bad_out.iterdir()):
            result.err("failure cleanup left artifacts")
        else:
            result.extras["failure_cleanup_ok"] = True

        # Scenario: gap / not calibrated
        rid, vid = generate_run_id(), "video_gap"
        t_fp = pitch_template_fingerprint(build_pitch_template())
        H = perspective_H()
        seg = make_segment(
            run_id=rid,
            video_id=vid,
            segment_id="seg_early",
            calibration_id=1,
            start_time_us=0,
            end_time_us=40_000,
            H=H,
            template_fp=t_fp,
        )
        obs = obs_row(
            run_id=rid,
            video_id=vid,
            frame_index=5,
            track_id=1,
            entity_type="human",
            bbox=human_bbox_for_footpoint(H, 50.0, 30.0),
        )
        sel = select_segment_for_time(
            [seg],
            run_id=rid,
            video_id=vid,
            video_time_us=200_000,
            config=config,
            pitch_template_fingerprint=t_fp,
        )
        if sel.status != "not_calibrated":
            result.err("gap should be not_calibrated")

        # Scenario: overlapping conflict
        seg_a = make_segment(
            run_id=rid,
            video_id=vid,
            segment_id="seg_a",
            calibration_id=1,
            start_time_us=0,
            end_time_us=100_000,
            H=H,
            template_fp=t_fp,
        )
        seg_b = make_segment(
            run_id=rid,
            video_id=vid,
            segment_id="seg_b",
            calibration_id=2,
            start_time_us=50_000,
            end_time_us=150_000,
            H=H,
            template_fp=t_fp,
        )
        conflict = run_pitch_projection(
            output_dir=session / "conflict",
            config=config,
            contain_root=RUNTIME_ROOT,
            observations_rows=[
                obs_row(
                    run_id=rid,
                    video_id=vid,
                    frame_index=2,
                    track_id=1,
                    entity_type="human",
                    bbox=human_bbox_for_footpoint(H, 50.0, 30.0),
                )
            ],
            segments_rows=[seg_a, seg_b],
            frame_times={2: 60_000},
        )
        if conflict.accepted or conflict.error_code != "SEGMENT_OVERLAP_CONFLICT":
            result.err(f"overlap conflict expected, got {conflict.error_code}")

        # Homogeneous w≈0
        Hs = singular_w_H()
        geom = apply_image_to_pitch_projection(
            image_x=50.0,
            image_y=50.0,
            H_row_major=[float(x) for x in Hs.reshape(9)],
            H_inv_row_major=None,
            pitch_length_m=105.0,
            pitch_width_m=68.0,
            config=config,
        )
        if geom.mapping_status != "failed":
            result.err("singular w should fail")

        # Matrix direction: projecting with H_inv must not be the service path
        # (service uses image_to_pitch only — verified via provenance)
        if svc.accepted:
            for p in svc.summary["projections"]:
                prov = json.loads(p["provenance_json"])
                if prov.get("used_h_inv_for_projection") is not False:
                    result.err("used_h_inv_for_projection must be false")
                    break
            else:
                result.extras["direction_ok"] = True

        # Identity H smoke
        _ = identity_H()
        _ = coverage_hull_for_H(perspective_H())
        _ = build_projection_for_observation
        _ = obs

    except Exception as exc:  # noqa: BLE001
        result.err(f"synthetic validator failed: {type(exc).__name__}: {exc}")
    finally:
        if not args.keep:
            shutil.rmtree(session, ignore_errors=True)

    result.finding(
        "NBJW/SV adapter remains evaluation_only / GPL-2.0 linking risk (Stage 8B; not vendored)"
    )
    result.finding("Real football projected-position accuracy not validated — no reviewed GT")
    result.finding("Human footpoint is bbox_bottom_centre approximation (no pose/foot model)")
    result.finding(
        "Ball projection is image-plane centre; airborne/grounded unknown; never metric-eligible"
    )
    result.finding("Attack direction remains unknown; no distance/speed/sprint/heatmap/events")
    result.warn("Synthetic known-H projection metrics are not match accuracy")

    result.extras["gate"] = GATE
    result.extras["runtime_root"] = str(RUNTIME_ROOT)
    result.extras["stage_8_closed"] = True
    return result.finalize()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/calibration/pitch_projection_pipeline.yaml",
        help="Pitch projection pipeline config path",
    )
    parser.add_argument("--keep", action="store_true", help="Keep validator session dir")
    parser.add_argument("--json", action="store_true", help="Emit JSON report")
    args = parser.parse_args(argv)
    result = run_checks(args)
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status: {payload['status']}")
        print(f"gate: {payload.get('gate')}")
        print(f"exit_code: {payload['exit_code']}")
        for e in payload["errors"]:
            print(f"error: {e}")
        for w in payload["warnings"]:
            print(f"warning: {w}")
        for f in payload["findings"]:
            print(f"finding: {f}")
    return int(payload["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
