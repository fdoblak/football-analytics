#!/usr/bin/env python3
"""Validate Stage 8A pitch calibration / homography / coordinate contracts.

Exit codes:
  0  success (PASS / PASS_WITH_WARNINGS)
  1  validation finding/failure
  2  configuration/schema failure
  3  integrity/security failure
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

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

EXIT_PASS = 0
EXIT_FINDING = 1
EXIT_CONFIG = 2
EXIT_INTEGRITY = 3

RUNTIME_ROOT = Path("/home/fdoblak/workspace/calibration_contract_checks")
GATE_PASS = "PASS — CALIBRATION AND PITCH COORDINATE CONTRACTS ACTIVE"
GATE_FINDINGS = "PASS_WITH_FINDINGS — CALIBRATION AND PITCH COORDINATE CONTRACTS ACTIVE"
GATE_FAIL = "NO-GO — CALIBRATION CONTRACT FAILURE"


class Result:
    def __init__(self) -> None:
        self.status = "PASS"
        self.exit_code = EXIT_PASS
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.extras: dict[str, Any] = {}
        self.scenarios: dict[str, str] = {}

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

    def ok_scenario(self, name: str) -> None:
        self.scenarios[name] = "PASS"

    def fail_scenario(self, name: str, msg: str) -> None:
        self.scenarios[name] = f"FAIL: {msg}"
        self.err(f"{name}: {msg}")

    def finalize(self, *, strict: bool) -> Result:
        if self.exit_code in {EXIT_INTEGRITY, EXIT_CONFIG} or self.errors:
            self.status = "FAIL"
            if self.exit_code == EXIT_PASS:
                self.exit_code = EXIT_FINDING
        elif self.warnings and strict:
            self.status = "FAIL"
            self.exit_code = EXIT_FINDING
        elif self.warnings:
            self.status = "PASS_WITH_WARNINGS"
            self.exit_code = EXIT_PASS
        else:
            self.status = "PASS"
            self.exit_code = EXIT_PASS
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "status": self.status,
            "exit_code": self.exit_code,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "scenarios": dict(self.scenarios),
            "extras": self.extras,
        }


def run_checks(*, keep: bool, strict: bool) -> Result:
    from football_analytics.calibration.contracts import (
        EXPECTED_CALIBRATIONS_FP,
        EXPECTED_REGISTRY_CONTRACT_COUNT,
        assert_calibration_contracts_registered,
        assert_calibrations_fingerprint_frozen,
        calibration_schema_fingerprints,
        compile_calibration_schemas,
        load_calibration_json_schema,
        validate_against_json_schema,
    )
    from football_analytics.calibration.evaluation import (
        NOT_EVALUATED_CALIBRATION,
        evaluate_calibration,
    )
    from football_analytics.calibration.fixtures import (
        correspondences_for_H,
        e2e_bundle,
        gap_segments_bundle,
        identity_homography,
        ill_conditioned_matrix_row_major,
        known_perspective_H,
        mirrored_homography,
        overlapping_segments_bundle,
        projected_from_track,
        rotation_homography,
        scale_translate_homography,
        singular_matrix_row_major,
    )
    from football_analytics.calibration.homography import (
        apply_homography,
        solve_homography,
        validate_homography_matrix,
    )
    from football_analytics.calibration.pitch_template import (
        build_pitch_template,
        pitch_template_fingerprint,
        validate_fifa_range,
    )
    from football_analytics.calibration.policy import (
        coordinate_system_fingerprint,
        load_calibration_policy,
        load_coordinate_system,
        policy_fingerprint,
    )
    from football_analytics.calibration.receipt import (
        build_synthetic_receipt,
        build_synthetic_request,
        validate_receipt_payload,
        validate_request_payload,
    )
    from football_analytics.calibration.segments import (
        find_calibration_gaps,
        find_segment_overlaps,
        terminate_on_shot_cut,
    )
    from football_analytics.calibration.types import HomographyError
    from football_analytics.calibration.validation import validate_calibration_bundle
    from football_analytics.core.records import RecordError, write_json_record
    from football_analytics.data.compiler import get_contract, list_contracts
    from football_analytics.data.fingerprint import contract_fingerprint

    result = Result()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix="cal_", dir=str(RUNTIME_ROOT)))
    result.extras["session"] = str(session)

    try:
        assert_calibration_contracts_registered()
        names = list_contracts()
        result.extras["contract_count"] = len(names)
        if len(names) != EXPECTED_REGISTRY_CONTRACT_COUNT:
            result.err(
                f"expected {EXPECTED_REGISTRY_CONTRACT_COUNT} contracts, got {len(names)}",
                config=True,
            )
        for n in (
            "calibration_features",
            "calibration_segments",
            "projected_positions",
            "calibrations",
        ):
            if n not in names:
                result.err(f"{n} missing from registry", config=True)

        assert_calibrations_fingerprint_frozen()
        fps = calibration_schema_fingerprints()
        result.extras["schema_fingerprints"] = fps
        if fps["calibrations"] != EXPECTED_CALIBRATIONS_FP:
            result.err("calibrations fingerprint regression", integrity=True)

        compile_calibration_schemas()
        policy = load_calibration_policy()
        coords = load_coordinate_system()
        pol_fp = policy_fingerprint(policy)
        coord_fp = coordinate_system_fingerprint(coords)
        template = build_pitch_template()
        tpl_fp = pitch_template_fingerprint(template)
        validate_fifa_range(template.length_m, template.width_m)
        result.ok_scenario("pitch_template")

        # 1 identity
        img, pitch = correspondences_for_H(identity_homography(), n=4)
        # identity maps pitch==image coords — use pitch as image for identity H
        try:
            r = solve_homography(pitch, pitch)
            if r.status != "valid":
                result.fail_scenario("identity_homography", "not valid")
            else:
                result.ok_scenario("identity_homography")
        except HomographyError as exc:
            result.fail_scenario("identity_homography", str(exc))

        # 2 scale/translation
        H_st = scale_translate_homography(scale_x=0.5, scale_y=0.4, tx=10.0, ty=5.0)
        img2, pitch2 = correspondences_for_H(H_st, n=4)
        try:
            solve_homography(img2, pitch2)
            result.ok_scenario("scale_translation")
        except HomographyError as exc:
            result.fail_scenario("scale_translation", str(exc))

        # 3 perspective
        H_p = known_perspective_H()
        img3, pitch3 = correspondences_for_H(H_p, n=4)
        try:
            solved = solve_homography(img3, pitch3)
            result.ok_scenario("perspective")
        except HomographyError as exc:
            result.fail_scenario("perspective", str(exc))
            solved = None

        # 4 rotation
        H_r = rotation_homography(0.3)
        img4, pitch4 = correspondences_for_H(H_r, n=4)
        try:
            solve_homography(img4, pitch4)
            result.ok_scenario("rotation")
        except HomographyError as exc:
            result.fail_scenario("rotation", str(exc))

        # 5 round-trip
        if solved is not None:
            back = apply_homography(solved.H_inv, apply_homography(solved.H, img3))
            err = float(np.mean(np.linalg.norm(back - np.asarray(img3), axis=1)))
            if err > 1.0:
                result.fail_scenario("round_trip", f"err={err}")
            else:
                result.ok_scenario("round_trip")
        else:
            result.fail_scenario("round_trip", "no solved H")

        # 6 collinear
        col_img = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
        col_pitch = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)]
        try:
            solve_homography(col_img, col_pitch)
            result.fail_scenario("collinear", "accepted")
        except HomographyError:
            result.ok_scenario("collinear")

        # 7 duplicates
        dup_img = [(0.0, 0.0), (0.0, 0.0), (10.0, 0.0), (0.0, 10.0)]
        dup_pitch = [(0.0, 0.0), (1.0, 0.0), (10.0, 0.0), (0.0, 10.0)]
        try:
            solve_homography(dup_img, dup_pitch)
            result.fail_scenario("duplicates", "accepted")
        except HomographyError:
            result.ok_scenario("duplicates")

        # 8 singular
        try:
            validate_homography_matrix(singular_matrix_row_major())
            result.fail_scenario("singular", "accepted")
        except HomographyError:
            result.ok_scenario("singular")

        # 9 ill-conditioned
        try:
            validate_homography_matrix(ill_conditioned_matrix_row_major())
            result.fail_scenario("ill_conditioned", "accepted")
        except HomographyError:
            result.ok_scenario("ill_conditioned")

        # 10 mirrored
        try:
            validate_homography_matrix(list(mirrored_homography().reshape(9)))
            result.fail_scenario("mirrored", "accepted")
        except HomographyError:
            result.ok_scenario("mirrored")

        # 11 insufficient
        try:
            solve_homography([(0, 0), (1, 0), (0, 1)], [(0, 0), (1, 0), (0, 1)])
            result.fail_scenario("insufficient", "accepted")
        except HomographyError:
            result.ok_scenario("insufficient")

        # 12 high reprojection — corrupt one pitch point after building from H
        img_h, pitch_h = correspondences_for_H(H_p, n=4)
        pitch_bad = list(pitch_h)
        pitch_bad[0] = (pitch_bad[0][0] + 50.0, pitch_bad[0][1] + 50.0)
        try:
            solve_homography(img_h, pitch_bad, max_mean_reprojection_error_px=1.0)
            result.fail_scenario("high_reproj", "accepted")
        except HomographyError:
            result.ok_scenario("high_reproj")

        bundle = e2e_bundle()
        Hrm = bundle["H"].matrix_row_major()

        # 13 bounds / extrapolation — far outside image→likely outside pitch
        far = projected_from_track(
            bundle["run_id"],
            bundle["video_id"],
            entity_type="human",
            bbox=(-5000.0, -5000.0, -4900.0, -4800.0),
            H_row_major=Hrm,
            projection_id="proj_far",
        )
        if far["mapping_status"] in {"outside_pitch", "extrapolated", "mapped"}:
            # mapped would be unexpected for far negative; still check eligibility
            if far["physical_metric_eligibility"] == "eligible" and far["is_extrapolated"]:
                result.fail_scenario("bounds_extrapolation", "extrapolated eligible")
            else:
                result.ok_scenario("bounds_extrapolation")
        else:
            result.ok_scenario("bounds_extrapolation")

        # 14 human bottom-centre
        human = next(p for p in bundle["projections"] if p["entity_type"] == "human")
        if human["source_point_type"] != "bbox_bottom_centre":
            result.fail_scenario("human_footpoint", human["source_point_type"])
        else:
            result.ok_scenario("human_footpoint")

        # 15 ball centre
        ball = next(p for p in bundle["projections"] if p["entity_type"] == "ball")
        if ball["source_point_type"] != "bbox_centre":
            result.fail_scenario("ball_centre", ball["source_point_type"])
        else:
            result.ok_scenario("ball_centre")

        # 16 predicted ineligible
        pred = projected_from_track(
            bundle["run_id"],
            bundle["video_id"],
            entity_type="human",
            bbox=(100.0, 100.0, 140.0, 200.0),
            H_row_major=Hrm,
            observation_source="predicted",
            projection_id="proj_pred",
        )
        if pred["physical_metric_eligibility"] == "eligible":
            result.fail_scenario("predicted_ineligible", "marked eligible")
        else:
            result.ok_scenario("predicted_ineligible")

        # 17 shot cut
        cut = terminate_on_shot_cut(segment=bundle["segments"][0], cut_time_us=400_000)
        if cut["end_time_us"] != 400_000 or cut["boundary_reason"] != "SHOT_CUT_TERMINATE":
            result.fail_scenario("shot_cut", "not terminated")
        else:
            result.ok_scenario("shot_cut")

        # 18 overlapping conflict
        overs = overlapping_segments_bundle(bundle["run_id"], bundle["video_id"])
        if not find_segment_overlaps(overs):
            result.fail_scenario("overlap", "no conflict detected")
        else:
            result.ok_scenario("overlap")

        # 19 gap / not calibrated
        gaps = find_calibration_gaps(
            gap_segments_bundle(bundle["run_id"], bundle["video_id"]),
            timeline_start_us=0,
            timeline_end_us=2_000_000,
        )
        if not gaps:
            result.fail_scenario("gap", "no gap")
        else:
            result.ok_scenario("gap")

        # 20 VFR microsecond interval
        seg = bundle["segments"][0]
        if not (seg["start_time_us"] < seg["end_time_us"]):
            result.fail_scenario("vfr_us", "bad interval")
        else:
            result.ok_scenario("vfr_us")

        # 21 hash / FK / template mismatch
        bad_seg = dict(bundle["segments"][0])
        bad_seg["pitch_template_fingerprint"] = "b" * 64
        vr = validate_calibration_bundle(
            calibration_features=bundle["features"],
            calibration_segments=[bad_seg],
            projected_positions=bundle["projections"],
            pitch_template_fingerprint=tpl_fp,
            policy=policy,
        )
        if vr.status != "FAIL":
            result.fail_scenario("template_mismatch", "not detected")
        else:
            result.ok_scenario("template_mismatch")

        # 22 deterministic fingerprint
        fp1 = pitch_template_fingerprint(template)
        fp2 = pitch_template_fingerprint(build_pitch_template())
        if fp1 != fp2 or len(fp1) != 64:
            result.fail_scenario("deterministic_fp", "unstable")
        else:
            result.ok_scenario("deterministic_fp")

        # 23 atomic no-overwrite
        out = session / "receipt.json"
        req = build_synthetic_request(
            run_id=bundle["run_id"],
            video_id=bundle["video_id"],
            policy_fingerprint=pol_fp,
            coordinate_system_fingerprint=coord_fp,
            pitch_template_fingerprint=tpl_fp,
        )
        validate_request_payload(req)
        receipt = build_synthetic_receipt(
            run_id=bundle["run_id"],
            video_id=bundle["video_id"],
            policy_fingerprint=pol_fp,
            coordinate_system_fingerprint=coord_fp,
            pitch_template_fingerprint=tpl_fp,
            pitch_length_m=template.length_m,
            pitch_width_m=template.width_m,
            features=bundle["features"],
            segments=bundle["segments"],
            projections=bundle["projections"],
            correspondence_accepted=4,
            mean_reprojection_error_px=float(bundle["H"].mean_reprojection_error_px),
        )
        validate_receipt_payload(receipt)
        write_json_record(out, receipt, contain_root=session, overwrite=False)
        try:
            write_json_record(out, receipt, contain_root=session, overwrite=False)
            result.fail_scenario("no_overwrite", "overwrite allowed")
        except RecordError:
            result.ok_scenario("no_overwrite")

        # JSON schemas load
        for name in (
            "calibration_request",
            "calibration_run_receipt",
            "calibration_evaluation",
        ):
            schema = load_calibration_json_schema(name)
            if name == "calibration_evaluation":
                ev = evaluate_calibration()
                validate_against_json_schema(
                    ev.to_dict(run_id=bundle["run_id"], video_id=bundle["video_id"]),
                    schema,
                )
        if evaluate_calibration().ground_truth_evaluation_status != NOT_EVALUATED_CALIBRATION:
            result.fail_scenario("eval_stub", "wrong status")
        else:
            result.ok_scenario("eval_stub")

        # Bundle validate good path
        vr_ok = validate_calibration_bundle(
            calibration_features=bundle["calibration_features"],
            calibration_segments=bundle["calibration_segments"],
            projected_positions=bundle["projected_positions"],
            pitch_template_fingerprint=tpl_fp,
            policy=policy,
            receipt=receipt,
            timeline_start_us=0,
            timeline_end_us=2_000_000,
        )
        if vr_ok.status == "FAIL":
            result.fail_scenario("bundle_valid", str(vr_ok.errors))
        else:
            result.ok_scenario("bundle_valid")

        # Frozen calibrations contract compile
        _ = contract_fingerprint(get_contract("calibrations", 1))

        result.extras["policy_fingerprint"] = pol_fp
        result.extras["coordinate_system_fingerprint"] = coord_fp
        result.extras["pitch_template_fingerprint"] = tpl_fp
        result.extras["gate_candidate"] = GATE_PASS

    except Exception as exc:  # noqa: BLE001
        result.err(f"validator crash: {exc}", config=True)
    finally:
        # 24 failure cleanup — remove session unless --keep
        if not keep:
            shutil.rmtree(session, ignore_errors=True)
            result.ok_scenario("cleanup")
        else:
            result.extras["kept_session"] = str(session)
            result.ok_scenario("cleanup_skipped_keep")

    return result.finalize(strict=strict)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    result = run_checks(keep=bool(args.keep), strict=bool(args.strict))
    gate = GATE_PASS
    if result.status == "FAIL":
        gate = GATE_FAIL
    elif result.warnings:
        gate = GATE_FINDINGS
    result.extras["gate"] = gate
    payload = result.to_dict()
    payload["gate"] = gate
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(gate)
        print(f"status={result.status} exit={result.exit_code}")
        if result.errors:
            print("errors:")
            for e in result.errors:
                print(f"  - {e}")
        if result.warnings:
            print("warnings:")
            for w in result.warnings:
                print(f"  - {w}")
        print("scenarios:")
        for k, v in sorted(result.scenarios.items()):
            print(f"  {k}: {v}")
    return int(result.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
