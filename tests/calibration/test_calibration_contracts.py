#!/usr/bin/env python3
"""Stage 8A calibration / homography / coordinate contract tests (§14)."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from football_analytics.calibration.contracts import (
    EXPECTED_CALIBRATIONS_FP,
    EXPECTED_REGISTRY_CONTRACT_COUNT,
    assert_calibration_contracts_registered,
    assert_calibrations_fingerprint_frozen,
    calibration_schema_fingerprints,
    compile_calibration_schemas,
    load_calibration_json_schema,
)
from football_analytics.calibration.coordinates import (
    ball_centre_from_bbox,
    default_attack_direction,
    human_footpoint_from_bbox,
)
from football_analytics.calibration.evaluation import (
    NOT_EVALUATED_CALIBRATION,
    evaluate_calibration,
)
from football_analytics.calibration.fixtures import (
    correspondences_for_H,
    e2e_bundle,
    gap_segments_bundle,
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
from football_analytics.data.registry import (
    default_project_root,
    default_registry_path,
    load_schema_registry,
)


class CalibrationContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = default_project_root()
        self.reg = load_schema_registry(default_registry_path(), project_root=self.root)

    def test_01_registry_count_and_frozen_calibrations(self) -> None:
        assert_calibration_contracts_registered(registry=self.reg)
        self.assertEqual(len(list_contracts(registry=self.reg)), EXPECTED_REGISTRY_CONTRACT_COUNT)
        assert_calibrations_fingerprint_frozen(registry=self.reg)
        fp = contract_fingerprint(get_contract("calibrations", 1, registry=self.reg))
        self.assertEqual(fp, EXPECTED_CALIBRATIONS_FP)
        fps = calibration_schema_fingerprints(registry=self.reg)
        self.assertEqual(fps["calibrations"], EXPECTED_CALIBRATIONS_FP)
        compile_calibration_schemas(registry=self.reg)

    def test_02_pitch_template_and_coords(self) -> None:
        t = build_pitch_template()
        self.assertEqual(t.length_m, 105.0)
        self.assertEqual(t.width_m, 68.0)
        validate_fifa_range(t.length_m, t.width_m)
        self.assertEqual(len(pitch_template_fingerprint(t)), 64)
        self.assertEqual(default_attack_direction(), "unknown")
        coords = load_coordinate_system()
        self.assertEqual(coords["image"]["origin"], "top_left")
        self.assertEqual(coords["image"]["y_axis"], "down")
        self.assertEqual(coords["attack_direction"]["default"], "unknown")

    def test_03_identity_scale_perspective_rotation(self) -> None:
        pitch = [(0.0, 0.0), (100.0, 0.0), (100.0, 50.0), (0.0, 50.0)]
        r = solve_homography(pitch, pitch)
        self.assertEqual(r.status, "valid")

        H_st = scale_translate_homography(scale_x=0.5, scale_y=0.4, tx=10.0, ty=5.0)
        img, pit = correspondences_for_H(H_st, n=4)
        self.assertEqual(solve_homography(img, pit).status, "valid")

        img, pit = correspondences_for_H(known_perspective_H(), n=4)
        solved = solve_homography(img, pit)
        self.assertEqual(solved.status, "valid")
        back = apply_homography(solved.H_inv, apply_homography(solved.H, img))
        err = float(np.mean(np.linalg.norm(back - np.asarray(img), axis=1)))
        self.assertLess(err, 1.0)

        img, pit = correspondences_for_H(rotation_homography(0.25), n=4)
        self.assertEqual(solve_homography(img, pit).status, "valid")

    def test_04_negatives_collinear_dup_singular_ill_mirror_insufficient(self) -> None:
        with self.assertRaises(HomographyError):
            solve_homography(
                [(0, 0), (1, 1), (2, 2), (3, 3)],
                [(0, 0), (10, 0), (20, 0), (30, 0)],
            )
        with self.assertRaises(HomographyError):
            solve_homography(
                [(0, 0), (0, 0), (10, 0), (0, 10)],
                [(0, 0), (1, 0), (10, 0), (0, 10)],
            )
        with self.assertRaises(HomographyError):
            validate_homography_matrix(singular_matrix_row_major())
        with self.assertRaises(HomographyError):
            validate_homography_matrix(ill_conditioned_matrix_row_major())
        with self.assertRaises(HomographyError):
            validate_homography_matrix(list(mirrored_homography().reshape(9)))
        with self.assertRaises(HomographyError):
            solve_homography([(0, 0), (1, 0), (0, 1)], [(0, 0), (1, 0), (0, 1)])

    def test_05_high_reprojection(self) -> None:
        img, pitch = correspondences_for_H(known_perspective_H(), n=4)
        pitch_bad = list(pitch)
        pitch_bad[0] = (pitch_bad[0][0] + 80.0, pitch_bad[0][1] + 80.0)
        with self.assertRaises(HomographyError):
            solve_homography(img, pitch_bad, max_mean_reprojection_error_px=1.0)

    def test_06_projection_human_ball_predicted(self) -> None:
        fp = human_footpoint_from_bbox((10.0, 20.0, 30.0, 50.0))
        self.assertEqual(fp.x_px, 20.0)
        self.assertEqual(fp.y_px, 50.0)
        bc = ball_centre_from_bbox((10.0, 20.0, 30.0, 40.0))
        self.assertEqual(bc.x_px, 20.0)
        self.assertEqual(bc.y_px, 30.0)

        bundle = e2e_bundle()
        Hrm = bundle["H"].matrix_row_major()
        human = next(p for p in bundle["projections"] if p["entity_type"] == "human")
        self.assertEqual(human["source_point_type"], "bbox_bottom_centre")
        ball = next(p for p in bundle["projections"] if p["entity_type"] == "ball")
        self.assertEqual(ball["source_point_type"], "bbox_centre")
        pred = projected_from_track(
            bundle["run_id"],
            bundle["video_id"],
            entity_type="human",
            bbox=(100.0, 100.0, 140.0, 200.0),
            H_row_major=Hrm,
            observation_source="predicted",
            projection_id="p_pred",
        )
        self.assertNotEqual(pred["physical_metric_eligibility"], "eligible")

    def test_07_segments_shot_cut_overlap_gap_vfr(self) -> None:
        bundle = e2e_bundle()
        cut = terminate_on_shot_cut(segment=bundle["segments"][0], cut_time_us=250_000)
        self.assertEqual(cut["end_time_us"], 250_000)
        self.assertEqual(cut["boundary_reason"], "SHOT_CUT_TERMINATE")
        overs = overlapping_segments_bundle(bundle["run_id"], bundle["video_id"])
        self.assertTrue(find_segment_overlaps(overs))
        gaps = find_calibration_gaps(
            gap_segments_bundle(bundle["run_id"], bundle["video_id"]),
            timeline_start_us=0,
            timeline_end_us=2_000_000,
        )
        self.assertTrue(gaps)
        seg = bundle["segments"][0]
        self.assertLess(seg["start_time_us"], seg["end_time_us"])

    def test_08_template_mismatch_and_bundle(self) -> None:
        policy = load_calibration_policy()
        bundle = e2e_bundle()
        tpl_fp = bundle["template_fp"]
        bad = dict(bundle["segments"][0])
        bad["pitch_template_fingerprint"] = "c" * 64
        vr = validate_calibration_bundle(
            calibration_features=bundle["features"],
            calibration_segments=[bad],
            projected_positions=bundle["projections"],
            pitch_template_fingerprint=tpl_fp,
            policy=policy,
        )
        self.assertEqual(vr.status, "FAIL")
        vr_ok = validate_calibration_bundle(
            calibration_features=bundle["calibration_features"],
            calibration_segments=bundle["calibration_segments"],
            projected_positions=bundle["projected_positions"],
            pitch_template_fingerprint=tpl_fp,
            policy=policy,
        )
        self.assertNotEqual(vr_ok.status, "FAIL")

    def test_09_receipt_eval_no_overwrite(self) -> None:
        policy = load_calibration_policy()
        coords = load_coordinate_system()
        bundle = e2e_bundle()
        pol_fp = policy_fingerprint(policy)
        from football_analytics.calibration.policy import coordinate_system_fingerprint

        coord_fp = coordinate_system_fingerprint(coords)
        req = build_synthetic_request(
            run_id=bundle["run_id"],
            video_id=bundle["video_id"],
            policy_fingerprint=pol_fp,
            coordinate_system_fingerprint=coord_fp,
            pitch_template_fingerprint=bundle["template_fp"],
        )
        validate_request_payload(req)
        receipt = build_synthetic_receipt(
            run_id=bundle["run_id"],
            video_id=bundle["video_id"],
            policy_fingerprint=pol_fp,
            coordinate_system_fingerprint=coord_fp,
            pitch_template_fingerprint=bundle["template_fp"],
            pitch_length_m=105.0,
            pitch_width_m=68.0,
            features=bundle["features"],
            segments=bundle["segments"],
            projections=bundle["projections"],
        )
        validate_receipt_payload(receipt)
        self.assertEqual(receipt["ground_truth_evaluation_status"], NOT_EVALUATED_CALIBRATION)
        self.assertEqual(
            evaluate_calibration().ground_truth_evaluation_status, NOT_EVALUATED_CALIBRATION
        )
        for name in (
            "calibration_request",
            "calibration_run_receipt",
            "calibration_evaluation",
        ):
            self.assertTrue(isinstance(load_calibration_json_schema(name), dict))

        import tempfile

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "r.json"
            write_json_record(path, receipt, contain_root=root, overwrite=False)
            with self.assertRaises(RecordError):
                write_json_record(path, receipt, contain_root=root, overwrite=False)

    def test_10_policy_safety(self) -> None:
        policy = load_calibration_policy()
        self.assertTrue(policy["safety"]["no_sv_kp_inference"])
        self.assertTrue(policy["safety"]["no_sv_lines_inference"])
        self.assertFalse(policy["segments"]["silent_gap_fill"])
        self.assertEqual(policy["attack_direction"]["default"], "unknown")


if __name__ == "__main__":
    unittest.main()
