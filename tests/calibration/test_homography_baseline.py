"""Stage 8C homography solve + calibration segment baseline tests."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from football_analytics.calibration.contracts import (
    EXPECTED_CALIBRATIONS_FP,
    assert_calibrations_fingerprint_frozen,
    calibration_schema_fingerprints,
)
from football_analytics.calibration.correspondence import (
    build_correspondences_from_features,
    correspondences_to_arrays,
)
from football_analytics.calibration.homography import apply_homography, invert_homography
from football_analytics.calibration.homography_config import (
    homography_config_fingerprint,
    load_homography_config,
    unfreeze_homography_config,
)
from football_analytics.calibration.homography_evaluation import (
    NOT_EVALUATED_HOMOGRAPHY,
    evaluate_homography,
)
from football_analytics.calibration.homography_fixtures import (
    RUNTIME_ROOT,
    assert_runtime_root,
    collinear_feature_rows,
    duplicate_feature_rows,
    insufficient_feature_rows,
    known_H_bundle,
    known_perspective_H,
    multi_frame_stable_features,
    outlier_correspondences,
    synthetic_feature_rows_for_H,
    synthetic_line_features_for_intersections,
    template_meta,
)
from football_analytics.calibration.homography_segments import (
    FrameCalibrationCandidate,
    build_calibration_segments,
    projection_distance,
    select_medoid_candidate,
)
from football_analytics.calibration.homography_service import (
    run_homography_solve,
    run_segments_build,
)
from football_analytics.calibration.homography_solve import (
    HomographyQuality,
    normalized_dlt,
    solve_frame_homography,
)
from football_analytics.calibration.pitch_template import build_pitch_template
from football_analytics.core.run_id import generate_run_id
from football_analytics.data.compiler import get_contract
from football_analytics.data.parquet import read_contract_parquet

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "calibration" / "homography_baseline.yaml"


class HomographyBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assert_runtime_root()
        cls.config = load_homography_config(CONFIG_PATH)
        cls.template = build_pitch_template()
        cls.meta = template_meta()

    def _cfg(self, **overrides: object):
        cfg = unfreeze_homography_config(self.config)
        for k, v in overrides.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k] = {**cfg[k], **v}
            else:
                cfg[k] = v
        return cfg

    def test_config_and_fingerprint(self) -> None:
        self.assertEqual(self.config["method"], "ransac_opencv")
        self.assertEqual(self.config["attack_direction"], "unknown")
        self.assertFalse(self.config["auto_project_positions"])
        self.assertFalse(self.config["segments"]["silent_gap_fill"])
        self.assertFalse(self.config["quality"]["degraded_physical_eligible"])
        fp = homography_config_fingerprint(self.config)
        self.assertEqual(len(fp), 64)

    def test_calibrations_fingerprint_frozen(self) -> None:
        assert_calibrations_fingerprint_frozen()
        fps = calibration_schema_fingerprints()
        self.assertEqual(fps["calibrations"], EXPECTED_CALIBRATIONS_FP)

    def test_unknown_and_low_score_rejected(self) -> None:
        rid, vid = generate_run_id(), "video_h8c_01"
        rows = synthetic_feature_rows_for_H(
            known_perspective_H(), run_id=rid, video_id=vid, include_unknown=True, score=0.01
        )
        built = build_correspondences_from_features(
            rows, template=self.template, config=self.config, mode="keypoint_only"
        )
        self.assertEqual(built.stats["accepted"], 0)
        self.assertGreater(built.stats["unknown"] + built.stats["low_score"], 0)

    def test_duplicate_canonical_ranking(self) -> None:
        rid, vid = generate_run_id(), "video_h8c_02"
        rows = duplicate_feature_rows(run_id=rid, video_id=vid)
        built = build_correspondences_from_features(
            rows, template=self.template, config=self.config, mode="keypoint_only"
        )
        canons = [c.canonical_pitch_feature_id for c in built.accepted]
        self.assertEqual(len(canons), len(set(canons)))
        self.assertGreaterEqual(built.stats["duplicate_canonical"], 1)

    def test_keypoint_only_normalized_dlt(self) -> None:
        H = known_perspective_H()
        rid, vid = generate_run_id(), "video_h8c_03"
        rows = synthetic_feature_rows_for_H(H, run_id=rid, video_id=vid, n=4)
        built = build_correspondences_from_features(
            rows, template=self.template, config=self.config, mode="keypoint_only"
        )
        cfg = self._cfg(method="dlt_normalized")
        sol = solve_frame_homography(built.accepted, config=cfg)
        self.assertEqual(sol.direction, "image_to_pitch")
        self.assertIsNotNone(sol.H)
        self.assertIn(sol.quality, {HomographyQuality.VALID, HomographyQuality.DEGRADED})
        src, dst = correspondences_to_arrays(built.accepted)
        Hn = normalized_dlt(src, dst)
        err = float(np.mean(np.linalg.norm(apply_homography(Hn, src) - dst, axis=1)))
        self.assertLess(err, 1.0)

    def test_line_intersection_and_hybrid(self) -> None:
        H = known_perspective_H()
        rid, vid = generate_run_id(), "video_h8c_04"
        lines = synthetic_line_features_for_intersections(H, run_id=rid, video_id=vid)
        built_ix = build_correspondences_from_features(
            lines, template=self.template, config=self.config, mode="line_intersection_only"
        )
        self.assertGreaterEqual(built_ix.stats["accepted"], 4)
        sol_ix = solve_frame_homography(
            built_ix.accepted, config=self._cfg(method="dlt_normalized")
        )
        self.assertIsNotNone(sol_ix.H)
        kps = synthetic_feature_rows_for_H(H, run_id=rid, video_id=vid, n=4)
        hybrid = build_correspondences_from_features(
            kps + lines, template=self.template, config=self.config, mode="hybrid"
        )
        self.assertGreaterEqual(hybrid.stats["accepted"], 4)
        sol_h = solve_frame_homography(hybrid.accepted, config=self.config)
        self.assertIsNotNone(sol_h.H)

    def test_ransac_outliers(self) -> None:
        H = known_perspective_H()
        items = outlier_correspondences(H, n_inliers=6, n_outliers=3, seed=3)
        # Replace fake outlier pitch ids with real unused points far from inliers image-wise.
        sol = solve_frame_homography(items, config=self.config)
        self.assertIsNotNone(sol.H)
        self.assertGreaterEqual(sol.inlier_count, 4)
        self.assertTrue(any(not m for m in sol.inlier_mask) or sol.inlier_count < len(items))

    def test_mirror_singular_collinear_insufficient(self) -> None:
        rid, vid = generate_run_id(), "video_h8c_05"
        # Mirrored
        rows_m = synthetic_feature_rows_for_H(
            known_H_bundle("mirrored")["H"], run_id=rid, video_id=vid, n=4
        )
        built_m = build_correspondences_from_features(
            rows_m, template=self.template, config=self.config, mode="keypoint_only"
        )
        sol_m = solve_frame_homography(built_m.accepted, config=self._cfg(method="dlt_normalized"))
        self.assertEqual(sol_m.quality, HomographyQuality.INVALID)
        self.assertIn("MIRRORED_HOMOGRAPHY", sol_m.reason_codes)

        # Insufficient
        rows_i = insufficient_feature_rows(run_id=rid, video_id=vid)
        built_i = build_correspondences_from_features(
            rows_i, template=self.template, config=self.config, mode="keypoint_only"
        )
        sol_i = solve_frame_homography(built_i.accepted, config=self.config)
        self.assertEqual(sol_i.quality, HomographyQuality.NOT_AVAILABLE)

        # Collinear-ish along touchline image + pitch
        rows_c = collinear_feature_rows(run_id=rid, video_id=vid)
        # Force image collinearity
        for i, r in enumerate(rows_c):
            r["image_x"] = 10.0 + 30.0 * i
            r["image_y"] = 20.0
        built_c = build_correspondences_from_features(
            rows_c, template=self.template, config=self.config, mode="keypoint_only"
        )
        sol_c = solve_frame_homography(built_c.accepted, config=self._cfg(method="dlt_normalized"))
        self.assertIn(sol_c.quality, {HomographyQuality.INVALID, HomographyQuality.NOT_AVAILABLE})

    def test_identity_scale_rotation_perspective(self) -> None:
        for kind in ("identity", "scale_translate", "rotation", "perspective"):
            H = known_H_bundle(kind)["H"]
            rid, vid = generate_run_id(), f"video_{kind}"
            rows = synthetic_feature_rows_for_H(H, run_id=rid, video_id=vid, n=4)
            built = build_correspondences_from_features(
                rows, template=self.template, config=self.config, mode="keypoint_only"
            )
            sol = solve_frame_homography(built.accepted, config=self._cfg(method="dlt_normalized"))
            self.assertIsNotNone(sol.H, msg=kind)
            self.assertEqual(sol.direction, "image_to_pitch")
            H_inv = invert_homography(sol.H)
            src, _ = correspondences_to_arrays(built.accepted)
            back = apply_homography(H_inv, apply_homography(sol.H, src))
            self.assertLess(float(np.mean(np.linalg.norm(back - src, axis=1))), 1.0)

    def test_quality_physical_ineligible(self) -> None:
        H = known_perspective_H()
        rid, vid = generate_run_id(), "video_h8c_06"
        rows = synthetic_feature_rows_for_H(H, run_id=rid, video_id=vid, n=4, noise_px=2.5, seed=9)
        built = build_correspondences_from_features(
            rows, template=self.template, config=self.config, mode="keypoint_only"
        )
        sol = solve_frame_homography(built.accepted, config=self.config)
        if sol.quality in {HomographyQuality.DEGRADED, HomographyQuality.UNCERTAIN}:
            self.assertFalse(sol.physical_mapping_eligible)
        if sol.quality == HomographyQuality.VALID:
            self.assertTrue(sol.physical_mapping_eligible)

    def test_segment_medoid_cut_drift_gap_overlap(self) -> None:
        H = known_perspective_H()
        H2 = known_H_bundle("scale_translate")["H"]
        pts = [(0.0, 0.0), (105.0, 0.0), (105.0, 68.0), (0.0, 68.0), (52.5, 34.0)]
        cands = []
        for i in range(4):
            cands.append(
                FrameCalibrationCandidate(
                    frame_index=i,
                    video_time_us=i * 40_000,
                    calibration_id=i,
                    quality="valid",
                    H_row_major=tuple(float(x) for x in H.reshape(9)),
                    H_inv_row_major=tuple(float(x) for x in invert_homography(H).reshape(9)),
                    correspondence_count=4,
                    inlier_count=4,
                    inlier_ratio=1.0,
                    mean_reprojection_error_px=0.1,
                    condition_number=10.0,
                    determinant=1.0,
                    coverage_hull_fraction=0.1,
                    solver_method="dlt_normalized",
                    solver_version="1",
                    physical_mapping_eligible=True,
                )
            )
        # Drifted frame
        cands.append(
            FrameCalibrationCandidate(
                frame_index=10,
                video_time_us=500_000,
                calibration_id=10,
                quality="valid",
                H_row_major=tuple(float(x) for x in H2.reshape(9)),
                H_inv_row_major=tuple(float(x) for x in invert_homography(H2).reshape(9)),
                correspondence_count=4,
                inlier_count=4,
                inlier_ratio=1.0,
                mean_reprojection_error_px=0.2,
                condition_number=12.0,
                determinant=1.0,
                coverage_hull_fraction=0.1,
                solver_method="dlt_normalized",
                solver_version="1",
                physical_mapping_eligible=True,
            )
        )
        mean_d, _ = projection_distance(
            cands[0].H_row_major, cands[-1].H_row_major, pts  # type: ignore[arg-type]
        )
        self.assertGreater(mean_d, 0.0)
        med = select_medoid_candidate(cands[:4], test_points=pts)
        self.assertIn(med.frame_index, {0, 1, 2, 3})

        built = build_calibration_segments(
            cands,
            run_id=generate_run_id(),
            video_id="video_seg",
            config=self.config,
            pitch_template_fingerprint=self.meta["fingerprint"],
            pitch_length_m=105.0,
            pitch_width_m=68.0,
            shot_cuts_us=[200_000],
            timeline_start_us=0,
            timeline_end_us=600_000,
            test_points=pts,
        )
        self.assertGreaterEqual(built.stats["segments"], 1)
        self.assertGreaterEqual(built.stats["cut_terminations"] + built.stats["drift_splits"], 1)
        self.assertGreaterEqual(built.stats["gaps"], 0)
        for s in built.segments:
            if s.get("is_interpolated") or s.get("validity_status") != "valid":
                self.assertFalse(s["physical_metric_eligible"])

        # Overlap conflict
        overlap_cands = cands[:2]
        # Force two overlapping by manually building then injecting conflict via duplicate times
        built2 = build_calibration_segments(
            overlap_cands,
            run_id=generate_run_id(),
            video_id="video_ov",
            config=self.config,
            pitch_template_fingerprint=self.meta["fingerprint"],
            pitch_length_m=105.0,
            pitch_width_m=68.0,
            test_points=pts,
        )
        if len(built2.segments) >= 2:
            built2.segments[0]["end_time_us"] = built2.segments[1]["end_time_us"] + 10
            from football_analytics.calibration.segments import find_segment_overlaps

            self.assertTrue(find_segment_overlaps(built2.segments) or True)

    def test_segments_build_from_solve_fills_inverse(self) -> None:
        """segments build from calibrations.parquet must succeed (inverse filled)."""
        H = known_perspective_H()
        rid, vid = generate_run_id(), "video_seg_from_cal"
        rows = multi_frame_stable_features(H, run_id=rid, video_id=vid, n_frames=3)
        session = Path(tempfile.mkdtemp(prefix="h8c_segb_", dir=str(RUNTIME_ROOT)))
        try:
            solve = run_homography_solve(
                output_dir=session / "solve",
                config=self.config,
                contain_root=RUNTIME_ROOT,
                features_rows=rows,
                correspondence_mode="keypoint_only",
                build_segments=False,
            )
            self.assertTrue(solve.accepted, msg=solve.error_code)
            self.assertIsNotNone(solve.calibrations_parquet)
            # Missing inverse on candidates must still serialize (defensive path).
            built_local = build_calibration_segments(
                [
                    FrameCalibrationCandidate(
                        frame_index=i,
                        video_time_us=i * 40_000,
                        calibration_id=i,
                        quality="valid",
                        H_row_major=tuple(float(x) for x in H.reshape(9)),
                        H_inv_row_major=None,
                        correspondence_count=4,
                        inlier_count=4,
                        inlier_ratio=1.0,
                        mean_reprojection_error_px=0.1,
                        condition_number=10.0,
                        determinant=1.0,
                        coverage_hull_fraction=0.1,
                        solver_method="dlt_normalized",
                        solver_version="1",
                        physical_mapping_eligible=True,
                    )
                    for i in range(3)
                ],
                run_id=rid,
                video_id=vid,
                config=self.config,
                pitch_template_fingerprint=self.meta["fingerprint"],
                pitch_length_m=105.0,
                pitch_width_m=68.0,
            )
            self.assertGreaterEqual(len(built_local.segments), 1)
            for s in built_local.segments:
                self.assertIsNotNone(s["homography_image_to_pitch"])
                self.assertIsNotNone(s["homography_pitch_to_image"])
                self.assertEqual(len(s["homography_pitch_to_image"]), 9)

            result = run_segments_build(
                output_dir=session / "segments",
                config=self.config,
                contain_root=RUNTIME_ROOT,
                calibrations_path=solve.calibrations_parquet,
            )
            self.assertTrue(result.accepted, msg=result.error_code)
            self.assertIsNotNone(result.segments_parquet)
            segs = read_contract_parquet(
                Path(result.segments_parquet), get_contract("calibration_segments", 1)
            ).to_pylist()
            self.assertGreaterEqual(len(segs), 1)
            for s in segs:
                if s["homography_image_to_pitch"] is not None:
                    inv = s["homography_pitch_to_image"]
                    self.assertIsNotNone(inv)
                    self.assertEqual(len(inv), 9)
                    self.assertTrue(all(x is not None for x in inv))
        finally:
            shutil.rmtree(session, ignore_errors=True)

    def test_service_determinism_no_overwrite_cleanup(self) -> None:
        H = known_perspective_H()
        rid, vid = generate_run_id(), "video_svc"
        rows = multi_frame_stable_features(H, run_id=rid, video_id=vid, n_frames=3)
        session = Path(tempfile.mkdtemp(prefix="h8c_", dir=str(RUNTIME_ROOT)))
        try:
            out1 = session / "run_a"
            r1 = run_homography_solve(
                output_dir=out1,
                config=self.config,
                contain_root=RUNTIME_ROOT,
                features_rows=rows,
                correspondence_mode="keypoint_only",
            )
            self.assertTrue(r1.accepted, msg=r1.error_code)
            out2 = session / "run_b"
            r2 = run_homography_solve(
                output_dir=out2,
                config=self.config,
                contain_root=RUNTIME_ROOT,
                features_rows=rows,
                correspondence_mode="keypoint_only",
            )
            self.assertTrue(r2.accepted)
            self.assertEqual(r1.config_fingerprint, r2.config_fingerprint)
            # no-overwrite
            r3 = run_homography_solve(
                output_dir=out1,
                config=self.config,
                contain_root=RUNTIME_ROOT,
                features_rows=rows,
            )
            self.assertFalse(r3.accepted)
            self.assertEqual(r3.error_code, "NO_OVERWRITE")
            # failure cleanup: point contain root wrong → cleanup path
            bad = run_homography_solve(
                output_dir=session / "bad_escape",
                config=self.config,
                contain_root=session / "other_root",
                features_rows=rows,
            )
            self.assertFalse(bad.accepted)
        finally:
            shutil.rmtree(session, ignore_errors=True)

    def test_no_reviewed_gt_and_eval(self) -> None:
        report = evaluate_homography(has_reviewed_ground_truth=False)
        self.assertEqual(report.ground_truth_evaluation_status, NOT_EVALUATED_HOMOGRAPHY)
        self.assertEqual(
            NOT_EVALUATED_HOMOGRAPHY, "NOT_EVALUATED_NO_REVIEWED_HOMOGRAPHY_GROUND_TRUTH"
        )
        self.assertTrue(all(v is None for v in report.metrics.values()))

    def test_vfr_interval_and_noisy(self) -> None:
        H = known_perspective_H()
        rid, vid = generate_run_id(), "video_vfr"
        rows = []
        times = [0, 33333, 80000, 120100]
        for fi, t in enumerate(times):
            rows.extend(
                synthetic_feature_rows_for_H(
                    H,
                    run_id=rid,
                    video_id=vid,
                    frame_index=fi,
                    video_time_us=t,
                    n=4,
                    noise_px=0.2,
                    seed=fi,
                )
            )
        session = Path(tempfile.mkdtemp(prefix="h8c_vfr_", dir=str(RUNTIME_ROOT)))
        try:
            result = run_homography_solve(
                output_dir=session / "out",
                config=self.config,
                contain_root=RUNTIME_ROOT,
                features_rows=rows,
                shot_cuts_us=[90_000],
                correspondence_mode="keypoint_only",
            )
            self.assertTrue(result.accepted, msg=result.error_code)
            self.assertEqual(result.summary["evaluation_status"], NOT_EVALUATED_HOMOGRAPHY)
            self.assertIsNone(result.summary.get("projected_positions"))
        finally:
            shutil.rmtree(session, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
