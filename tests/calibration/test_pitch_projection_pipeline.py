"""Stage 8D pitch projection pipeline + Stage 8 close tests."""

from __future__ import annotations

import json
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
from football_analytics.calibration.homography import apply_homography, invert_homography
from football_analytics.calibration.pitch_projection import (
    apply_image_to_pitch_projection,
    build_projection_for_observation,
    extract_source_point,
    select_segment_for_time,
    target_customer_metric_eligible,
)
from football_analytics.calibration.pitch_projection_config import (
    load_pitch_projection_config,
    pitch_projection_config_fingerprint,
    unfreeze_pitch_projection_config,
)
from football_analytics.calibration.pitch_projection_evaluation import (
    NOT_EVALUATED_PROJECTED_POS,
    evaluate_pitch_projection,
)
from football_analytics.calibration.pitch_projection_fixtures import (
    RUNTIME_ROOT,
    assert_runtime_root,
    ball_bbox_for_centre,
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
from football_analytics.data.compiler import get_contract
from football_analytics.data.parquet import read_contract_parquet

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "calibration" / "pitch_projection_pipeline.yaml"


class PitchProjectionPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assert_runtime_root()
        cls.config = load_pitch_projection_config(CONFIG_PATH)
        cls.template = build_pitch_template()
        cls.t_fp = pitch_template_fingerprint(cls.template)

    def _cfg(self, **overrides: object):
        cfg = unfreeze_pitch_projection_config(self.config)
        for k, v in overrides.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k] = {**cfg[k], **v}
            else:
                cfg[k] = v
        return cfg

    def test_config_and_fingerprint(self) -> None:
        self.assertEqual(self.config["direction"], "image_to_pitch")
        self.assertEqual(self.config["attack_direction"], "unknown")
        self.assertFalse(self.config["compute_physical_metrics"])
        self.assertFalse(self.config["ball_source"]["physical_metric_eligible"])
        self.assertFalse(self.config["ball_source"]["event_metric_eligible"])
        fp = pitch_projection_config_fingerprint(self.config)
        self.assertEqual(len(fp), 64)

    def test_calibrations_and_projected_fp_pins(self) -> None:
        assert_calibrations_fingerprint_frozen()
        fps = calibration_schema_fingerprints()
        self.assertEqual(fps["calibrations"], EXPECTED_CALIBRATIONS_FP)
        self.assertEqual(
            fps["projected_positions"],
            "1860638e13c101a2c4b52ecb86e31c264f1b09ee8f255b3813fb9da8325055ba",
        )

    def test_human_footpoint_and_ball_centre(self) -> None:
        human = extract_source_point(
            entity_type="human",
            bbox_xyxy=(100.0, 100.0, 140.0, 200.0),
            observation_source="detection_associated",
            config=self.config,
            frame_width=1280.0,
            frame_height=720.0,
        )
        self.assertTrue(human.ok)
        self.assertEqual(human.source_point_type, "bbox_bottom_centre")
        self.assertAlmostEqual(human.image_x or 0.0, 120.0)
        self.assertAlmostEqual(human.image_y or 0.0, 200.0)

        ball = extract_source_point(
            entity_type="ball",
            bbox_xyxy=(50.0, 50.0, 58.0, 58.0),
            observation_source="detection_associated",
            config=self.config,
        )
        self.assertTrue(ball.ok)
        self.assertEqual(ball.source_point_type, "bbox_centre")
        self.assertAlmostEqual(ball.image_x or 0.0, 54.0)
        self.assertAlmostEqual(ball.image_y or 0.0, 54.0)

    def test_identity_and_perspective_projection(self) -> None:
        for kind, H in (("identity", identity_H()), ("perspective", perspective_H())):
            with self.subTest(kind=kind):
                bundle = base_bundle(H=H, video_id=f"video_{kind}")
                session = Path(tempfile.mkdtemp(prefix=f"proj_{kind}_", dir=RUNTIME_ROOT))
                try:
                    svc = run_pitch_projection(
                        output_dir=session / "out",
                        config=self.config,
                        contain_root=Path(RUNTIME_ROOT),
                        observations_rows=bundle["observations"],
                        segments_rows=bundle["segments"],
                        frame_times=bundle["frame_times"],
                        coverage_hulls=bundle["coverage_hulls"],
                        eligibility_timeline=bundle["eligibility_timeline"],
                        analysis_windows=bundle["analysis_windows"],
                        fingerprints=bundle["fingerprints"],
                        frame_width=1280.0,
                        frame_height=720.0,
                    )
                    self.assertTrue(svc.accepted, svc.error_code)
                    human = next(
                        p for p in svc.summary["projections"] if p["entity_type"] == "human"
                    )
                    self.assertEqual(human["mapping_status"], "mapped")
                    self.assertEqual(human["physical_metric_eligibility"], "eligible")
                    # Footpoint should land near intended pitch coords.
                    self.assertIsNotNone(human["pitch_x_m"])
                    self.assertLess(abs(float(human["pitch_x_m"]) - 52.5), 2.0)
                finally:
                    shutil.rmtree(session, ignore_errors=True)

    def test_ball_eligibility_always_zero(self) -> None:
        bundle = base_bundle()
        session = Path(tempfile.mkdtemp(prefix="proj_ball_", dir=RUNTIME_ROOT))
        try:
            svc = run_pitch_projection(
                output_dir=session / "out",
                config=self.config,
                contain_root=Path(RUNTIME_ROOT),
                observations_rows=bundle["observations"],
                segments_rows=bundle["segments"],
                frame_times=bundle["frame_times"],
                coverage_hulls=bundle["coverage_hulls"],
                eligibility_timeline=bundle["eligibility_timeline"],
                analysis_windows=bundle["analysis_windows"],
                fingerprints=bundle["fingerprints"],
            )
            self.assertTrue(svc.accepted, svc.error_code)
            receipt = json.loads(Path(str(svc.receipt_json)).read_text(encoding="utf-8"))
            self.assertEqual(receipt["ball_physical_metric_eligible_count"], 0)
            self.assertEqual(receipt["ball_event_metric_eligible_count"], 0)
            for p in svc.summary["projections"]:
                if p["entity_type"] == "ball":
                    self.assertNotEqual(p["physical_metric_eligibility"], "eligible")
                    prov = json.loads(p["provenance_json"])
                    self.assertFalse(prov["event_metric_eligible"])
                    self.assertEqual(prov["airborne_status"], "unknown")
        finally:
            shutil.rmtree(session, ignore_errors=True)

    def test_segment_boundary_gap_and_conflict(self) -> None:
        rid, vid = generate_run_id(), "video_bound"
        H = perspective_H()
        seg = make_segment(
            run_id=rid,
            video_id=vid,
            segment_id="seg0",
            calibration_id=1,
            start_time_us=0,
            end_time_us=100_000,
            H=H,
            template_fp=self.t_fp,
        )
        # Boundary: end exclusive → time==end is gap
        sel_end = select_segment_for_time(
            [seg],
            run_id=rid,
            video_id=vid,
            video_time_us=100_000,
            config=self.config,
            pitch_template_fingerprint=self.t_fp,
        )
        self.assertEqual(sel_end.status, "not_calibrated")
        sel_ok = select_segment_for_time(
            [seg],
            run_id=rid,
            video_id=vid,
            video_time_us=99_999,
            config=self.config,
            pitch_template_fingerprint=self.t_fp,
        )
        self.assertEqual(sel_ok.status, "ok")

        seg2 = make_segment(
            run_id=rid,
            video_id=vid,
            segment_id="seg1",
            calibration_id=2,
            start_time_us=50_000,
            end_time_us=150_000,
            H=H,
            template_fp=self.t_fp,
        )
        sel_c = select_segment_for_time(
            [seg, seg2],
            run_id=rid,
            video_id=vid,
            video_time_us=60_000,
            config=self.config,
            pitch_template_fingerprint=self.t_fp,
        )
        self.assertEqual(sel_c.status, "conflict")

    def test_degraded_uncertain_not_used(self) -> None:
        rid, vid = generate_run_id(), "video_deg"
        H = perspective_H()
        seg = make_segment(
            run_id=rid,
            video_id=vid,
            segment_id="seg_deg",
            calibration_id=1,
            start_time_us=0,
            end_time_us=1_000_000,
            H=H,
            validity_status="uncertain",
            physical_metric_eligible=False,
            template_fp=self.t_fp,
        )
        sel = select_segment_for_time(
            [seg],
            run_id=rid,
            video_id=vid,
            video_time_us=10_000,
            config=self.config,
            pitch_template_fingerprint=self.t_fp,
        )
        self.assertEqual(sel.status, "not_calibrated")

    def test_predicted_human_and_ball_not_eligible(self) -> None:
        bundle = base_bundle()
        H = bundle["H"]
        rid, vid = bundle["run_id"], bundle["video_id"]
        for entity, tid, bbox in (
            ("human", 1, human_bbox_for_footpoint(H, 52.5, 34.0)),
            ("ball", 10, ball_bbox_for_centre(H, 60.0, 30.0)),
        ):
            with self.subTest(entity=entity):
                obs = obs_row(
                    run_id=rid,
                    video_id=vid,
                    frame_index=0,
                    track_id=tid,
                    entity_type=entity,
                    bbox=bbox,
                    observation_state="predicted",
                    detection_id=None,
                )
                row = build_projection_for_observation(
                    observation=obs,
                    segments=bundle["segments"],
                    config=self.config,
                    run_id=rid,
                    video_id=vid,
                    video_time_us=0,
                    projection_id=f"proj_{entity}_pred",
                    pitch_template_fingerprint=bundle["pitch_template_fingerprint"],
                    coverage_hulls=bundle["coverage_hulls"],
                    eligibility_timeline=bundle["eligibility_timeline"],
                    analysis_windows=bundle["analysis_windows"],
                )
                self.assertNotEqual(row["physical_metric_eligibility"], "eligible")

    def test_target_confirmed_provisional_revoked(self) -> None:
        bundle = base_bundle()
        # Confirmed track 1
        ok, _ = target_customer_metric_eligible(
            track_id=1,
            frame_index=0,
            eligibility_timeline=bundle["eligibility_timeline"],
            human_physical_eligible=True,
            config=self.config,
        )
        self.assertTrue(ok)
        # Provisional track 2
        ok_p, reasons_p = target_customer_metric_eligible(
            track_id=2,
            frame_index=0,
            eligibility_timeline=bundle["eligibility_timeline"],
            human_physical_eligible=True,
            config=self.config,
        )
        self.assertFalse(ok_p)
        self.assertTrue(any("PROVISIONAL" in r for r in reasons_p))
        # Revoked track 3
        ok_r, _ = target_customer_metric_eligible(
            track_id=3,
            frame_index=0,
            eligibility_timeline=bundle["eligibility_timeline"],
            human_physical_eligible=True,
            config=self.config,
        )
        self.assertFalse(ok_r)

    def test_outside_pitch_and_coverage_extrapolation(self) -> None:
        bundle = base_bundle()
        H = bundle["H"]
        rid, vid = bundle["run_id"], bundle["video_id"]
        # Outside: far image point
        obs_out = obs_row(
            run_id=rid,
            video_id=vid,
            frame_index=0,
            track_id=1,
            entity_type="human",
            bbox=(-400.0, -400.0, -360.0, -320.0),
        )
        row_out = build_projection_for_observation(
            observation=obs_out,
            segments=bundle["segments"],
            config=self.config,
            run_id=rid,
            video_id=vid,
            video_time_us=0,
            projection_id="proj_out",
            pitch_template_fingerprint=bundle["pitch_template_fingerprint"],
            coverage_hulls=bundle["coverage_hulls"],
            analysis_windows=bundle["analysis_windows"],
        )
        self.assertIn(row_out["mapping_status"], {"outside_pitch", "extrapolated", "failed"})
        self.assertNotEqual(row_out["physical_metric_eligibility"], "eligible")

        # Coverage hull extrapolation: point inside pitch but outside hull
        hull = coverage_hull_for_H(
            H,
            pitch_pts=[(40.0, 25.0), (50.0, 25.0), (50.0, 35.0), (40.0, 35.0)],
        )
        # Place footpoint at pitch (90, 60) — likely outside small hull
        obs_ex = obs_row(
            run_id=rid,
            video_id=vid,
            frame_index=0,
            track_id=1,
            entity_type="human",
            bbox=human_bbox_for_footpoint(H, 90.0, 60.0),
        )
        row_ex = build_projection_for_observation(
            observation=obs_ex,
            segments=bundle["segments"],
            config=self.config,
            run_id=rid,
            video_id=vid,
            video_time_us=0,
            projection_id="proj_ex",
            pitch_template_fingerprint=bundle["pitch_template_fingerprint"],
            coverage_hulls={"seg_main": hull},
            analysis_windows=bundle["analysis_windows"],
        )
        self.assertEqual(row_ex["mapping_status"], "extrapolated")
        self.assertTrue(row_ex["is_extrapolated"])
        self.assertNotEqual(row_ex["physical_metric_eligibility"], "eligible")

    def test_truncated_bbox_and_singular_w(self) -> None:
        src = extract_source_point(
            entity_type="human",
            bbox_xyxy=(0.0, 640.0, 30.0, 720.0),
            observation_source="detection_associated",
            config=self.config,
            frame_width=1280.0,
            frame_height=720.0,
        )
        self.assertTrue(src.truncated)

        Hs = singular_w_H()
        geom = apply_image_to_pitch_projection(
            image_x=50.0,
            image_y=50.0,
            H_row_major=[float(x) for x in Hs.reshape(9)],
            H_inv_row_major=None,
            pitch_length_m=105.0,
            pitch_width_m=68.0,
            config=self.config,
        )
        self.assertEqual(geom.mapping_status, "failed")

    def test_matrix_direction_not_h_inv(self) -> None:
        H = perspective_H()
        H_inv = invert_homography(H)
        # Projecting a known image point with H vs H_inv must differ.
        ix, iy = 200.0, 150.0
        with_h = apply_homography(H, [(ix, iy)])[0]
        with_inv = apply_homography(H_inv, [(ix, iy)])[0]
        self.assertGreater(float(np.linalg.norm(np.asarray(with_h) - np.asarray(with_inv))), 1.0)
        geom = apply_image_to_pitch_projection(
            image_x=ix,
            image_y=iy,
            H_row_major=[float(x) for x in H.reshape(9)],
            H_inv_row_major=[float(x) for x in H_inv.reshape(9)],
            pitch_length_m=105.0,
            pitch_width_m=68.0,
            config=self.config,
        )
        self.assertAlmostEqual(float(geom.pitch_x_m or 0), float(with_h[0]), places=5)

    def test_fingerprint_mismatch_and_cleanup(self) -> None:
        bundle = base_bundle()
        session = Path(tempfile.mkdtemp(prefix="proj_fp_", dir=RUNTIME_ROOT))
        try:
            bad = dict(bundle["fingerprints"])
            bad["pitch_template"] = "0" * 64
            out = session / "out"
            svc = run_pitch_projection(
                output_dir=out,
                config=self.config,
                contain_root=Path(RUNTIME_ROOT),
                observations_rows=bundle["observations"],
                segments_rows=bundle["segments"],
                fingerprints=bad,
            )
            self.assertFalse(svc.accepted)
            self.assertIn("FINGERPRINT_MISMATCH", str(svc.error_code))
            self.assertFalse(out.exists() and any(out.iterdir()))
        finally:
            shutil.rmtree(session, ignore_errors=True)

    def test_no_overwrite_deterministic_and_eval(self) -> None:
        bundle = base_bundle()
        session = Path(tempfile.mkdtemp(prefix="proj_e2e_", dir=RUNTIME_ROOT))
        try:
            out = session / "out"
            kwargs = dict(
                config=self.config,
                contain_root=Path(RUNTIME_ROOT),
                observations_rows=bundle["observations"],
                segments_rows=bundle["segments"],
                frame_times=bundle["frame_times"],
                coverage_hulls=bundle["coverage_hulls"],
                eligibility_timeline=bundle["eligibility_timeline"],
                analysis_windows=bundle["analysis_windows"],
                fingerprints=bundle["fingerprints"],
            )
            a = run_pitch_projection(output_dir=out, **kwargs)
            self.assertTrue(a.accepted, a.error_code)
            b = run_pitch_projection(output_dir=out, **kwargs)
            self.assertFalse(b.accepted)
            self.assertEqual(b.error_code, "NO_OVERWRITE")

            out2 = session / "out2"
            c = run_pitch_projection(output_dir=out2, **kwargs)
            self.assertTrue(c.accepted, c.error_code)
            pa = [(p["pitch_x_m"], p["mapping_status"]) for p in a.summary["projections"]]
            pc = [(p["pitch_x_m"], p["mapping_status"]) for p in c.summary["projections"]]
            self.assertEqual(pa, pc)

            table = read_contract_parquet(
                Path(str(a.projected_positions_parquet)),
                get_contract("projected_positions", 1),
            )
            self.assertGreater(table.num_rows, 0)

            report = evaluate_pitch_projection(has_reviewed_ground_truth=False)
            self.assertEqual(report.ground_truth_evaluation_status, NOT_EVALUATED_PROJECTED_POS)
        finally:
            shutil.rmtree(session, ignore_errors=True)

    def test_stage8_regression_lazy_import(self) -> None:
        import sys

        before = {k for k in sys.modules if "cls_hrnet" in k.lower() or k.startswith("fa_nbjw_")}
        import football_analytics.calibration as cal  # noqa: F401

        after = {k for k in sys.modules if "cls_hrnet" in k.lower() or k.startswith("fa_nbjw_")}
        self.assertEqual(after - before, set())

    def test_ambiguous_ball_uncertain(self) -> None:
        bundle = base_bundle()
        rid, vid = bundle["run_id"], bundle["video_id"]
        obs = obs_row(
            run_id=rid,
            video_id=vid,
            frame_index=0,
            track_id=10,
            entity_type="ball",
            bbox=ball_bbox_for_centre(bundle["H"], 60.0, 30.0),
            detection_id=1,
        )
        row = build_projection_for_observation(
            observation=obs,
            segments=bundle["segments"],
            config=self.config,
            run_id=rid,
            video_id=vid,
            video_time_us=0,
            projection_id="proj_amb",
            pitch_template_fingerprint=bundle["pitch_template_fingerprint"],
            coverage_hulls=bundle["coverage_hulls"],
            ambiguous_ball=True,
        )
        self.assertEqual(row["mapping_status"], "uncertain")
        self.assertNotEqual(row["physical_metric_eligibility"], "eligible")


if __name__ == "__main__":
    unittest.main()
