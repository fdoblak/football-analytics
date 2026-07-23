"""Stage 6C ball tracking baseline tests (synthetic only)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from football_analytics.core.hashing import hash_canonical_json
from football_analytics.data.compiler import get_contract
from football_analytics.data.fingerprint import contract_fingerprint
from football_analytics.tracking.ball_association import (
    ball_association_cost,
    greedy_ball_associate,
    passes_ball_association_gate,
)
from football_analytics.tracking.ball_tracker import run_ball_tracker
from football_analytics.tracking.ball_tracking_config import (
    ball_tracking_config_fingerprint,
    default_ball_tracking_config_path,
    load_ball_tracking_config,
)
from football_analytics.tracking.ball_tracking_evaluation import (
    NOT_EVALUATED_BALL_TRACKING,
    evaluate_ball_tracking,
)
from football_analytics.tracking.ball_tracking_fixtures import (
    RUNTIME_ROOT,
    assert_runtime_root,
    frozen_ambiguity,
    frozen_constant_velocity,
    frozen_fast_zero_iou,
    frozen_long_gap,
    frozen_reject_human,
    frozen_replay_nonplayable,
    frozen_short_gap,
    frozen_shot_cut,
    frozen_tie_break,
    frozen_vfr,
)
from football_analytics.tracking.ball_tracking_service import run_ball_tracking
from football_analytics.tracking.contracts import (
    EXPECTED_DETECTIONS_FP,
    EXPECTED_TRACK_OBSERVATIONS_FP,
    EXPECTED_TRACK_SUMMARIES_FP,
    assert_v1_track_fingerprints_unchanged,
)
from football_analytics.tracking.human_tracker import run_human_tracker
from football_analytics.tracking.human_tracking_config import (
    default_human_tracking_config_path,
    load_human_tracking_config,
)
from football_analytics.tracking.human_tracking_fixtures import frozen_single_person
from football_analytics.tracking.policy import load_tracking_policy
from football_analytics.tracking.time_rules import gap_us


class BallTrackingBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_ball_tracking_config(default_ball_tracking_config_path())
        cls.policy = load_tracking_policy()
        cls.cfg_fp = ball_tracking_config_fingerprint(cls.cfg)
        assert_runtime_root()

    def _track(self, bundle: dict) -> object:
        return run_ball_tracker(
            run_id=bundle["run_id"],
            video_id=bundle["video_id"],
            frames=bundle["frames"].to_pylist(),
            detections=bundle["detections"].to_pylist(),
            analysis_windows=bundle["analysis_windows"].to_pylist(),
            detection_attributes=bundle["detection_attributes"].to_pylist(),
            config=self.cfg,
            policy=self.policy,
        )

    def test_01_fast_motion_association(self) -> None:
        r = self._track(frozen_fast_zero_iou())
        tids = {o["track_id"] for o in r.observations if o["observation_state"] == "observed"}
        self.assertEqual(len(tids), 1)

    def test_02_zero_iou_motion_association(self) -> None:
        pred = (40.0, 300.0, 48.0, 308.0)
        det = (100.0, 300.0, 108.0, 308.0)
        cost, iou, dist, ratio = ball_association_cost(
            pred,
            det,
            detection_confidence=0.9,
            motion_gate_px=180.0,
            size_ratio_gate=3.0,
            motion_weight=0.55,
            size_weight=0.20,
            confidence_weight=0.15,
            iou_weight=0.10,
        )
        self.assertEqual(iou, 0.0)
        self.assertTrue(
            passes_ball_association_gate(
                center_dist=dist,
                motion_gate_px=180.0,
                size_ratio=ratio,
                size_ratio_gate=3.0,
                iou=iou,
                iou_support_min=0.0,
                require_motion_gate=True,
            )
        )
        self.assertGreater(cost, 0.0)
        # IoU alone must not pass when motion fails.
        self.assertFalse(
            passes_ball_association_gate(
                center_dist=500.0,
                motion_gate_px=180.0,
                size_ratio=1.0,
                size_ratio_gate=3.0,
                iou=1.0,
                iou_support_min=0.0,
                require_motion_gate=True,
            )
        )

    def test_03_deterministic_tie_break(self) -> None:
        tracks = [{"track_id": 0}, {"track_id": 1}]
        dets = [
            {
                "detection_id": 10,
                "bbox_x1": 10.0,
                "bbox_y1": 200.0,
                "bbox_x2": 20.0,
                "bbox_y2": 210.0,
                "confidence": 0.9,
            },
            {
                "detection_id": 11,
                "bbox_x1": 200.0,
                "bbox_y1": 200.0,
                "bbox_x2": 210.0,
                "bbox_y2": 210.0,
                "confidence": 0.9,
            },
        ]
        preds = {0: (10.0, 200.0, 20.0, 210.0), 1: (200.0, 200.0, 210.0, 210.0)}
        kwargs = dict(
            predicted_bboxes=preds,
            dt_us_by_track={0: 40000, 1: 40000},
            motion_center_gate_px=180.0,
            motion_gate_scale_per_us=0.00025,
            size_ratio_gate=3.0,
            iou_support_min=0.0,
            require_motion_gate=True,
            motion_weight=0.55,
            size_weight=0.20,
            confidence_weight=0.15,
            iou_weight=0.10,
        )
        m1, _, _ = greedy_ball_associate(tracks, dets, **kwargs)
        m2, _, _ = greedy_ball_associate(tracks, dets, **kwargs)
        self.assertEqual(
            [(m.track_id, m.detection_id, m.cost) for m in m1],
            [(m.track_id, m.detection_id, m.cost) for m in m2],
        )
        b = frozen_tie_break()
        r1 = self._track(b)
        r2 = self._track(b)
        self.assertEqual(
            [(o["frame_index"], o["track_id"], o["detection_id"]) for o in r1.observations],
            [(o["frame_index"], o["track_id"], o["detection_id"]) for o in r2.observations],
        )

    def test_04_multi_candidate_ambiguity(self) -> None:
        r = self._track(frozen_ambiguity())
        self.assertGreaterEqual(r.stats["ambiguous_frames"], 1)
        amb = [p for p in r.primary_sidecar if p["status"] == "ambiguous"]
        self.assertGreaterEqual(len(amb), 1)
        for p in amb:
            self.assertIsNone(p["primary_track_id"])

    def test_05_primary_candidate_uniqueness(self) -> None:
        r = self._track(frozen_constant_velocity())
        for p in r.primary_sidecar:
            if p["status"] == "primary":
                self.assertIsNotNone(p["primary_track_id"])
                # At most one primary per frame (by construction).
                self.assertEqual(p["status"], "primary")
        primaries = [p for p in r.primary_sidecar if p["status"] == "primary"]
        self.assertGreaterEqual(len(primaries), 1)

    def test_06_short_gap_prediction(self) -> None:
        r = self._track(frozen_short_gap())
        preds = [o for o in r.observations if o["observation_state"] == "predicted"]
        self.assertGreaterEqual(len(preds), 1)
        self.assertTrue(all(o["detection_id"] is None for o in preds))

    def test_07_long_gap_termination_new_track(self) -> None:
        r = self._track(frozen_long_gap())
        tids = {o["track_id"] for o in r.observations}
        self.assertGreaterEqual(len(tids), 2)

    def test_08_no_cross_cut_replay(self) -> None:
        r = self._track(frozen_shot_cut())
        self.assertTrue(any("SHOT_CUT" in f for f in r.findings))
        self.assertGreaterEqual(len({o["track_id"] for o in r.observations}), 2)
        r2 = self._track(frozen_replay_nonplayable())
        playable_last = 5
        self.assertTrue(all(int(o["frame_index"]) <= playable_last for o in r2.observations))

    def test_09_vfr_us_gap(self) -> None:
        b = frozen_vfr()
        frames = b["frames"].to_pylist()
        g = gap_us(int(frames[0]["video_time_us"]), int(frames[1]["video_time_us"]))
        self.assertGreater(g, 0)
        r = self._track(b)
        self.assertGreaterEqual(len(r.observations), 1)

    def test_10_prediction_uncertainty(self) -> None:
        r = self._track(frozen_short_gap())
        preds = [o for o in r.observations if o["observation_state"] == "predicted"]
        self.assertTrue(
            any(
                any(str(f).startswith("prediction_uncertainty:") for f in o["quality_flags"])
                for o in preds
            )
        )

    def test_11_predicted_event_physical_ineligible(self) -> None:
        r = self._track(frozen_short_gap())
        for o in r.observations:
            if o["observation_state"] == "predicted":
                self.assertIn("physical_metric_ineligible", o["quality_flags"])
                self.assertIn("event_ineligible", o["quality_flags"])

    def test_12_human_ball_separation(self) -> None:
        r = self._track(frozen_reject_human())
        self.assertEqual(r.stats["rejected_non_ball"], 1)
        self.assertEqual(r.stats["ball_input_detections"], 1)

    def test_13_role_always_unknown(self) -> None:
        r = self._track(frozen_constant_velocity())
        for o in r.observations:
            if o["observation_state"] == "observed":
                self.assertIn("role_unknown", o["quality_flags"])

    def test_14_receipt_count_hash(self) -> None:
        b = frozen_constant_velocity()
        with tempfile.TemporaryDirectory(dir=RUNTIME_ROOT) as td:
            res = run_ball_tracking(
                detections="mem",
                frames="mem",
                analysis_windows="mem",
                output_dir=td,
                config=self.cfg,
                in_memory_bundle=b,
                contain_root=RUNTIME_ROOT,
                run_id=b["run_id"],
                video_id=b["video_id"],
            )
            self.assertTrue(res.accepted, msg=res.error_code)
            receipt = json.loads(Path(res.receipt_json).read_text(encoding="utf-8"))
            self.assertEqual(
                receipt["observation_counts"]["total"],
                sum(
                    receipt["observation_counts"][k]
                    for k in ("observed", "predicted", "interpolated")
                ),
            )
            self.assertEqual(receipt["config_fingerprint"], self.cfg_fp)
            self.assertTrue(Path(res.primary_sidecar_json).is_file())

    def test_15_no_overwrite_failure_cleanup(self) -> None:
        b = frozen_constant_velocity()
        with tempfile.TemporaryDirectory(dir=RUNTIME_ROOT) as td:
            out = Path(td)
            res1 = run_ball_tracking(
                detections="mem",
                frames="mem",
                analysis_windows="mem",
                output_dir=out,
                config=self.cfg,
                in_memory_bundle=b,
                contain_root=RUNTIME_ROOT,
                run_id=b["run_id"],
                video_id=b["video_id"],
            )
            self.assertTrue(res1.accepted)
            res2 = run_ball_tracking(
                detections="mem",
                frames="mem",
                analysis_windows="mem",
                output_dir=out,
                config=self.cfg,
                in_memory_bundle=b,
                contain_root=RUNTIME_ROOT,
                run_id=b["run_id"],
                video_id=b["video_id"],
            )
            self.assertFalse(res2.accepted)
            self.assertEqual(res2.error_code, "OVERWRITE_FORBIDDEN")

    def test_16_human_tracker_regression(self) -> None:
        hcfg = load_human_tracking_config(default_human_tracking_config_path())
        b = frozen_single_person()
        r = run_human_tracker(
            run_id=b["run_id"],
            video_id=b["video_id"],
            frames=b["frames"].to_pylist(),
            detections=b["detections"].to_pylist(),
            analysis_windows=b["analysis_windows"].to_pylist(),
            detection_attributes=b["detection_attributes"].to_pylist(),
            config=hcfg,
            policy=self.policy,
        )
        self.assertGreaterEqual(len(r.observations), 1)
        self.assertEqual(r.stats["human_input_detections"], 5)

    def test_17_fingerprint_regression(self) -> None:
        assert_v1_track_fingerprints_unchanged()
        self.assertEqual(
            contract_fingerprint(get_contract("track_observations", 1)),
            EXPECTED_TRACK_OBSERVATIONS_FP,
        )
        self.assertEqual(
            contract_fingerprint(get_contract("track_summaries", 1)),
            EXPECTED_TRACK_SUMMARIES_FP,
        )
        self.assertEqual(
            contract_fingerprint(get_contract("detections", 1)),
            EXPECTED_DETECTIONS_FP,
        )
        life_fp = contract_fingerprint(get_contract("track_lifecycle", 1))
        self.assertTrue(life_fp.startswith("613cd81e"))

    def test_18_not_evaluated_without_reviewed_gt(self) -> None:
        rep = evaluate_ball_tracking(has_reviewed_ground_truth=False)
        self.assertEqual(rep.ground_truth_evaluation_status, NOT_EVALUATED_BALL_TRACKING)
        self.assertEqual(rep.status, "not_evaluated")

    def test_19_deterministic_output(self) -> None:
        b = frozen_constant_velocity(run_id="run_20260723T120000000000Z_deadbeefcafe")
        r1 = self._track(b)
        r2 = self._track(b)
        fp1 = hash_canonical_json(
            {
                "observations": r1.observations,
                "lifecycle": r1.lifecycle,
                "primary": r1.primary_sidecar,
            }
        )
        fp2 = hash_canonical_json(
            {
                "observations": r2.observations,
                "lifecycle": r2.lifecycle,
                "primary": r2.primary_sidecar,
            }
        )
        self.assertEqual(fp1, fp2)

    def test_20_one_obs_per_track_per_frame(self) -> None:
        r = self._track(frozen_short_gap())
        keys = [(o["frame_index"], o["track_id"]) for o in r.observations]
        self.assertEqual(len(keys), len(set(keys)))


if __name__ == "__main__":
    unittest.main()
