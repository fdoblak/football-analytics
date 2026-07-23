"""Stage 6B human multi-object tracking baseline tests (synthetic only)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from football_analytics.core.hashing import hash_canonical_json
from football_analytics.data.compiler import get_contract
from football_analytics.data.fingerprint import contract_fingerprint
from football_analytics.tracking.contracts import (
    EXPECTED_DETECTIONS_FP,
    EXPECTED_TRACK_OBSERVATIONS_FP,
    EXPECTED_TRACK_SUMMARIES_FP,
    assert_v1_track_fingerprints_unchanged,
)
from football_analytics.tracking.human_association import (
    association_cost,
    greedy_associate,
    passes_association_gate,
)
from football_analytics.tracking.human_motion import (
    bbox_center,
    predict_bbox_constant_velocity,
    velocity_from_centers,
)
from football_analytics.tracking.human_tracker import run_human_tracker
from football_analytics.tracking.human_tracking_config import (
    default_human_tracking_config_path,
    human_tracking_config_fingerprint,
    load_human_tracking_config,
)
from football_analytics.tracking.human_tracking_evaluation import (
    NOT_EVALUATED_HUMAN_TRACKING,
    evaluate_human_tracking,
)
from football_analytics.tracking.human_tracking_fixtures import (
    RUNTIME_ROOT,
    assert_runtime_root,
    frozen_long_occlusion,
    frozen_multi_human,
    frozen_non_playable,
    frozen_reject_ball,
    frozen_role_unknown_and_conflict,
    frozen_short_occlusion,
    frozen_shot_cut,
    frozen_single_person,
    frozen_tie_break,
    frozen_vfr,
)
from football_analytics.tracking.human_tracking_service import run_human_tracking
from football_analytics.tracking.lifecycle import assert_transition_allowed
from football_analytics.tracking.policy import load_tracking_policy
from football_analytics.tracking.time_rules import gap_us
from football_analytics.tracking.types import LifecycleState, TransitionError


class HumanTrackingBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_human_tracking_config(default_human_tracking_config_path())
        cls.policy = load_tracking_policy()
        cls.cfg_fp = human_tracking_config_fingerprint(cls.cfg)
        assert_runtime_root()

    def _track(self, bundle: dict) -> object:
        return run_human_tracker(
            run_id=bundle["run_id"],
            video_id=bundle["video_id"],
            frames=bundle["frames"].to_pylist(),
            detections=bundle["detections"].to_pylist(),
            analysis_windows=bundle["analysis_windows"].to_pylist(),
            detection_attributes=bundle["detection_attributes"].to_pylist(),
            config=self.cfg,
            policy=self.policy,
        )

    def test_01_deterministic_association_tie_break(self) -> None:
        tracks = [{"track_id": 0}, {"track_id": 1}]
        dets = [
            {
                "detection_id": 10,
                "bbox_x1": 10.0,
                "bbox_y1": 10.0,
                "bbox_x2": 40.0,
                "bbox_y2": 80.0,
            },
            {
                "detection_id": 11,
                "bbox_x1": 200.0,
                "bbox_y1": 10.0,
                "bbox_x2": 230.0,
                "bbox_y2": 80.0,
            },
        ]
        preds = {
            0: (10.0, 10.0, 40.0, 80.0),
            1: (200.0, 10.0, 230.0, 80.0),
        }
        m1, _, _ = greedy_associate(
            tracks,
            dets,
            predicted_bboxes=preds,
            iou_gate=0.2,
            motion_center_gate_px=120.0,
            iou_weight=0.7,
            motion_weight=0.3,
        )
        m2, _, _ = greedy_associate(
            tracks,
            dets,
            predicted_bboxes=preds,
            iou_gate=0.2,
            motion_center_gate_px=120.0,
            iou_weight=0.7,
            motion_weight=0.3,
        )
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

    def test_02_human_only_routing(self) -> None:
        r = self._track(frozen_reject_ball())
        self.assertEqual(r.stats["rejected_non_human"], 1)
        self.assertEqual(r.stats["human_input_detections"], 1)

    def test_03_one_detection_one_track(self) -> None:
        r = self._track(frozen_single_person())
        used = [
            (o["frame_index"], o["detection_id"])
            for o in r.observations
            if o["detection_id"] is not None
        ]
        self.assertEqual(len(used), len(set(used)))

    def test_04_one_track_one_obs_per_frame(self) -> None:
        r = self._track(frozen_short_occlusion())
        keys = [(o["frame_index"], o["track_id"]) for o in r.observations]
        self.assertEqual(len(keys), len(set(keys)))

    def test_05_lifecycle_transitions(self) -> None:
        r = self._track(frozen_short_occlusion())
        states = [e["lifecycle_state"] for e in r.lifecycle]
        self.assertIn("tentative", states)
        self.assertIn("confirmed", states)
        self.assertIn("lost", states)
        self.assertIn("terminated", states)
        assert_transition_allowed(None, LifecycleState.TENTATIVE, policy=self.policy)
        with self.assertRaises(TransitionError):
            assert_transition_allowed(
                LifecycleState.TERMINATED, LifecycleState.CONFIRMED, policy=self.policy
            )

    def test_06_short_gap_recovery(self) -> None:
        r = self._track(frozen_short_occlusion())
        self.assertGreaterEqual(r.stats["recoveries"], 1)
        self.assertTrue(any(e["transition_reason"] == "recover" for e in r.lifecycle))

    def test_07_long_gap_split(self) -> None:
        r = self._track(frozen_long_occlusion())
        tids = {o["track_id"] for o in r.observations}
        self.assertGreaterEqual(len(tids), 2)

    def test_08_terminated_no_reopen(self) -> None:
        r = self._track(frozen_single_person())
        by_tid: dict[int, list] = {}
        for e in r.lifecycle:
            by_tid.setdefault(int(e["track_id"]), []).append(e)
        for events in by_tid.values():
            ordered = sorted(events, key=lambda e: int(e["event_index"]))
            seen_term = False
            for ev in ordered:
                if seen_term:
                    self.fail("reopen after terminated")
                if ev["lifecycle_state"] == "terminated":
                    seen_term = True

    def test_09_shot_cut_no_cross_continuation(self) -> None:
        r = self._track(frozen_shot_cut())
        self.assertTrue(any("SHOT_CUT" in f for f in r.findings))
        self.assertGreaterEqual(len({o["track_id"] for o in r.observations}), 2)

    def test_10_non_playable_gap(self) -> None:
        r = self._track(frozen_non_playable())
        # Tracks must terminate before/at non-playable; no obs in graphics window frames.
        playable_last = 5
        self.assertTrue(all(int(o["frame_index"]) <= playable_last for o in r.observations))

    def test_11_vfr_pts_gap(self) -> None:
        b = frozen_vfr()
        frames = b["frames"].to_pylist()
        g = gap_us(int(frames[0]["video_time_us"]), int(frames[1]["video_time_us"]))
        self.assertGreater(g, 0)
        r = self._track(b)
        self.assertGreaterEqual(len(r.observations), 1)

    def test_12_iou_motion_gate(self) -> None:
        cost, iou, dist = association_cost(
            (0.0, 0.0, 10.0, 10.0),
            (1000.0, 1000.0, 1010.0, 1010.0),
            iou_weight=0.7,
            motion_weight=0.3,
            motion_center_gate_px=50.0,
        )
        self.assertFalse(
            passes_association_gate(
                iou=iou, center_dist=dist, iou_gate=0.2, motion_center_gate_px=50.0
            )
        )
        self.assertGreater(cost, 0.0)
        pred = predict_bbox_constant_velocity((0.0, 0.0, 10.0, 10.0), vx=1e-5, vy=0.0, dt_us=100000)
        self.assertAlmostEqual(pred[0], 1.0, places=5)
        vx, vy = velocity_from_centers((0.0, 0.0), 0, (10.0, 0.0), 1000000)
        self.assertAlmostEqual(vx, 1e-5, places=12)
        self.assertEqual(bbox_center((0.0, 0.0, 10.0, 10.0)), (5.0, 5.0))

    def test_13_crossing_tracks(self) -> None:
        r = self._track(frozen_multi_human())
        self.assertGreaterEqual(len({o["track_id"] for o in r.observations}), 2)

    def test_14_unknown_role_preserved(self) -> None:
        b = frozen_single_person()
        # all unknown roles
        r = self._track(b)
        self.assertFalse(any("ROLE_CONFLICT" in f for f in r.findings))

    def test_15_role_conflict_review(self) -> None:
        r = self._track(frozen_role_unknown_and_conflict())
        self.assertTrue(any("ROLE_CONFLICT" in f for f in r.findings))
        self.assertGreaterEqual(r.stats["review_required_count"], 1)

    def test_16_predicted_observation_marked(self) -> None:
        r = self._track(frozen_short_occlusion())
        preds = [o for o in r.observations if o["observation_state"] == "predicted"]
        self.assertGreaterEqual(len(preds), 1)
        self.assertTrue(all(o["detection_id"] is None for o in preds))

    def test_17_predicted_physical_metric_ineligible(self) -> None:
        r = self._track(frozen_short_occlusion())
        for o in r.observations:
            if o["observation_state"] == "predicted":
                self.assertIn("physical_metric_ineligible", o["quality_flags"])

    def test_18_detection_frame_fk(self) -> None:
        b = frozen_single_person()
        r = self._track(b)
        fset = {int(f["frame_index"]) for f in b["frames"].to_pylist()}
        dset = {
            (int(d["frame_index"]), int(d["detection_id"])) for d in b["detections"].to_pylist()
        }
        for o in r.observations:
            self.assertIn(int(o["frame_index"]), fset)
            if o["detection_id"] is not None:
                self.assertIn((int(o["frame_index"]), int(o["detection_id"])), dset)

    def test_19_receipt_count_hash_consistency(self) -> None:
        b = frozen_single_person()
        with tempfile.TemporaryDirectory(dir=RUNTIME_ROOT) as td:
            res = run_human_tracking(
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
            self.assertTrue(res.accepted)
            receipt = json.loads(Path(res.receipt_json).read_text(encoding="utf-8"))
            self.assertEqual(
                receipt["observation_counts"]["total"],
                sum(
                    receipt["observation_counts"][k]
                    for k in ("observed", "predicted", "interpolated")
                ),
            )
            self.assertEqual(
                receipt["config_fingerprint"],
                self.cfg_fp,
            )

    def test_20_no_overwrite_failure_cleanup(self) -> None:
        b = frozen_single_person()
        with tempfile.TemporaryDirectory(dir=RUNTIME_ROOT) as td:
            out = Path(td)
            res1 = run_human_tracking(
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
            res2 = run_human_tracking(
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

    def test_21_deterministic_output_fingerprint(self) -> None:
        b = frozen_single_person(run_id="run_20260723T120000000000Z_deadbeefcafe")
        # Fix run_id in tables
        rid = b["run_id"]
        r1 = self._track(b)
        r2 = self._track(b)
        fp1 = hash_canonical_json(
            {"observations": r1.observations, "lifecycle": r1.lifecycle, "summaries": r1.summaries}
        )
        fp2 = hash_canonical_json(
            {"observations": r2.observations, "lifecycle": r2.lifecycle, "summaries": r2.summaries}
        )
        self.assertEqual(fp1, fp2)
        self.assertEqual(rid, b["run_id"])

    def test_22_fingerprint_regression(self) -> None:
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

    def test_23_not_evaluated_without_reviewed_gt(self) -> None:
        rep = evaluate_human_tracking(has_reviewed_ground_truth=False)
        self.assertEqual(rep.ground_truth_evaluation_status, NOT_EVALUATED_HUMAN_TRACKING)
        self.assertEqual(rep.status, "not_evaluated")
        self.assertTrue(all(v is None for k, v in rep.metrics.items() if k != "gap_recovery"))


if __name__ == "__main__":
    unittest.main()
