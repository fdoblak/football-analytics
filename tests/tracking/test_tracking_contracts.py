"""Stage 6A tracking contracts / lifecycle tests (synthetic only)."""

from __future__ import annotations

import unittest
from pathlib import Path

from football_analytics.core.records import RecordError, write_json_record
from football_analytics.core.run_id import generate_run_id
from football_analytics.data.compiler import get_contract, list_contracts
from football_analytics.data.fingerprint import contract_fingerprint
from football_analytics.data.registry import (
    default_project_root,
    default_registry_path,
    load_schema_registry,
)
from football_analytics.tracking import (
    EXPECTED_DETECTIONS_FP,
    EXPECTED_TRACK_OBSERVATIONS_FP,
    EXPECTED_TRACK_SUMMARIES_FP,
    NOT_EVALUATED_TRACKING,
    TrackIdAllocator,
    allocate_hash_track_id,
    assert_track_contracts_registered,
    assert_transition_allowed,
    assert_v1_track_fingerprints_unchanged,
    evaluate_tracking,
    gap_us,
    load_tracking_json_schema,
    load_tracking_policy,
    observation_state_for_source,
    policy_fingerprint,
    tracking_schema_fingerprints,
    validate_against_json_schema,
    validate_track_bbox,
    validate_track_bundle,
)
from football_analytics.tracking.fixtures import (
    _attr_row,
    _cast,
    _det_row,
    _life_row,
    _obs_row,
    _summary_row,
    base_context,
    lost_recover_bundle,
    mutate_lifecycle_reopen,
    terminated_bundle,
    valid_birth_confirmed_bundle,
)
from football_analytics.tracking.receipt import (
    build_synthetic_receipt,
    build_synthetic_request,
    validate_receipt_payload,
    validate_request_payload,
)
from football_analytics.tracking.types import TrackingContractError, TransitionError

ROOT = default_project_root()


class TrackingContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reg = load_schema_registry(default_registry_path(), project_root=ROOT)
        self.policy = load_tracking_policy()
        self.pol_fp = policy_fingerprint(self.policy)
        self.specs = {
            n: get_contract(n, 1, registry=self.reg)
            for n in (
                "videos",
                "frames",
                "detections",
                "detection_attributes",
                "analysis_windows",
                "track_observations",
                "track_summaries",
                "track_lifecycle",
            )
        }

    def _validate(self, bundle: dict, receipt=None):
        return validate_track_bundle(
            track_observations=bundle.get("track_observations"),
            track_summaries=bundle.get("track_summaries"),
            track_lifecycle=bundle.get("track_lifecycle"),
            frames=bundle.get("frames"),
            detections=bundle.get("detections"),
            detection_attributes=bundle.get("detection_attributes"),
            videos=bundle.get("videos"),
            analysis_windows=bundle.get("analysis_windows"),
            specs=self.specs,
            policy=self.policy,
            receipt=receipt,
            frame_width=1280,
            frame_height=720,
        )

    def test_01_registry_and_fingerprints(self) -> None:
        assert_track_contracts_registered(registry=self.reg)
        self.assertEqual(len(list_contracts(registry=self.reg)), 16)
        assert_v1_track_fingerprints_unchanged(registry=self.reg)
        fps = tracking_schema_fingerprints(registry=self.reg)
        self.assertEqual(fps["track_observations"], EXPECTED_TRACK_OBSERVATIONS_FP)
        self.assertEqual(fps["track_summaries"], EXPECTED_TRACK_SUMMARIES_FP)
        self.assertEqual(fps["detections"], EXPECTED_DETECTIONS_FP)
        self.assertEqual(
            contract_fingerprint(get_contract("track_observations", 1, registry=self.reg)),
            EXPECTED_TRACK_OBSERVATIONS_FP,
        )

    def test_02_observation_source_mapping(self) -> None:
        self.assertEqual(observation_state_for_source("detection_associated"), "observed")
        self.assertEqual(observation_state_for_source("predicted"), "predicted")
        self.assertEqual(observation_state_for_source("interpolated"), "interpolated")
        self.assertEqual(observation_state_for_source("not_observed"), "prefer_no_row")

    def test_03_birth_confirmed(self) -> None:
        bundle = valid_birth_confirmed_bundle()
        receipt = build_synthetic_receipt(
            run_id=bundle["run_id"],
            video_id=bundle["video_id"],
            policy_fingerprint=self.pol_fp,
            observations=bundle["track_observations"].to_pylist(),
            lifecycle=bundle["track_lifecycle"].to_pylist(),
            detections=bundle["detections"].to_pylist(),
        )
        validate_request_payload(
            build_synthetic_request(
                run_id=bundle["run_id"],
                video_id=bundle["video_id"],
                policy_fingerprint=self.pol_fp,
            )
        )
        validate_receipt_payload(receipt)
        vr = self._validate(bundle, receipt)
        self.assertNotEqual(vr.status, "FAIL", msg=vr.errors)

    def test_04_lost_recover_and_terminate(self) -> None:
        self.assertNotEqual(self._validate(lost_recover_bundle()).status, "FAIL")
        self.assertNotEqual(self._validate(terminated_bundle()).status, "FAIL")

    def test_05_no_reopen_terminated(self) -> None:
        with self.assertRaises(TransitionError):
            assert_transition_allowed("terminated", "confirmed", policy=self.policy)
        vr = self._validate(mutate_lifecycle_reopen(terminated_bundle()))
        self.assertEqual(vr.status, "FAIL")

    def test_06_duplicate_detection_assignment(self) -> None:
        b = valid_birth_confirmed_bundle()
        obs = b["track_observations"].to_pylist()
        obs.append(
            _obs_row(b["run_id"], b["video_id"], 0, 1, detection_id=0, observation_state="observed")
        )
        b["track_observations"] = _cast("track_observations", obs)
        self.assertEqual(self._validate(b).status, "FAIL")

    def test_07_human_ball_merge_rejected(self) -> None:
        ctx = base_context(n_frames=4)
        rid, vid = ctx["run_id"], ctx["video_id"]
        times = ctx["times"]
        dets = [
            _det_row(rid, vid, 0, 0),
            _det_row(rid, vid, 1, 1, class_id=32, class_name="sports_ball", bbox=(5, 5, 15, 15)),
        ]
        attrs = [
            _attr_row(rid, vid, 0, 0, entity_type="human"),
            _attr_row(rid, vid, 1, 1, entity_type="ball"),
        ]
        obs = [
            _obs_row(rid, vid, 0, 0, detection_id=0, observation_state="observed"),
            _obs_row(
                rid,
                vid,
                1,
                0,
                detection_id=1,
                observation_state="observed",
                class_id=32,
                bbox=(5, 5, 15, 15),
            ),
        ]
        life = [_life_row(rid, vid, 0, 0, 0, times[0], "tentative", None, entity_type="human")]
        bundle = {
            **ctx,
            "detections": _cast("detections", dets),
            "detection_attributes": _cast("detection_attributes", attrs),
            "track_observations": _cast("track_observations", obs),
            "track_summaries": _cast("track_summaries", [_summary_row(rid, vid, 0, obs)]),
            "track_lifecycle": _cast("track_lifecycle", life),
        }
        self.assertEqual(self._validate(bundle).status, "FAIL")

    def test_08_cross_video_and_dangling_fk(self) -> None:
        b = valid_birth_confirmed_bundle()
        obs = b["track_observations"].to_pylist()
        obs[0]["video_id"] = "other_video"
        b["track_observations"] = _cast("track_observations", obs)
        self.assertEqual(self._validate(b).status, "FAIL")

        b2 = valid_birth_confirmed_bundle()
        obs2 = b2["track_observations"].to_pylist()
        obs2[0]["detection_id"] = 999
        b2["track_observations"] = _cast("track_observations", obs2)
        self.assertEqual(self._validate(b2).status, "FAIL")

    def test_09_vfr_gap_and_timestamp_reverse(self) -> None:
        ctx = base_context(n_frames=5, vfr=True)
        self.assertEqual(
            gap_us(ctx["times"][0], ctx["times"][1]),
            ctx["times"][1] - ctx["times"][0],
        )
        b = lost_recover_bundle()
        life = b["track_lifecycle"].to_pylist()
        life[-1]["video_time_us"] = 0
        b["track_lifecycle"] = _cast("track_lifecycle", life)
        self.assertEqual(self._validate(b).status, "FAIL")

    def test_10_predicted_physical_ineligible(self) -> None:
        b = lost_recover_bundle()
        obs = b["track_observations"].to_pylist()
        for o in obs:
            if o["observation_state"] == "predicted":
                o["quality_flags"] = []
        b["track_observations"] = _cast("track_observations", obs)
        self.assertEqual(self._validate(b).status, "FAIL")

    def test_11_role_conflict_and_unknown(self) -> None:
        b = valid_birth_confirmed_bundle()
        roles = {
            a["role_label"]
            for a in b["detection_attributes"].to_pylist()
            if a["entity_type"] == "human"
        }
        self.assertEqual(roles, {"unknown"})

        ctx = base_context(n_frames=4)
        rid, vid = ctx["run_id"], ctx["video_id"]
        times = ctx["times"]
        dets = [_det_row(rid, vid, 0, 0), _det_row(rid, vid, 1, 1)]
        attrs = [
            _attr_row(rid, vid, 0, 0, role_label="player"),
            _attr_row(rid, vid, 1, 1, role_label="referee"),
        ]
        obs = [
            _obs_row(rid, vid, 0, 0, detection_id=0, observation_state="observed"),
            _obs_row(rid, vid, 1, 0, detection_id=1, observation_state="observed"),
        ]
        life = [_life_row(rid, vid, 0, 0, 0, times[0], "tentative", None)]
        bundle = {
            **ctx,
            "detections": _cast("detections", dets),
            "detection_attributes": _cast("detection_attributes", attrs),
            "track_observations": _cast("track_observations", obs),
            "track_summaries": _cast("track_summaries", [_summary_row(rid, vid, 0, obs)]),
            "track_lifecycle": _cast("track_lifecycle", life),
        }
        self.assertEqual(self._validate(bundle).status, "FAIL")

    def test_12_track_id_no_reuse(self) -> None:
        rid = generate_run_id()
        alloc = TrackIdAllocator(run_id=rid, video_id="clip_a")
        a = alloc.allocate()
        b = alloc.allocate()
        self.assertNotEqual(a, b)
        with self.assertRaises(TrackingContractError):
            alloc.register_external(a)
        self.assertEqual(
            allocate_hash_track_id(run_id=rid, video_id="clip_a", seed="s"),
            allocate_hash_track_id(run_id=rid, video_id="clip_a", seed="s"),
        )

    def test_13_fingerprint_mismatch_and_eval(self) -> None:
        b = valid_birth_confirmed_bundle()
        vr = validate_track_bundle(
            track_observations=b["track_observations"],
            track_summaries=b["track_summaries"],
            track_lifecycle=b["track_lifecycle"],
            frames=b["frames"],
            detections=b["detections"],
            detection_attributes=b["detection_attributes"],
            videos=b["videos"],
            specs=self.specs,
            policy=self.policy,
            expected_input_fingerprint="a" * 64,
            actual_input_fingerprint="b" * 64,
        )
        self.assertEqual(vr.status, "FAIL")
        rep = evaluate_tracking()
        self.assertEqual(rep.ground_truth_evaluation_status, NOT_EVALUATED_TRACKING)
        validate_against_json_schema(
            rep.to_dict(run_id=b["run_id"], video_id=b["video_id"], config_fingerprint=self.pol_fp),
            load_tracking_json_schema("tracking_evaluation"),
        )

    def test_14_bbox_and_atomic(self) -> None:
        with self.assertRaises(TrackingContractError):
            validate_track_bbox((0.0, 0.0, 0.0, 1.0))
        validate_track_bbox((10.0, 20.0, 40.0, 80.0), frame_width=1280, frame_height=720)
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "x.json"
            write_json_record(path, {"a": 1}, contain_root=root, overwrite=False)
            with self.assertRaises(RecordError):
                write_json_record(path, {"a": 2}, contain_root=root, overwrite=False)


if __name__ == "__main__":
    unittest.main()
