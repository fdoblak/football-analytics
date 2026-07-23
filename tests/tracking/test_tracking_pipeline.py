"""Stage 6D tracking fusion + quality pipeline tests."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pyarrow as pa

from football_analytics.core.run_id import generate_run_id
from football_analytics.data.compiler import compile_arrow_schema, get_contract
from football_analytics.data.fingerprint import contract_fingerprint
from football_analytics.data.parquet import write_contract_parquet
from football_analytics.data.registry import default_project_root
from football_analytics.tracking.contracts import tracking_schema_fingerprints
from football_analytics.tracking.evaluation import NOT_EVALUATED_TRACKING
from football_analytics.tracking.tracking_fusion import (
    TrackingFusionError,
    compute_ball_track_remap,
    detect_cross_cut_violations,
    detect_terminated_reopen,
    fuse_tracking_bundle,
    merge_track_tables,
)
from football_analytics.tracking.tracking_pipeline import run_tracking_integrate
from football_analytics.tracking.tracking_pipeline_config import (
    load_tracking_pipeline_config,
)
from football_analytics.tracking.tracking_pipeline_fixtures import (
    DETECTION_FP_A,
    DETECTION_FP_B,
    SOURCE_SHA_A,
    SOURCE_SHA_B,
    TIMELINE_FP_A,
    TIMELINE_FP_B,
    WINDOW_FP_A,
    build_minimal_tracking_fusion_inputs,
    make_analysis_window_row,
    write_json,
)
from football_analytics.tracking.tracking_quality import (
    build_tracking_review_queue,
    evaluate_tracking_quality,
)

EXPECTED_DETECTIONS_FP = "04ae8dd7a7e92bf7bd468db7a263e5e28258a30887d43c8f603c69d56f5c18b6"
EXPECTED_OBS_FP = "9ca2f7af56e69b47ec8db8d644164c84aa7fe3a62da40e247ed6db4f2c4c5f01"
EXPECTED_SUM_FP = "7b04e31d641c49e66ad06baec53e1075e2bc286b9f08f1497aa0571bf7c1c168"


def _cfg() -> dict[str, Any]:
    return load_tracking_pipeline_config(
        default_project_root() / "configs/tracking/tracking_pipeline.yaml"
    )


def _tbl(name: str, rows: list[dict[str, Any]]) -> Any:
    schema = compile_arrow_schema(get_contract(name, 1))
    return pa.Table.from_pylist(rows, schema=schema) if rows else schema.empty_table()


def _write_bundle(root: Path, inputs: dict[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    mapping = {
        "detections": ("detections", "detections.parquet"),
        "detection_attributes": ("detection_attributes", "detection_attributes.parquet"),
        "frames": ("frames", "frames.parquet"),
        "analysis_windows": ("analysis_windows", "analysis_windows.parquet"),
        "human_observations": ("track_observations", "human_observations.parquet"),
        "human_summaries": ("track_summaries", "human_summaries.parquet"),
        "human_lifecycle": ("track_lifecycle", "human_lifecycle.parquet"),
        "ball_observations": ("track_observations", "ball_observations.parquet"),
        "ball_summaries": ("track_summaries", "ball_summaries.parquet"),
        "ball_lifecycle": ("track_lifecycle", "ball_lifecycle.parquet"),
    }
    for key, (contract, fname) in mapping.items():
        p = root / fname
        write_contract_parquet(
            _tbl(contract, list(inputs[key])),
            p,
            get_contract(contract, 1),
            contain_root=root,
        )
        paths[key] = p
    paths["detection_receipt"] = write_json(
        root / "detection_receipt.json", inputs["detection_receipt"]
    )
    paths["human_receipt"] = write_json(root / "human_receipt.json", inputs["human_receipt"])
    paths["ball_receipt"] = write_json(root / "ball_receipt.json", inputs["ball_receipt"])
    paths["primary"] = write_json(
        root / "ball_primary_candidates.json",
        {
            "schema_version": 1,
            "run_id": inputs["detection_receipt"]["run_id"],
            "video_id": inputs["detection_receipt"]["video_id"],
            "frames": inputs["primary_sidecar"],
        },
    )
    return paths


class TrackingPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = _cfg()
        self.run_id = generate_run_id()

    def test_01_human_ball_fusion_and_namespace(self) -> None:
        inputs = build_minimal_tracking_fusion_inputs(
            self.run_id, n_frames=3, collide_track_ids=True
        )
        fused = fuse_tracking_bundle(
            human_observations=inputs["human_observations"],
            human_summaries=inputs["human_summaries"],
            human_lifecycle=inputs["human_lifecycle"],
            ball_observations=inputs["ball_observations"],
            ball_summaries=inputs["ball_summaries"],
            ball_lifecycle=inputs["ball_lifecycle"],
            primary_sidecar=inputs["primary_sidecar"],
            detections=inputs["detections"],
            detection_attributes=inputs["detection_attributes"],
            frames=inputs["frames"],
            analysis_windows=inputs["analysis_windows"],
            detection_receipt=inputs["detection_receipt"],
            human_receipt=inputs["human_receipt"],
            ball_receipt=inputs["ball_receipt"],
            config=self.config,
            validate=True,
        )
        self.assertEqual(fused.human_track_count, 1)
        self.assertEqual(fused.ball_track_count, 1)
        tids = {int(o["track_id"]) for o in fused.observations}
        self.assertEqual(len(tids), 2)
        self.assertIn(0, tids)
        self.assertTrue(fused.track_id_remap)  # ball remapped off human 0

    def test_02_track_id_uniqueness_remap(self) -> None:
        remap = compute_ball_track_remap(
            [{"track_id": 0}, {"track_id": 1}],
            [{"track_id": 0}, {"track_id": 1}],
            enabled=True,
        )
        self.assertEqual(remap[0], 2)
        self.assertEqual(remap[1], 3)
        values = set(remap.values())
        self.assertTrue(values.isdisjoint({0, 1}))

    def test_03_human_ball_fk_separation(self) -> None:
        inputs = build_minimal_tracking_fusion_inputs(self.run_id, n_frames=1)
        # Bind human observation to ball detection → fail
        bad = copy.deepcopy(inputs["human_observations"])
        bad[0]["detection_id"] = 1
        with self.assertRaises(TrackingFusionError) as ctx:
            merge_track_tables(
                human_observations=bad,
                human_summaries=inputs["human_summaries"],
                human_lifecycle=inputs["human_lifecycle"],
                ball_observations=inputs["ball_observations"],
                ball_summaries=inputs["ball_summaries"],
                ball_lifecycle=inputs["ball_lifecycle"],
                primary_sidecar=inputs["primary_sidecar"],
                detection_attributes=inputs["detection_attributes"],
                analysis_windows=inputs["analysis_windows"],
                config=self.config,
            )
        self.assertEqual(ctx.exception.code, "ENTITY_FK_MISMATCH")

    def test_04_duplicate_detection_assignment_rejected(self) -> None:
        inputs = build_minimal_tracking_fusion_inputs(
            self.run_id, n_frames=1, collide_track_ids=False
        )
        dup = copy.deepcopy(inputs["ball_observations"][0])
        dup["track_id"] = 99
        with self.assertRaises(TrackingFusionError) as ctx:
            merge_track_tables(
                human_observations=inputs["human_observations"],
                human_summaries=inputs["human_summaries"],
                human_lifecycle=inputs["human_lifecycle"],
                ball_observations=inputs["ball_observations"] + [dup],
                ball_summaries=inputs["ball_summaries"],
                ball_lifecycle=inputs["ball_lifecycle"],
                primary_sidecar=inputs["primary_sidecar"],
                detection_attributes=inputs["detection_attributes"],
                analysis_windows=inputs["analysis_windows"],
                config=self.config,
            )
        self.assertEqual(ctx.exception.code, "DUPLICATE_DETECTION_ASSIGNMENT")

    def test_05_lifecycle_summary_consistency(self) -> None:
        inputs = build_minimal_tracking_fusion_inputs(self.run_id, n_frames=3)
        fused = fuse_tracking_bundle(
            human_observations=inputs["human_observations"],
            human_summaries=inputs["human_summaries"],
            human_lifecycle=inputs["human_lifecycle"],
            ball_observations=inputs["ball_observations"],
            ball_summaries=inputs["ball_summaries"],
            ball_lifecycle=inputs["ball_lifecycle"],
            primary_sidecar=inputs["primary_sidecar"],
            detections=inputs["detections"],
            detection_attributes=inputs["detection_attributes"],
            frames=inputs["frames"],
            analysis_windows=inputs["analysis_windows"],
            detection_receipt=inputs["detection_receipt"],
            human_receipt=inputs["human_receipt"],
            ball_receipt=inputs["ball_receipt"],
            config=self.config,
            validate=True,
        )
        for s in fused.summaries:
            obs = [o for o in fused.observations if int(o["track_id"]) == int(s["track_id"])]
            self.assertEqual(int(s["observation_count"]), len(obs))
            self.assertEqual(
                int(s["observed_count"]),
                sum(1 for o in obs if o["observation_state"] == "observed"),
            )

    def test_06_source_timeline_mismatch_hard_fail(self) -> None:
        inputs = build_minimal_tracking_fusion_inputs(self.run_id, n_frames=1)
        inputs["ball_receipt"]["source_video_sha256"] = SOURCE_SHA_B
        inputs["ball_receipt"]["artifacts"]["source_video_sha256"] = SOURCE_SHA_B
        with self.assertRaises(TrackingFusionError) as ctx:
            fuse_tracking_bundle(
                human_observations=inputs["human_observations"],
                human_summaries=inputs["human_summaries"],
                human_lifecycle=inputs["human_lifecycle"],
                ball_observations=inputs["ball_observations"],
                ball_summaries=inputs["ball_summaries"],
                ball_lifecycle=inputs["ball_lifecycle"],
                primary_sidecar=inputs["primary_sidecar"],
                detections=inputs["detections"],
                detection_attributes=inputs["detection_attributes"],
                frames=inputs["frames"],
                analysis_windows=inputs["analysis_windows"],
                detection_receipt=inputs["detection_receipt"],
                human_receipt=inputs["human_receipt"],
                ball_receipt=inputs["ball_receipt"],
                config=self.config,
                validate=False,
            )
        self.assertEqual(ctx.exception.code, "SOURCE_SHA_MISMATCH")

        inputs2 = build_minimal_tracking_fusion_inputs(generate_run_id(), n_frames=1)
        inputs2["ball_receipt"]["timeline_fingerprint"] = TIMELINE_FP_B
        inputs2["ball_receipt"]["artifacts"]["timeline_fingerprint"] = TIMELINE_FP_B
        with self.assertRaises(TrackingFusionError) as ctx2:
            fuse_tracking_bundle(
                human_observations=inputs2["human_observations"],
                human_summaries=inputs2["human_summaries"],
                human_lifecycle=inputs2["human_lifecycle"],
                ball_observations=inputs2["ball_observations"],
                ball_summaries=inputs2["ball_summaries"],
                ball_lifecycle=inputs2["ball_lifecycle"],
                primary_sidecar=inputs2["primary_sidecar"],
                detections=inputs2["detections"],
                detection_attributes=inputs2["detection_attributes"],
                frames=inputs2["frames"],
                analysis_windows=inputs2["analysis_windows"],
                detection_receipt=inputs2["detection_receipt"],
                human_receipt=inputs2["human_receipt"],
                ball_receipt=inputs2["ball_receipt"],
                config=self.config,
                validate=False,
            )
        self.assertEqual(ctx2.exception.code, "TIMELINE_FINGERPRINT_MISMATCH")

        inputs3 = build_minimal_tracking_fusion_inputs(generate_run_id(), n_frames=1)
        inputs3["human_receipt"]["detection_bundle_fingerprint"] = DETECTION_FP_B
        inputs3["human_receipt"]["artifacts"]["detection_bundle_fingerprint"] = DETECTION_FP_B
        with self.assertRaises(TrackingFusionError) as ctx3:
            fuse_tracking_bundle(
                human_observations=inputs3["human_observations"],
                human_summaries=inputs3["human_summaries"],
                human_lifecycle=inputs3["human_lifecycle"],
                ball_observations=inputs3["ball_observations"],
                ball_summaries=inputs3["ball_summaries"],
                ball_lifecycle=inputs3["ball_lifecycle"],
                primary_sidecar=inputs3["primary_sidecar"],
                detections=inputs3["detections"],
                detection_attributes=inputs3["detection_attributes"],
                frames=inputs3["frames"],
                analysis_windows=inputs3["analysis_windows"],
                detection_receipt=inputs3["detection_receipt"],
                human_receipt=inputs3["human_receipt"],
                ball_receipt=inputs3["ball_receipt"],
                config=self.config,
                validate=False,
            )
        self.assertEqual(ctx3.exception.code, "DETECTION_FINGERPRINT_MISMATCH")

    def test_07_cross_cut_and_terminated_reopen(self) -> None:
        rid = self.run_id
        vid = "v1"
        windows = [
            make_analysis_window_row(rid, vid, n_frames=2, shot_id="shot_a", window_id="aw1"),
            make_analysis_window_row(
                rid,
                vid,
                n_frames=2,
                shot_id="shot_b",
                window_id="aw2",
                start_frame=2,
                replay_status="replay",
            ),
        ]
        obs = [
            {"run_id": rid, "video_id": vid, "track_id": 0, "frame_index": 0},
            {"run_id": rid, "video_id": vid, "track_id": 0, "frame_index": 3},
        ]
        viol = detect_cross_cut_violations(obs, [], windows)
        self.assertTrue(viol)

        life = [
            {
                "run_id": rid,
                "video_id": vid,
                "track_id": 0,
                "event_index": 0,
                "lifecycle_state": "terminated",
            },
            {
                "run_id": rid,
                "video_id": vid,
                "track_id": 0,
                "event_index": 1,
                "lifecycle_state": "confirmed",
            },
        ]
        self.assertEqual(detect_terminated_reopen(life), 1)

    def test_08_predicted_flags_unknown_role_ambiguous_ball(self) -> None:
        inputs = build_minimal_tracking_fusion_inputs(self.run_id, n_frames=3)
        fused = fuse_tracking_bundle(
            human_observations=inputs["human_observations"],
            human_summaries=inputs["human_summaries"],
            human_lifecycle=inputs["human_lifecycle"],
            ball_observations=inputs["ball_observations"],
            ball_summaries=inputs["ball_summaries"],
            ball_lifecycle=inputs["ball_lifecycle"],
            primary_sidecar=inputs["primary_sidecar"],
            detections=inputs["detections"],
            detection_attributes=inputs["detection_attributes"],
            frames=inputs["frames"],
            analysis_windows=inputs["analysis_windows"],
            detection_receipt=inputs["detection_receipt"],
            human_receipt=inputs["human_receipt"],
            ball_receipt=inputs["ball_receipt"],
            config=self.config,
            validate=True,
        )
        preds = [o for o in fused.observations if o["observation_state"] == "predicted"]
        self.assertTrue(preds)
        for p in preds:
            flags = list(p.get("quality_flags") or [])
            self.assertIn("physical_metric_ineligible", flags)
            self.assertIn("event_ineligible", flags)
            self.assertIsNone(p.get("detection_id"))

        unknowns = [
            a
            for a in inputs["detection_attributes"]
            if a["entity_type"] == "human" and a["role_label"] == "unknown"
        ]
        self.assertTrue(unknowns)

        amb = [f for f in fused.primary_sidecar if f.get("status") == "ambiguous"]
        self.assertTrue(amb)
        for a in amb:
            self.assertIsNone(a.get("primary_track_id"))

    def test_09_quality_statuses_review_sampling_gt_code(self) -> None:
        inputs = build_minimal_tracking_fusion_inputs(self.run_id, n_frames=6)
        fused = fuse_tracking_bundle(
            human_observations=inputs["human_observations"],
            human_summaries=inputs["human_summaries"],
            human_lifecycle=inputs["human_lifecycle"],
            ball_observations=inputs["ball_observations"],
            ball_summaries=inputs["ball_summaries"],
            ball_lifecycle=inputs["ball_lifecycle"],
            primary_sidecar=inputs["primary_sidecar"],
            detections=inputs["detections"],
            detection_attributes=inputs["detection_attributes"],
            frames=inputs["frames"],
            analysis_windows=inputs["analysis_windows"],
            detection_receipt=inputs["detection_receipt"],
            human_receipt=inputs["human_receipt"],
            ball_receipt=inputs["ball_receipt"],
            config=self.config,
            validate=True,
        )
        q = evaluate_tracking_quality(
            observations=fused.observations,
            summaries=fused.summaries,
            lifecycle=fused.lifecycle,
            detection_attributes=inputs["detection_attributes"],
            primary_sidecar=fused.primary_sidecar,
            frames=inputs["frames"],
            analysis_windows=inputs["analysis_windows"],
            config=self.config,
            receipt_counts=fused.counts,
            has_reviewed_ground_truth=False,
        )
        self.assertEqual(q.ground_truth_evaluation_status, NOT_EVALUATED_TRACKING)
        self.assertIn(q.status, {"pass", "pass_with_findings"})
        self.assertIn(NOT_EVALUATED_TRACKING, q.findings)

        review = build_tracking_review_queue(
            observations=fused.observations,
            lifecycle=fused.lifecycle,
            detection_attributes=inputs["detection_attributes"],
            primary_sidecar=fused.primary_sidecar,
            config=self.config,
            quality=q,
            run_id=fused.run_id,
            video_id=fused.video_id,
        )
        self.assertLessEqual(len(review["items"]), 20)

        q_fail = evaluate_tracking_quality(
            observations=fused.observations,
            summaries=fused.summaries,
            lifecycle=fused.lifecycle,
            detection_attributes=inputs["detection_attributes"],
            primary_sidecar=fused.primary_sidecar,
            frames=inputs["frames"],
            analysis_windows=inputs["analysis_windows"],
            config=self.config,
            dangling_fk_count=1,
            has_reviewed_ground_truth=False,
        )
        self.assertEqual(q_fail.status, "fail")

    def test_10_atomic_overwrite_cleanup_determinism(self) -> None:
        inputs = build_minimal_tracking_fusion_inputs(self.run_id, n_frames=4)
        with tempfile.TemporaryDirectory(
            prefix="trk_pipe_test_", dir="/home/fdoblak/workspace/tracking_pipeline_checks"
        ) as td:
            root = Path(td)
            paths = _write_bundle(root, inputs)
            out1 = root / "out1"
            out1.mkdir()
            res1 = run_tracking_integrate(
                detections=str(paths["detections"]),
                detection_attributes=str(paths["detection_attributes"]),
                detection_receipt=str(paths["detection_receipt"]),
                human_observations=str(paths["human_observations"]),
                human_summaries=str(paths["human_summaries"]),
                human_lifecycle=str(paths["human_lifecycle"]),
                human_receipt=str(paths["human_receipt"]),
                ball_observations=str(paths["ball_observations"]),
                ball_summaries=str(paths["ball_summaries"]),
                ball_lifecycle=str(paths["ball_lifecycle"]),
                ball_receipt=str(paths["ball_receipt"]),
                output_dir=str(out1),
                config=self.config,
                contain_root=root,
                frames=str(paths["frames"]),
                analysis_windows=str(paths["analysis_windows"]),
                ball_primary_sidecar=str(paths["primary"]),
                expected_source_sha=SOURCE_SHA_A,
                expected_timeline_fp=TIMELINE_FP_A,
                expected_detection_fp=DETECTION_FP_A,
                expected_analysis_window_fp=WINDOW_FP_A,
            )
            self.assertTrue(res1.accepted, res1.error_code)
            receipt = json.loads(Path(str(res1.pipeline_receipt_json)).read_text(encoding="utf-8"))
            self.assertEqual(receipt["observed_count"], fused_obs_count(out1))
            self.assertIn("output_hashes", receipt)

            res_ow = run_tracking_integrate(
                detections=str(paths["detections"]),
                detection_attributes=str(paths["detection_attributes"]),
                detection_receipt=str(paths["detection_receipt"]),
                human_observations=str(paths["human_observations"]),
                human_summaries=str(paths["human_summaries"]),
                human_lifecycle=str(paths["human_lifecycle"]),
                human_receipt=str(paths["human_receipt"]),
                ball_observations=str(paths["ball_observations"]),
                ball_summaries=str(paths["ball_summaries"]),
                ball_lifecycle=str(paths["ball_lifecycle"]),
                ball_receipt=str(paths["ball_receipt"]),
                output_dir=str(out1),
                config=self.config,
                contain_root=root,
                expected_source_sha=SOURCE_SHA_A,
                expected_timeline_fp=TIMELINE_FP_A,
                expected_detection_fp=DETECTION_FP_A,
                expected_analysis_window_fp=WINDOW_FP_A,
            )
            self.assertFalse(res_ow.accepted)
            self.assertEqual(res_ow.error_code, "OVERWRITE_FORBIDDEN")

            out2 = root / "out2"
            out2.mkdir()
            res2 = run_tracking_integrate(
                detections=str(paths["detections"]),
                detection_attributes=str(paths["detection_attributes"]),
                detection_receipt=str(paths["detection_receipt"]),
                human_observations=str(paths["human_observations"]),
                human_summaries=str(paths["human_summaries"]),
                human_lifecycle=str(paths["human_lifecycle"]),
                human_receipt=str(paths["human_receipt"]),
                ball_observations=str(paths["ball_observations"]),
                ball_summaries=str(paths["ball_summaries"]),
                ball_lifecycle=str(paths["ball_lifecycle"]),
                ball_receipt=str(paths["ball_receipt"]),
                output_dir=str(out2),
                config=self.config,
                contain_root=root,
                frames=str(paths["frames"]),
                analysis_windows=str(paths["analysis_windows"]),
                ball_primary_sidecar=str(paths["primary"]),
                expected_source_sha=SOURCE_SHA_A,
                expected_timeline_fp=TIMELINE_FP_A,
                expected_detection_fp=DETECTION_FP_A,
                expected_analysis_window_fp=WINDOW_FP_A,
            )
            self.assertTrue(res2.accepted)
            self.assertEqual(res1.total_track_count, res2.total_track_count)

    def test_11_fingerprint_regression(self) -> None:
        fps = tracking_schema_fingerprints()
        self.assertEqual(fps["detections"], EXPECTED_DETECTIONS_FP)
        self.assertEqual(fps["track_observations"], EXPECTED_OBS_FP)
        self.assertEqual(fps["track_summaries"], EXPECTED_SUM_FP)
        self.assertTrue(fps["track_lifecycle"].startswith("613cd81e"))
        self.assertEqual(
            contract_fingerprint(get_contract("detections", 1)), EXPECTED_DETECTIONS_FP
        )


def fused_obs_count(out_dir: Path) -> int:
    from football_analytics.data.parquet import read_contract_parquet

    table = read_contract_parquet(
        out_dir / "track_observations.parquet",
        get_contract("track_observations", 1),
        contain_root=out_dir.parent,
    )
    return sum(1 for r in table.to_pylist() if r["observation_state"] == "observed")


if __name__ == "__main__":
    unittest.main()
