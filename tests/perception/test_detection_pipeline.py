"""Stage 5E detection fusion + quality pipeline tests."""

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
from football_analytics.perception.contracts import detection_schema_fingerprints
from football_analytics.perception.detection_fusion import (
    DetectionFusionError,
    fuse_detection_bundle,
    merge_detections_and_attributes,
    merge_frame_status,
)
from football_analytics.perception.detection_pipeline import run_detection_integrate
from football_analytics.perception.detection_pipeline_config import (
    detection_pipeline_config_fingerprint,
    load_detection_pipeline_config,
)
from football_analytics.perception.detection_pipeline_fixtures import (
    SOURCE_SHA_A,
    SOURCE_SHA_B,
    TIMELINE_FP_A,
    TIMELINE_FP_B,
    build_minimal_fusion_inputs,
    make_detection_row,
    make_frame_status_row,
    make_human_attribute_row,
    write_json,
)
from football_analytics.perception.detection_quality import (
    NOT_EVALUATED_DETECTION,
    build_detection_review_queue,
    evaluate_detection_quality,
)

EXPECTED_DETECTIONS_FP = "04ae8dd7a7e92bf7bd468db7a263e5e28258a30887d43c8f603c69d56f5c18b6"


def _cfg() -> dict[str, Any]:
    return load_detection_pipeline_config(
        default_project_root() / "configs/perception/detection_pipeline.yaml"
    )


def _tbl(name: str, rows: list[dict[str, Any]]) -> Any:
    schema = compile_arrow_schema(get_contract(name, 1))
    return pa.Table.from_pylist(rows, schema=schema) if rows else schema.empty_table()


def _write_bundle(root: Path, inputs: dict[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    mapping = {
        "human_detections": ("detections", "human_detections.parquet"),
        "human_frame_status": ("detection_frame_status", "human_frame_status.parquet"),
        "human_attributes": ("detection_attributes", "human_attributes.parquet"),
        "ball_detections": ("detections", "ball_detections.parquet"),
        "ball_frame_status": ("detection_frame_status", "ball_frame_status.parquet"),
        "ball_attributes": ("detection_attributes", "ball_attributes.parquet"),
        "role_attributes": ("detection_attributes", "role_attributes.parquet"),
        "frames": ("frames", "frames.parquet"),
        "analysis_windows": ("analysis_windows", "analysis_windows.parquet"),
    }
    for key, (contract, fname) in mapping.items():
        p = root / fname
        write_contract_parquet(
            _tbl(contract, list(inputs[key]) if key != "analysis_windows" else list(inputs[key])),
            p,
            get_contract(contract, 1),
            contain_root=root,
        )
        paths[key] = p
    paths["human_receipt"] = write_json(root / "human_receipt.json", inputs["human_receipt"])
    paths["ball_receipt"] = write_json(root / "ball_receipt.json", inputs["ball_receipt"])
    paths["role_receipt"] = write_json(root / "role_receipt.json", inputs["role_receipt"])
    return paths


class DetectionPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = _cfg()
        self.run_id = generate_run_id()

    def test_01_human_ball_merge_and_uniqueness(self) -> None:
        inputs = build_minimal_fusion_inputs(self.run_id, n_frames=3, collide_ball_id=True)
        fused = fuse_detection_bundle(
            human_detections=inputs["human_detections"],
            human_frame_status=inputs["human_frame_status"],
            human_attributes=inputs["human_attributes"],
            ball_detections=inputs["ball_detections"],
            ball_frame_status=inputs["ball_frame_status"],
            ball_attributes=inputs["ball_attributes"],
            role_attributes=inputs["role_attributes"],
            human_receipt=inputs["human_receipt"],
            ball_receipt=inputs["ball_receipt"],
            role_receipt=inputs["role_receipt"],
            config=self.config,
            validate=True,
        )
        self.assertEqual(fused.human_detection_count, 3)
        self.assertEqual(fused.ball_detection_count, 3)
        pks = {(d["frame_index"], d["detection_id"]) for d in fused.detections}
        self.assertEqual(len(pks), len(fused.detections))
        # Ball remapped off human id 0
        self.assertTrue(any(v == 1 for v in fused.id_remap.values()) or fused.id_remap)

    def test_02_cross_class_preserve_no_nms(self) -> None:
        inputs = build_minimal_fusion_inputs(self.run_id, n_frames=1, collide_ball_id=False)
        # Overlapping boxes intentionally — both must survive (no cross-class NMS).
        inputs["human_detections"][0]["bbox_x1"] = 10
        inputs["human_detections"][0]["bbox_x2"] = 60
        inputs["ball_detections"][0]["bbox_x1"] = 40
        inputs["ball_detections"][0]["bbox_x2"] = 55
        fused = fuse_detection_bundle(
            human_detections=inputs["human_detections"],
            human_frame_status=inputs["human_frame_status"],
            human_attributes=inputs["human_attributes"],
            ball_detections=inputs["ball_detections"],
            ball_frame_status=inputs["ball_frame_status"],
            ball_attributes=inputs["ball_attributes"],
            role_attributes=inputs["role_attributes"],
            human_receipt=inputs["human_receipt"],
            ball_receipt=inputs["ball_receipt"],
            role_receipt=inputs["role_receipt"],
            config=self.config,
            validate=True,
        )
        self.assertEqual(len(fused.detections), 2)
        entities = {a["entity_type"] for a in fused.attributes}
        self.assertEqual(entities, {"human", "ball"})

    def test_03_attribute_fk_and_ball_role_reject(self) -> None:
        inputs = build_minimal_fusion_inputs(self.run_id, n_frames=1, collide_ball_id=False)
        bad = copy.deepcopy(inputs["ball_attributes"][0])
        bad["role_label"] = "player"
        with self.assertRaises(DetectionFusionError) as ctx:
            merge_detections_and_attributes(
                human_detections=inputs["human_detections"],
                ball_detections=inputs["ball_detections"],
                human_attributes=inputs["human_attributes"],
                ball_attributes=[bad],
                role_attributes=inputs["role_attributes"],
                config=self.config,
            )
        self.assertEqual(ctx.exception.code, "BALL_ROLE_FORBIDDEN")

        # Missing attr → dangling FK
        with self.assertRaises(DetectionFusionError) as ctx2:
            merge_detections_and_attributes(
                human_detections=inputs["human_detections"],
                ball_detections=inputs["ball_detections"],
                human_attributes=[],
                ball_attributes=inputs["ball_attributes"],
                role_attributes=[],
                config=self.config,
            )
        self.assertEqual(ctx2.exception.code, "DANGLING_FK")

    def test_04_fingerprint_and_source_sha_mismatch(self) -> None:
        inputs = build_minimal_fusion_inputs(self.run_id, n_frames=1)
        inputs["ball_receipt"]["source_video_sha256"] = SOURCE_SHA_B
        inputs["ball_receipt"]["artifacts"]["source_video_sha256"] = SOURCE_SHA_B
        with self.assertRaises(DetectionFusionError) as ctx:
            fuse_detection_bundle(
                human_detections=inputs["human_detections"],
                human_frame_status=inputs["human_frame_status"],
                human_attributes=inputs["human_attributes"],
                ball_detections=inputs["ball_detections"],
                ball_frame_status=inputs["ball_frame_status"],
                ball_attributes=inputs["ball_attributes"],
                role_attributes=inputs["role_attributes"],
                human_receipt=inputs["human_receipt"],
                ball_receipt=inputs["ball_receipt"],
                role_receipt=inputs["role_receipt"],
                config=self.config,
                validate=False,
            )
        self.assertEqual(ctx.exception.code, "SOURCE_SHA_MISMATCH")

        inputs2 = build_minimal_fusion_inputs(generate_run_id(), n_frames=1)
        inputs2["ball_receipt"]["timeline_fingerprint"] = TIMELINE_FP_B
        inputs2["ball_receipt"]["artifacts"]["timeline_fingerprint"] = TIMELINE_FP_B
        with self.assertRaises(DetectionFusionError) as ctx3:
            fuse_detection_bundle(
                human_detections=inputs2["human_detections"],
                human_frame_status=inputs2["human_frame_status"],
                human_attributes=inputs2["human_attributes"],
                ball_detections=inputs2["ball_detections"],
                ball_frame_status=inputs2["ball_frame_status"],
                ball_attributes=inputs2["ball_attributes"],
                role_attributes=inputs2["role_attributes"],
                human_receipt=inputs2["human_receipt"],
                ball_receipt=inputs2["ball_receipt"],
                role_receipt=inputs2["role_receipt"],
                config=self.config,
                validate=False,
            )
        self.assertEqual(ctx3.exception.code, "TIMELINE_FINGERPRINT_MISMATCH")

    def test_05_invalid_bbox_and_receipt_mismatch_quality(self) -> None:
        rid = self.run_id
        det = make_detection_row(rid, "video_01", frame_index=0, detection_id=0, bbox=[1, 1, 2, 2])
        det["bbox_x2"] = float("nan")
        with self.assertRaises(DetectionFusionError) as ctx:
            merge_detections_and_attributes(
                human_detections=[det],
                ball_detections=[],
                human_attributes=[
                    make_human_attribute_row(rid, "video_01", frame_index=0, detection_id=0)
                ],
                ball_attributes=[],
                role_attributes=None,
                config=self.config,
            )
        self.assertEqual(ctx.exception.code, "INVALID_BBOX")

        q = evaluate_detection_quality(
            detections=[],
            frame_status=[
                make_frame_status_row(rid, "video_01", frame_index=0, processing_status="failed")
            ],
            attributes=[],
            config=self.config,
            receipt_counts={"processed_frame_count": 99},
            receipt_mismatch_count=1,
        )
        self.assertEqual(q.status, "fail")
        self.assertIn("receipt_count_mismatch", q.findings)

    def test_06_frame_status_eligibility_and_processed_no_det(self) -> None:
        rid = self.run_id
        human = [
            make_frame_status_row(
                rid, "v1", frame_index=0, processing_status="processed", human_count=1
            ),
            make_frame_status_row(
                rid, "v1", frame_index=1, processing_status="skipped", human_count=0
            ),
            make_frame_status_row(
                rid,
                "v1",
                frame_index=2,
                processing_status="processed_no_detections",
                human_count=0,
            ),
        ]
        ball = [
            make_frame_status_row(
                rid,
                "v1",
                frame_index=0,
                processing_status="skipped",
                ball_count=0,
                detector_id="ball",
            ),
            make_frame_status_row(
                rid,
                "v1",
                frame_index=1,
                processing_status="processed",
                ball_count=1,
                detector_id="ball",
            ),
            make_frame_status_row(
                rid,
                "v1",
                frame_index=2,
                processing_status="processed_no_detections",
                ball_count=0,
                detector_id="ball",
            ),
        ]
        merged = merge_frame_status(human, ball, config=self.config)
        by_fi = {r["frame_index"]: r for r in merged}
        self.assertEqual(by_fi[0]["processing_status"], "processed")
        self.assertEqual(by_fi[1]["processing_status"], "processed")
        self.assertEqual(by_fi[2]["processing_status"], "processed_no_detections")

        # Eligibility conflict
        with self.assertRaises(DetectionFusionError) as ctx:
            merge_frame_status(
                [make_frame_status_row(rid, "v1", frame_index=0, processing_status="not_eligible")],
                [
                    make_frame_status_row(
                        rid,
                        "v1",
                        frame_index=0,
                        processing_status="processed",
                        ball_count=1,
                        detector_id="ball",
                    )
                ],
                config=self.config,
            )
        self.assertEqual(ctx.exception.code, "ELIGIBILITY_CONFLICT")

    def test_07_unknown_preserved_review_sampling_quality_statuses(self) -> None:
        inputs = build_minimal_fusion_inputs(self.run_id, n_frames=6, collide_ball_id=True)
        fused = fuse_detection_bundle(
            human_detections=inputs["human_detections"],
            human_frame_status=inputs["human_frame_status"],
            human_attributes=inputs["human_attributes"],
            ball_detections=inputs["ball_detections"],
            ball_frame_status=inputs["ball_frame_status"],
            ball_attributes=inputs["ball_attributes"],
            role_attributes=inputs["role_attributes"],
            human_receipt=inputs["human_receipt"],
            ball_receipt=inputs["ball_receipt"],
            role_receipt=inputs["role_receipt"],
            config=self.config,
            validate=True,
        )
        unknowns = [
            a
            for a in fused.attributes
            if a["entity_type"] == "human" and a["role_label"] == "unknown"
        ]
        self.assertGreaterEqual(len(unknowns), 1)
        # Never auto-promote unknown
        human_roles = [a["role_label"] for a in fused.attributes if a["entity_type"] == "human"]
        self.assertTrue(all(r in {"player", "unknown"} for r in human_roles))

        q = evaluate_detection_quality(
            detections=fused.detections,
            frame_status=fused.frame_status,
            attributes=fused.attributes,
            config=self.config,
            has_reviewed_ground_truth=False,
        )
        self.assertEqual(q.ground_truth_evaluation_status, NOT_EVALUATED_DETECTION)
        self.assertIn(q.status, {"pass", "pass_with_findings"})

        review = build_detection_review_queue(
            attributes=fused.attributes,
            frame_status=fused.frame_status,
            detections=fused.detections,
            config=self.config,
            quality=q,
            run_id=fused.run_id,
            video_id=fused.video_id,
        )
        unknown_items = [
            i for i in review["items"] if "ROLE_UNKNOWN_ABSTENTION" in i["reason_codes"]
        ]
        self.assertLessEqual(
            len(unknown_items), self.config["review_policy"]["max_unknown_review_items"]
        )
        self.assertLess(len(unknown_items), max(len(unknowns), 1) + 1)

    def test_08_deterministic_atomic_cleanup_and_e2e(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = build_minimal_fusion_inputs(self.run_id, n_frames=4, collide_ball_id=True)
            paths = _write_bundle(root, inputs)
            out1 = root / "out1"
            out1.mkdir()
            res1 = run_detection_integrate(
                human_detections=str(paths["human_detections"]),
                human_frame_status=str(paths["human_frame_status"]),
                human_attributes=str(paths["human_attributes"]),
                human_receipt=str(paths["human_receipt"]),
                ball_detections=str(paths["ball_detections"]),
                ball_frame_status=str(paths["ball_frame_status"]),
                ball_attributes=str(paths["ball_attributes"]),
                ball_receipt=str(paths["ball_receipt"]),
                role_attributes=str(paths["role_attributes"]),
                role_receipt=str(paths["role_receipt"]),
                output_dir=str(out1),
                config=self.config,
                contain_root=root,
                analysis_windows=str(paths["analysis_windows"]),
                frames=str(paths["frames"]),
                expected_source_sha=SOURCE_SHA_A,
                expected_timeline_fp=TIMELINE_FP_A,
            )
            self.assertTrue(res1.accepted, res1.error_code)
            receipt1 = json.loads(Path(str(res1.pipeline_receipt_json)).read_text(encoding="utf-8"))
            self.assertEqual(receipt1["ground_truth_evaluation_status"], NOT_EVALUATED_DETECTION)

            # Deterministic second run (fresh out dir)
            out2 = root / "out2"
            out2.mkdir()
            res2 = run_detection_integrate(
                human_detections=str(paths["human_detections"]),
                human_frame_status=str(paths["human_frame_status"]),
                human_attributes=str(paths["human_attributes"]),
                human_receipt=str(paths["human_receipt"]),
                ball_detections=str(paths["ball_detections"]),
                ball_frame_status=str(paths["ball_frame_status"]),
                ball_attributes=str(paths["ball_attributes"]),
                ball_receipt=str(paths["ball_receipt"]),
                role_attributes=str(paths["role_attributes"]),
                role_receipt=str(paths["role_receipt"]),
                output_dir=str(out2),
                config=self.config,
                contain_root=root,
                analysis_windows=str(paths["analysis_windows"]),
                frames=str(paths["frames"]),
                expected_source_sha=SOURCE_SHA_A,
                expected_timeline_fp=TIMELINE_FP_A,
            )
            self.assertTrue(res2.accepted, res2.error_code)
            self.assertEqual(res1.total_detection_count, res2.total_detection_count)
            self.assertEqual(res1.config_fingerprint, res2.config_fingerprint)

            # Atomic no-overwrite
            res3 = run_detection_integrate(
                human_detections=str(paths["human_detections"]),
                human_frame_status=str(paths["human_frame_status"]),
                human_attributes=str(paths["human_attributes"]),
                human_receipt=str(paths["human_receipt"]),
                ball_detections=str(paths["ball_detections"]),
                ball_frame_status=str(paths["ball_frame_status"]),
                ball_attributes=str(paths["ball_attributes"]),
                ball_receipt=str(paths["ball_receipt"]),
                role_attributes=str(paths["role_attributes"]),
                role_receipt=str(paths["role_receipt"]),
                output_dir=str(out1),
                config=self.config,
                contain_root=root,
                expected_source_sha=SOURCE_SHA_A,
                expected_timeline_fp=TIMELINE_FP_A,
            )
            self.assertFalse(res3.accepted)
            self.assertEqual(res3.error_code, "OVERWRITE_FORBIDDEN")

            # Failure cleanup: force SOURCE_SHA mismatch → no partial publish in new dir
            out_fail = root / "out_fail"
            out_fail.mkdir()
            bad_ball = copy.deepcopy(inputs["ball_receipt"])
            bad_ball["source_video_sha256"] = SOURCE_SHA_B
            bad_ball["artifacts"]["source_video_sha256"] = SOURCE_SHA_B
            write_json(root / "ball_receipt_bad.json", bad_ball)
            res_fail = run_detection_integrate(
                human_detections=str(paths["human_detections"]),
                human_frame_status=str(paths["human_frame_status"]),
                human_attributes=str(paths["human_attributes"]),
                human_receipt=str(paths["human_receipt"]),
                ball_detections=str(paths["ball_detections"]),
                ball_frame_status=str(paths["ball_frame_status"]),
                ball_attributes=str(paths["ball_attributes"]),
                ball_receipt=str(root / "ball_receipt_bad.json"),
                role_attributes=str(paths["role_attributes"]),
                role_receipt=str(paths["role_receipt"]),
                output_dir=str(out_fail),
                config=self.config,
                contain_root=root,
                expected_source_sha=SOURCE_SHA_A,
            )
            self.assertFalse(res_fail.accepted)
            self.assertFalse((out_fail / "detections.parquet").exists())
            # No leftover tmp dirs
            tmp_left = list(out_fail.glob(".tmp_fusion_*"))
            self.assertEqual(tmp_left, [])

    def test_09_detections_fingerprint_unchanged(self) -> None:
        fps = detection_schema_fingerprints()
        self.assertEqual(fps["detections"], EXPECTED_DETECTIONS_FP)
        self.assertEqual(
            contract_fingerprint(get_contract("detections", 1)), EXPECTED_DETECTIONS_FP
        )

    def test_10_config_fingerprint_stable(self) -> None:
        a = detection_pipeline_config_fingerprint(self.config)
        b = detection_pipeline_config_fingerprint(_cfg())
        self.assertEqual(a, b)
        self.assertEqual(len(a), 64)

    def test_11_cross_class_nms_flag_rejected(self) -> None:
        cfg = copy.deepcopy(self.config)
        cfg["fusion"]["cross_class_nms"] = True
        with self.assertRaises(DetectionFusionError):
            merge_detections_and_attributes(
                human_detections=[],
                ball_detections=[],
                human_attributes=[],
                ball_attributes=[],
                role_attributes=None,
                config=cfg,
            )


if __name__ == "__main__":
    unittest.main()
