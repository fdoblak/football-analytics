"""Stage 5A detection contract compile / typed model tests."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

from football_analytics.core.run_id import generate_run_id
from football_analytics.data.compiler import list_contracts
from football_analytics.data.registry import (
    default_project_root,
    default_registry_path,
    load_schema_registry,
)
from football_analytics.data.validation import validate_table
from football_analytics.perception import (
    CONTRACT_NAMES,
    CONTRACT_VERSION,
    DetectionAttributes,
    DetectionFrameStatus,
    DetectionRunReceipt,
    EntityType,
    RoleLabel,
    RoleSource,
    assert_detection_contracts_registered,
    compile_detection_schemas,
    detection_schema_fingerprints,
    load_all_detection_contracts,
    load_perception_json_schema,
    validate_against_json_schema,
)
from football_analytics.perception.types import PerceptionContractError

ROOT = default_project_root()
DETECTIONS_JSON = ROOT / "schemas/data/v1/detections.json"
DETECTIONS_SHA256 = "957a41ca2ded9580bc18d39bc7902e133b34ec866077ccc944ab334b9e2681fd"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DetectionContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reg = load_schema_registry(default_registry_path(), project_root=ROOT)
        self.run_id = generate_run_id()

    def test_01_contracts_registered(self) -> None:
        assert_detection_contracts_registered(registry=self.reg)
        names = load_all_detection_contracts(registry=self.reg)
        self.assertEqual(len(list_contracts(registry=self.reg)), 27)
        for n in CONTRACT_NAMES:
            self.assertIn(n, names)

    def test_02_compile_and_fingerprint(self) -> None:
        schemas = compile_detection_schemas(registry=self.reg)
        fps = detection_schema_fingerprints(registry=self.reg)
        self.assertEqual(fps, detection_schema_fingerprints(registry=self.reg))
        for name in (*CONTRACT_NAMES, "detections"):
            self.assertIn(name, schemas)
            self.assertEqual(len(fps[name]), 64)

    def test_03_detections_json_unchanged(self) -> None:
        self.assertEqual(_sha256(DETECTIONS_JSON), DETECTIONS_SHA256)

    def test_04_frame_status_roundtrip(self) -> None:
        payload = {
            "run_id": self.run_id,
            "video_id": "clip_demo_01",
            "frame_index": 0,
            "video_time_us": 0,
            "analysis_window_id": "aw_001",
            "processing_status": "processed_no_detections",
            "eligibility": "eligible",
            "detector_id": "det_dummy_v1",
            "input_artifact_ref": None,
            "detection_count": 0,
            "human_count": 0,
            "ball_count": 0,
            "skip_reason": None,
            "error_code": None,
            "coverage": 1.0,
            "provenance_json": None,
            "contract_version": CONTRACT_VERSION,
        }
        obj = DetectionFrameStatus.from_dict(payload)
        self.assertEqual(DetectionFrameStatus.from_dict(obj.to_dict()).to_dict(), obj.to_dict())
        schema = compile_detection_schemas(registry=self.reg)["detection_frame_status"]
        import pyarrow as pa

        table = pa.Table.from_pylist([obj.to_dict()], schema=schema)
        vr = validate_table(table, self.reg.load_contract("detection_frame_status", 1))
        self.assertEqual(vr.status, "PASS", vr.errors)

    def test_05_processed_requires_positive_count(self) -> None:
        base = {
            "run_id": self.run_id,
            "video_id": "clip_demo_01",
            "frame_index": 0,
            "video_time_us": 0,
            "analysis_window_id": None,
            "processing_status": "processed",
            "eligibility": "eligible",
            "detector_id": "det_dummy_v1",
            "input_artifact_ref": None,
            "detection_count": 0,
            "human_count": 0,
            "ball_count": 0,
            "skip_reason": None,
            "error_code": None,
            "coverage": 1.0,
            "provenance_json": None,
            "contract_version": 1,
        }
        with self.assertRaises(PerceptionContractError):
            DetectionFrameStatus.from_dict(base)

    def test_06_attributes_person_unknown_and_ball_role(self) -> None:
        human = DetectionAttributes.from_dict(
            {
                "run_id": self.run_id,
                "video_id": "clip_demo_01",
                "frame_index": 0,
                "detection_id": 0,
                "entity_type": "human",
                "role_label": "unknown",
                "role_source": "unknown",
                "role_score": None,
                "occlusion": None,
                "truncation": None,
                "visibility": 1.0,
                "review_status": "unreviewed",
                "attribute_source_ref": None,
                "provenance_json": None,
                "contract_version": 1,
            }
        )
        self.assertEqual(human.entity_type, EntityType.HUMAN)
        self.assertEqual(human.role_label, RoleLabel.UNKNOWN)
        with self.assertRaises(PerceptionContractError):
            DetectionAttributes.from_dict(
                {
                    "run_id": self.run_id,
                    "video_id": "clip_demo_01",
                    "frame_index": 0,
                    "detection_id": 1,
                    "entity_type": "ball",
                    "role_label": "player",
                    "role_source": "unknown",
                    "role_score": None,
                    "occlusion": None,
                    "truncation": None,
                    "visibility": None,
                    "review_status": "unreviewed",
                    "attribute_source_ref": None,
                    "provenance_json": None,
                    "contract_version": 1,
                }
            )

    def test_07_goalkeeper_manual_role(self) -> None:
        obj = DetectionAttributes.from_dict(
            {
                "run_id": self.run_id,
                "video_id": "clip_demo_01",
                "frame_index": 0,
                "detection_id": 2,
                "entity_type": "human",
                "role_label": "goalkeeper",
                "role_source": "manual_review",
                "role_score": 0.0,
                "occlusion": 0.2,
                "truncation": 0.0,
                "visibility": 0.8,
                "review_status": "accepted",
                "attribute_source_ref": "reviewer_a",
                "provenance_json": None,
                "contract_version": 1,
            }
        )
        self.assertEqual(obj.role_source, RoleSource.MANUAL_REVIEW)

    def test_08_receipt_rejects_fake_sha_format_and_validates_schema(self) -> None:
        with self.assertRaises(PerceptionContractError):
            DetectionRunReceipt.from_dict(
                {
                    "schema_version": 1,
                    "receipt_id": "det_receipt_01",
                    "run_id": self.run_id,
                    "detector_id": "det_dummy_v1",
                    "model_registry_id": None,
                    "model_sha256": "notasha",
                    "adapter_id": "a",
                    "adapter_version": "0",
                    "config_fingerprint": "a" * 64,
                    "taxonomy_version": "1.0.0",
                    "source_video_ref": "v",
                    "frames_ref": "f",
                    "analysis_windows_ref": None,
                    "eligible_frame_count": 0,
                    "processed_frame_count": 0,
                    "skipped_frame_count": 0,
                    "failed_frame_count": 0,
                    "processed_no_detection_count": 0,
                    "total_detection_count": 0,
                    "human_detection_count": 0,
                    "ball_detection_count": 0,
                    "pre_nms_count": None,
                    "post_nms_count": None,
                    "started_at_utc": "2026-07-23T00:00:00.000000Z",
                    "completed_at_utc": "2026-07-23T00:00:01.000000Z",
                    "status": "succeeded",
                    "warnings": [],
                    "errors": [],
                    "artifacts": {},
                    "environment_ref": None,
                    "provenance": {"stage": "5A", "label": "t", "notes": None},
                }
            )
        ok = DetectionRunReceipt.from_dict(
            {
                "schema_version": 1,
                "receipt_id": "det_receipt_01",
                "run_id": self.run_id,
                "detector_id": "det_dummy_v1",
                "model_registry_id": None,
                "model_sha256": None,
                "adapter_id": "synthetic_adapter",
                "adapter_version": "0.0.0",
                "config_fingerprint": "b" * 64,
                "taxonomy_version": "1.0.0",
                "source_video_ref": "logical",
                "frames_ref": "frames.parquet",
                "analysis_windows_ref": None,
                "eligible_frame_count": 1,
                "processed_frame_count": 1,
                "skipped_frame_count": 0,
                "failed_frame_count": 0,
                "processed_no_detection_count": 1,
                "total_detection_count": 0,
                "human_detection_count": 0,
                "ball_detection_count": 0,
                "pre_nms_count": 0,
                "post_nms_count": 0,
                "started_at_utc": "2026-07-23T00:00:00.000000Z",
                "completed_at_utc": "2026-07-23T00:00:01.000000Z",
                "status": "succeeded",
                "warnings": [],
                "errors": [],
                "artifacts": {},
                "environment_ref": None,
                "provenance": {"stage": "5A", "label": "unit", "notes": None},
            }
        )
        validate_against_json_schema(
            ok.to_dict(), load_perception_json_schema("detection_run_receipt")
        )


if __name__ == "__main__":
    unittest.main()
