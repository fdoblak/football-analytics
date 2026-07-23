"""Detection semantics: empty vs unprocessed, roles, routing, receipt totals."""

from __future__ import annotations

import unittest
from typing import Any

from football_analytics.core.run_id import generate_run_id
from football_analytics.data.compiler import compile_arrow_schema, get_contract
from football_analytics.perception import (
    DetectionAttributes,
    DetectionFrameStatus,
    DetectionRunReceipt,
    map_model_class,
    validate_detection_bundle,
)
from football_analytics.perception.policy import load_detection_policy, resolve_frame_routing
from football_analytics.perception.taxonomy import load_detection_taxonomy
from football_analytics.perception.types import PerceptionContractError, RoleLabel


def _cast(name: str, rows: list[dict[str, Any]]) -> Any:
    import pyarrow as pa

    return pa.Table.from_pylist(rows, schema=compile_arrow_schema(get_contract(name, 1)))


class DetectionSemanticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.run_id = generate_run_id()
        self.tax = load_detection_taxonomy()
        self.pol = load_detection_policy()

    def test_01_empty_processed_vs_unprocessed(self) -> None:
        empty = DetectionFrameStatus.from_dict(
            {
                "run_id": self.run_id,
                "video_id": "clip_demo_01",
                "frame_index": 0,
                "video_time_us": 0,
                "analysis_window_id": None,
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
                "contract_version": 1,
            }
        )
        skipped = DetectionFrameStatus.from_dict(
            {
                "run_id": self.run_id,
                "video_id": "clip_demo_01",
                "frame_index": 1,
                "video_time_us": 40000,
                "analysis_window_id": None,
                "processing_status": "skipped",
                "eligibility": "ineligible",
                "detector_id": "det_dummy_v1",
                "input_artifact_ref": None,
                "detection_count": 0,
                "human_count": 0,
                "ball_count": 0,
                "skip_reason": "FRAME_NOT_ELIGIBLE",
                "error_code": "FRAME_NOT_ELIGIBLE",
                "coverage": 1.0,
                "provenance_json": None,
                "contract_version": 1,
            }
        )
        self.assertNotEqual(empty.processing_status, skipped.processing_status)

    def test_02_downstream_role_player_after_person(self) -> None:
        mapped = map_model_class(0, "person", taxonomy=self.tax)
        self.assertEqual(mapped.role_label, RoleLabel.UNKNOWN)
        attrs = DetectionAttributes.from_dict(
            {
                "run_id": self.run_id,
                "video_id": "clip_demo_01",
                "frame_index": 0,
                "detection_id": 0,
                "entity_type": "human",
                "role_label": "player",
                "role_source": "downstream_classifier",
                "role_score": 0.6,
                "occlusion": None,
                "truncation": None,
                "visibility": 1.0,
                "review_status": "unreviewed",
                "attribute_source_ref": "role_clf",
                "provenance_json": None,
                "contract_version": 1,
            }
        )
        self.assertEqual(attrs.role_label, RoleLabel.PLAYER)

    def test_03_referee_imported(self) -> None:
        attrs = DetectionAttributes.from_dict(
            {
                "run_id": self.run_id,
                "video_id": "clip_demo_01",
                "frame_index": 0,
                "detection_id": 1,
                "entity_type": "human",
                "role_label": "referee",
                "role_source": "imported",
                "role_score": None,
                "occlusion": None,
                "truncation": None,
                "visibility": None,
                "review_status": "accepted",
                "attribute_source_ref": "label_set",
                "provenance_json": None,
                "contract_version": 1,
            }
        )
        self.assertEqual(attrs.role_label.value, "referee")

    def test_04_ball_ineligible_routing(self) -> None:
        window = {
            "playability": "playable",
            "graphics_status": "none",
            "tracking_eligibility": "eligible",
            "identity_eligibility": "unknown",
            "ball_analysis_eligibility": "ineligible",
        }
        r = resolve_frame_routing(window, policy=self.pol)
        self.assertTrue(r["process_human"])
        self.assertFalse(r["process_ball"])

    def test_05_score_bounds_and_receipt_totals(self) -> None:
        detections = _cast(
            "detections",
            [
                {
                    "run_id": self.run_id,
                    "video_id": "clip_demo_01",
                    "frame_index": 0,
                    "detection_id": 0,
                    "class_id": 0,
                    "class_name": "person",
                    "confidence": 0.9,
                    "bbox_x1": 1.0,
                    "bbox_y1": 1.0,
                    "bbox_x2": 5.0,
                    "bbox_y2": 5.0,
                    "model_id": "det_dummy_v1",
                    "is_interpolated": False,
                    "quality_flags": [],
                },
                {
                    "run_id": self.run_id,
                    "video_id": "clip_demo_01",
                    "frame_index": 0,
                    "detection_id": 1,
                    "class_id": 32,
                    "class_name": "ball",
                    "confidence": 0.4,
                    "bbox_x1": 10.0,
                    "bbox_y1": 10.0,
                    "bbox_x2": 12.0,
                    "bbox_y2": 12.0,
                    "model_id": "det_dummy_v1",
                    "is_interpolated": False,
                    "quality_flags": [],
                },
            ],
        )
        attrs = _cast(
            "detection_attributes",
            [
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
                },
                {
                    "run_id": self.run_id,
                    "video_id": "clip_demo_01",
                    "frame_index": 0,
                    "detection_id": 1,
                    "entity_type": "ball",
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
                },
            ],
        )
        status = _cast(
            "detection_frame_status",
            [
                {
                    "run_id": self.run_id,
                    "video_id": "clip_demo_01",
                    "frame_index": 0,
                    "video_time_us": 0,
                    "analysis_window_id": None,
                    "processing_status": "processed",
                    "eligibility": "eligible",
                    "detector_id": "det_dummy_v1",
                    "input_artifact_ref": None,
                    "detection_count": 2,
                    "human_count": 1,
                    "ball_count": 1,
                    "skip_reason": None,
                    "error_code": None,
                    "coverage": 1.0,
                    "provenance_json": None,
                    "contract_version": 1,
                },
                {
                    "run_id": self.run_id,
                    "video_id": "clip_demo_01",
                    "frame_index": 1,
                    "video_time_us": 40000,
                    "analysis_window_id": None,
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
                    "contract_version": 1,
                },
            ],
        )
        frames = _cast(
            "frames",
            [
                {
                    "run_id": self.run_id,
                    "video_id": "clip_demo_01",
                    "frame_index": i,
                    "pts": i,
                    "video_time_us": i * 40000,
                    "duration_us": 40000,
                    "is_key_frame": True,
                    "decode_status": "ok",
                }
                for i in (0, 1)
            ],
        )
        receipt = DetectionRunReceipt.from_dict(
            {
                "schema_version": 1,
                "receipt_id": "det_receipt_01",
                "run_id": self.run_id,
                "detector_id": "det_dummy_v1",
                "model_registry_id": None,
                "model_sha256": None,
                "adapter_id": "synthetic",
                "adapter_version": "0",
                "config_fingerprint": "c" * 64,
                "taxonomy_version": "1.0.0",
                "source_video_ref": "logical",
                "frames_ref": "frames.parquet",
                "analysis_windows_ref": None,
                "eligible_frame_count": 2,
                "processed_frame_count": 2,
                "skipped_frame_count": 0,
                "failed_frame_count": 0,
                "processed_no_detection_count": 1,
                "total_detection_count": 2,
                "human_detection_count": 1,
                "ball_detection_count": 1,
                "pre_nms_count": 4,
                "post_nms_count": 2,
                "started_at_utc": "2026-07-23T00:00:00.000000Z",
                "completed_at_utc": "2026-07-23T00:00:01.000000Z",
                "status": "succeeded",
                "warnings": [],
                "errors": [],
                "artifacts": {},
                "environment_ref": None,
                "provenance": {"stage": "5A", "label": "semantics", "notes": None},
            }
        )
        vr = validate_detection_bundle(
            detections=detections,
            frame_status=status,
            attributes=attrs,
            frames=frames,
            receipt=receipt.to_dict(),
        )
        self.assertEqual(vr.status, "PASS", vr.errors)

        bad_receipt = dict(receipt.to_dict())
        bad_receipt["total_detection_count"] = 99
        vr_bad = validate_detection_bundle(
            detections=detections,
            frame_status=status,
            attributes=attrs,
            frames=frames,
            receipt=bad_receipt,
        )
        self.assertEqual(vr_bad.status, "FAIL")

    def test_06_failed_requires_error_code(self) -> None:
        with self.assertRaises(PerceptionContractError):
            DetectionFrameStatus.from_dict(
                {
                    "run_id": self.run_id,
                    "video_id": "clip_demo_01",
                    "frame_index": 0,
                    "video_time_us": 0,
                    "analysis_window_id": None,
                    "processing_status": "failed",
                    "eligibility": "eligible",
                    "detector_id": "det_dummy_v1",
                    "input_artifact_ref": None,
                    "detection_count": 0,
                    "human_count": 0,
                    "ball_count": 0,
                    "skip_reason": None,
                    "error_code": None,
                    "coverage": 0.0,
                    "provenance_json": None,
                    "contract_version": 1,
                }
            )

    def test_07_nms_suppressed_not_in_canonical_policy(self) -> None:
        self.assertFalse(self.pol["nms"]["keep_suppressed_in_canonical"])
        self.assertTrue(self.pol["thresholds"]["calibrated_confidence_separate"])
        self.assertFalse(self.pol["provenance_requirements"]["invent_model_sha_when_missing"])


if __name__ == "__main__":
    unittest.main()
