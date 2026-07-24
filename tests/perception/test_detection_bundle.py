"""Detection bundle FK / count consistency tests."""

from __future__ import annotations

import unittest
from typing import Any

from football_analytics.core.run_id import generate_run_id
from football_analytics.data.bundle import build_synthetic_bundle, validate_contract_bundle
from football_analytics.data.compiler import compile_arrow_schema, get_contract, list_contracts
from football_analytics.data.registry import (
    default_project_root,
    default_registry_path,
    load_schema_registry,
)
from football_analytics.perception.validation import validate_detection_bundle


def _cast(name: str, rows: list[dict[str, Any]]) -> Any:
    import pyarrow as pa

    return pa.Table.from_pylist(rows, schema=compile_arrow_schema(get_contract(name, 1)))


def _attr(
    run_id: str,
    *,
    frame: int,
    det: int,
    entity: str = "human",
    role: str = "unknown",
    source: str = "unknown",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": "clip_demo_01",
        "frame_index": frame,
        "detection_id": det,
        "entity_type": entity,
        "role_label": role,
        "role_source": source,
        "role_score": None,
        "occlusion": None,
        "truncation": None,
        "visibility": 1.0,
        "review_status": "unreviewed",
        "attribute_source_ref": None,
        "provenance_json": None,
        "contract_version": 1,
    }


def _status(
    run_id: str,
    *,
    frame: int,
    status: str,
    det_count: int,
    human: int = 0,
    ball: int = 0,
    window: str | None = "aw_001",
    eligibility: str = "eligible",
    skip: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": "clip_demo_01",
        "frame_index": frame,
        "video_time_us": frame * 40000,
        "analysis_window_id": window,
        "processing_status": status,
        "eligibility": eligibility,
        "detector_id": "det_dummy_v1",
        "input_artifact_ref": None,
        "detection_count": det_count,
        "human_count": human,
        "ball_count": ball,
        "skip_reason": skip,
        "error_code": error,
        "coverage": 1.0,
        "provenance_json": None,
        "contract_version": 1,
    }


class DetectionBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.run_id = generate_run_id()
        self.reg = load_schema_registry(
            default_registry_path(), project_root=default_project_root()
        )

    def test_01_synthetic_nine_core_unchanged(self) -> None:
        bundle = build_synthetic_bundle(self.run_id)
        self.assertEqual(len(bundle), 9)
        self.assertNotIn("detection_frame_status", bundle)
        specs = {n: self.reg.load_contract(n, 1) for n in bundle}
        vr = validate_contract_bundle(bundle, specs)
        self.assertEqual(vr.status, "PASS", vr.errors)
        self.assertEqual(len(list_contracts(registry=self.reg)), 27)

    def test_02_valid_detection_sidecar_bundle(self) -> None:
        base = build_synthetic_bundle(self.run_id)
        # Replace detections with known rows matching attributes
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
                    "bbox_x1": 10.0,
                    "bbox_y1": 20.0,
                    "bbox_x2": 40.0,
                    "bbox_y2": 80.0,
                    "model_id": "det_dummy_v1",
                    "is_interpolated": False,
                    "quality_flags": [],
                }
            ],
        )
        attrs = _cast("detection_attributes", [_attr(self.run_id, frame=0, det=0)])
        status = _cast(
            "detection_frame_status",
            [
                _status(
                    self.run_id,
                    frame=0,
                    status="processed",
                    det_count=1,
                    human=1,
                    window=None,
                ),
                _status(
                    self.run_id,
                    frame=1,
                    status="processed_no_detections",
                    det_count=0,
                    window=None,
                ),
                _status(
                    self.run_id,
                    frame=2,
                    status="skipped",
                    det_count=0,
                    window=None,
                    eligibility="unknown",
                    skip="UNKNOWN_PLAYABILITY",
                    error="UNKNOWN_PLAYABILITY",
                ),
                _status(
                    self.run_id,
                    frame=3,
                    status="failed",
                    det_count=0,
                    window=None,
                    eligibility="eligible",
                    error="INFERENCE_FAILED",
                ),
            ],
        )
        tables = {
            **base,
            "detections": detections,
            "detection_attributes": attrs,
            "detection_frame_status": status,
        }
        specs = {n: self.reg.load_contract(n, 1) for n in tables}
        vr = validate_contract_bundle(tables, specs)
        self.assertEqual(vr.status, "PASS", vr.errors)

    def test_03_processed_no_detections_with_rows_fails(self) -> None:
        base = build_synthetic_bundle(self.run_id)
        status = _cast(
            "detection_frame_status",
            [
                _status(
                    self.run_id,
                    frame=0,
                    status="processed_no_detections",
                    det_count=0,
                    window=None,
                )
            ],
        )
        # base already has a detection on frame 0
        vr = validate_detection_bundle(
            detections=base["detections"],
            frame_status=status,
            attributes=None,
            frames=base["frames"],
            videos=base["videos"],
        )
        self.assertEqual(vr.status, "FAIL")

    def test_04_duplicate_detection_id_fails(self) -> None:
        rows = [
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
                "bbox_x2": 2.0,
                "bbox_y2": 2.0,
                "model_id": "det_dummy_v1",
                "is_interpolated": False,
                "quality_flags": [],
            },
            {
                "run_id": self.run_id,
                "video_id": "clip_demo_01",
                "frame_index": 0,
                "detection_id": 0,
                "class_id": 0,
                "class_name": "person",
                "confidence": 0.8,
                "bbox_x1": 3.0,
                "bbox_y1": 3.0,
                "bbox_x2": 4.0,
                "bbox_y2": 4.0,
                "model_id": "det_dummy_v1",
                "is_interpolated": False,
                "quality_flags": [],
            },
        ]
        import pyarrow as pa

        # Bypass schema cast uniqueness by building table without validate
        schema = compile_arrow_schema(get_contract("detections", 1))
        table = pa.Table.from_pylist(rows, schema=schema)
        vr = validate_detection_bundle(
            detections=table,
            frame_status=None,
            attributes=None,
        )
        self.assertEqual(vr.status, "FAIL")

    def test_05_missing_attribute_fk_fails(self) -> None:
        base = build_synthetic_bundle(self.run_id)
        attrs = _cast(
            "detection_attributes",
            [_attr(self.run_id, frame=0, det=99)],
        )
        vr = validate_detection_bundle(
            detections=base["detections"],
            frame_status=None,
            attributes=attrs,
            frames=base["frames"],
        )
        self.assertEqual(vr.status, "FAIL")

    def test_06_skipped_with_detection_rows_fails(self) -> None:
        base = build_synthetic_bundle(self.run_id)
        status = _cast(
            "detection_frame_status",
            [
                _status(
                    self.run_id,
                    frame=0,
                    status="skipped",
                    det_count=0,
                    window=None,
                    skip="FRAME_NOT_ELIGIBLE",
                    error="FRAME_NOT_ELIGIBLE",
                    eligibility="ineligible",
                )
            ],
        )
        vr = validate_detection_bundle(
            detections=base["detections"],
            frame_status=status,
            attributes=None,
            frames=base["frames"],
        )
        self.assertEqual(vr.status, "FAIL")


if __name__ == "__main__":
    unittest.main()
