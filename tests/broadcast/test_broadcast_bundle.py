"""Broadcast bundle integration with data.bundle."""

from __future__ import annotations

import unittest
from typing import Any

from football_analytics.broadcast.validation import validate_broadcast_bundle
from football_analytics.core.run_id import generate_run_id
from football_analytics.data.bundle import build_synthetic_bundle, validate_contract_bundle
from football_analytics.data.compiler import compile_arrow_schema, get_contract, list_contracts
from football_analytics.data.registry import (
    default_project_root,
    default_registry_path,
    load_schema_registry,
)


def _cast(name: str, rows: list[dict[str, Any]]) -> Any:
    import pyarrow as pa

    return pa.Table.from_pylist(rows, schema=compile_arrow_schema(get_contract(name, 1)))


class BroadcastBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.run_id = generate_run_id()
        self.reg = load_schema_registry(
            default_registry_path(), project_root=default_project_root()
        )

    def test_01_synthetic_bundle_still_nine_core_tables(self) -> None:
        bundle = build_synthetic_bundle(self.run_id)
        self.assertEqual(len(bundle), 9)
        self.assertNotIn("shot_boundaries", bundle)
        specs = {n: self.reg.load_contract(n, 1) for n in bundle}
        vr = validate_contract_bundle(bundle, specs)
        self.assertEqual(vr.status, "PASS", vr.errors)

    def test_02_registry_has_fifteen(self) -> None:
        self.assertEqual(len(list_contracts(registry=self.reg)), 15)

    def test_03_bundle_with_broadcast_tables(self) -> None:
        base = build_synthetic_bundle(self.run_id)
        video_id = "clip_demo_01"
        boundaries = _cast(
            "shot_boundaries",
            [
                {
                    "run_id": self.run_id,
                    "video_id": video_id,
                    "boundary_id": "bnd_001",
                    "boundary_time_us": 160000,
                    "left_frame_index": 3,
                    "right_frame_index": 4,
                    "transition_type": "hard_cut",
                    "transition_duration_us": 0,
                    "confidence": 0.9,
                    "detection_source": "manual",
                    "evidence_ref": None,
                    "review_status": "accepted",
                    "provenance_json": None,
                    "contract_version": 1,
                }
            ],
        )
        shots = _cast(
            "shot_segments",
            [
                {
                    "run_id": self.run_id,
                    "video_id": video_id,
                    "shot_id": "shot_001",
                    "start_time_us": 0,
                    "end_time_us": 160000,
                    "start_frame_index": 0,
                    "end_frame_index_exclusive": 4,
                    "start_boundary_id": None,
                    "end_boundary_id": "bnd_001",
                    "duration_us": 160000,
                    "frame_count": 4,
                    "timeline_mapping_quality": "exact_identity",
                    "segment_status": "active",
                    "provenance_json": None,
                    "contract_version": 1,
                },
                {
                    "run_id": self.run_id,
                    "video_id": video_id,
                    "shot_id": "shot_002",
                    "start_time_us": 160000,
                    "end_time_us": 320000,
                    "start_frame_index": 4,
                    "end_frame_index_exclusive": 8,
                    "start_boundary_id": "bnd_001",
                    "end_boundary_id": None,
                    "duration_us": 160000,
                    "frame_count": 4,
                    "timeline_mapping_quality": "exact_identity",
                    "segment_status": "active",
                    "provenance_json": None,
                    "contract_version": 1,
                },
            ],
        )
        cameras = _cast(
            "camera_view_segments",
            [
                {
                    "run_id": self.run_id,
                    "video_id": video_id,
                    "camera_segment_id": "cam_001",
                    "shot_id": "shot_001",
                    "start_time_us": 0,
                    "end_time_us": 160000,
                    "start_frame_index": 0,
                    "end_frame_index_exclusive": 4,
                    "view_family": "main_broadcast",
                    "framing_scale": "wide",
                    "camera_position": "sideline",
                    "camera_motion": "pan",
                    "replay_status": "live",
                    "graphics_status": "none",
                    "playability": "playable",
                    "calibration_suitability": "suitable",
                    "tracking_suitability": "suitable",
                    "target_identity_suitability": "conditionally_suitable",
                    "classification_source": "manual",
                    "confidence": 0.9,
                    "coverage": 1.0,
                    "review_status": "accepted",
                    "evidence_refs": [],
                    "provenance_json": None,
                    "contract_version": 1,
                }
            ],
        )
        tables = {
            **base,
            "shot_boundaries": boundaries,
            "shot_segments": shots,
            "camera_view_segments": cameras,
        }
        specs = {n: self.reg.load_contract(n, 1) for n in tables}
        vr = validate_contract_bundle(tables, specs)
        self.assertEqual(vr.status, "PASS", vr.errors)
        br = validate_broadcast_bundle(
            boundaries,
            shots,
            cameras,
            videos=base["videos"],
            frames=base["frames"],
        )
        self.assertEqual(br.status, "PASS", br.errors)

    def test_04_camera_containment_failure_in_bundle(self) -> None:
        base = build_synthetic_bundle(self.run_id)
        video_id = "clip_demo_01"
        shots = _cast(
            "shot_segments",
            [
                {
                    "run_id": self.run_id,
                    "video_id": video_id,
                    "shot_id": "shot_001",
                    "start_time_us": 0,
                    "end_time_us": 80000,
                    "start_frame_index": None,
                    "end_frame_index_exclusive": None,
                    "start_boundary_id": None,
                    "end_boundary_id": None,
                    "duration_us": 80000,
                    "frame_count": None,
                    "timeline_mapping_quality": "uncertain",
                    "segment_status": "active",
                    "provenance_json": None,
                    "contract_version": 1,
                }
            ],
        )
        cameras = _cast(
            "camera_view_segments",
            [
                {
                    "run_id": self.run_id,
                    "video_id": video_id,
                    "camera_segment_id": "cam_overflow",
                    "shot_id": "shot_001",
                    "start_time_us": 0,
                    "end_time_us": 160000,
                    "start_frame_index": None,
                    "end_frame_index_exclusive": None,
                    "view_family": "main_broadcast",
                    "framing_scale": "wide",
                    "camera_position": "sideline",
                    "camera_motion": "static",
                    "replay_status": "live",
                    "graphics_status": "none",
                    "playability": "playable",
                    "calibration_suitability": "suitable",
                    "tracking_suitability": "suitable",
                    "target_identity_suitability": "unknown",
                    "classification_source": "manual",
                    "confidence": 0.5,
                    "coverage": 1.0,
                    "review_status": "unreviewed",
                    "evidence_refs": [],
                    "provenance_json": None,
                    "contract_version": 1,
                }
            ],
        )
        tables = {**base, "shot_segments": shots, "camera_view_segments": cameras}
        specs = {n: self.reg.load_contract(n, 1) for n in tables}
        vr = validate_contract_bundle(tables, specs)
        self.assertEqual(vr.status, "FAIL")


if __name__ == "__main__":
    unittest.main()
