"""Stage 4A broadcast contract compile / typed model tests."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

from football_analytics.broadcast import (
    CONTRACT_NAMES,
    CONTRACT_VERSION,
    CameraViewSegment,
    MappingQuality,
    ShotBoundary,
    ShotSegment,
    TransitionType,
    assert_broadcast_contracts_registered,
    broadcast_schema_fingerprints,
    compile_broadcast_schemas,
    load_all_broadcast_contracts,
)
from football_analytics.broadcast.types import BroadcastContractError
from football_analytics.core.run_id import generate_run_id
from football_analytics.data.compiler import compile_arrow_schema
from football_analytics.data.registry import (
    default_project_root,
    default_registry_path,
    load_schema_registry,
)
from football_analytics.data.validation import validate_table

ROOT = default_project_root()
FRAMES_JSON = ROOT / "schemas/data/v1/frames.json"
FRAMES_JSON_SHA256 = "8fd233af820aa6242d575a2005f6552e43b37dd535140f26858782cc116d0437"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BroadcastContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reg = load_schema_registry(default_registry_path(), project_root=ROOT)
        self.run_id = generate_run_id()

    def test_01_contracts_registered(self) -> None:
        assert_broadcast_contracts_registered(registry=self.reg)
        names = load_all_broadcast_contracts(registry=self.reg)
        self.assertEqual(set(names), set(CONTRACT_NAMES))

    def test_02_compile_and_fingerprint(self) -> None:
        schemas = compile_broadcast_schemas(registry=self.reg)
        fps = broadcast_schema_fingerprints(registry=self.reg)
        self.assertEqual(fps, broadcast_schema_fingerprints(registry=self.reg))
        for name in CONTRACT_NAMES:
            self.assertIn(name, schemas)
            self.assertEqual(len(fps[name]), 64)

    def test_03_frames_json_unchanged(self) -> None:
        self.assertEqual(_sha256(FRAMES_JSON), FRAMES_JSON_SHA256)

    def test_04_shot_boundary_roundtrip(self) -> None:
        payload = {
            "run_id": self.run_id,
            "video_id": "clip_demo_01",
            "boundary_id": "bnd_hard_001",
            "boundary_time_us": 120000,
            "left_frame_index": 2,
            "right_frame_index": 3,
            "transition_type": "hard_cut",
            "transition_duration_us": 0,
            "confidence": 0.91,
            "detection_source": "manual",
            "evidence_ref": None,
            "review_status": "accepted",
            "provenance_json": '{"origin":"fixture"}',
            "contract_version": CONTRACT_VERSION,
        }
        obj = ShotBoundary.from_dict(payload)
        self.assertEqual(obj.transition_type, TransitionType.HARD_CUT)
        self.assertEqual(ShotBoundary.from_dict(obj.to_dict()).to_dict(), obj.to_dict())

    def test_05_shot_segment_duration_enforced(self) -> None:
        base = {
            "run_id": self.run_id,
            "video_id": "clip_demo_01",
            "shot_id": "shot_001",
            "start_time_us": 0,
            "end_time_us": 100000,
            "start_frame_index": None,
            "end_frame_index_exclusive": None,
            "start_boundary_id": None,
            "end_boundary_id": None,
            "duration_us": 100000,
            "frame_count": None,
            "timeline_mapping_quality": MappingQuality.EXACT_IDENTITY.value,
            "segment_status": "active",
            "provenance_json": None,
            "contract_version": 1,
        }
        shot = ShotSegment.from_dict(base)
        self.assertEqual(shot.duration_us, 100000)
        bad = dict(base)
        bad["duration_us"] = 50
        with self.assertRaises(BroadcastContractError):
            ShotSegment.from_dict(bad)

    def test_06_dissolve_and_null_edge_boundaries(self) -> None:
        dissolve = ShotBoundary.from_dict(
            {
                "run_id": self.run_id,
                "video_id": "clip_demo_01",
                "boundary_id": "bnd_dissolve_001",
                "boundary_time_us": 200000,
                "left_frame_index": None,
                "right_frame_index": None,
                "transition_type": "dissolve",
                "transition_duration_us": 80000,
                "confidence": None,
                "detection_source": "rule",
                "evidence_ref": None,
                "review_status": "unreviewed",
                "provenance_json": None,
                "contract_version": 1,
            }
        )
        self.assertEqual(dissolve.transition_type.value, "dissolve")
        first = ShotSegment.from_dict(
            {
                "run_id": self.run_id,
                "video_id": "clip_demo_01",
                "shot_id": "shot_first",
                "start_time_us": 0,
                "end_time_us": 200000,
                "start_frame_index": None,
                "end_frame_index_exclusive": None,
                "start_boundary_id": None,
                "end_boundary_id": "bnd_dissolve_001",
                "duration_us": 200000,
                "frame_count": None,
                "timeline_mapping_quality": "derived_with_constant_offset",
                "segment_status": "active",
                "provenance_json": None,
                "contract_version": 1,
            }
        )
        self.assertIsNone(first.start_boundary_id)

    def test_07_camera_axes_separate(self) -> None:
        cam = CameraViewSegment.from_dict(
            {
                "run_id": self.run_id,
                "video_id": "clip_demo_01",
                "camera_segment_id": "cam_wide_live",
                "shot_id": "shot_001",
                "start_time_us": 0,
                "end_time_us": 40000,
                "start_frame_index": 0,
                "end_frame_index_exclusive": 1,
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
                "confidence": 0.88,
                "coverage": 1.0,
                "review_status": "accepted",
                "evidence_refs": [],
                "provenance_json": None,
                "contract_version": 1,
            }
        )
        self.assertEqual(cam.view_family.value, "main_broadcast")
        self.assertEqual(cam.framing_scale.value, "wide")
        self.assertEqual(cam.replay_status.value, "live")

    def test_08_table_semantic_valid(self) -> None:
        import pyarrow as pa

        spec = self.reg.load_contract("shot_boundaries", 1)
        schema = compile_arrow_schema(spec)
        table = pa.Table.from_pylist(
            [
                {
                    "run_id": self.run_id,
                    "video_id": "clip_demo_01",
                    "boundary_id": "bnd_001",
                    "boundary_time_us": 0,
                    "left_frame_index": None,
                    "right_frame_index": None,
                    "transition_type": "unknown",
                    "transition_duration_us": None,
                    "confidence": None,
                    "detection_source": "imported",
                    "evidence_ref": None,
                    "review_status": "needs_review",
                    "provenance_json": None,
                    "contract_version": 1,
                }
            ],
            schema=schema,
        )
        vr = validate_table(table, spec)
        self.assertEqual(vr.status, "PASS", vr.errors)


if __name__ == "__main__":
    unittest.main()
