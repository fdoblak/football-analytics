"""Broadcast validation scenario tests."""

from __future__ import annotations

import unittest
from typing import Any

from football_analytics.broadcast.validation import validate_broadcast_bundle
from football_analytics.core.run_id import generate_run_id
from football_analytics.data.compiler import compile_arrow_schema, get_contract


def _cast(name: str, rows: list[dict[str, Any]]) -> Any:
    import pyarrow as pa

    spec = get_contract(name, 1)
    return pa.Table.from_pylist(rows, schema=compile_arrow_schema(spec))


def _videos(run_id: str) -> Any:
    return _cast(
        "videos",
        [
            {
                "run_id": run_id,
                "video_id": "clip_demo_01",
                "source_sha256": "b" * 64,
                "container": "mp4",
                "codec": "h264",
                "width_px": 1280,
                "height_px": 720,
                "fps_numerator": 25,
                "fps_denominator": 1,
                "time_base_numerator": 1,
                "time_base_denominator": 25,
                "frame_count": 10,
                "duration_us": 400000,
                "has_audio": False,
                "source_ref": "logical_clip",
            }
        ],
    )


def _frames(run_id: str, n: int = 10) -> Any:
    rows = [
        {
            "run_id": run_id,
            "video_id": "clip_demo_01",
            "frame_index": i,
            "pts": i,
            "video_time_us": i * 40000,
            "duration_us": 40000,
            "is_key_frame": i % 4 == 0,
            "decode_status": "ok",
        }
        for i in range(n)
    ]
    return _cast("frames", rows)


class BroadcastValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.run_id = generate_run_id()
        self.videos = _videos(self.run_id)
        self.frames = _frames(self.run_id)

    def test_01_single_shot_valid(self) -> None:
        shots = _cast(
            "shot_segments",
            [
                {
                    "run_id": self.run_id,
                    "video_id": "clip_demo_01",
                    "shot_id": "shot_only",
                    "start_time_us": 0,
                    "end_time_us": 400000,
                    "start_frame_index": 0,
                    "end_frame_index_exclusive": 10,
                    "start_boundary_id": None,
                    "end_boundary_id": None,
                    "duration_us": 400000,
                    "frame_count": 10,
                    "timeline_mapping_quality": "exact_identity",
                    "segment_status": "active",
                    "provenance_json": None,
                    "contract_version": 1,
                }
            ],
        )
        vr = validate_broadcast_bundle(None, shots, None, videos=self.videos, frames=self.frames)
        self.assertEqual(vr.status, "PASS", vr.errors)

    def test_02_boundary_order_and_multi_shot(self) -> None:
        boundaries = _cast(
            "shot_boundaries",
            [
                {
                    "run_id": self.run_id,
                    "video_id": "clip_demo_01",
                    "boundary_id": "bnd_b",
                    "boundary_time_us": 200000,
                    "left_frame_index": 4,
                    "right_frame_index": 5,
                    "transition_type": "hard_cut",
                    "transition_duration_us": 0,
                    "confidence": 0.9,
                    "detection_source": "manual",
                    "evidence_ref": None,
                    "review_status": "accepted",
                    "provenance_json": None,
                    "contract_version": 1,
                },
                {
                    "run_id": self.run_id,
                    "video_id": "clip_demo_01",
                    "boundary_id": "bnd_a",
                    "boundary_time_us": 80000,
                    "left_frame_index": 1,
                    "right_frame_index": 2,
                    "transition_type": "flash",
                    "transition_duration_us": 0,
                    "confidence": 0.6,
                    "detection_source": "model",
                    "evidence_ref": None,
                    "review_status": "unreviewed",
                    "provenance_json": None,
                    "contract_version": 1,
                },
            ],
        )
        shots = _cast(
            "shot_segments",
            [
                {
                    "run_id": self.run_id,
                    "video_id": "clip_demo_01",
                    "shot_id": "shot_1",
                    "start_time_us": 0,
                    "end_time_us": 80000,
                    "start_frame_index": 0,
                    "end_frame_index_exclusive": 2,
                    "start_boundary_id": None,
                    "end_boundary_id": "bnd_a",
                    "duration_us": 80000,
                    "frame_count": 2,
                    "timeline_mapping_quality": "timestamp_preserved",
                    "segment_status": "active",
                    "provenance_json": None,
                    "contract_version": 1,
                },
                {
                    "run_id": self.run_id,
                    "video_id": "clip_demo_01",
                    "shot_id": "shot_2",
                    "start_time_us": 80000,
                    "end_time_us": 200000,
                    "start_frame_index": 2,
                    "end_frame_index_exclusive": 5,
                    "start_boundary_id": "bnd_a",
                    "end_boundary_id": "bnd_b",
                    "duration_us": 120000,
                    "frame_count": 3,
                    "timeline_mapping_quality": "timestamp_preserved",
                    "segment_status": "active",
                    "provenance_json": None,
                    "contract_version": 1,
                },
                {
                    "run_id": self.run_id,
                    "video_id": "clip_demo_01",
                    "shot_id": "shot_3",
                    "start_time_us": 200000,
                    "end_time_us": 400000,
                    "start_frame_index": 5,
                    "end_frame_index_exclusive": 10,
                    "start_boundary_id": "bnd_b",
                    "end_boundary_id": None,
                    "duration_us": 200000,
                    "frame_count": 5,
                    "timeline_mapping_quality": "derived_with_resampling",
                    "segment_status": "active",
                    "provenance_json": None,
                    "contract_version": 1,
                },
            ],
        )
        vr = validate_broadcast_bundle(
            boundaries, shots, None, videos=self.videos, frames=self.frames
        )
        self.assertEqual(vr.status, "PASS", vr.errors)

    def test_03_overlap_rejected(self) -> None:
        shots = _cast(
            "shot_segments",
            [
                {
                    "run_id": self.run_id,
                    "video_id": "clip_demo_01",
                    "shot_id": "a",
                    "start_time_us": 0,
                    "end_time_us": 100000,
                    "start_frame_index": None,
                    "end_frame_index_exclusive": None,
                    "start_boundary_id": None,
                    "end_boundary_id": None,
                    "duration_us": 100000,
                    "frame_count": None,
                    "timeline_mapping_quality": "uncertain",
                    "segment_status": "active",
                    "provenance_json": None,
                    "contract_version": 1,
                },
                {
                    "run_id": self.run_id,
                    "video_id": "clip_demo_01",
                    "shot_id": "b",
                    "start_time_us": 50000,
                    "end_time_us": 150000,
                    "start_frame_index": None,
                    "end_frame_index_exclusive": None,
                    "start_boundary_id": None,
                    "end_boundary_id": None,
                    "duration_us": 100000,
                    "frame_count": None,
                    "timeline_mapping_quality": "uncertain",
                    "segment_status": "active",
                    "provenance_json": None,
                    "contract_version": 1,
                },
            ],
        )
        vr = validate_broadcast_bundle(None, shots, None, videos=self.videos)
        self.assertEqual(vr.status, "FAIL")
        self.assertTrue(any("overlap" in e for e in vr.errors))

    def test_04_invalid_duration(self) -> None:
        shots = _cast(
            "shot_segments",
            [
                {
                    "run_id": self.run_id,
                    "video_id": "clip_demo_01",
                    "shot_id": "bad_dur",
                    "start_time_us": 0,
                    "end_time_us": 100000,
                    "start_frame_index": None,
                    "end_frame_index_exclusive": None,
                    "start_boundary_id": None,
                    "end_boundary_id": None,
                    "duration_us": 50,
                    "frame_count": None,
                    "timeline_mapping_quality": "uncertain",
                    "segment_status": "active",
                    "provenance_json": None,
                    "contract_version": 1,
                }
            ],
        )
        vr = validate_broadcast_bundle(None, shots, None, videos=self.videos)
        self.assertEqual(vr.status, "FAIL")

    def test_05_invalid_fk_boundary(self) -> None:
        boundaries = _cast(
            "shot_boundaries",
            [
                {
                    "run_id": self.run_id,
                    "video_id": "clip_demo_01",
                    "boundary_id": "bnd_real",
                    "boundary_time_us": 100000,
                    "left_frame_index": None,
                    "right_frame_index": None,
                    "transition_type": "wipe",
                    "transition_duration_us": None,
                    "confidence": 0.5,
                    "detection_source": "manual",
                    "evidence_ref": None,
                    "review_status": "reviewed",
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
                    "video_id": "clip_demo_01",
                    "shot_id": "shot_x",
                    "start_time_us": 0,
                    "end_time_us": 100000,
                    "start_frame_index": None,
                    "end_frame_index_exclusive": None,
                    "start_boundary_id": None,
                    "end_boundary_id": "bnd_missing",
                    "duration_us": 100000,
                    "frame_count": None,
                    "timeline_mapping_quality": "not_available",
                    "segment_status": "incomplete",
                    "provenance_json": None,
                    "contract_version": 1,
                }
            ],
        )
        vr = validate_broadcast_bundle(boundaries, shots, None, videos=self.videos)
        self.assertEqual(vr.status, "FAIL")

    def test_06_confidence_coverage_bounds(self) -> None:
        cams = _cast(
            "camera_view_segments",
            [
                {
                    "run_id": self.run_id,
                    "video_id": "clip_demo_01",
                    "camera_segment_id": "cam_bad_cov",
                    "shot_id": None,
                    "start_time_us": 0,
                    "end_time_us": 40000,
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
                    "classification_source": "model",
                    "confidence": 1.5,
                    "coverage": 0.5,
                    "review_status": "unreviewed",
                    "evidence_refs": [],
                    "provenance_json": None,
                    "contract_version": 1,
                }
            ],
        )
        vr = validate_broadcast_bundle(None, None, cams, videos=self.videos)
        self.assertEqual(vr.status, "FAIL")

    def test_07_gap_with_status(self) -> None:
        shots = _cast(
            "shot_segments",
            [
                {
                    "run_id": self.run_id,
                    "video_id": "clip_demo_01",
                    "shot_id": "shot_a",
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
                },
                {
                    "run_id": self.run_id,
                    "video_id": "clip_demo_01",
                    "shot_id": "shot_gap",
                    "start_time_us": 80000,
                    "end_time_us": 120000,
                    "start_frame_index": None,
                    "end_frame_index_exclusive": None,
                    "start_boundary_id": None,
                    "end_boundary_id": None,
                    "duration_us": 40000,
                    "frame_count": None,
                    "timeline_mapping_quality": "uncertain",
                    "segment_status": "gap_coverage",
                    "provenance_json": None,
                    "contract_version": 1,
                },
                {
                    "run_id": self.run_id,
                    "video_id": "clip_demo_01",
                    "shot_id": "shot_b",
                    "start_time_us": 120000,
                    "end_time_us": 200000,
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
                },
            ],
        )
        vr = validate_broadcast_bundle(None, shots, None, videos=self.videos)
        self.assertIn(vr.status, {"PASS", "PASS_WITH_WARNINGS"}, vr.errors)


if __name__ == "__main__":
    unittest.main()
