"""Single-player suitability and camera-axis semantic tests."""

from __future__ import annotations

import unittest
from typing import Any

from football_analytics.broadcast.validation import validate_broadcast_bundle
from football_analytics.core.run_id import generate_run_id
from football_analytics.data.compiler import compile_arrow_schema, get_contract


def _cast(name: str, rows: list[dict[str, Any]]) -> Any:
    import pyarrow as pa

    return pa.Table.from_pylist(rows, schema=compile_arrow_schema(get_contract(name, 1)))


def _videos(run_id: str) -> Any:
    return _cast(
        "videos",
        [
            {
                "run_id": run_id,
                "video_id": "clip_demo_01",
                "source_sha256": "c" * 64,
                "container": "mp4",
                "codec": "h264",
                "width_px": 1280,
                "height_px": 720,
                "fps_numerator": 25,
                "fps_denominator": 1,
                "time_base_numerator": 1,
                "time_base_denominator": 25,
                "frame_count": 8,
                "duration_us": 320000,
                "has_audio": False,
                "source_ref": "logical",
            }
        ],
    )


def _cam(run_id: str, **overrides: Any) -> dict[str, Any]:
    base = {
        "run_id": run_id,
        "video_id": "clip_demo_01",
        "camera_segment_id": "cam_001",
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
        "target_identity_suitability": "conditionally_suitable",
        "classification_source": "manual",
        "confidence": 0.9,
        "coverage": 1.0,
        "review_status": "accepted",
        "evidence_refs": [],
        "provenance_json": None,
        "contract_version": 1,
    }
    base.update(overrides)
    return base


class BroadcastSemanticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.run_id = generate_run_id()
        self.videos = _videos(self.run_id)

    def _validate(self, *rows: dict[str, Any]):
        cams = _cast("camera_view_segments", list(rows))
        return validate_broadcast_bundle(None, None, cams, videos=self.videos)

    def test_01_wide_live_playable(self) -> None:
        vr = self._validate(_cam(self.run_id, camera_segment_id="cam_wide"))
        self.assertEqual(vr.status, "PASS", vr.errors)

    def test_02_close_up_identity_suitable_tracking_caution(self) -> None:
        vr = self._validate(
            _cam(
                self.run_id,
                camera_segment_id="cam_close",
                framing_scale="close_up",
                playability="partially_playable",
                calibration_suitability="unsuitable",
                tracking_suitability="conditionally_suitable",
                target_identity_suitability="suitable",
            )
        )
        self.assertIn(vr.status, {"PASS", "PASS_WITH_WARNINGS"}, vr.errors)

    def test_03_replay_not_fully_playable(self) -> None:
        vr = self._validate(
            _cam(
                self.run_id,
                camera_segment_id="cam_replay",
                replay_status="replay",
                playability="playable",
            )
        )
        self.assertEqual(vr.status, "FAIL")

    def test_04_graphics_fullscreen_hard_rules(self) -> None:
        vr = self._validate(
            _cam(
                self.run_id,
                camera_segment_id="cam_gfx",
                view_family="graphics",
                graphics_status="full_screen",
                playability="playable",
                calibration_suitability="suitable",
                tracking_suitability="suitable",
            )
        )
        self.assertEqual(vr.status, "FAIL")
        ok = self._validate(
            _cam(
                self.run_id,
                camera_segment_id="cam_gfx_ok",
                view_family="graphics",
                graphics_status="full_screen",
                playability="non_playable",
                calibration_suitability="unsuitable",
                tracking_suitability="unsuitable",
                target_identity_suitability="unsuitable",
                confidence=None,
            )
        )
        self.assertEqual(ok.status, "PASS", ok.errors)

    def test_05_crowd_and_bench_non_playable(self) -> None:
        for family in ("crowd", "bench", "studio"):
            vr = self._validate(
                _cam(
                    self.run_id,
                    camera_segment_id=f"cam_{family}",
                    view_family=family,
                    playability="playable",
                )
            )
            self.assertEqual(vr.status, "FAIL", family)
            ok = self._validate(
                _cam(
                    self.run_id,
                    camera_segment_id=f"cam_{family}_ok",
                    view_family=family,
                    playability="non_playable",
                    calibration_suitability="unsuitable",
                    tracking_suitability="unsuitable",
                    target_identity_suitability="unsuitable",
                )
            )
            self.assertEqual(ok.status, "PASS", ok.errors)

    def test_06_unknown_view_warns_when_playable(self) -> None:
        vr = self._validate(_cam(self.run_id, camera_segment_id="cam_unk", view_family="unknown"))
        self.assertEqual(vr.status, "PASS_WITH_WARNINGS")
        self.assertTrue(any("unknown view" in w for w in vr.warnings))

    def test_07_playable_camera_overlap_rejected(self) -> None:
        vr = self._validate(
            _cam(self.run_id, camera_segment_id="cam_a", start_time_us=0, end_time_us=80000),
            _cam(
                self.run_id,
                camera_segment_id="cam_b",
                start_time_us=40000,
                end_time_us=120000,
            ),
        )
        self.assertEqual(vr.status, "FAIL")

    def test_08_camera_contained_in_shot(self) -> None:
        shots = _cast(
            "shot_segments",
            [
                {
                    "run_id": self.run_id,
                    "video_id": "clip_demo_01",
                    "shot_id": "shot_001",
                    "start_time_us": 0,
                    "end_time_us": 80000,
                    "start_frame_index": None,
                    "end_frame_index_exclusive": None,
                    "start_boundary_id": None,
                    "end_boundary_id": None,
                    "duration_us": 80000,
                    "frame_count": None,
                    "timeline_mapping_quality": "exact_identity",
                    "segment_status": "active",
                    "provenance_json": None,
                    "contract_version": 1,
                }
            ],
        )
        good = _cast(
            "camera_view_segments",
            [_cam(self.run_id, shot_id="shot_001", start_time_us=0, end_time_us=40000)],
        )
        vr = validate_broadcast_bundle(None, shots, good, videos=self.videos)
        self.assertEqual(vr.status, "PASS", vr.errors)
        bad = _cast(
            "camera_view_segments",
            [_cam(self.run_id, shot_id="shot_001", start_time_us=0, end_time_us=120000)],
        )
        vr2 = validate_broadcast_bundle(None, shots, bad, videos=self.videos)
        self.assertEqual(vr2.status, "FAIL")

    def test_09_suitability_axes_independent(self) -> None:
        # Wide: tracking/calibration suitable, identity only conditional — valid
        vr = self._validate(
            _cam(
                self.run_id,
                camera_segment_id="cam_axes",
                framing_scale="wide",
                calibration_suitability="suitable",
                tracking_suitability="suitable",
                target_identity_suitability="conditionally_suitable",
            )
        )
        self.assertEqual(vr.status, "PASS", vr.errors)


if __name__ == "__main__":
    unittest.main()
