"""Feature extraction smoke tests (OpenCV)."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from football_analytics.broadcast.camera_config import (
    default_camera_config_path,
    load_camera_view_config,
)
from football_analytics.broadcast.camera_features import extract_features_for_samples
from football_analytics.broadcast.camera_fixtures import RUNTIME_ROOT, generate_wide_pitch
from football_analytics.broadcast.camera_sampling import plan_sample_points
from football_analytics.broadcast.shot_features import build_cfr_timeline
from football_analytics.broadcast.shot_service import count_decoded_frames


class CameraFeaturesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_camera_view_config(default_camera_config_path())
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)

    def test_features_finite_and_pitch_high(self) -> None:
        session = Path(tempfile.mkdtemp(prefix="cam_feat_", dir=str(RUNTIME_ROOT)))
        try:
            video = session / "wide.mp4"
            spec = generate_wide_pitch(video)
            n = count_decoded_frames(video)
            tl = build_cfr_timeline(frame_count=n, fps_num=spec.fps, fps_den=1)
            samples = plan_sample_points(
                tl, start_time_us=0, end_time_us=spec.duration_us, config=self.config
            )
            feats = extract_features_for_samples(video, samples, self.config)
            self.assertEqual(len(feats), len(samples))
            self.assertGreater(feats[0].pitch_green_fraction, 0.3)
            for f in feats:
                self.assertGreaterEqual(f.pitch_green_fraction, 0.0)
                self.assertLessEqual(f.pitch_green_fraction, 1.0)
        finally:
            shutil.rmtree(session, ignore_errors=True)

    def test_empty_samples_fail(self) -> None:
        from football_analytics.broadcast.camera_features import CameraFeatureError

        with self.assertRaises(CameraFeatureError):
            extract_features_for_samples("/tmp/nope.mp4", [], self.config)


if __name__ == "__main__":
    unittest.main()
