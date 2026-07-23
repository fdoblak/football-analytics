"""Detection unit tests for Stage 4B baseline."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from football_analytics.broadcast.shot_config import (
    default_shot_config_path,
    load_shot_boundary_config,
)
from football_analytics.broadcast.shot_detection import detect_shots
from football_analytics.broadcast.shot_features import build_cfr_timeline, extract_feature_frames
from football_analytics.broadcast.shot_fixtures import (
    RUNTIME_ROOT,
    generate_flash,
    generate_hard_cut,
)
from football_analytics.core.run_id import generate_run_id
from football_analytics.video.types import MappingQuality


class ShotDetectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_shot_boundary_config(default_shot_config_path())
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)

    def _decode_n(self, video: Path) -> int:
        import cv2

        cap = cv2.VideoCapture(str(video))
        n = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            n += 1
        cap.release()
        return n

    def test_hard_cut_detected(self) -> None:
        session = Path(tempfile.mkdtemp(prefix="det_", dir=str(RUNTIME_ROOT)))
        try:
            video = session / "hard.mp4"
            generate_hard_cut(video)
            n = self._decode_n(video)
            tl = build_cfr_timeline(frame_count=n, fps_num=25, fps_den=1)
            feats = extract_feature_frames(video, tl, self.config)
            duration = tl[-1][1] + 40000
            b, s, _sc = detect_shots(
                feats,
                run_id=generate_run_id(),
                video_id="vid_hard",
                duration_us=duration,
                timeline=tl,
                config=self.config,
                mapping_quality=MappingQuality.DERIVED_WITH_CONSTANT_OFFSET,
            )
            self.assertGreaterEqual(len(b), 1)
            self.assertEqual(b[0].transition_type.value, "hard_cut")
            self.assertEqual(b[0].detection_source.value, "rule")
            self.assertIsNone(b[0].confidence)
            self.assertGreaterEqual(len(s), 2)
            self.assertEqual(s[0].start_time_us, 0)
            self.assertEqual(s[-1].end_time_us, duration)
        finally:
            shutil.rmtree(session, ignore_errors=True)

    def test_flash_suppressed(self) -> None:
        session = Path(tempfile.mkdtemp(prefix="det_f_", dir=str(RUNTIME_ROOT)))
        try:
            video = session / "flash.mp4"
            generate_flash(video)
            n = self._decode_n(video)
            tl = build_cfr_timeline(frame_count=n, fps_num=25, fps_den=1)
            feats = extract_feature_frames(video, tl, self.config)
            duration = tl[-1][1] + 40000
            b, _s, _sc = detect_shots(
                feats,
                run_id=generate_run_id(),
                video_id="vid_flash",
                duration_us=duration,
                timeline=tl,
                config=self.config,
                mapping_quality=MappingQuality.NOT_AVAILABLE,
            )
            self.assertEqual(len(b), 0)
        finally:
            shutil.rmtree(session, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
