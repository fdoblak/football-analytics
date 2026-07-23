"""Unit tests for shot feature extraction."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from football_analytics.broadcast.shot_config import (
    default_shot_config_path,
    load_shot_boundary_config,
)
from football_analytics.broadcast.shot_features import (
    ShotFeatureError,
    build_cfr_timeline,
    extract_feature_frames,
)
from football_analytics.broadcast.shot_fixtures import (
    RUNTIME_ROOT,
    generate_hard_cut,
    generate_static,
)


class ShotFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_shot_boundary_config(default_shot_config_path())
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)

    def test_cfr_timeline_contiguous(self) -> None:
        tl = build_cfr_timeline(frame_count=5, fps_num=25, fps_den=1)
        self.assertEqual([i for i, _ in tl], list(range(5)))
        self.assertEqual(tl[0][1], 0)
        self.assertEqual(tl[1][1], 40000)

    def test_hard_cut_features_spike(self) -> None:
        session = Path(tempfile.mkdtemp(prefix="feat_", dir=str(RUNTIME_ROOT)))
        try:
            video = session / "hard.mp4"
            generate_hard_cut(video)
            # Count via extract with matching timeline length discovered by OpenCV
            import cv2

            cap = cv2.VideoCapture(str(video))
            n = 0
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                n += 1
            cap.release()
            tl = build_cfr_timeline(frame_count=n, fps_num=25, fps_den=1)
            feats = extract_feature_frames(video, tl, self.config)
            self.assertEqual(len(feats), n)
            scores = [
                0.45 * f.luma_mae + 0.35 * f.hist_distance + 0.20 * f.edge_change_ratio
                for f in feats
            ]
            self.assertTrue(all(math.isfinite(s) for s in scores))
            self.assertGreater(max(scores), 0.3)
        finally:
            import shutil

            shutil.rmtree(session, ignore_errors=True)

    def test_static_near_zero(self) -> None:
        session = Path(tempfile.mkdtemp(prefix="feat_s_", dir=str(RUNTIME_ROOT)))
        try:
            video = session / "static.mp4"
            generate_static(video)
            import cv2

            cap = cv2.VideoCapture(str(video))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 25
            # Prefer actual decode count
            n = 0
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                n += 1
            cap.release()
            tl = build_cfr_timeline(frame_count=n, fps_num=25, fps_den=1)
            feats = extract_feature_frames(video, tl, self.config)
            for f in feats[1:]:
                self.assertLess(f.luma_mae, 0.05)
        finally:
            import shutil

            shutil.rmtree(session, ignore_errors=True)

    def test_rejects_nan_in_manual_path(self) -> None:
        with self.assertRaises(ShotFeatureError):
            build_cfr_timeline(frame_count=0)


if __name__ == "__main__":
    unittest.main()
