"""Unit tests for Stage 3D time mapping."""

from __future__ import annotations

import unittest

from football_analytics.video.time_mapping import (
    MappingStats,
    classify_mapping_quality,
    duration_ts_to_us,
    pts_to_video_time_us,
)
from football_analytics.video.types import (
    FrameRateMode,
    MappingQuality,
    Rational,
    VideoContractError,
)


class TimeMappingTests(unittest.TestCase):
    def test_rational_pts_mapping(self) -> None:
        # time_base 1/12800 → pts 512 → 40000 us
        tb = Rational(1, 12800)
        self.assertEqual(pts_to_video_time_us(512, tb), 40000)
        self.assertEqual(pts_to_video_time_us(0, tb), 0)

    def test_duration_ts(self) -> None:
        tb = Rational(1, 12800)
        self.assertEqual(duration_ts_to_us(512, tb), 40000)
        self.assertIsNone(duration_ts_to_us(None, tb))
        self.assertIsNone(duration_ts_to_us(0, tb))

    def test_rejects_negative_pts(self) -> None:
        with self.assertRaises(VideoContractError):
            pts_to_video_time_us(-1, Rational(1, 25))

    def test_quality_exact_cfr(self) -> None:
        stats = MappingStats(frame_count=10, ok_count=10)
        self.assertEqual(
            classify_mapping_quality(stats, frame_rate_mode=FrameRateMode.CFR),
            MappingQuality.EXACT,
        )

    def test_quality_vfr_good_not_exact(self) -> None:
        stats = MappingStats(frame_count=10, ok_count=10)
        self.assertEqual(
            classify_mapping_quality(stats, frame_rate_mode=FrameRateMode.VFR),
            MappingQuality.GOOD,
        )

    def test_quality_degraded_on_missing(self) -> None:
        stats = MappingStats(frame_count=10, ok_count=9, skipped_count=1, missing_pts_count=1)
        self.assertEqual(
            classify_mapping_quality(stats, frame_rate_mode=FrameRateMode.CFR),
            MappingQuality.DEGRADED,
        )

    def test_quality_failed_on_invention(self) -> None:
        stats = MappingStats(frame_count=5, ok_count=5, invented_from_index_or_fps=True)
        self.assertEqual(
            classify_mapping_quality(stats, frame_rate_mode=FrameRateMode.CFR),
            MappingQuality.FAILED,
        )


if __name__ == "__main__":
    unittest.main()
