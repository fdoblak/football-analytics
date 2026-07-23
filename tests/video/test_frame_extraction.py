"""Tests for Stage 3D frame materialization helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

from football_analytics.video.frame_extraction import (
    build_ffmpeg_extract_argv,
    select_materialize_indices,
)
from football_analytics.video.frame_timeline import FrameTimelineError
from football_analytics.video.types import FrameTimelineMode


class FrameExtractionTests(unittest.TestCase):
    def test_select_indices(self) -> None:
        self.assertEqual(
            select_materialize_indices(10, mode=FrameTimelineMode.TIMELINE_ONLY, sample_every=2),
            [],
        )
        self.assertEqual(
            select_materialize_indices(10, mode=FrameTimelineMode.SAMPLED, sample_every=3),
            [0, 3, 6, 9],
        )
        self.assertEqual(
            select_materialize_indices(4, mode=FrameTimelineMode.ALL_FRAMES, sample_every=1),
            [0, 1, 2, 3],
        )

    def test_sample_every_invalid(self) -> None:
        with self.assertRaises(FrameTimelineError):
            select_materialize_indices(5, mode=FrameTimelineMode.SAMPLED, sample_every=0)

    def test_ffmpeg_argv_sampled(self) -> None:
        argv = build_ffmpeg_extract_argv(
            Path("/usr/bin/ffmpeg"),
            Path("/tmp/in.mp4"),
            Path("/tmp/out/frame_%06d.png"),
            mode=FrameTimelineMode.SAMPLED,
            sample_every=25,
            image_format="png",
        )
        self.assertEqual(argv[0], "/usr/bin/ffmpeg")
        self.assertIn("-nostdin", argv)
        joined = " ".join(argv)
        self.assertIn("select=", joined)
        self.assertNotIn("shell", joined.lower())


if __name__ == "__main__":
    unittest.main()
