"""Synthetic fixture design tests for Stage 3A."""

from __future__ import annotations

import unittest
from pathlib import Path

from football_analytics.core.hashing import sha256_bytes
from football_analytics.video.fixtures import (
    RUNTIME_ROOT,
    SCENARIOS,
    ffmpeg_available,
    generate_cfr_video,
    hash_file,
    metadata_fixture,
    open_fixture_session,
)
from football_analytics.video.types import VideoProbe
from football_analytics.video.validation import verify_source_integrity


class VideoFixtureDesignTests(unittest.TestCase):
    def test_scenarios_defined(self) -> None:
        names = {s.name for s in SCENARIOS}
        for required in (
            "cfr_tiny",
            "cfr_with_audio",
            "rotation_metadata",
            "vfr_metadata",
            "unknown_frame_count",
            "unsupported_codec",
            "zero_byte",
            "symlink_negative",
            "hash_mismatch",
        ):
            self.assertIn(required, names)

    def test_metadata_fixtures_parse(self) -> None:
        sha = "f" * 64
        for name in (
            "rotation_metadata",
            "vfr_metadata",
            "unknown_frame_count",
            "unsupported_codec",
        ):
            probe = VideoProbe.from_dict(metadata_fixture(name, source_sha256=sha))
            self.assertEqual(probe.source_sha256, sha)

    def test_ffmpeg_tiny_fixture_and_cleanup(self) -> None:
        self.assertTrue(RUNTIME_ROOT.as_posix().startswith("/home/fdoblak/workspace/"))
        self.assertTrue(ffmpeg_available())
        session = open_fixture_session()
        try:
            path = session.track(session.root / "cfr.mp4")
            generate_cfr_video(path, with_audio=False)
            self.assertTrue(path.is_file())
            size, digest = hash_file(path)
            self.assertGreater(size, 0)
            self.assertEqual(len(digest), 64)
            verify_source_integrity(path, expected_sha256=digest, expected_size_bytes=size)
            # audio variant
            audio_path = session.track(session.root / "cfr_audio.mp4")
            generate_cfr_video(audio_path, with_audio=True)
            self.assertTrue(audio_path.is_file())
            # negatives
            zero = session.track(session.root / "zero.mp4")
            zero.write_bytes(b"")
            self.assertEqual(zero.stat().st_size, 0)
            link = session.track(session.root / "link.mp4")
            link.symlink_to(path)
            self.assertTrue(link.is_symlink())
            # hash mismatch case
            from football_analytics.video.types import VideoSourceError

            with self.assertRaises(VideoSourceError):
                verify_source_integrity(
                    path, expected_sha256=sha256_bytes(b"nope"), expected_size_bytes=size
                )
        finally:
            report = session.cleanup()
            self.assertTrue(report["cleanup_ok"], report)
            self.assertFalse(session.root.exists())

    def test_no_git_tracked_binary_paths(self) -> None:
        # Runtime root is outside the repo
        repo = Path("/home/fdoblak/projects/football-analytics")
        self.assertFalse(str(RUNTIME_ROOT).startswith(str(repo)))


if __name__ == "__main__":
    unittest.main()
