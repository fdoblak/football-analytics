"""Unit tests for Stage 3C FFmpeg runner."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from football_analytics.video.contracts import default_repo_root, load_ingest_policy
from football_analytics.video.ffmpeg import (
    FfmpegError,
    assert_libx264_available,
    build_normalize_argv,
    get_ffmpeg_version,
    parse_ffmpeg_version_output,
    resolve_ffmpeg_binary,
    run_ffmpeg_normalize,
)


class FfmpegRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_ingest_policy(default_repo_root() / "configs/video/ingest_policy.yaml")

    def test_exact_binary_and_version(self) -> None:
        binary = resolve_ffmpeg_binary(
            "/usr/bin/ffmpeg",
            allowed_realpaths=self.policy["ffmpeg_policy"]["allowed_binary_realpaths"],
        )
        self.assertEqual(binary.as_posix(), "/usr/bin/ffmpeg")
        ver = get_ffmpeg_version(binary)
        self.assertTrue(ver.version_token)
        self.assertIn("ffmpeg", ver.version_line.lower())
        assert_libx264_available(binary)

    def test_argv_no_y_and_maps(self) -> None:
        argv = build_normalize_argv(
            Path("/usr/bin/ffmpeg"),
            source_path=Path("/tmp/a.mp4"),
            temp_output=Path("/tmp/a.tmp.mp4"),
            video_stream_ordinal=0,
            audio_stream_ordinal=None,
            audio_action="absent",
            target_pixel_format="yuv420p",
            video_preset="veryfast",
            video_crf=23,
            ffmpeg_threads=2,
            movflags_faststart=True,
            rotation_degrees=0,
            bake_rotation=False,
            target_width=160,
            target_height=120,
            resize=False,
            force_setsar=True,
            frame_rate_conversion=False,
            target_frame_rate_num=None,
            target_frame_rate_den=None,
        )
        self.assertEqual(argv[0], "/usr/bin/ffmpeg")
        self.assertNotIn("-y", argv)
        self.assertIn("-nostdin", argv)
        self.assertIn("0:v:0", argv)
        self.assertIn("+faststart", argv)
        self.assertIn("libx264", argv)
        self.assertIn("-vf", argv)
        vf = argv[argv.index("-vf") + 1]
        self.assertIn("setsar=1", vf)

    def test_fps_conversion_uses_vsync_cfr(self) -> None:
        argv = build_normalize_argv(
            Path("/usr/bin/ffmpeg"),
            source_path=Path("/tmp/a.mp4"),
            temp_output=Path("/tmp/a.tmp.mp4"),
            video_stream_ordinal=0,
            audio_stream_ordinal=None,
            audio_action="none",
            target_pixel_format="yuv420p",
            video_preset="veryfast",
            video_crf=23,
            ffmpeg_threads=2,
            movflags_faststart=False,
            rotation_degrees=90,
            bake_rotation=True,
            target_width=120,
            target_height=160,
            resize=True,
            force_setsar=False,
            frame_rate_conversion=True,
            target_frame_rate_num=25,
            target_frame_rate_den=1,
        )
        self.assertIn("-vsync", argv)
        self.assertIn("cfr", argv)
        self.assertIn("-r", argv)
        self.assertIn("25/1", argv)
        self.assertIn("transpose=1", ",".join(argv))

    def test_dash_filename_uses_double_dash(self) -> None:
        argv = build_normalize_argv(
            Path("/usr/bin/ffmpeg"),
            source_path=Path("/tmp/-evil.mp4"),
            temp_output=Path("/tmp/out.mp4"),
            video_stream_ordinal=0,
            audio_stream_ordinal=None,
            audio_action="absent",
            target_pixel_format="yuv420p",
            video_preset="veryfast",
            video_crf=23,
            ffmpeg_threads=2,
            movflags_faststart=True,
            rotation_degrees=0,
            bake_rotation=False,
            target_width=None,
            target_height=None,
            resize=False,
            force_setsar=False,
            frame_rate_conversion=False,
            target_frame_rate_num=None,
            target_frame_rate_den=None,
        )
        self.assertIn("--", argv)

    def test_version_parse(self) -> None:
        token = parse_ffmpeg_version_output("ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright")
        self.assertTrue(token.startswith("4.4.2"))

    def test_missing_binary(self) -> None:
        with self.assertRaises(FfmpegError) as ctx:
            resolve_ffmpeg_binary(
                "/usr/bin/ffmpeg-missing-xyz",
                allowed_realpaths=["/usr/bin/ffmpeg-missing-xyz"],
            )
        self.assertEqual(ctx.exception.code, "FFMPEG_NOT_AVAILABLE")

    def test_shell_false_and_timeout_maps_error(self) -> None:
        with mock.patch("football_analytics.video.ffmpeg.subprocess.Popen") as popen:
            proc = mock.Mock()
            TimeoutExpired = __import__("subprocess").TimeoutExpired

            def communicate_side_effect(*_a, **_k):
                if not getattr(communicate_side_effect, "called", False):
                    communicate_side_effect.called = True
                    raise TimeoutExpired(cmd="x", timeout=1)
                return b"", b""

            proc.communicate.side_effect = communicate_side_effect
            proc.poll.return_value = None
            proc.pid = 1
            proc.returncode = -9
            popen.return_value = proc
            with mock.patch("football_analytics.video.ffmpeg.get_ffmpeg_version") as gv:
                from football_analytics.video.ffmpeg import FfmpegVersion

                gv.return_value = FfmpegVersion("/usr/bin/ffmpeg", "/usr/bin/ffmpeg", "v", "4.4.2")
                with (
                    mock.patch("football_analytics.video.ffmpeg.os.killpg"),
                    self.assertRaises(FfmpegError) as ctx,
                ):
                    run_ffmpeg_normalize(
                        Path("/tmp/does-not-need-exist-for-mock.mp4"),
                        Path("/tmp/out-tmp-mock.mp4"),
                        [
                            "/usr/bin/ffmpeg",
                            "-nostdin",
                            "-i",
                            "/tmp/x.mp4",
                            "/tmp/out-tmp-mock.mp4",
                        ],
                        self.policy,
                    )
                self.assertEqual(ctx.exception.code, "FFMPEG_TIMEOUT")
                self.assertTrue(popen.called)
                self.assertIs(popen.call_args.kwargs.get("shell"), False)


if __name__ == "__main__":
    unittest.main()
