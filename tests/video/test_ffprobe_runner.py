"""Unit tests for Stage 3B FFprobe runner."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from football_analytics.video.contracts import default_repo_root, load_ingest_policy
from football_analytics.video.ffprobe import (
    ProbeError,
    build_ffprobe_argv,
    decode_ffprobe_json,
    get_ffprobe_version,
    parse_ffprobe_version_output,
    resolve_ffprobe_binary,
    run_ffprobe,
)


class FfprobeRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_ingest_policy(default_repo_root() / "configs/video/ingest_policy.yaml")

    def test_exact_binary_and_version(self) -> None:
        binary = resolve_ffprobe_binary(
            "/usr/bin/ffprobe",
            allowed_realpaths=self.policy["ffprobe_policy"]["allowed_binary_realpaths"],
        )
        self.assertEqual(binary.as_posix(), "/usr/bin/ffprobe")
        ver = get_ffprobe_version(binary)
        self.assertTrue(ver.version_token)
        self.assertIn("ffprobe", ver.version_line.lower())

    def test_argv_no_shell_and_no_extra_args(self) -> None:
        argv = build_ffprobe_argv(Path("/usr/bin/ffprobe"), Path("/tmp/a.mp4"))
        self.assertEqual(argv[0], "/usr/bin/ffprobe")
        self.assertNotIn("-count_frames", argv)
        self.assertIn("-print_format", argv)
        self.assertTrue(argv[-1].endswith("a.mp4"))

    def test_dash_filename_uses_double_dash(self) -> None:
        argv = build_ffprobe_argv(Path("/usr/bin/ffprobe"), Path("/tmp/-evil.mp4"))
        self.assertIn("--", argv)

    def test_version_parse(self) -> None:
        token = parse_ffprobe_version_output("ffprobe version 4.4.2-0ubuntu0.22.04.1 Copyright")
        self.assertTrue(token.startswith("4.4.2"))

    def test_decode_json_and_depth(self) -> None:
        data = decode_ffprobe_json(b'{"streams":[],"format":{}}', max_depth=12)
        self.assertIn("streams", data)
        with self.assertRaises(ProbeError):
            decode_ffprobe_json(b"not-json", max_depth=12)

    def test_missing_binary(self) -> None:
        with self.assertRaises(ProbeError) as ctx:
            resolve_ffprobe_binary(
                "/usr/bin/ffprobe-missing-xyz",
                allowed_realpaths=["/usr/bin/ffprobe-missing-xyz"],
            )
        self.assertEqual(ctx.exception.code, "FFPROBE_NOT_AVAILABLE")

    def test_run_ffprobe_rejects_relative(self) -> None:
        with self.assertRaises(ProbeError):
            run_ffprobe(Path("rel.mp4"), policy=self.policy)

    def test_timeout_maps_error(self) -> None:
        with mock.patch("football_analytics.video.ffprobe.subprocess.Popen") as popen:
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
            with mock.patch("football_analytics.video.ffprobe.get_ffprobe_version") as gv:
                from football_analytics.video.ffprobe import FfprobeVersion

                gv.return_value = FfprobeVersion(
                    "/usr/bin/ffprobe", "/usr/bin/ffprobe", "v", "4.4.2"
                )
                with (
                    mock.patch("football_analytics.video.ffprobe.os.killpg"),
                    self.assertRaises(ProbeError) as ctx,
                ):
                    run_ffprobe(
                        Path("/tmp/does-not-need-exist-for-mock.mp4"),
                        policy=self.policy,
                    )
                self.assertEqual(ctx.exception.code, "PROBE_TIMEOUT")


if __name__ == "__main__":
    unittest.main()
