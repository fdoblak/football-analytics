"""Integration tests for Stage 3C normalization service."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path

from football_analytics.core.hashing import sha256_file
from football_analytics.video.contracts import default_repo_root, load_ingest_policy
from football_analytics.video.fixtures import generate_cfr_video
from football_analytics.video.normalization_service import run_video_normalization
from football_analytics.video.types import NormalizationStatus

RUNTIME = Path("/home/fdoblak/workspace/video_normalization_checks")
REPO = default_repo_root()
PY = sys.executable
DiskUsage = namedtuple("DiskUsage", "total used free")


def _ample_disk(_path: str) -> DiskUsage:
    return DiskUsage(total=10**15, used=0, free=10**15)


def _generate_mpeg4(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "/usr/bin/ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=160x120:d=1:r=25",
        "-c:v",
        "mpeg4",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-500:])
    return path


class NormalizationServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_ingest_policy(REPO / "configs/video/ingest_policy.yaml")
        RUNTIME.mkdir(parents=True, exist_ok=True)

    def test_mpeg4_to_h264_execute(self) -> None:
        session = Path(tempfile.mkdtemp(prefix="s3c_", dir=str(RUNTIME)))
        try:
            media = session / "src_mpeg4.mp4"
            _generate_mpeg4(media)
            digest = sha256_file(media)
            out = session / "norm" / "out.mp4"
            out.parent.mkdir(parents=True, exist_ok=True)
            res = run_video_normalization(
                source=str(media),
                output=str(out),
                policy=self.policy,
                expected_source_sha256=digest,
                execute=True,
                contain_root=RUNTIME,
                receipt_dir=str(session / "receipts"),
                disk_usage_fn=_ample_disk,
            )
            self.assertEqual(res.status, NormalizationStatus.SUCCEEDED, res.error_code)
            self.assertTrue(out.is_file())
            self.assertTrue((session / "receipts" / "normalization_receipt.json").is_file())
            self.assertIsNotNone(res.output_probe)
            assert res.output_probe is not None
            # h264 output
            from football_analytics.video.types import VideoStreamInfo

            v = next(s for s in res.output_probe.streams if isinstance(s, VideoStreamInfo))
            self.assertEqual(v.codec_name, "h264")
        finally:
            shutil.rmtree(session, ignore_errors=True)
            self.assertFalse(session.exists())

    def test_canonical_skip_and_dry_run(self) -> None:
        session = Path(tempfile.mkdtemp(prefix="s3cskip_", dir=str(RUNTIME)))
        try:
            media = session / "ok.mp4"
            generate_cfr_video(media)
            digest = sha256_file(media)
            out = session / "out.mp4"
            # dry-run
            dry = run_video_normalization(
                source=str(media),
                output=str(out),
                policy=self.policy,
                expected_source_sha256=digest,
                execute=False,
                contain_root=RUNTIME,
                receipt_dir=str(session / "r_dry"),
                disk_usage_fn=_ample_disk,
            )
            self.assertEqual(dry.status, NormalizationStatus.SKIPPED)
            self.assertFalse(out.exists())
            # execute skip
            exe = run_video_normalization(
                source=str(media),
                output=str(out),
                policy=self.policy,
                expected_source_sha256=digest,
                execute=True,
                contain_root=RUNTIME,
                receipt_dir=str(session / "r_exe"),
                disk_usage_fn=_ample_disk,
            )
            self.assertEqual(exe.status, NormalizationStatus.SKIPPED)
            self.assertFalse(out.exists())
        finally:
            shutil.rmtree(session, ignore_errors=True)

    def test_overwrite_rejected(self) -> None:
        session = Path(tempfile.mkdtemp(prefix="s3cow_", dir=str(RUNTIME)))
        try:
            media = session / "src.mp4"
            _generate_mpeg4(media)
            digest = sha256_file(media)
            out = session / "exists.mp4"
            out.write_bytes(b"x")
            res = run_video_normalization(
                source=str(media),
                output=str(out),
                policy=self.policy,
                expected_source_sha256=digest,
                execute=True,
                contain_root=RUNTIME,
                disk_usage_fn=_ample_disk,
            )
            self.assertNotEqual(res.status, NormalizationStatus.SUCCEEDED)
            self.assertEqual(res.exit_code, 3)
        finally:
            shutil.rmtree(session, ignore_errors=True)

    def test_shell_metachar_filename(self) -> None:
        session = Path(tempfile.mkdtemp(prefix="s3cmeta_", dir=str(RUNTIME)))
        try:
            media = session / "name;id;.mp4"
            _generate_mpeg4(media)
            digest = sha256_file(media)
            out = session / "out_meta.mp4"
            res = run_video_normalization(
                source=str(media),
                output=str(out),
                policy=self.policy,
                expected_source_sha256=digest,
                execute=True,
                contain_root=RUNTIME,
                receipt_dir=str(session / "r"),
                disk_usage_fn=_ample_disk,
            )
            self.assertEqual(res.status, NormalizationStatus.SUCCEEDED, res.error_code)
            self.assertTrue(out.is_file())
        finally:
            shutil.rmtree(session, ignore_errors=True)

    def test_mutation_detected(self) -> None:
        session = Path(tempfile.mkdtemp(prefix="s3cmut_", dir=str(RUNTIME)))
        try:
            media = session / "mut.mp4"
            _generate_mpeg4(media)
            digest = sha256_file(media)
            out = session / "out_mut.mp4"

            real_runner = None
            from football_analytics.video import normalization_service as ns

            real_runner = ns.run_ffmpeg_normalize

            def mutating_runner(*args, **kwargs):
                result = real_runner(*args, **kwargs)
                media.write_bytes(media.read_bytes() + b"\x00")
                return result

            res = run_video_normalization(
                source=str(media),
                output=str(out),
                policy=self.policy,
                expected_source_sha256=digest,
                execute=True,
                contain_root=RUNTIME,
                receipt_dir=str(session / "r"),
                disk_usage_fn=_ample_disk,
                ffmpeg_runner=mutating_runner,
            )
            self.assertEqual(res.error_code, "SOURCE_MUTATED_DURING_NORMALIZATION")
            self.assertEqual(res.exit_code, 3)
            self.assertFalse(out.exists())
        finally:
            shutil.rmtree(session, ignore_errors=True)

    def test_cli_normalize_help_and_dry_run(self) -> None:
        session = Path(tempfile.mkdtemp(prefix="s3ccli_", dir=str(RUNTIME)))
        try:
            media = session / "cli.mp4"
            generate_cfr_video(media)
            digest = sha256_file(media)
            out = session / "cli_out.mp4"
            env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
            help_p = subprocess.run(
                [PY, "-m", "football_analytics", "video", "normalize", "--help"],
                cwd=str(REPO),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(help_p.returncode, 0)
            self.assertIn("--execute", help_p.stdout)
            proc = subprocess.run(
                [
                    PY,
                    "-m",
                    "football_analytics",
                    "video",
                    "normalize",
                    "--source",
                    str(media),
                    "--output",
                    str(out),
                    "--contain-root",
                    str(RUNTIME),
                    "--expected-source-sha256",
                    digest,
                    "--receipt-dir",
                    str(session / "cli_r"),
                ],
                cwd=str(REPO),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertFalse(out.exists())
        finally:
            shutil.rmtree(session, ignore_errors=True)

    def test_insufficient_space(self) -> None:
        session = Path(tempfile.mkdtemp(prefix="s3cspace_", dir=str(RUNTIME)))
        try:
            media = session / "sp.mp4"
            _generate_mpeg4(media)
            digest = sha256_file(media)
            out = session / "out_sp.mp4"

            def tiny_disk(_path: str) -> DiskUsage:
                return DiskUsage(total=1000, used=900, free=100)

            res = run_video_normalization(
                source=str(media),
                output=str(out),
                policy=self.policy,
                expected_source_sha256=digest,
                execute=True,
                contain_root=RUNTIME,
                disk_usage_fn=tiny_disk,
            )
            self.assertEqual(res.error_code, "INSUFFICIENT_OUTPUT_SPACE")
        finally:
            shutil.rmtree(session, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
