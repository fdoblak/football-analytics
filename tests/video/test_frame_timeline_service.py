"""Integration + CLI tests for Stage 3D frame timeline service."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from football_analytics.core.hashing import sha256_file
from football_analytics.video.contracts import default_repo_root, load_ingest_policy
from football_analytics.video.fixtures import generate_cfr_video
from football_analytics.video.frame_timeline_service import run_frame_timeline
from football_analytics.video.types import FrameTimelineMode, FrameTimelineStatus

RUNTIME = Path("/home/fdoblak/workspace/frame_timeline_checks")
REPO = default_repo_root()
PY = sys.executable


class FrameTimelineServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_ingest_policy(REPO / "configs/video/ingest_policy.yaml")
        RUNTIME.mkdir(parents=True, exist_ok=True)

    def test_timeline_only_cfr(self) -> None:
        session = Path(tempfile.mkdtemp(prefix="s3d_", dir=str(RUNTIME)))
        try:
            media = session / "src.mp4"
            generate_cfr_video(media)
            out = session / "out"
            out.mkdir()
            res = run_frame_timeline(
                source=str(media),
                output_dir=str(out),
                policy=self.policy,
                mode=FrameTimelineMode.TIMELINE_ONLY,
                contain_root=RUNTIME,
                expected_source_sha256=sha256_file(media),
                video_id="vid_cfr",
            )
            self.assertTrue(res.accepted, res.error_code)
            self.assertEqual(res.status, FrameTimelineStatus.SUCCEEDED)
            self.assertTrue(Path(str(res.frames_parquet)).is_file())
            self.assertTrue((out / "frame_timeline_receipt.json").is_file())
            self.assertGreater(res.receipt.frame_count, 0)
            self.assertFalse(res.receipt.materialized)
        finally:
            shutil.rmtree(session, ignore_errors=True)

    def test_sampled_requires_flag(self) -> None:
        session = Path(tempfile.mkdtemp(prefix="s3dreq_", dir=str(RUNTIME)))
        try:
            media = session / "src.mp4"
            generate_cfr_video(media)
            out = session / "out"
            out.mkdir()
            res = run_frame_timeline(
                source=str(media),
                output_dir=str(out),
                policy=self.policy,
                mode=FrameTimelineMode.SAMPLED,
                contain_root=RUNTIME,
                execute_materialize=False,
            )
            self.assertFalse(res.accepted)
            self.assertEqual(res.error_code, "MATERIALIZE_FLAG_REQUIRED")
        finally:
            shutil.rmtree(session, ignore_errors=True)

    def test_sampled_materialize(self) -> None:
        session = Path(tempfile.mkdtemp(prefix="s3dsamp_", dir=str(RUNTIME)))
        try:
            media = session / "src.mp4"
            generate_cfr_video(media)
            out = session / "out"
            out.mkdir()
            res = run_frame_timeline(
                source=str(media),
                output_dir=str(out),
                policy=self.policy,
                mode=FrameTimelineMode.SAMPLED,
                contain_root=RUNTIME,
                execute_materialize=True,
                sample_every=2,
                expected_source_sha256=sha256_file(media),
                video_id="vid_samp",
            )
            self.assertTrue(res.accepted, res.error_code)
            self.assertTrue(res.receipt.materialized)
            self.assertIsNotNone(res.artifact_manifest)
            assert res.artifact_manifest is not None
            self.assertTrue(Path(res.artifact_manifest).is_file())
            self.assertTrue((out / "frame_artifacts.jsonl").is_file())
            self.assertTrue((out / "frames").is_dir())
        finally:
            shutil.rmtree(session, ignore_errors=True)

    def test_security_rejects_symlink_and_url(self) -> None:
        session = Path(tempfile.mkdtemp(prefix="s3dsec_", dir=str(RUNTIME)))
        try:
            media = session / "src.mp4"
            generate_cfr_video(media)
            link = session / "link.mp4"
            link.symlink_to(media)
            out = session / "out"
            out.mkdir()
            res = run_frame_timeline(
                source=str(link),
                output_dir=str(out),
                policy=self.policy,
                contain_root=RUNTIME,
            )
            self.assertFalse(res.accepted)
            res2 = run_frame_timeline(
                source="https://example.com/x.mp4",
                output_dir=str(out),
                policy=self.policy,
                contain_root=RUNTIME,
            )
            self.assertFalse(res2.accepted)
        finally:
            shutil.rmtree(session, ignore_errors=True)

    def test_overwrite_forbidden(self) -> None:
        session = Path(tempfile.mkdtemp(prefix="s3dow_", dir=str(RUNTIME)))
        try:
            media = session / "src.mp4"
            generate_cfr_video(media)
            out = session / "out"
            out.mkdir()
            r1 = run_frame_timeline(
                source=str(media),
                output_dir=str(out),
                policy=self.policy,
                contain_root=RUNTIME,
                video_id="vid_ow",
            )
            self.assertTrue(r1.accepted, r1.error_code)
            r2 = run_frame_timeline(
                source=str(media),
                output_dir=str(out),
                policy=self.policy,
                contain_root=RUNTIME,
                video_id="vid_ow2",
            )
            self.assertFalse(r2.accepted)
        finally:
            shutil.rmtree(session, ignore_errors=True)

    def test_cli_timeline_only(self) -> None:
        session = Path(tempfile.mkdtemp(prefix="s3dcli_", dir=str(RUNTIME)))
        try:
            media = session / "src.mp4"
            generate_cfr_video(media)
            out = session / "out"
            out.mkdir()
            cmd = [
                PY,
                "-m",
                "football_analytics",
                "video",
                "frames",
                "--source",
                str(media),
                "--output-dir",
                str(out),
                "--mode",
                "timeline_only",
                "--contain-root",
                str(RUNTIME),
                "--video-id",
                "vid_cli",
                "--expected-source-sha256",
                sha256_file(media),
            ]
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=str(REPO))
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("accepted: True", proc.stdout)
            self.assertTrue((out / "frames.parquet").is_file())
        finally:
            shutil.rmtree(session, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
