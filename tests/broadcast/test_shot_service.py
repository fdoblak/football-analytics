"""Service / CLI / security tests for Stage 4B shot boundary baseline."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from football_analytics.broadcast.shot_config import (
    default_shot_config_path,
    load_shot_boundary_config,
    shot_config_fingerprint,
)
from football_analytics.broadcast.shot_fixtures import RUNTIME_ROOT, generate_hard_cut
from football_analytics.broadcast.shot_service import (
    prepare_cfr_timeline_for_video,
    run_shot_boundary_detection,
)
from football_analytics.core.run_id import generate_run_id
from football_analytics.data.registry import default_project_root
from football_analytics.video.types import MappingQuality

REPO = default_project_root()
PY = sys.executable


class ShotServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_shot_boundary_config(default_shot_config_path())
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)

    def test_service_writes_artifacts(self) -> None:
        session = Path(tempfile.mkdtemp(prefix="svc_", dir=str(RUNTIME_ROOT)))
        try:
            video = session / "hard.mp4"
            generate_hard_cut(video)
            frames = session / "frames.parquet"
            rid = generate_run_id()
            prepare_cfr_timeline_for_video(
                video,
                frames_out=frames,
                run_id=rid,
                video_id="vid_svc",
                fps=25,
                contain_root=RUNTIME_ROOT,
            )
            out = session / "out"
            out.mkdir()
            res = run_shot_boundary_detection(
                source=str(video),
                timeline=str(frames),
                output_dir=str(out),
                config=self.config,
                contain_root=RUNTIME_ROOT,
                run_id=rid,
                video_id="vid_svc",
                mapping_quality=MappingQuality.DERIVED_WITH_CONSTANT_OFFSET,
            )
            self.assertTrue(res.accepted, res.error_code)
            self.assertTrue(Path(str(res.boundaries_parquet)).is_file())
            self.assertTrue(Path(str(res.segments_parquet)).is_file())
            self.assertTrue(Path(str(res.detection_receipt)).is_file())
            self.assertGreaterEqual(res.boundary_count, 1)
        finally:
            shutil.rmtree(session, ignore_errors=True)

    def test_security_rejects_symlink_and_url(self) -> None:
        session = Path(tempfile.mkdtemp(prefix="svc_sec_", dir=str(RUNTIME_ROOT)))
        try:
            video = session / "hard.mp4"
            generate_hard_cut(video)
            frames = session / "frames.parquet"
            rid = generate_run_id()
            prepare_cfr_timeline_for_video(
                video,
                frames_out=frames,
                run_id=rid,
                video_id="vid_sec",
                fps=25,
                contain_root=RUNTIME_ROOT,
            )
            link = session / "link.mp4"
            link.symlink_to(video)
            out = session / "out"
            out.mkdir()
            res = run_shot_boundary_detection(
                source=str(link),
                timeline=str(frames),
                output_dir=str(out),
                config=self.config,
                contain_root=RUNTIME_ROOT,
                run_id=rid,
                video_id="vid_sec",
            )
            self.assertFalse(res.accepted)
            res2 = run_shot_boundary_detection(
                source="https://example.com/x.mp4",
                timeline=str(frames),
                output_dir=str(out),
                config=self.config,
                contain_root=RUNTIME_ROOT,
            )
            self.assertFalse(res2.accepted)
        finally:
            shutil.rmtree(session, ignore_errors=True)

    def test_overwrite_forbidden(self) -> None:
        session = Path(tempfile.mkdtemp(prefix="svc_ow_", dir=str(RUNTIME_ROOT)))
        try:
            video = session / "hard.mp4"
            generate_hard_cut(video)
            frames = session / "frames.parquet"
            rid = generate_run_id()
            prepare_cfr_timeline_for_video(
                video,
                frames_out=frames,
                run_id=rid,
                video_id="vid_ow",
                fps=25,
                contain_root=RUNTIME_ROOT,
            )
            out = session / "out"
            out.mkdir()
            r1 = run_shot_boundary_detection(
                source=str(video),
                timeline=str(frames),
                output_dir=str(out),
                config=self.config,
                contain_root=RUNTIME_ROOT,
                run_id=rid,
                video_id="vid_ow",
            )
            self.assertTrue(r1.accepted, r1.error_code)
            r2 = run_shot_boundary_detection(
                source=str(video),
                timeline=str(frames),
                output_dir=str(out),
                config=self.config,
                contain_root=RUNTIME_ROOT,
                run_id=rid,
                video_id="vid_ow2",
            )
            self.assertFalse(r2.accepted)
            self.assertEqual(r2.error_code, "OVERWRITE_FORBIDDEN")
        finally:
            shutil.rmtree(session, ignore_errors=True)

    def test_determinism(self) -> None:
        session = Path(tempfile.mkdtemp(prefix="svc_det_", dir=str(RUNTIME_ROOT)))
        try:
            video = session / "hard.mp4"
            generate_hard_cut(video)
            frames = session / "frames.parquet"
            rid = generate_run_id()
            prepare_cfr_timeline_for_video(
                video,
                frames_out=frames,
                run_id=rid,
                video_id="vid_d",
                fps=25,
                contain_root=RUNTIME_ROOT,
            )
            outs = []
            for label in ("a", "b"):
                out = session / f"out_{label}"
                out.mkdir()
                res = run_shot_boundary_detection(
                    source=str(video),
                    timeline=str(frames),
                    output_dir=str(out),
                    config=self.config,
                    contain_root=RUNTIME_ROOT,
                    run_id=rid,
                    video_id=f"vid_{label}",
                )
                self.assertTrue(res.accepted, res.error_code)
                outs.append(res.boundary_count)
            # Different video_id → different row content; compare scores instead
            scores_a = (session / "out_a" / "scores.jsonl").read_text()
            scores_b = (session / "out_b" / "scores.jsonl").read_text()
            self.assertEqual(scores_a, scores_b)
            self.assertEqual(outs[0], outs[1])
            self.assertEqual(
                shot_config_fingerprint(self.config),
                shot_config_fingerprint(load_shot_boundary_config(default_shot_config_path())),
            )
        finally:
            shutil.rmtree(session, ignore_errors=True)

    def test_cli_detect_smoke(self) -> None:
        session = Path(tempfile.mkdtemp(prefix="svc_cli_", dir=str(RUNTIME_ROOT)))
        try:
            video = session / "hard.mp4"
            generate_hard_cut(video)
            frames = session / "frames.parquet"
            rid = generate_run_id()
            prepare_cfr_timeline_for_video(
                video,
                frames_out=frames,
                run_id=rid,
                video_id="vid_cli",
                fps=25,
                contain_root=RUNTIME_ROOT,
            )
            out = session / "out"
            out.mkdir()
            cmd = [
                PY,
                "-m",
                "football_analytics",
                "broadcast",
                "shots",
                "detect",
                "--source",
                str(video),
                "--timeline",
                str(frames),
                "--output-dir",
                str(out),
                "--contain-root",
                str(RUNTIME_ROOT),
                "--run-id",
                rid,
                "--video-id",
                "vid_cli",
            ]
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=str(REPO))
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("accepted: True", proc.stdout)
        finally:
            shutil.rmtree(session, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
