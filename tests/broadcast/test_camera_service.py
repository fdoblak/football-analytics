"""Service / security tests for Stage 4C camera-view baseline."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from football_analytics.broadcast.camera_config import (
    default_camera_config_path,
    load_camera_view_config,
)
from football_analytics.broadcast.camera_fixtures import RUNTIME_ROOT, generate_wide_pitch
from football_analytics.broadcast.camera_service import (
    prepare_cfr_timeline_for_video,
    run_camera_view_classification,
    write_single_shot_parquet,
)
from football_analytics.core.run_id import generate_run_id
from football_analytics.data.registry import default_project_root

REPO = default_project_root()
PY = sys.executable


class CameraServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_camera_view_config(default_camera_config_path())
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)

    def test_service_writes_artifacts(self) -> None:
        session = Path(tempfile.mkdtemp(prefix="cam_svc_", dir=str(RUNTIME_ROOT)))
        try:
            video = session / "wide.mp4"
            spec = generate_wide_pitch(video)
            frames = session / "frames.parquet"
            shots = session / "shots.parquet"
            rid = generate_run_id()
            timeline = prepare_cfr_timeline_for_video(
                video,
                frames_out=frames,
                run_id=rid,
                video_id="vid_cam",
                fps=spec.fps,
                contain_root=RUNTIME_ROOT,
            )
            write_single_shot_parquet(
                shots,
                run_id=rid,
                video_id="vid_cam",
                shot_id="shot_wide",
                start_time_us=0,
                end_time_us=spec.duration_us,
                start_frame_index=0,
                end_frame_index_exclusive=len(timeline),
                contain_root=RUNTIME_ROOT,
            )
            out = session / "out"
            out.mkdir()
            res = run_camera_view_classification(
                source=str(video),
                timeline=str(frames),
                shots=str(shots),
                output_dir=str(out),
                config=self.config,
                contain_root=RUNTIME_ROOT,
                run_id=rid,
                video_id="vid_cam",
            )
            self.assertTrue(res.accepted, res.error_code)
            self.assertTrue(Path(str(res.cameras_parquet)).is_file())
            self.assertTrue(Path(str(res.classification_receipt)).is_file())
            self.assertEqual(res.segment_count, 1)
            receipt = json.loads(Path(str(res.classification_receipt)).read_text(encoding="utf-8"))
            self.assertEqual(receipt["limitation"], "one_camera_view_segment_per_shot")
        finally:
            shutil.rmtree(session, ignore_errors=True)

    def test_security_rejects_url(self) -> None:
        session = Path(tempfile.mkdtemp(prefix="cam_sec_", dir=str(RUNTIME_ROOT)))
        try:
            out = session / "out"
            out.mkdir()
            res = run_camera_view_classification(
                source="https://example.com/x.mp4",
                timeline=str(session / "frames.parquet"),
                shots=str(session / "shots.parquet"),
                output_dir=str(out),
                config=self.config,
                contain_root=RUNTIME_ROOT,
            )
            self.assertFalse(res.accepted)
        finally:
            shutil.rmtree(session, ignore_errors=True)

    def test_cli_help(self) -> None:
        proc = subprocess.run(
            [PY, "-m", "football_analytics", "broadcast", "camera", "classify", "--help"],
            cwd=str(REPO),
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("--shots", proc.stdout)


if __name__ == "__main__":
    unittest.main()
