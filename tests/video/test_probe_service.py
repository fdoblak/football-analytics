"""Policy media validation and probe service/integration tests (Stage 3B)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from football_analytics.video.contracts import default_repo_root, load_ingest_policy
from football_analytics.video.fixtures import generate_cfr_video
from football_analytics.video.media_validation import validate_probe_against_policy
from football_analytics.video.probe_parser import map_ffprobe_json_to_video_probe
from football_analytics.video.probe_service import run_media_probe

RUNTIME = Path("/home/fdoblak/workspace/video_probe_checks")
REPO = default_repo_root()
PY = sys.executable


class MediaValidationAndServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_ingest_policy(REPO / "configs/video/ingest_policy.yaml")
        RUNTIME.mkdir(parents=True, exist_ok=True)

    def test_policy_rejects_bad_codec(self) -> None:
        probe = map_ffprobe_json_to_video_probe(
            {
                "format": {"format_name": "mp4", "duration": "1.0"},
                "streams": [
                    {
                        "index": 0,
                        "codec_type": "video",
                        "codec_name": "bogus",
                        "width": 320,
                        "height": 240,
                        "pix_fmt": "yuv420p",
                        "r_frame_rate": "25/1",
                        "avg_frame_rate": "25/1",
                        "time_base": "1/25",
                        "disposition": {"default": 1, "attached_pic": 0, "forced": 0},
                    }
                ],
            },
            source_id="src_badcodec",
            source_sha256="a" * 64,
            file_size_bytes=1000,
            probe_tool_version="4.4.2",
            probed_at_utc="2026-07-22T22:00:00Z",
        )
        result = validate_probe_against_policy(probe, self.policy, source_size_bytes=1000)
        self.assertFalse(result.accepted)
        self.assertTrue(any(e.code == "UNSUPPORTED_VIDEO_CODEC" for e in result.errors))

    def test_real_ffprobe_cfr_and_cleanup(self) -> None:
        session = Path(tempfile.mkdtemp(prefix="s3b_", dir=str(RUNTIME)))
        try:
            media = session / "ok.mp4"
            generate_cfr_video(media, with_audio=False)
            out = session / "out"
            out.mkdir()
            res = run_media_probe(
                source=str(media),
                output_dir=str(out),
                policy=self.policy,
                contain_root=RUNTIME,
            )
            self.assertTrue(res.accepted, res.error_code)
            self.assertTrue((out / "video_probe.json").is_file())
            self.assertTrue((out / "media_validation.json").is_file())
            self.assertTrue((out / "probe_execution_receipt.json").is_file())
            mode = (out / "video_probe.json").stat().st_mode & 0o777
            self.assertEqual(mode, 0o600)
            # no overwrite
            res2 = run_media_probe(
                source=str(media),
                output_dir=str(out),
                policy=self.policy,
                contain_root=RUNTIME,
            )
            self.assertFalse(res2.accepted)
            # video+audio
            media_a = session / "ok_a.mp4"
            generate_cfr_video(media_a, with_audio=True)
            out_a = session / "out_a"
            out_a.mkdir()
            res_a = run_media_probe(
                source=str(media_a),
                output_dir=str(out_a),
                policy=self.policy,
                contain_root=RUNTIME,
            )
            self.assertTrue(res_a.accepted, res_a.error_code)
            assert res_a.probe is not None
            self.assertIsNotNone(res_a.probe.selected_audio_stream_index)
        finally:
            shutil.rmtree(session, ignore_errors=True)
            self.assertFalse(session.exists())

    def test_invalid_zero_symlink_url_mutation(self) -> None:
        session = Path(tempfile.mkdtemp(prefix="s3bsec_", dir=str(RUNTIME)))
        try:
            zero = session / "zero.mp4"
            zero.write_bytes(b"")
            out = session / "out_zero"
            out.mkdir()
            res = run_media_probe(
                source=str(zero), output_dir=str(out), policy=self.policy, contain_root=RUNTIME
            )
            self.assertFalse(res.accepted)

            good = session / "g.mp4"
            generate_cfr_video(good)
            link = session / "link.mp4"
            link.symlink_to(good)
            out_l = session / "out_link"
            out_l.mkdir()
            res_l = run_media_probe(
                source=str(link), output_dir=str(out_l), policy=self.policy, contain_root=RUNTIME
            )
            self.assertFalse(res_l.accepted)

            out_u = session / "out_url"
            out_u.mkdir()
            res_u = run_media_probe(
                source="https://example.com/x.mp4",
                output_dir=str(out_u),
                policy=self.policy,
                contain_root=RUNTIME,
            )
            self.assertFalse(res_u.accepted)

            # shell metacharacter filename — must not execute
            weird = session / "name;id;.mp4"
            generate_cfr_video(weird)
            out_w = session / "out_weird"
            out_w.mkdir()
            res_w = run_media_probe(
                source=str(weird), output_dir=str(out_w), policy=self.policy, contain_root=RUNTIME
            )
            self.assertTrue(res_w.accepted, res_w.error_code)

            # space in name
            spaced = session / "my clip.mp4"
            generate_cfr_video(spaced)
            out_s = session / "out_space"
            out_s.mkdir()
            res_s = run_media_probe(
                source=str(spaced), output_dir=str(out_s), policy=self.policy, contain_root=RUNTIME
            )
            self.assertTrue(res_s.accepted, res_s.error_code)

            # FIFO
            fifo = session / "pipe.mp4"
            os.mkfifo(fifo)
            out_f = session / "out_fifo"
            out_f.mkdir()
            res_f = run_media_probe(
                source=str(fifo), output_dir=str(out_f), policy=self.policy, contain_root=RUNTIME
            )
            self.assertFalse(res_f.accepted)
            fifo.unlink(missing_ok=True)
        finally:
            shutil.rmtree(session, ignore_errors=True)

    def test_cli_probe_and_help(self) -> None:
        session = Path(tempfile.mkdtemp(prefix="s3bcli_", dir=str(RUNTIME)))
        try:
            media = session / "cli.mp4"
            generate_cfr_video(media)
            out = session / "cli_out"
            out.mkdir()
            env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
            help_p = subprocess.run(
                [PY, "-m", "football_analytics", "video", "probe", "--help"],
                cwd=str(REPO),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(help_p.returncode, 0)
            self.assertIn("--source", help_p.stdout)
            proc = subprocess.run(
                [
                    PY,
                    "-m",
                    "football_analytics",
                    "video",
                    "probe",
                    "--source",
                    str(media),
                    "--output-dir",
                    str(out),
                    "--contain-root",
                    str(RUNTIME),
                ],
                cwd=str(REPO),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            # previous CLI still works
            ver = subprocess.run(
                [PY, "-m", "football_analytics", "--version"],
                cwd=str(REPO),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(ver.returncode, 0)
            # no heavy imports from cli module load
            iso = subprocess.run(
                [
                    PY,
                    "-c",
                    "import sys; "
                    "banned={'torch','cv2','ultralytics','SoccerNet'}; "
                    "[sys.modules.pop(k) for k in list(sys.modules) if k.split('.')[0] in banned]; "
                    "import football_analytics.cli; "
                    "bad={m for m in sys.modules if m.split('.')[0] in banned}; "
                    "assert not bad, bad",
                ],
                cwd=str(REPO),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(iso.returncode, 0, iso.stderr)
        finally:
            shutil.rmtree(session, ignore_errors=True)

    def test_corrupt_media_rejected(self) -> None:
        session = Path(tempfile.mkdtemp(prefix="s3bbad_", dir=str(RUNTIME)))
        try:
            bad = session / "x.mp4"
            bad.write_bytes(b"not-really-mp4-content")
            out = session / "out"
            out.mkdir()
            res = run_media_probe(
                source=str(bad), output_dir=str(out), policy=self.policy, contain_root=RUNTIME
            )
            self.assertFalse(res.accepted)
            self.assertNotEqual(res.receipt.status.value, "validated")
        finally:
            shutil.rmtree(session, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
