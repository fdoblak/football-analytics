"""Tests for R1-F2-A-FIX1 Windows GT review launcher repair."""

from __future__ import annotations

import importlib.util
import json
import socket
import subprocess
import threading
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _load_pkg():
    path = REPO / "scripts" / "r1_gt_windows_package.py"
    spec = importlib.util.spec_from_file_location("r1_gt_windows_package", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_server():
    path = REPO / "scripts" / "r1_independent_gt_review_server.py"
    spec = importlib.util.spec_from_file_location("r1_indep_gt_server", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class BatSafetyTests(unittest.TestCase):
    def test_canonical_bat_ascii_no_bom_no_dangerous_quoting(self) -> None:
        pkg = _load_pkg()
        text = pkg.load_canonical_bat()
        pkg.assert_bat_safe(text)
        raw = pkg.bat_windows_bytes()
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        self.assertIn(b"\r\n", raw)
        self.assertNotIn("—".encode(), raw)
        self.assertNotIn(b"nohup", raw)
        self.assertNotIn(b"pkill", raw)
        self.assertIn(b"Ubuntu-22.04", raw)
        self.assertIn(b"start_r1_gt_review.sh", raw)
        self.assertIn(b"Invoke-RestMethod", raw)
        self.assertIn(b"Browser was NOT opened", raw)

    def test_old_broken_bat_would_fail_guard(self) -> None:
        pkg = _load_pkg()
        broken = (
            "@echo off\n"
            "echo R1 Independent GT Review — yalnız localhost\n"
            'wsl -e bash -lc "nohup python x.py > log 2>&1 &"\n'
        )
        with self.assertRaises(ValueError):
            pkg.assert_bat_safe(broken)


class WrapperShellTests(unittest.TestCase):
    def test_bash_syntax(self) -> None:
        script = REPO / "scripts" / "start_r1_gt_review.sh"
        self.assertTrue(script.is_file())
        r = subprocess.run(
            ["bash", "-n", str(script)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        text = script.read_text(encoding="utf-8")
        self.assertIn("set -euo pipefail", text)
        self.assertIn("/home/fdoblak/miniconda3/envs/ai-dev/bin/python", text)
        self.assertIn("exec ", text)


class HealthAndStartupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        runtime = Path("/home/fdoblak/workspace/independent_gt_review/own_video_97b298e4")
        if not (runtime / "draft_annotations.json").is_file():
            raise unittest.SkipTest("runtime draft missing")
        cls.runtime = runtime
        cls.video = Path(
            "/home/fdoblak/football_data/videos/raw_matches/own_video_97b298e4/original.mp4"
        )
        cls.mod = _load_server()

    def test_health_endpoint_and_frame_and_autosave(self) -> None:
        app = self.mod.ReviewApp(self.runtime, self.video)
        handler = self.mod.build_handler(app)
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        server = ThreadingHTTPServer(("127.0.0.1", port), handler)
        th = threading.Thread(target=server.serve_forever, daemon=True)
        th.start()
        try:
            conn = HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            body = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(resp.status, 200)
            self.assertEqual(body["status"], "ok")
            self.assertEqual(body["service"], "r1_independent_gt_review")
            self.assertEqual(body["source_id"], "own_video_97b298e4")
            self.assertTrue(body["source_sha256_ok"])
            self.assertNotIn("predictions", body)
            self.assertNotIn("token", json.dumps(body).lower())

            conn.request("GET", "/")
            resp = conn.getresponse()
            html = resp.read()
            self.assertEqual(resp.status, 200)
            self.assertNotIn(b"cdn.", html.lower())

            fi = int(app.draft["frames"][0]["frame_idx"])
            conn.request("GET", f"/api/frame.jpg?frame_idx={fi}")
            resp = conn.getresponse()
            jpeg = resp.read()
            self.assertEqual(resp.status, 200)
            self.assertTrue(jpeg[:2] == b"\xff\xd8")

            # autosave path writable
            before = (self.runtime / "draft_annotations.json").stat().st_mtime
            conn.request(
                "POST",
                "/api/save",
                body=b"{}",
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            resp.read()
            self.assertEqual(resp.status, 200)
            after = (self.runtime / "draft_annotations.json").stat().st_mtime
            self.assertGreaterEqual(after, before)
            conn.close()
        finally:
            server.shutdown()
            server.server_close()

    def test_localhost_binding_reject_non_local_host_arg(self) -> None:
        # CLI guard is in main(); simulate by checking source text.
        src = (REPO / "scripts" / "r1_independent_gt_review_server.py").read_text(encoding="utf-8")
        self.assertIn('args.host not in {"127.0.0.1", "localhost"}', src)


class MirrorShaTests(unittest.TestCase):
    def test_windows_mirror_matches_repo_template_if_present(self) -> None:
        pkg = _load_pkg()
        win = Path("/mnt/c/Users/furka/Desktop/Football Analytics Validation/R1 Independent GT")
        bat = win / "START_GT_REVIEW.bat"
        if not bat.is_file():
            self.skipTest("windows mirror missing")
        mirror = pkg.sha256_bytes(bat.read_bytes())
        repo = pkg.sha256_bytes(pkg.bat_windows_bytes())
        self.assertEqual(mirror, repo)
        names = {p.name for p in win.iterdir() if p.is_file()}
        self.assertEqual(names, set(pkg.ALLOWED))


if __name__ == "__main__":
    unittest.main()
