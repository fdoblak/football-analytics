"""Unit tests for camera config loader."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from football_analytics.broadcast.camera_config import (
    CameraConfigError,
    camera_config_fingerprint,
    default_camera_config_path,
    load_camera_view_config,
)


class CameraConfigTests(unittest.TestCase):
    def test_default_loads_and_fingerprint_stable(self) -> None:
        cfg = load_camera_view_config(default_camera_config_path())
        a = camera_config_fingerprint(cfg)
        b = camera_config_fingerprint(cfg)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 64)
        self.assertFalse(cfg["overwrite_allowed"])

    def test_rejects_unknown_key(self) -> None:
        src = default_camera_config_path().read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.yaml"
            p.write_text(src + "\nextra_key: 1\n", encoding="utf-8")
            with self.assertRaises(CameraConfigError):
                load_camera_view_config(p)


if __name__ == "__main__":
    unittest.main()
