"""Portable final_delivery media validation (Stage 17-R1 jersey-5 own-video)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import pytest
from PIL import Image

from football_analytics.acceptance.portable_final_media import validate_portable_mp4

ROOT = Path(__file__).resolve().parents[2]
DELIVERY = ROOT / "artifacts/final_delivery"
WIN = Path("/mnt/c/Users/furka/Desktop/Football Analytics Final")

REQUIRED = [
    "OPEN_RESULTS.html",
    "README.md",
    "FUTBOLCU_5_ANALIZ_RAPORU_TR.pdf",
    "FUTBOLCU_5_ANALIZ_RAPORU_TR.json",
    "futbolcu_5_analiz_ozeti.png",
    "futbolcu_5_video_analiz.mp4",
    "evidence_manifest.json",
    "checksums.sha256",
    "cleanup_manifest.json",
]


def test_final_mp4_ffprobe_and_portable_profile() -> None:
    mp4 = DELIVERY / "futbolcu_5_video_analiz.mp4"
    assert mp4.is_file()
    info = validate_portable_mp4(mp4)
    assert info["codec_name"] == "h264"
    assert info["codec_tag_string"] == "avc1"
    assert info["pix_fmt"] == "yuv420p"
    assert info["profile"] == "Main"
    assert info["width"] == 1336
    assert info["height"] == 744
    assert info["moov_before_mdat"] is True


def test_final_mp4_full_decode_and_frames() -> None:
    mp4 = DELIVERY / "futbolcu_5_video_analiz.mp4"
    info = validate_portable_mp4(mp4)
    assert info["sequential_frames"] >= 100
    assert info["frame0_shape"][0] == 744
    assert info["middle_shape"][0] == 744
    assert info["last_shape"][0] == 744


def test_final_png_pillow_and_opencv() -> None:
    png = DELIVERY / "futbolcu_5_analiz_ozeti.png"
    im = Image.open(png)
    im.verify()
    im = Image.open(png)
    im.load()
    assert im.mode == "RGB"
    assert im.size[0] >= 640 and im.size[1] >= 400
    arr = cv2.imread(str(png), cv2.IMREAD_COLOR)
    assert arr is not None
    assert arr.ndim == 3


def test_turkish_report_json_metrics() -> None:
    path = DELIVERY / "FUTBOLCU_5_ANALIZ_RAPORU_TR.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["target_jersey"] == 5
    assert data["jersey7_revoked"] is True
    assert "34 SANİYELİK" in data["label"]
    assert data["fusion"]["identity_status"] in {"confirmed", "provisional", "requested"}
    assert data["ball"]["precision"] == "ÖLÇÜLEMEDİ"
    metrics = data["metrics"]
    assert "measured" in metrics
    assert "unmeasured" in metrics
    assert metrics["calibration"]["status"] == "ÖLÇÜLEMEDİ"


def test_html_references_existing_files() -> None:
    html = (DELIVERY / "OPEN_RESULTS.html").read_text(encoding="utf-8")
    for name in (
        "FUTBOLCU_5_ANALIZ_RAPORU_TR.pdf",
        "FUTBOLCU_5_ANALIZ_RAPORU_TR.json",
        "futbolcu_5_analiz_ozeti.png",
        "futbolcu_5_video_analiz.mp4",
        "evidence_manifest.json",
    ):
        assert name in html
        assert (DELIVERY / name).is_file()


def test_windows_mirror_hash_equality() -> None:
    if not WIN.is_dir():
        pytest.skip("Windows Desktop mirror path unavailable")
    lines = (DELIVERY / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    from football_analytics.acceptance.download_manifest import sha256_file

    for line in lines:
        digest, name = line.split("  ")
        assert sha256_file(WIN / name) == digest


def test_only_jersey5_final_delivery_media() -> None:
    names = {p.name for p in DELIVERY.iterdir() if p.is_file()}
    for n in REQUIRED:
        assert n in names
    # Jersey 7 customer dual delivery must not remain
    assert not any(n.startswith("7_NUMARA_") for n in names)
    assert not any("ADAY_" in n for n in names)


def test_required_files_nonempty() -> None:
    for n in REQUIRED:
        p = DELIVERY / n
        assert p.is_file()
        assert p.stat().st_size > 0
