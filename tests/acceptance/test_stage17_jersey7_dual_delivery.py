"""Acceptance checks for Stage 17-R1 jersey-5 own-video final delivery."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import pytest

from football_analytics.acceptance.portable_final_media import validate_portable_mp4

REPO = Path(__file__).resolve().parents[2]
FINAL = REPO / "artifacts" / "final_delivery"


@pytest.mark.skipif(not FINAL.is_dir(), reason="final_delivery missing")
def test_jersey5_final_delivery_presence() -> None:
    names = {p.name for p in FINAL.iterdir() if p.is_file()}
    for n in (
        "FUTBOLCU_5_ANALIZ_RAPORU_TR.pdf",
        "FUTBOLCU_5_ANALIZ_RAPORU_TR.json",
        "futbolcu_5_analiz_ozeti.png",
        "futbolcu_5_video_analiz.mp4",
        "OPEN_RESULTS.html",
        "README.md",
        "evidence_manifest.json",
        "cleanup_manifest.json",
        "checksums.sha256",
    ):
        assert n in names
    assert not any(n.startswith("7_NUMARA_") for n in names)


@pytest.mark.skipif(not FINAL.is_dir(), reason="final_delivery missing")
def test_metrics_honest_and_jersey7_revoked() -> None:
    data = json.loads((FINAL / "FUTBOLCU_5_ANALIZ_RAPORU_TR.json").read_text(encoding="utf-8"))
    assert data["target_jersey"] == 5
    assert data["jersey7_revoked"] is True
    assert data["source_video_id"] == "own_video_97b298e4"
    assert "SoccerTrack" not in json.dumps(data)
    assert data["ball"]["f1"] == "ÖLÇÜLEMEDİ"
    # unmeasured must not be fabricated as zero counts without status
    for name in data["metrics"]["unmeasured"]:
        assert name


@pytest.mark.skipif(not FINAL.is_dir(), reason="final_delivery missing")
def test_png_rgb_and_mp4_portable() -> None:
    png = FINAL / "futbolcu_5_analiz_ozeti.png"
    arr = cv2.imread(str(png), cv2.IMREAD_COLOR)
    assert arr is not None and arr.ndim == 3
    info = validate_portable_mp4(FINAL / "futbolcu_5_video_analiz.mp4")
    assert info["profile"] == "Main"
    assert info["sequential_frames"] >= 100


@pytest.mark.skipif(not FINAL.is_dir(), reason="final_delivery missing")
def test_html_references_jersey5() -> None:
    html = (FINAL / "OPEN_RESULTS.html").read_text(encoding="utf-8")
    assert "Forma 5" in html or "forma 5" in html.lower() or "Forma5" in html
    assert "FUTBOLCU_5_ANALIZ_RAPORU_TR.pdf" in html
    assert "7_NUMARA_ADAY" not in html


@pytest.mark.skipif(not FINAL.is_dir(), reason="final_delivery missing")
def test_cleanup_data_loss_false() -> None:
    cleanup = json.loads((FINAL / "cleanup_manifest.json").read_text(encoding="utf-8"))
    assert cleanup.get("data_loss") is False
    assert cleanup.get("source_video_preserved") is True
