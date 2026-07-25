"""Portable final_delivery media and Turkish report validation (Stage 16-R4-FIX3)."""

from __future__ import annotations

import json
import subprocess
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
    "FUTBOLCU_ANALIZ_RAPORU_TR.pdf",
    "FUTBOLCU_ANALIZ_RAPORU_TR.json",
    "single_player_analysis_summary.png",
    "real_video_analysis_proof.mp4",
    "evidence_manifest.json",
    "checksums.sha256",
    "cleanup_manifest.json",
]


def test_final_mp4_ffprobe_and_portable_profile() -> None:
    mp4 = DELIVERY / "real_video_analysis_proof.mp4"
    assert mp4.is_file()
    info = validate_portable_mp4(mp4, expected_frames=750, expected_duration=30.0)
    assert info["codec_name"] == "h264"
    assert info["codec_tag_string"] == "avc1"
    assert info["pix_fmt"] == "yuv420p"
    assert info["profile"] == "Main"
    assert info["width"] == 1280
    assert info["height"] == 720
    assert info["moov_before_mdat"] is True


def test_final_mp4_full_decode_and_frames() -> None:
    mp4 = DELIVERY / "real_video_analysis_proof.mp4"
    info = validate_portable_mp4(mp4, expected_frames=750, expected_duration=30.0)
    assert info["sequential_frames"] == 750
    assert info["frame0_shape"][0] == 720
    assert info["middle_shape"][0] == 720
    assert info["last_shape"][0] == 720


def test_final_png_pillow_and_opencv() -> None:
    png = DELIVERY / "single_player_analysis_summary.png"
    im = Image.open(png)
    im.verify()
    im = Image.open(png)
    im.load()
    assert im.mode == "RGB"
    assert im.size[0] >= 1920 and im.size[1] >= 1080
    arr = cv2.imread(str(png), cv2.IMREAD_COLOR)
    assert arr is not None
    assert arr.ndim == 3


def test_turkish_report_json_metrics() -> None:
    payload = json.loads((DELIVERY / "FUTBOLCU_ANALIZ_RAPORU_TR.json").read_text(encoding="utf-8"))
    assert payload["target"]["player_id"] == "506469"
    assert int(payload["target"]["jersey_number"]) == 24
    assert payload["namespace_isolation"]["not_the_same_person"] is True
    table = {r["metric"]: r for r in payload["metric_table"]}
    assert table["Pas girişimi"]["value"] == 30
    assert table["Koşu mesafesi"]["value"] == 10969.106
    for row in payload["metric_table"]:
        if row["status"] == "ÖLÇÜLEMEDİ":
            assert row["value"] not in (0, "0", 0.0)
            assert "ÖLÇÜLEMEDİ" in str(row["value"])
    assert (DELIVERY / "FUTBOLCU_ANALIZ_RAPORU_TR.pdf").is_file()
    assert (DELIVERY / "FUTBOLCU_ANALIZ_RAPORU_TR.pdf").stat().st_size > 10_000


def test_html_references_existing_files() -> None:
    html = (DELIVERY / "OPEN_RESULTS.html").read_text(encoding="utf-8")
    for name in (
        "real_video_analysis_proof.mp4",
        "single_player_analysis_summary.png",
        "FUTBOLCU_ANALIZ_RAPORU_TR.pdf",
        "FUTBOLCU_ANALIZ_RAPORU_TR.json",
    ):
        assert name in html
        assert (DELIVERY / name).is_file()
    assert "real_video_tracking_proof" not in html
    assert "autoplay" not in html.lower()
    assert "controls" in html


def test_windows_mirror_hash_equality() -> None:
    if not WIN.is_dir():
        pytest.skip("Windows Desktop mirror path unavailable")
    for name in REQUIRED:
        a = (DELIVERY / name).read_bytes()
        b = (WIN / name).read_bytes()
        assert a == b, name


def test_git_blob_not_lfs_pointer_and_decode() -> None:
    path = "artifacts/final_delivery/real_video_analysis_proof.mp4"
    proc = subprocess.run(
        ["git", "ls-files", "-s", path],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if not proc.stdout.strip():
        pytest.skip("mp4 not yet staged/committed")
    blob = subprocess.check_output(["git", "show", f":{path}"], cwd=str(ROOT))
    assert not blob.startswith(b"version https://git-lfs.github.com"), "LFS pointer"
    assert blob[:8] != b"version ", "unexpected pointer"
    import tempfile
    from pathlib import Path as P

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(blob)
        tmp_path = P(tmp.name)
    try:
        validate_portable_mp4(tmp_path, expected_frames=750, expected_duration=30.0)
    finally:
        tmp_path.unlink(missing_ok=True)


def test_only_one_final_delivery_media() -> None:
    assert sorted(p.name for p in DELIVERY.iterdir() if p.is_file()) == sorted(REQUIRED)
    pngs = list(DELIVERY.glob("*.png"))
    mp4s = list(DELIVERY.glob("*.mp4"))
    pdfs = list(DELIVERY.glob("*.pdf"))
    assert len(pngs) == 1
    assert len(mp4s) == 1
    assert len(pdfs) == 1
    assert pngs[0].name == "single_player_analysis_summary.png"
    assert mp4s[0].name == "real_video_analysis_proof.mp4"
    assert pdfs[0].name == "FUTBOLCU_ANALIZ_RAPORU_TR.pdf"


def test_metric_recount_from_perception_evidence() -> None:
    evidence = (
        ROOT / "artifacts/evidence/stage_16_r4_fix3_turkish_perception/perception_report.json"
    )
    if not evidence.is_file():
        pytest.skip("perception evidence missing")
    d = json.loads(evidence.read_text())
    det = d["detection"]["full_selected"]
    tp, fp, fn = det["tp"], det["fp"], det["fn"]
    assert abs(det["precision"] - tp / (tp + fp)) < 1e-12
    assert abs(det["recall"] - tp / (tp + fn)) < 1e-12
    p, r = det["precision"], det["recall"]
    assert abs(det["f1"] - (2 * p * r / (p + r))) < 1e-12
