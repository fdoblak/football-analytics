"""Portable final_delivery media and recount validation (Stage 16-R4-FIX2)."""

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


def test_final_mp4_ffprobe_and_portable_profile() -> None:
    mp4 = DELIVERY / "real_video_tracking_proof.mp4"
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
    mp4 = DELIVERY / "real_video_tracking_proof.mp4"
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
    assert im.size == (1920, 1080)
    arr = cv2.imread(str(png), cv2.IMREAD_COLOR)
    assert arr is not None
    assert arr.shape == (1080, 1920, 3)


def test_final_json_schema_and_metrics() -> None:
    payload = json.loads((DELIVERY / "single_player_analysis_summary.json").read_text())
    assert payload["target"]["player_id"] == "506469"
    assert payload["target"]["jersey_number"] == 24
    assert payload["target"]["deprecated_invalid_not_used"]["player_id"] == "506466"
    rv = payload["real_video_validation"]["metrics"]
    tp, fp, fn = rv["tp"]["value"], rv["fp"]["value"], rv["fn"]["value"]
    assert abs(rv["precision"]["value"] - tp / (tp + fp)) < 1e-12
    assert abs(rv["recall"]["value"] - tp / (tp + fn)) < 1e-12
    p, r = rv["precision"]["value"], rv["recall"]["value"]
    assert abs(rv["f1"]["value"] - (2 * p * r / (p + r))) < 1e-12
    assert rv["detections"]["value"] == tp + fp
    assert payload["annotation_derived_metrics"]["bas_pass_attempts"]["value"] == 30
    assert payload["annotation_derived_metrics"]["measured_distance_m"]["value"] == 10969.106
    assert "pass_accuracy" in payload["not_evaluable_metrics"]


def test_html_references_existing_files() -> None:
    html = (DELIVERY / "OPEN_RESULTS.html").read_text(encoding="utf-8")
    for name in (
        "real_video_tracking_proof.mp4",
        "single_player_analysis_summary.png",
        "single_player_analysis_summary.json",
    ):
        assert name in html
        assert (DELIVERY / name).is_file()
    assert "autoplay" not in html.lower()
    assert "controls" in html
    assert 'preload="metadata"' in html


def test_windows_mirror_hash_equality() -> None:
    if not WIN.is_dir():
        pytest.skip("Windows Desktop mirror path unavailable")
    for name in (
        "OPEN_RESULTS.html",
        "README.md",
        "single_player_analysis_summary.png",
        "single_player_analysis_summary.json",
        "real_video_tracking_proof.mp4",
        "evidence_manifest.json",
        "checksums.sha256",
    ):
        a = (DELIVERY / name).read_bytes()
        b = (WIN / name).read_bytes()
        assert a == b, name


def test_git_blob_not_lfs_pointer_and_decode() -> None:
    path = "artifacts/final_delivery/real_video_tracking_proof.mp4"
    # working tree vs index may differ before commit; after commit HEAD blob must decode
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


def test_only_one_final_delivery_folder() -> None:
    pngs = list((ROOT / "artifacts").rglob("*.png"))
    mp4s = list((ROOT / "artifacts").rglob("*.mp4"))
    assert len(pngs) == 1
    assert len(mp4s) == 1
    assert "final_delivery" in str(pngs[0])
    assert "final_delivery" in str(mp4s[0])
    assert not (ROOT / "artifacts/final").exists() or not any((ROOT / "artifacts/final").glob("*"))


def test_metric_recount_formula() -> None:
    payload = json.loads((DELIVERY / "single_player_analysis_summary.json").read_text())
    rv = payload["real_video_validation"]["metrics"]
    tp, fp, fn = rv["tp"]["value"], rv["fp"]["value"], rv["fn"]["value"]
    assert rv["precision"]["value"] == tp / (tp + fp)
    assert rv["recall"]["value"] == tp / (tp + fn)
    p = rv["precision"]["value"]
    r = rv["recall"]["value"]
    assert rv["f1"]["value"] == 2 * p * r / (p + r)
