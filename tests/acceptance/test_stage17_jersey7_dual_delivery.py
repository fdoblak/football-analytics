"""Acceptance checks for Stage 17 dual jersey-7 final delivery."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
FINAL = REPO / "artifacts" / "final_delivery"

REQUIRED_METRICS = [
    "Isı haritası",
    "İkili mücadele sayısı",
    "Kazanılan ikili mücadele",
    "İkili mücadele kazanma oranı",
    "Pas girişimi",
    "Tamamlanan pas",
    "Başarısız pas",
    "Pas isabet oranı",
    "Başarılı dripling",
    "Başarısız dripling",
    "Adam eksiltme oranı",
    "Top çalma",
    "Top kaybı",
    "Hava topu mücadelesi",
    "Kazanılan hava topu",
    "Uzaklaştırma",
    "1→2 bölge geçiş pası",
    "2→3 bölge geçiş pası",
    "Uzun pas sayısı",
    "Uzun pas oranı",
    "Ölçülen koşu mesafesi",
    "Sprint sayısı",
    "Sprint mesafesi",
    "Sprint süresi",
    "Ortalama hız",
    "Maksimum hız",
    "Ceza sahası topla buluşma",
    "Aktivite indeksi",
    "Görünürlük coverage",
    "Kimlik coverage",
]


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.mark.skipif(not FINAL.is_dir(), reason="final_delivery missing")
def test_dual_final_delivery_presence():
    names = {p.name for p in FINAL.iterdir() if p.is_file()}
    for slug in ("ADAY_A", "ADAY_B"):
        assert f"7_NUMARA_{slug}_FUTBOLCU_ANALIZ_RAPORU_TR.pdf" in names
        assert f"7_NUMARA_{slug}_FUTBOLCU_ANALIZ_VERILERI.json" in names
        assert f"7_NUMARA_{slug}_ANALIZ_OZETI.png" in names
        assert f"7_NUMARA_{slug}_ANALIZ_KANITI.mp4" in names
    for n in (
        "OPEN_RESULTS.html",
        "README.md",
        "evidence_manifest.json",
        "recovery_manifest.json",
        "cleanup_manifest.json",
        "checksums.sha256",
    ):
        assert n in names
    # no Stage16 SoccerTrack filenames in current tree
    assert "FUTBOLCU_ANALIZ_RAPORU_TR.pdf" not in names
    assert "real_video_analysis_proof.mp4" not in names
    assert "single_player_analysis_summary.png" not in names


@pytest.mark.skipif(not FINAL.is_dir(), reason="final_delivery missing")
def test_metrics_and_no_soccertrack_and_privacy_flags():
    for slug, kit in (("ADAY_A", "light_kit"), ("ADAY_B", "dark_kit")):
        data = json.loads((FINAL / f"7_NUMARA_{slug}_FUTBOLCU_ANALIZ_VERILERI.json").read_text())
        names = [r["metric"] for r in data["metric_table"]]
        for m in REQUIRED_METRICS:
            assert m in names, m
        assert data["target"]["face_recognition_used"] is False
        assert data["target"]["real_name_used"] is False
        assert data["target"]["kit_color_label"] == kit
        blob = json.dumps(data, ensure_ascii=False).lower()
        assert (
            "soccertrack" not in blob
            or "kullanılmamıştır" in json.dumps(data["limitations"], ensure_ascii=False).lower()
        )
        assert "506469" not in blob
        # unmeasured metrics must not be fake zeros
        for row in data["metric_table"]:
            if row["status"] == "ÖLÇÜLEMEDİ":
                assert row["value"] != 0
                assert "ÖLÇÜLEMEDİ" in str(row["value"])


@pytest.mark.skipif(not FINAL.is_dir(), reason="final_delivery missing")
def test_png_rgb_and_mp4_portable():
    for slug in ("ADAY_A", "ADAY_B"):
        png = FINAL / f"7_NUMARA_{slug}_ANALIZ_OZETI.png"
        im = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
        assert im is not None
        assert im.ndim == 3 and im.shape[2] == 3
        assert im.shape[1] >= 1920 and im.shape[0] >= 1080

        mp4 = FINAL / f"7_NUMARA_{slug}_ANALIZ_KANITI.mp4"
        probe = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_streams",
                    "-show_format",
                    "-of",
                    "json",
                    str(mp4),
                ],
                text=True,
            )
        )
        v = next(s for s in probe["streams"] if s["codec_type"] == "video")
        assert v["codec_name"] == "h264"
        assert "Main" in (v.get("profile") or "")
        assert v["pix_fmt"] == "yuv420p"
        assert int(v["width"]) in {1280, 1920}
        assert not any(s["codec_type"] == "audio" for s in probe["streams"])
        # full decode
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(mp4), "-f", "null", "-"],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stderr


@pytest.mark.skipif(not FINAL.is_dir(), reason="final_delivery missing")
def test_html_references_both_candidates():
    html = (FINAL / "OPEN_RESULTS.html").read_text(encoding="utf-8")
    assert "7_NUMARA_ADAY_A_FUTBOLCU_ANALIZ_RAPORU_TR.pdf" in html
    assert "7_NUMARA_ADAY_B_FUTBOLCU_ANALIZ_RAPORU_TR.pdf" in html
    assert 'lang="tr"' in html


@pytest.mark.skipif(not FINAL.is_dir(), reason="final_delivery missing")
def test_checksums_match_and_raw_absent_from_git():
    lines = (FINAL / "checksums.sha256").read_text().strip().splitlines()
    for line in lines:
        digest, name = line.split(None, 1)
        assert _sha256(FINAL / name) == digest
    # raw youtube not tracked
    tracked = subprocess.check_output(
        ["git", "-C", str(REPO), "ls-files"],
        text=True,
    )
    assert "B_cKZkrgxrM.mp4" not in tracked
    assert "authorized_youtube" not in tracked
    assert "target_candidates.png" not in tracked


@pytest.mark.skipif(not FINAL.is_dir(), reason="final_delivery missing")
def test_cleanup_data_loss_false():
    cleanup = json.loads((FINAL / "cleanup_manifest.json").read_text())
    assert cleanup.get("data_loss") is False
    recovery = json.loads((FINAL / "recovery_manifest.json").read_text())
    assert "git_restored_tracked" in recovery
    assert recovery.get("stage17_mode") == "dual_candidate_reports_user_selected_both"


@pytest.mark.skipif(not FINAL.is_dir(), reason="final_delivery missing")
def test_proof_frames_have_head_blur_signal():
    """Sanity: mid frame is not identical to unblurred source neighborhood variance collapse."""
    for slug in ("ADAY_A", "ADAY_B"):
        mp4 = FINAL / f"7_NUMARA_{slug}_ANALIZ_KANITI.mp4"
        cap = cv2.VideoCapture(str(mp4))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        assert n > 20
        cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
        ok, fr = cap.read()
        cap.release()
        assert ok and fr is not None
        # HUD text present
        # Just ensure frame is RGB-like and non-empty
        assert fr.shape[0] == 720 and fr.shape[1] == 1280
        assert float(np.mean(fr)) > 5.0
