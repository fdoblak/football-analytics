"""Final delivery policy during Stage 17-R2 WAITING / recovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "artifacts/final_delivery"
REJECTED = ROOT / "artifacts/rejected_v0.18.0"
DIAG = ROOT / "artifacts/diagnostics/own_video_recovery"
WIN = Path("/mnt/c/Users/furka/Desktop/Football Analytics Final")


def test_active_final_not_misleading_customer_bundle() -> None:
    """Customer jersey-5 bundle must not remain as active final while WAITING."""
    assert not (FINAL / "FUTBOLCU_5_ANALIZ_RAPORU_TR.pdf").exists()
    assert not (FINAL / "futbolcu_5_video_analiz.mp4").exists()
    assert (FINAL / "REJECTED_SEE_rejected_v0.18.0.txt").is_file()


def test_v018_reclassified_rejected_archive() -> None:
    man = json.loads((REJECTED / "rejection_manifest.json").read_text(encoding="utf-8"))
    assert man["reclassified_gate"] == "NO-GO — OWN-VIDEO PERCEPTION NOT VALIDATED"
    assert "checksums_sha256" in man
    assert (REJECTED / "checksums.sha256").is_file()


def test_diagnostics_waiting_gate() -> None:
    gate = json.loads((DIAG / "GATE_STATUS.json").read_text(encoding="utf-8"))
    assert gate["gate"] in {
        "WAITING — REAL FRAME REVIEW REQUIRED",
        "NO-GO — OWN-VIDEO PERCEPTION ACCEPTANCE FAILED",
    }
    gt_h = json.loads((DIAG / "gt" / "gt_human.json").read_text(encoding="utf-8"))
    gt_b = json.loads((DIAG / "gt" / "gt_ball.json").read_text(encoding="utf-8"))
    assert gt_h["n_reviewed"] >= 180
    assert gt_b["n_reviewed"] >= 300
    # auto never silently equals reviewed count for full set
    assert gt_h["n_reviewed"] >= 180 and gt_b["n_reviewed"] >= 300


def test_windows_mirror_not_old_final() -> None:
    if not WIN.is_dir():
        pytest.skip("Windows mirror unavailable")
    names = {p.name for p in WIN.iterdir()}
    assert "futbolcu_5_video_analiz.mp4" not in names
    assert (
        "GATE_STATUS.json" in names or "STATUS.txt" in names or "AUTHORITATIVE_STATUS.txt" in names
    )
