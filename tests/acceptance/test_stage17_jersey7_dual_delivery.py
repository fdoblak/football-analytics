"""Stage 17-R2: jersey-5 customer delivery deferred until perception validated."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
FINAL = REPO / "artifacts" / "final_delivery"
DIAG = REPO / "artifacts" / "diagnostics" / "own_video_recovery"
REJECTED = REPO / "artifacts" / "rejected_v0.18.0"


@pytest.mark.skipif(not DIAG.is_dir(), reason="diagnostics missing")
def test_no_active_jersey5_customer_delivery_while_waiting() -> None:
    assert not (FINAL / "FUTBOLCU_5_ANALIZ_RAPORU_TR.json").exists()
    gate = json.loads((DIAG / "GATE_STATUS.json").read_text(encoding="utf-8"))
    assert ("WAITING" in gate["gate"]) or ("NO-GO" in gate["gate"])


@pytest.mark.skipif(not REJECTED.is_dir(), reason="rejected archive missing")
def test_rejected_v018_manifest_present() -> None:
    man = json.loads((REJECTED / "rejection_manifest.json").read_text(encoding="utf-8"))
    assert man["previous_claimed_gate"].startswith("PASS_WITH_FINDINGS")
    assert man["reclassified_gate"].startswith("NO-GO")


@pytest.mark.skipif(not DIAG.is_dir(), reason="diagnostics missing")
def test_cleanup_data_loss_false() -> None:
    cleanup = json.loads((DIAG / "cleanup_receipt.json").read_text(encoding="utf-8"))
    assert cleanup.get("data_loss") is False
    assert cleanup.get("source_video_preserved") is True


@pytest.mark.skipif(not DIAG.is_dir(), reason="diagnostics missing")
def test_holdout_evaluation_is_nogo() -> None:
    ev = json.loads((DIAG / "holdout_evaluation.json").read_text(encoding="utf-8"))
    assert ev["gate"].startswith("NO-GO")
    assert ev["gt_counts"]["human_reviewed"] >= 180
    assert ev["gt_counts"]["ball_reviewed"] >= 300
    assert "calibration" in ev["acceptance_blockers"]
    assert ev["role_macro_f1"] >= 0.90
    assert ev["ball_holdout"]["f1"] >= 0.75
