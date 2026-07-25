"""Unit/integration tests for Stage 16-R4 self-contained technical acceptance."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from football_analytics.acceptance.isolation import (
    NamespaceIsolationError,
    assert_namespaces_isolated,
)
from football_analytics.acceptance.namespaces import (
    DEPRECATED_INVALID_TARGET,
)
from football_analytics.acceptance.self_contained import (
    ScenarioConfig,
    run_self_contained_acceptance,
    validate_two_run_determinism,
)
from football_analytics.acceptance.soccertrack_v2.reference_analysis import (
    analyze_soccertrack_v2_reference,
    refuse_deprecated_target,
)

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_FP = ROOT / "tests/fixtures/acceptance/self_contained_expected_fingerprints.json"


def test_self_contained_two_run_hash_equality(tmp_path: Path) -> None:
    cfg = ScenarioConfig(seed=16040)
    a = run_self_contained_acceptance(output_dir=tmp_path / "a", config=cfg)
    b = run_self_contained_acceptance(output_dir=tmp_path / "b", config=cfg)
    check = validate_two_run_determinism(tmp_path / "a", tmp_path / "b")
    assert check["equal"] is True
    assert a["receipt"]["hf_required"] is False
    assert b["receipt"]["network_required"] is False
    expected = json.loads(EXPECTED_FP.read_text())
    assert a["receipt"]["metrics_fingerprint"] == expected["metrics_fingerprint"]
    assert a["receipt"]["config_fingerprint"] == expected["config_fingerprint"]


def test_deprecated_target_refused() -> None:
    with pytest.raises(ValueError):
        refuse_deprecated_target(
            str(DEPRECATED_INVALID_TARGET["player_id"]),
            int(DEPRECATED_INVALID_TARGET["jersey_number"]),
        )


def test_namespace_isolation() -> None:
    assert_namespaces_isolated(
        soccertrack_player_id="506469",
        teamtrack_track_id=7,
        claim_same_person=False,
    )
    with pytest.raises(NamespaceIsolationError):
        assert_namespaces_isolated(
            soccertrack_player_id="506469",
            claim_same_person=True,
        )
    with pytest.raises(NamespaceIsolationError):
        assert_namespaces_isolated(soccertrack_player_id="506466")


def test_reference_analysis_uses_authoritative_target() -> None:
    traj = Path(
        "/home/fdoblak/football_data/datasets/soccertrack_v2/runs/128057/"
        "reference_ground_truth/target_trajectory_reference.json"
    )
    bas = Path(
        "/home/fdoblak/football_data/datasets/soccertrack_v2/runs/128057/"
        "reference_ground_truth/bas_reference_events.json"
    )
    if not traj.is_file() or not bas.is_file():
        pytest.skip("local SoccerTrack reference exports unavailable")
    report = analyze_soccertrack_v2_reference(trajectory_path=traj, bas_path=bas)
    assert report["not_video_prediction"] is True
    assert report["target"]["player_id"] == "506469"
    assert report["metrics"]["pass_accuracy"]["status"] == "NOT_EVALUABLE"
    assert report["metrics"]["measured_distance_m"]["status"] == "REFERENCE_ANNOTATION_DERIVED"
    assert report["bas_target_label_counts"].get("Pass", 0) >= 1


def test_final_report_and_single_customer_png_policy() -> None:
    final_json = ROOT / "artifacts/final_delivery/FUTBOLCU_ANALIZ_RAPORU_TR.json"
    final_png = ROOT / "artifacts/final_delivery/single_player_analysis_summary.png"
    final_pdf = ROOT / "artifacts/final_delivery/FUTBOLCU_ANALIZ_RAPORU_TR.pdf"
    assert final_json.is_file()
    assert final_png.is_file()
    assert final_pdf.is_file()
    payload = json.loads(final_json.read_text(encoding="utf-8"))
    assert payload["target"]["player_id"] == "506469"
    assert int(payload["target"]["jersey_number"]) == 24
    pngs = list((ROOT / "artifacts/final_delivery").glob("*.png"))
    assert len(pngs) == 1
    assert pngs[0].name == "single_player_analysis_summary.png"
    assert not (
        ROOT / "artifacts/evidence/stage_16_real_video_pilot/real_video_pilot_summary.png"
    ).exists()
    assert not (ROOT / "artifacts/final/single_player_analysis_summary.png").exists()
    assert not (ROOT / "artifacts/final_delivery/real_video_tracking_proof.mp4").exists()


def test_offline_env_no_hf_token_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    os.environ.pop("HF_HUB_OFFLINE", None)
    result = run_self_contained_acceptance(output_dir=tmp_path / "off", config=ScenarioConfig())
    assert result["receipt"]["hf_required"] is False
    assert result["receipt"]["download_required"] is False


def test_cleanup_receipt_data_loss_false() -> None:
    receipt = ROOT / "artifacts/evidence/stage_16_r4/cleanup_receipt.json"
    assert receipt.is_file()
    data = json.loads(receipt.read_text())
    assert data["data_loss"] is False
    assert all(item.get("data_loss") is False for item in data["items"])
