"""Target selection + adapter namespace tests with mini fixtures."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from football_analytics.acceptance.contracts import (
    EXTERNAL_REFERENCE_CONFIRMATION,
    NAMESPACE_PREDICTIONS,
    NAMESPACE_REFERENCE_GT,
)
from football_analytics.acceptance.leakage import LeakageError
from football_analytics.acceptance.panorama_domain import classify_camera_domain
from football_analytics.acceptance.soccertrack_v2.adapter import (
    materialize_reference_bundle,
    write_bas_reference,
)
from football_analytics.acceptance.soccertrack_v2.loader import (
    iter_gsr_player_observations,
    load_bas_events,
)
from football_analytics.acceptance.soccertrack_v2.target_selection import select_target_player

FIXTURES = Path(__file__).parent / "fixtures" / "soccertrack_v2_mini"


def _layout_mini(tmp_path: Path) -> Path:
    root = tmp_path / "soccertrack_v2"
    match = "999001"
    (root / "gsr" / match).mkdir(parents=True)
    (root / "bas" / match).mkdir(parents=True)
    shutil.copy(
        FIXTURES / "gsr_999001_1st.json",
        root / "gsr" / match / "999001_1st.json",
    )
    shutil.copy(
        FIXTURES / "gsr_999001_2nd.json",
        root / "gsr" / match / "999001_2nd.json",
    )
    shutil.copy(
        FIXTURES / "bas_999001_12_class_events.json",
        root / "bas" / match / "999001_12_class_events.json",
    )
    return root


def test_bas_label_normalization(tmp_path: Path) -> None:
    root = _layout_mini(tmp_path)
    events = load_bas_events(root / "bas" / "999001" / "999001_12_class_events.json")
    assert {e.label for e in events} >= {"Pass", "Drive", "Shot"}


def test_gsr_stream_skips_pitch(tmp_path: Path) -> None:
    root = _layout_mini(tmp_path)
    obs = list(
        iter_gsr_player_observations(root / "gsr" / "999001" / "999001_1st.json", half=1)
    )
    assert all(o.role in {"player", "goalkeeper"} for o in obs)
    assert any(o.player_id == "1001" for o in obs)


def test_target_selection_prefers_outfield(tmp_path: Path) -> None:
    root = _layout_mini(tmp_path)
    receipt = select_target_player(root=root, match_id="999001", min_frames=2)
    assert receipt.selected_player_id == "1001"
    assert receipt.jersey_number == 9
    assert receipt.team_side == "left"
    assert receipt.confirmation_source == EXTERNAL_REFERENCE_CONFIRMATION
    assert "SoccerTrack v2 Match 999001" in receipt.display_name


def test_adapter_refuses_predictions_namespace(tmp_path: Path) -> None:
    root = _layout_mini(tmp_path)
    with pytest.raises(LeakageError):
        write_bas_reference(root, "999001", tmp_path / NAMESPACE_PREDICTIONS / "gt")


def test_materialize_reference_bundle(tmp_path: Path) -> None:
    root = _layout_mini(tmp_path)
    receipt = select_target_player(root=root, match_id="999001", min_frames=2)
    run = tmp_path / "run"
    paths = materialize_reference_bundle(
        root=root, match_id="999001", run_dir=run, receipt=receipt, stride=1
    )
    assert (run / NAMESPACE_REFERENCE_GT / "bas_reference_events.json").is_file()
    assert "trajectory" in paths
    payload = json.loads((run / NAMESPACE_REFERENCE_GT / "bas_reference_events.json").read_text())
    assert payload["annotation_provenance"].startswith("external_cc_by")


def test_panorama_domain_finding() -> None:
    d = classify_camera_domain()
    assert d["camera_domain"] == "panoramic_full_pitch"
    assert d["broadcast_acceptance"] == "not_covered_by_this_dataset"
