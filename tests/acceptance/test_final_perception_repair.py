"""Tests for final perception repair / Turkish delivery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from football_analytics.acceptance.final_perception_repair.pipeline import (
    ConfirmedTracker,
    DetConfig,
    _center_merge,
    _filter_boxes,
)
from football_analytics.acceptance.final_perception_repair.turkish_report import (
    EVIDENCE,
    build_turkish_metric_table,
)


def test_confirmed_tracker_gates_short_tracks() -> None:
    tr = ConfirmedTracker(iou_thresh=0.3, min_hits=3, max_age=5)
    box = (10.0, 10.0, 20.0, 40.0)
    r1 = tr.update([box])
    assert r1[0][2] is False
    r2 = tr.update([(10.5, 10.5, 20.0, 40.0)])
    assert r2[0][2] is False
    r3 = tr.update([(11.0, 11.0, 20.0, 40.0)])
    assert r3[0][2] is True


def test_center_merge_reduces_duplicates() -> None:
    boxes = [(0.0, 0.0, 10.0, 20.0), (2.0, 1.0, 10.0, 20.0), (100.0, 100.0, 10.0, 20.0)]
    scores = [0.9, 0.8, 0.7]
    out, sc = _center_merge(boxes, scores, dist=10.0)
    assert len(out) == 2
    assert len(sc) == 2


def test_filter_boxes_aspect() -> None:
    cfg = DetConfig(name="t", min_area=10, min_h=5, min_aspect=0.2, max_aspect=0.8)
    # w/h=1.0 rejected; w/h=0.25 kept
    boxes = [(0.0, 0.0, 40.0, 40.0), (0.0, 0.0, 10.0, 40.0)]
    scores = [0.5, 0.5]
    kept, _ = _filter_boxes(boxes, scores, cfg=cfg, frame_h=200)
    assert len(kept) == 1
    assert kept[0][2] == 10.0


def test_metric_table_never_empty_zero_for_missing() -> None:
    ref = {
        "annotation_derived_metrics": {
            "measured_distance_m": {"value": 100.0, "status": "REFERENCE_ANNOTATION_DERIVED"},
            "bas_pass_attempts": {"value": 30, "status": "REFERENCE_ANNOTATION_DERIVED"},
            "bas_drive_actions": {"value": 28, "status": "REFERENCE_ANNOTATION_DERIVED"},
            "bas_successful_tackles": {"value": 5, "status": "REFERENCE_ANNOTATION_DERIVED"},
            "bas_header_actions": {"value": 0, "status": "REFERENCE_ANNOTATION_DERIVED"},
            "bas_high_pass_attempts": {"value": 3, "status": "REFERENCE_ANNOTATION_DERIVED"},
            "heatmap": {"value": {"n_points": 10}, "status": "REFERENCE_ANNOTATION_DERIVED"},
            "mean_speed_m_s": {"value": 2.0, "status": "REFERENCE_ANNOTATION_DERIVED"},
            "peak_speed_m_s": {"value": 8.0, "status": "REFERENCE_ANNOTATION_DERIVED"},
            "sprint_count": {"value": 1, "status": "REFERENCE_ANNOTATION_DERIVED"},
            "sprint_distance_m": {"value": 20.0, "status": "REFERENCE_ANNOTATION_DERIVED"},
            "activity_index": {"value": 10.0, "status": "REFERENCE_ANNOTATION_DERIVED"},
            "duel_win_rate": {"value": None, "status": "NOT_EVALUABLE"},
            "clearances": {"value": None, "status": "NOT_EVALUABLE"},
            "box_touches": {"value": None, "status": "NOT_EVALUABLE"},
        }
    }
    perception = {
        "tracking": {
            "target_tracking_eval": {"target_coverage_ratio": 0.9},
            "confirmed_iou": {"id_switches": 1},
        }
    }
    rows = build_turkish_metric_table(ref=ref, perception=perception)
    names = [r["metric"] for r in rows]
    required = [
        "Isı haritası",
        "İkili mücadele sayısı",
        "Pas girişimi",
        "Başarılı dripling",
        "Top çalma",
        "Koşu mesafesi",
        "Coverage",
    ]
    for r in required:
        assert r in names
    for row in rows:
        if row["status"] == "ÖLÇÜLEMEDİ":
            assert row["value"] != 0
            assert row["value"] != "0"
            assert "ÖLÇÜLEMEDİ" in str(row["value"])
    assert EVIDENCE["NOT_MEASURED"]


def test_root_cause_json_exists_in_workspace_or_skip() -> None:
    p = Path("/home/fdoblak/workspace/final_perception_repair/artifacts/root_cause_diagnosis.json")
    if not p.is_file():
        pytest.skip("workspace root cause not present")
    data = json.loads(p.read_text())
    assert data["findings"]["gt_and_prediction_dual_boxes"]["present"] is True
