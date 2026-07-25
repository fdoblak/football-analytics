"""Unit tests for Stage 17-R1 jersey-5 appearance tracker and anchors."""

from __future__ import annotations

import numpy as np

from football_analytics.acceptance.stage17r1_jersey5.pipeline import (
    VISUAL_ANCHORS_JERSEY5,
    AppearanceConfirmedTracker,
    match_anchor_to_tracks,
)


def test_visual_anchors_jersey5_requirements() -> None:
    assert len(VISUAL_ANCHORS_JERSEY5) >= 3
    assert all(a["jersey_number"] == 5 for a in VISUAL_ANCHORS_JERSEY5)
    assert all(a["team"] == "team_yellow" for a in VISUAL_ANCHORS_JERSEY5)
    assert all(a["review_status"] == "reviewed" for a in VISUAL_ANCHORS_JERSEY5)
    times = sorted(a["t_s"] for a in VISUAL_ANCHORS_JERSEY5)
    assert times[-1] - times[0] >= 2.0


def test_appearance_tracker_confirms_after_min_hits() -> None:
    tracker = AppearanceConfirmedTracker(min_hits=3, max_age=5)
    hist = np.ones(48, dtype=np.float64)
    hist = hist / np.linalg.norm(hist)
    out = tracker.update([((10.0, 10.0, 20.0, 40.0), hist)])
    assert out[0][2] is False
    tracker.update([((11.0, 10.0, 20.0, 40.0), hist)])
    out = tracker.update([((12.0, 10.0, 20.0, 40.0), hist)])
    assert out[0][2] is True
    assert out[0][0] == 1


def test_match_anchor_to_tracks_iou() -> None:
    anchors = [
        {
            "anchor_id": "t",
            "frame": 1,
            "box": [10, 10, 20, 40],
        }
    ]
    tracks = {1: [(7, (10.0, 10.0, 20.0, 40.0), True)]}
    m = match_anchor_to_tracks(anchors=anchors, tracks_by_frame=tracks)
    assert m["n_matched_anchors"] == 1
    assert m["primary_track_id"] == 7
