"""Unit tests for TeamTrack MOT adapter and anonymous target selection."""

from __future__ import annotations

from pathlib import Path

from football_analytics.acceptance.teamtrack.loader import (
    MotBox,
    TeamTrackSequence,
    load_mot_gt,
    parse_seqinfo,
)
from football_analytics.acceptance.teamtrack.mot_eval import evaluate_detection_frames, iou_xywh
from football_analytics.acceptance.teamtrack.target_selection import select_anonymous_track


def test_parse_seqinfo_and_mot(tmp_path: Path) -> None:
    info = tmp_path / "seqinfo.ini"
    info.write_text(
        "[Sequence]\nname=demo\nimdir=img1\nframerate=25.0\nseqlength=3\nimwidth=100\nimheight=50\nimext=.jpg\n"
    )
    meta = parse_seqinfo(info)
    assert meta["seqlength"] == 3
    assert meta["framerate"] == 25.0
    gt = tmp_path / "gt.txt"
    gt.write_text("1,7,10,20,30,40,1,-1,-1\n2,7,11,21,30,40,1,-1,-1\n3,7,12,22,30,40,1,-1,-1\n")
    boxes = load_mot_gt(gt)
    assert len(boxes) == 3
    assert boxes[0].track_id == 7


def test_select_anonymous_track_deterministic() -> None:
    boxes = []
    for f in range(1, 11):
        boxes.append(MotBox(f, 2, 0, 0, 10, 10, 1, -1, -1))
        boxes.append(MotBox(f, 5, 0, 0, 50, 50, 1, -1, -1))
        boxes.append(MotBox(f, 9, 0, 0, 49, 49, 1, -1, -1))
    seq = TeamTrackSequence(
        dataset_id="teamtrack",
        sport_view="soccer_side",
        split="train",
        sequence_id="demo",
        root=Path("."),
        video_path=Path("x.mp4"),
        gt_path=Path("gt.txt"),
        seqinfo_path=Path("seqinfo.ini"),
        fps=25.0,
        seq_length=10,
        im_width=100,
        im_height=50,
        boxes=tuple(boxes),
    )
    a = select_anonymous_track(seq)
    b = select_anonymous_track(seq)
    assert a.persistent_track_id == b.persistent_track_id
    assert a.confirmation_source == "reviewed_ground_truth_target_selection"
    assert "jersey" not in a.display_name.lower()
    assert a.persistent_track_id == 5  # largest area in top quartile, then lowest id among top


def test_iou_and_detection_eval() -> None:
    assert iou_xywh((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    metrics = evaluate_detection_frames(
        gt_by_frame={1: [(0, 0, 10, 10)]},
        pred_by_frame={1: [(1, 1, 10, 10)]},
        iou_thresh=0.3,
    )
    assert metrics["tp"] == 1
