"""Synthetic fixtures for Stage 6B human tracking (no video/model).

Development fixtures and frozen evaluation fixtures are deliberately separate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from football_analytics.core.run_id import generate_run_id
from football_analytics.tracking.fixtures import _attr_row, _cast, _det_row, base_context

RUNTIME_ROOT = Path("/home/fdoblak/workspace/human_tracking_checks")


def assert_runtime_root() -> None:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not str(RUNTIME_ROOT).startswith("/home/fdoblak/workspace/"):
        raise RuntimeError("unexpected runtime root")


def _det(
    run_id: str,
    video_id: str,
    frame_index: int,
    detection_id: int,
    bbox: tuple[float, float, float, float],
    *,
    class_id: int = 0,
    class_name: str = "person",
    confidence: float = 0.9,
) -> dict[str, Any]:
    row = _det_row(
        run_id,
        video_id,
        frame_index,
        detection_id,
        class_id=class_id,
        class_name=class_name,
        bbox=bbox,
    )
    row["confidence"] = confidence
    return row


def _attr(
    run_id: str,
    video_id: str,
    frame_index: int,
    detection_id: int,
    *,
    entity_type: str = "human",
    role_label: str = "unknown",
) -> dict[str, Any]:
    return _attr_row(
        run_id,
        video_id,
        frame_index,
        detection_id,
        entity_type=entity_type,
        role_label=role_label,
    )


def _bundle(
    *,
    run_id: str | None,
    video_id: str,
    n_frames: int,
    dets: list[dict[str, Any]],
    attrs: list[dict[str, Any]],
    windows: Sequence[Mapping[str, Any]] | None = None,
    vfr: bool = False,
) -> dict[str, Any]:
    ctx = base_context(run_id=run_id, video_id=video_id, n_frames=n_frames, vfr=vfr)
    out = {
        **ctx,
        "detections": _cast("detections", dets),
        "detection_attributes": _cast("detection_attributes", attrs),
    }
    if windows is not None:
        # Rebuild analysis_windows from provided rows (must include required fields).
        out["analysis_windows"] = _cast("analysis_windows", [dict(w) for w in windows])
    return out


# --- Development fixtures (tuning-safe; not frozen eval) ---


def dev_single_person_linear(*, run_id: str | None = None) -> dict[str, Any]:
    """Dev: single person moving linearly."""
    rid = run_id or generate_run_id()
    vid = "dev_single_01"
    dets = []
    attrs = []
    for i in range(6):
        x = 10.0 + i * 15.0
        dets.append(_det(rid, vid, i, i, (x, 20.0, x + 30.0, 100.0)))
        attrs.append(_attr(rid, vid, i, i, role_label="unknown"))
    return _bundle(run_id=rid, video_id=vid, n_frames=6, dets=dets, attrs=attrs)


def dev_two_crossing(*, run_id: str | None = None) -> dict[str, Any]:
    """Dev: two humans crossing paths."""
    rid = run_id or generate_run_id()
    vid = "dev_cross_01"
    dets = []
    attrs = []
    did = 0
    for i in range(6):
        # A left→right
        xa = 20.0 + i * 40.0
        dets.append(_det(rid, vid, i, did, (xa, 30.0, xa + 25.0, 110.0)))
        attrs.append(_attr(rid, vid, i, did, role_label="player"))
        did += 1
        # B right→left
        xb = 260.0 - i * 40.0
        dets.append(_det(rid, vid, i, did, (xb, 40.0, xb + 25.0, 120.0)))
        attrs.append(_attr(rid, vid, i, did, role_label="player"))
        did += 1
    return _bundle(run_id=rid, video_id=vid, n_frames=6, dets=dets, attrs=attrs)


# --- Frozen evaluation fixtures (do not retune thresholds against these) ---


def frozen_single_person(*, run_id: str | None = None) -> dict[str, Any]:
    rid = run_id or generate_run_id()
    vid = "frozen_single_01"
    dets = []
    attrs = []
    gt = []
    for i in range(5):
        x = 50.0 + i * 10.0
        bbox = (x, 40.0, x + 28.0, 120.0)
        dets.append(_det(rid, vid, i, i, bbox))
        attrs.append(_attr(rid, vid, i, i))
        gt.append(
            {
                "frame_index": i,
                "track_id": 0,
                "bbox_x1": bbox[0],
                "bbox_y1": bbox[1],
                "bbox_x2": bbox[2],
                "bbox_y2": bbox[3],
            }
        )
    bundle = _bundle(run_id=rid, video_id=vid, n_frames=5, dets=dets, attrs=attrs)
    bundle["synthetic_gt"] = gt
    bundle["fixture_kind"] = "frozen_eval"
    return bundle


def frozen_multi_human(*, run_id: str | None = None) -> dict[str, Any]:
    rid = run_id or generate_run_id()
    vid = "frozen_multi_01"
    dets = []
    attrs = []
    did = 0
    for i in range(4):
        for j, y in enumerate((20.0, 200.0)):
            x = 30.0 + i * 12.0 + j * 80.0
            bbox = (x, y, x + 24.0, y + 70.0)
            dets.append(_det(rid, vid, i, did, bbox))
            attrs.append(_attr(rid, vid, i, did))
            did += 1
    bundle = _bundle(run_id=rid, video_id=vid, n_frames=4, dets=dets, attrs=attrs)
    bundle["fixture_kind"] = "frozen_eval"
    return bundle


def _playable_window(
    run_id: str,
    video_id: str,
    *,
    n_frames: int,
    times: Sequence[int],
    shot_id: str = "shot_001",
    window_id: str = "aw_play_full",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "analysis_window_id": window_id,
        "start_time_us": 0,
        "end_time_us": int(times[-1]) + 40000,
        "start_frame_index": 0,
        "end_frame_index_exclusive": n_frames,
        "shot_id": shot_id,
        "camera_segment_ids": ["cam_001"],
        "view_family": "main_broadcast",
        "framing_scale": "wide",
        "replay_status": "live",
        "graphics_status": "none",
        "playability": "playable",
        "tracking_eligibility": "eligible",
        "calibration_eligibility": "eligible",
        "identity_eligibility": "conditionally_eligible",
        "ball_analysis_eligibility": "eligible",
        "live_event_eligibility": "unknown",
        "physical_metric_eligibility": "eligible",
        "decision_codes": ["PLAYABLE_WIDE_VIEW"],
        "manual_review_required": False,
        "coverage": 1.0,
        "confidence": 0.95,
        "timeline_mapping_quality": "exact_identity",
        "source_refs": [shot_id],
        "policy_version": "1",
        "provenance_json": None,
        "contract_version": 1,
    }


def frozen_short_occlusion(*, run_id: str | None = None) -> dict[str, Any]:
    """Confirm (3 hits), miss one frame, recover (within max_lost / prediction)."""
    rid = run_id or generate_run_id()
    vid = "frozen_short_occ_01"
    dets = []
    attrs = []
    # frames 0,1,2 observed → confirmed; 3 miss; 4 recover
    for i, x in ((0, 40.0), (1, 55.0), (2, 70.0), (4, 100.0)):
        bbox = (x, 30.0, x + 30.0, 110.0)
        dets.append(_det(rid, vid, i, i, bbox))
        attrs.append(_attr(rid, vid, i, i))
    ctx = base_context(run_id=rid, video_id=vid, n_frames=5)
    window = _playable_window(rid, vid, n_frames=5, times=ctx["times"])
    bundle = _bundle(
        run_id=rid,
        video_id=vid,
        n_frames=5,
        dets=dets,
        attrs=attrs,
        windows=[window],
    )
    bundle["fixture_kind"] = "frozen_eval"
    bundle["expect_recovery"] = True
    return bundle


def frozen_long_occlusion(*, run_id: str | None = None) -> dict[str, Any]:
    """Long gap beyond max_lost_gap_us → new track (no ReID)."""
    rid = run_id or generate_run_id()
    vid = "frozen_long_occ_01"
    dets = []
    attrs = []
    for i, x in ((0, 40.0), (1, 50.0), (2, 60.0), (25, 70.0)):
        bbox = (x, 30.0, x + 30.0, 110.0)
        dets.append(_det(rid, vid, i, i, bbox))
        attrs.append(_attr(rid, vid, i, i))
    ctx = base_context(run_id=rid, video_id=vid, n_frames=30)
    window = _playable_window(rid, vid, n_frames=30, times=ctx["times"])
    bundle = _bundle(
        run_id=rid,
        video_id=vid,
        n_frames=30,
        dets=dets,
        attrs=attrs,
        windows=[window],
    )
    bundle["fixture_kind"] = "frozen_eval"
    bundle["expect_new_track_after_gap"] = True
    return bundle


def frozen_entry_exit(*, run_id: str | None = None) -> dict[str, Any]:
    rid = run_id or generate_run_id()
    vid = "frozen_entry_exit_01"
    dets = []
    attrs = []
    # Person A frames 0-2; Person B frames 3-5
    did = 0
    for i in range(3):
        x = 10.0 + i * 10.0
        dets.append(_det(rid, vid, i, did, (x, 20.0, x + 20.0, 90.0)))
        attrs.append(_attr(rid, vid, i, did))
        did += 1
    for i in range(3, 6):
        x = 300.0 + (i - 3) * 10.0
        dets.append(_det(rid, vid, i, did, (x, 200.0, x + 20.0, 270.0)))
        attrs.append(_attr(rid, vid, i, did))
        did += 1
    bundle = _bundle(run_id=rid, video_id=vid, n_frames=6, dets=dets, attrs=attrs)
    bundle["fixture_kind"] = "frozen_eval"
    return bundle


def frozen_tie_break(*, run_id: str | None = None) -> dict[str, Any]:
    """Two similar bboxes — association must be deterministic."""
    rid = run_id or generate_run_id()
    vid = "frozen_tie_01"
    dets = []
    attrs = []
    # Frame 0: two births far apart
    dets.append(_det(rid, vid, 0, 0, (10.0, 10.0, 40.0, 80.0)))
    attrs.append(_attr(rid, vid, 0, 0))
    dets.append(_det(rid, vid, 0, 1, (200.0, 10.0, 230.0, 80.0)))
    attrs.append(_attr(rid, vid, 0, 1))
    # Frame 1: nearly identical costs toward track 0 region — ordered detection_ids
    dets.append(_det(rid, vid, 1, 2, (12.0, 12.0, 42.0, 82.0)))
    attrs.append(_attr(rid, vid, 1, 2))
    dets.append(_det(rid, vid, 1, 3, (202.0, 12.0, 232.0, 82.0)))
    attrs.append(_attr(rid, vid, 1, 3))
    bundle = _bundle(run_id=rid, video_id=vid, n_frames=2, dets=dets, attrs=attrs)
    bundle["fixture_kind"] = "frozen_eval"
    return bundle


def frozen_miss_and_fp(*, run_id: str | None = None) -> dict[str, Any]:
    rid = run_id or generate_run_id()
    vid = "frozen_miss_fp_01"
    dets = [
        _det(rid, vid, 0, 0, (10.0, 10.0, 40.0, 80.0)),
        # frame 1 miss — no det
        _det(rid, vid, 2, 1, (30.0, 10.0, 60.0, 80.0)),
        # false positive far away
        _det(rid, vid, 2, 2, (500.0, 400.0, 530.0, 470.0), confidence=0.95),
    ]
    attrs = [
        _attr(rid, vid, 0, 0),
        _attr(rid, vid, 2, 1),
        _attr(rid, vid, 2, 2),
    ]
    bundle = _bundle(run_id=rid, video_id=vid, n_frames=3, dets=dets, attrs=attrs)
    bundle["fixture_kind"] = "frozen_eval"
    return bundle


def frozen_shot_cut(*, run_id: str | None = None) -> dict[str, Any]:
    """Two shots — track must not continue across cut."""
    ctx = base_context(run_id=run_id, video_id="frozen_shot_cut_01", n_frames=8)
    rid, vid = ctx["run_id"], ctx["video_id"]
    times = ctx["times"]
    # Replace windows: shot_001 frames 0-3, shot_002 frames 4-8
    windows = [
        {
            "run_id": rid,
            "video_id": vid,
            "analysis_window_id": "aw_shot_a",
            "start_time_us": 0,
            "end_time_us": times[3],
            "start_frame_index": 0,
            "end_frame_index_exclusive": 4,
            "shot_id": "shot_a",
            "camera_segment_ids": ["cam_a"],
            "view_family": "main_broadcast",
            "framing_scale": "wide",
            "replay_status": "live",
            "graphics_status": "none",
            "playability": "playable",
            "tracking_eligibility": "eligible",
            "calibration_eligibility": "eligible",
            "identity_eligibility": "conditionally_eligible",
            "ball_analysis_eligibility": "eligible",
            "live_event_eligibility": "unknown",
            "physical_metric_eligibility": "eligible",
            "decision_codes": ["PLAYABLE_WIDE_VIEW"],
            "manual_review_required": False,
            "coverage": 1.0,
            "confidence": 0.95,
            "timeline_mapping_quality": "exact_identity",
            "source_refs": ["shot_a"],
            "policy_version": "1",
            "provenance_json": None,
            "contract_version": 1,
        },
        {
            "run_id": rid,
            "video_id": vid,
            "analysis_window_id": "aw_shot_b",
            "start_time_us": times[4],
            "end_time_us": times[-1] + 40000,
            "start_frame_index": 4,
            "end_frame_index_exclusive": 8,
            "shot_id": "shot_b",
            "camera_segment_ids": ["cam_b"],
            "view_family": "main_broadcast",
            "framing_scale": "wide",
            "replay_status": "live",
            "graphics_status": "none",
            "playability": "playable",
            "tracking_eligibility": "eligible",
            "calibration_eligibility": "eligible",
            "identity_eligibility": "conditionally_eligible",
            "ball_analysis_eligibility": "eligible",
            "live_event_eligibility": "unknown",
            "physical_metric_eligibility": "eligible",
            "decision_codes": ["PLAYABLE_WIDE_VIEW"],
            "manual_review_required": False,
            "coverage": 1.0,
            "confidence": 0.95,
            "timeline_mapping_quality": "exact_identity",
            "source_refs": ["shot_b"],
            "policy_version": "1",
            "provenance_json": None,
            "contract_version": 1,
        },
    ]
    dets = []
    attrs = []
    for i in range(8):
        x = 40.0 + i * 5.0
        dets.append(_det(rid, vid, i, i, (x, 30.0, x + 30.0, 100.0)))
        attrs.append(_attr(rid, vid, i, i))
    bundle = _bundle(run_id=rid, video_id=vid, n_frames=8, dets=dets, attrs=attrs, windows=windows)
    bundle["fixture_kind"] = "frozen_eval"
    bundle["expect_shot_cut_split"] = True
    return bundle


def frozen_non_playable(*, run_id: str | None = None) -> dict[str, Any]:
    """Uses default second window (graphics/non-playable) from base_context."""
    ctx = base_context(run_id=run_id, video_id="frozen_nonplay_01", n_frames=10)
    rid, vid = ctx["run_id"], ctx["video_id"]
    dets = []
    attrs = []
    for i in range(10):
        x = 40.0 + i * 5.0
        dets.append(_det(rid, vid, i, i, (x, 30.0, x + 30.0, 100.0)))
        attrs.append(_attr(rid, vid, i, i))
    bundle = {
        **ctx,
        "detections": _cast("detections", dets),
        "detection_attributes": _cast("detection_attributes", attrs),
        "fixture_kind": "frozen_eval",
        "expect_non_playable_terminate": True,
    }
    return bundle


def frozen_vfr(*, run_id: str | None = None) -> dict[str, Any]:
    rid = run_id or generate_run_id()
    vid = "frozen_vfr_01"
    dets = []
    attrs = []
    for i in range(5):
        x = 20.0 + i * 8.0
        dets.append(_det(rid, vid, i, i, (x, 20.0, x + 25.0, 90.0)))
        attrs.append(_attr(rid, vid, i, i))
    bundle = _bundle(run_id=rid, video_id=vid, n_frames=5, dets=dets, attrs=attrs, vfr=True)
    bundle["fixture_kind"] = "frozen_eval"
    return bundle


def frozen_role_unknown_and_conflict(*, run_id: str | None = None) -> dict[str, Any]:
    rid = run_id or generate_run_id()
    vid = "frozen_role_01"
    dets = [
        _det(rid, vid, 0, 0, (10.0, 10.0, 40.0, 80.0)),
        _det(rid, vid, 1, 1, (15.0, 12.0, 45.0, 82.0)),
        _det(rid, vid, 2, 2, (20.0, 14.0, 50.0, 84.0)),
    ]
    attrs = [
        _attr(rid, vid, 0, 0, role_label="unknown"),
        _attr(rid, vid, 1, 1, role_label="player"),
        _attr(rid, vid, 2, 2, role_label="referee"),  # conflict with player
    ]
    bundle = _bundle(run_id=rid, video_id=vid, n_frames=3, dets=dets, attrs=attrs)
    bundle["fixture_kind"] = "frozen_eval"
    bundle["expect_role_conflict"] = True
    return bundle


def frozen_reject_ball(*, run_id: str | None = None) -> dict[str, Any]:
    rid = run_id or generate_run_id()
    vid = "frozen_ball_reject_01"
    dets = [
        _det(rid, vid, 0, 0, (10.0, 10.0, 40.0, 80.0), class_name="person"),
        _det(
            rid,
            vid,
            0,
            1,
            (100.0, 100.0, 110.0, 110.0),
            class_id=32,
            class_name="ball",
        ),
    ]
    attrs = [
        _attr(rid, vid, 0, 0, entity_type="human"),
        _attr(rid, vid, 0, 1, entity_type="ball"),
    ]
    bundle = _bundle(run_id=rid, video_id=vid, n_frames=1, dets=dets, attrs=attrs)
    bundle["fixture_kind"] = "frozen_eval"
    bundle["expect_ball_rejected"] = True
    return bundle


def all_frozen_fixtures() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("frozen_single_person", frozen_single_person()),
        ("frozen_multi_human", frozen_multi_human()),
        ("frozen_short_occlusion", frozen_short_occlusion()),
        ("frozen_long_occlusion", frozen_long_occlusion()),
        ("frozen_entry_exit", frozen_entry_exit()),
        ("frozen_tie_break", frozen_tie_break()),
        ("frozen_miss_and_fp", frozen_miss_and_fp()),
        ("frozen_shot_cut", frozen_shot_cut()),
        ("frozen_non_playable", frozen_non_playable()),
        ("frozen_vfr", frozen_vfr()),
        ("frozen_role_unknown_and_conflict", frozen_role_unknown_and_conflict()),
        ("frozen_reject_ball", frozen_reject_ball()),
    ]


__all__ = [
    "RUNTIME_ROOT",
    "assert_runtime_root",
    "dev_single_person_linear",
    "dev_two_crossing",
    "frozen_multi_human",
    "frozen_short_occlusion",
    "frozen_long_occlusion",
    "frozen_entry_exit",
    "frozen_tie_break",
    "frozen_miss_and_fp",
    "frozen_shot_cut",
    "frozen_non_playable",
    "frozen_vfr",
    "frozen_role_unknown_and_conflict",
    "frozen_reject_ball",
    "all_frozen_fixtures",
]
