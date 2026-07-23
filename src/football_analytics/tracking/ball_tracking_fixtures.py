"""Synthetic fixtures for Stage 6C ball tracking (no video/model).

Development fixtures and frozen evaluation fixtures are deliberately separate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from football_analytics.core.run_id import generate_run_id
from football_analytics.tracking.fixtures import _attr_row, _cast, _det_row, base_context

RUNTIME_ROOT = Path("/home/fdoblak/workspace/ball_tracking_checks")


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
    class_id: int = 32,
    class_name: str = "ball",
    confidence: float = 0.9,
    quality_flags: Sequence[str] | None = None,
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
    if quality_flags is not None:
        row["quality_flags"] = list(quality_flags)
    return row


def _attr(
    run_id: str,
    video_id: str,
    frame_index: int,
    detection_id: int,
    *,
    entity_type: str = "ball",
    role_label: str = "unknown",
    provenance_json: str | None = None,
) -> dict[str, Any]:
    row = _attr_row(
        run_id,
        video_id,
        frame_index,
        detection_id,
        entity_type=entity_type,
        role_label=role_label,
    )
    if provenance_json is not None:
        row["provenance_json"] = provenance_json
    return row


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
        out["analysis_windows"] = _cast("analysis_windows", [dict(w) for w in windows])
    return out


# --- Development fixtures ---


def dev_constant_velocity(*, run_id: str | None = None) -> dict[str, Any]:
    rid = run_id or generate_run_id()
    vid = "dev_ball_cv_01"
    dets, attrs = [], []
    for i in range(6):
        x = 100.0 + i * 20.0
        dets.append(
            _det(rid, vid, i, i, (x, 200.0, x + 12.0, 212.0), quality_flags=["src:full_frame"])
        )
        attrs.append(_attr(rid, vid, i, i, provenance_json='{"candidate_source":"full_frame"}'))
    return _bundle(run_id=rid, video_id=vid, n_frames=6, dets=dets, attrs=attrs)


# --- Frozen evaluation fixtures ---


def frozen_constant_velocity(*, run_id: str | None = None) -> dict[str, Any]:
    """1. Sabit hızlı top."""
    rid = run_id or generate_run_id()
    vid = "frozen_ball_cv_01"
    dets, attrs = [], []
    for i in range(5):
        x = 80.0 + i * 15.0
        bbox = (x, 180.0, x + 10.0, 190.0)
        dets.append(_det(rid, vid, i, i, bbox, quality_flags=["src:full_frame"]))
        attrs.append(_attr(rid, vid, i, i))
    ctx = base_context(run_id=rid, video_id=vid, n_frames=5)
    window = _playable_window(rid, vid, n_frames=5, times=ctx["times"])
    bundle = _bundle(run_id=rid, video_id=vid, n_frames=5, dets=dets, attrs=attrs, windows=[window])
    bundle["fixture_kind"] = "frozen_eval"
    return bundle


def frozen_accelerate(*, run_id: str | None = None) -> dict[str, Any]:
    """2. Hızlanan top."""
    rid = run_id or generate_run_id()
    vid = "frozen_ball_acc_01"
    dets, attrs = [], []
    x = 50.0
    for i in range(5):
        step = 8.0 + i * 12.0
        bbox = (x, 200.0, x + 10.0, 210.0)
        dets.append(_det(rid, vid, i, i, bbox))
        attrs.append(_attr(rid, vid, i, i))
        x += step
    bundle = _bundle(run_id=rid, video_id=vid, n_frames=5, dets=dets, attrs=attrs)
    bundle["fixture_kind"] = "frozen_eval"
    return bundle


def frozen_direction_change(*, run_id: str | None = None) -> dict[str, Any]:
    """3. Yön değişimi."""
    rid = run_id or generate_run_id()
    vid = "frozen_ball_dir_01"
    positions = [50.0, 80.0, 110.0, 90.0, 70.0]
    dets, attrs = [], []
    for i, x in enumerate(positions):
        dets.append(_det(rid, vid, i, i, (x, 200.0, x + 10.0, 210.0)))
        attrs.append(_attr(rid, vid, i, i))
    bundle = _bundle(run_id=rid, video_id=vid, n_frames=5, dets=dets, attrs=attrs)
    bundle["fixture_kind"] = "frozen_eval"
    return bundle


def frozen_fast_zero_iou(*, run_id: str | None = None) -> dict[str, Any]:
    """4. Hızlı hareket ve sıfır IoU (motion association)."""
    rid = run_id or generate_run_id()
    vid = "frozen_ball_fast_01"
    # Large center jumps — IoU=0 but within motion gate with CV.
    positions = [40.0, 100.0, 160.0, 220.0]
    dets, attrs = [], []
    for i, x in enumerate(positions):
        dets.append(_det(rid, vid, i, i, (x, 300.0, x + 8.0, 308.0)))
        attrs.append(_attr(rid, vid, i, i))
    ctx = base_context(run_id=rid, video_id=vid, n_frames=4)
    window = _playable_window(rid, vid, n_frames=4, times=ctx["times"])
    bundle = _bundle(run_id=rid, video_id=vid, n_frames=4, dets=dets, attrs=attrs, windows=[window])
    bundle["fixture_kind"] = "frozen_eval"
    bundle["expect_zero_iou_link"] = True
    return bundle


def frozen_short_gap(*, run_id: str | None = None) -> dict[str, Any]:
    """5. Kısa detection kaybı → predicted + recover."""
    rid = run_id or generate_run_id()
    vid = "frozen_ball_short_gap_01"
    dets, attrs = [], []
    for i, x in ((0, 40.0), (1, 55.0), (2, 70.0), (4, 100.0)):
        dets.append(_det(rid, vid, i, i, (x, 200.0, x + 10.0, 210.0)))
        attrs.append(_attr(rid, vid, i, i))
    ctx = base_context(run_id=rid, video_id=vid, n_frames=5)
    window = _playable_window(rid, vid, n_frames=5, times=ctx["times"])
    bundle = _bundle(run_id=rid, video_id=vid, n_frames=5, dets=dets, attrs=attrs, windows=[window])
    bundle["fixture_kind"] = "frozen_eval"
    bundle["expect_prediction"] = True
    return bundle


def frozen_long_gap(*, run_id: str | None = None) -> dict[str, Any]:
    """6. Uzun kayıp → terminate + new track (no ReID)."""
    rid = run_id or generate_run_id()
    vid = "frozen_ball_long_gap_01"
    dets, attrs = [], []
    for i, x in ((0, 40.0), (1, 50.0), (2, 60.0), (25, 70.0)):
        dets.append(_det(rid, vid, i, i, (x, 200.0, x + 10.0, 210.0)))
        attrs.append(_attr(rid, vid, i, i))
    ctx = base_context(run_id=rid, video_id=vid, n_frames=30)
    window = _playable_window(rid, vid, n_frames=30, times=ctx["times"])
    bundle = _bundle(
        run_id=rid, video_id=vid, n_frames=30, dets=dets, attrs=attrs, windows=[window]
    )
    bundle["fixture_kind"] = "frozen_eval"
    bundle["expect_new_track_after_gap"] = True
    return bundle


def frozen_multi_fp(*, run_id: str | None = None) -> dict[str, Any]:
    """7. Birden fazla false-positive aday."""
    rid = run_id or generate_run_id()
    vid = "frozen_ball_fp_01"
    dets, attrs = [], []
    did = 0
    for i in range(4):
        x = 100.0 + i * 12.0
        dets.append(_det(rid, vid, i, did, (x, 200.0, x + 10.0, 210.0), confidence=0.92))
        attrs.append(_attr(rid, vid, i, did))
        did += 1
        # Far FP (logo / ad board style)
        dets.append(
            _det(
                rid,
                vid,
                i,
                did,
                (900.0, 50.0, 912.0, 62.0),
                confidence=0.88,
                quality_flags=["src:tile"],
            )
        )
        attrs.append(_attr(rid, vid, i, did, provenance_json='{"candidate_source":"tile"}'))
        did += 1
    bundle = _bundle(run_id=rid, video_id=vid, n_frames=4, dets=dets, attrs=attrs)
    bundle["fixture_kind"] = "frozen_eval"
    return bundle


def frozen_ambiguity(*, run_id: str | None = None) -> dict[str, Any]:
    """8. Yakın skorlu iki aday → ambiguity."""
    rid = run_id or generate_run_id()
    vid = "frozen_ball_amb_01"
    dets, attrs = [], []
    # Frame 0: two births close in score, far in space
    dets.append(_det(rid, vid, 0, 0, (100.0, 200.0, 110.0, 210.0), confidence=0.85))
    attrs.append(_attr(rid, vid, 0, 0))
    dets.append(_det(rid, vid, 0, 1, (400.0, 200.0, 410.0, 210.0), confidence=0.84))
    attrs.append(_attr(rid, vid, 0, 1))
    # Frame 1: both continue — close scores → ambiguous primary
    dets.append(_det(rid, vid, 1, 2, (112.0, 200.0, 122.0, 210.0), confidence=0.86))
    attrs.append(_attr(rid, vid, 1, 2))
    dets.append(_det(rid, vid, 1, 3, (412.0, 200.0, 422.0, 210.0), confidence=0.85))
    attrs.append(_attr(rid, vid, 1, 3))
    bundle = _bundle(run_id=rid, video_id=vid, n_frames=2, dets=dets, attrs=attrs)
    bundle["fixture_kind"] = "frozen_eval"
    bundle["expect_ambiguity"] = True
    return bundle


def frozen_size_change(*, run_id: str | None = None) -> dict[str, Any]:
    """9. Boyut değişimi (moderate — within gate)."""
    rid = run_id or generate_run_id()
    vid = "frozen_ball_size_01"
    sizes = [8.0, 10.0, 12.0, 14.0]
    dets, attrs = [], []
    for i, s in enumerate(sizes):
        x = 100.0 + i * 10.0
        dets.append(_det(rid, vid, i, i, (x, 200.0, x + s, 200.0 + s)))
        attrs.append(_attr(rid, vid, i, i))
    bundle = _bundle(run_id=rid, video_id=vid, n_frames=4, dets=dets, attrs=attrs)
    bundle["fixture_kind"] = "frozen_eval"
    return bundle


def frozen_frame_edge(*, run_id: str | None = None) -> dict[str, Any]:
    """10. Frame kenarına çıkış."""
    rid = run_id or generate_run_id()
    vid = "frozen_ball_edge_01"
    xs = [10.0, 6.0, 2.0, -2.0]
    dets, attrs = [], []
    for i, x in enumerate(xs):
        x1 = max(0.0, x)
        dets.append(_det(rid, vid, i, i, (x1, 200.0, x1 + 10.0, 210.0)))
        attrs.append(_attr(rid, vid, i, i))
    bundle = _bundle(run_id=rid, video_id=vid, n_frames=4, dets=dets, attrs=attrs)
    bundle["fixture_kind"] = "frozen_eval"
    return bundle


def frozen_shot_cut(*, run_id: str | None = None) -> dict[str, Any]:
    """11. Shot cut — no cross-cut continuation."""
    ctx = base_context(run_id=run_id, video_id="frozen_ball_shot_01", n_frames=8)
    rid, vid = ctx["run_id"], ctx["video_id"]
    times = ctx["times"]
    windows = [
        {
            **_playable_window(
                rid, vid, n_frames=4, times=times[:4], shot_id="shot_a", window_id="aw_a"
            ),
            "end_frame_index_exclusive": 4,
            "end_time_us": times[3],
        },
        {
            **_playable_window(
                rid, vid, n_frames=8, times=times, shot_id="shot_b", window_id="aw_b"
            ),
            "start_frame_index": 4,
            "start_time_us": times[4],
            "end_frame_index_exclusive": 8,
        },
    ]
    dets, attrs = [], []
    for i in range(8):
        x = 40.0 + i * 8.0
        dets.append(_det(rid, vid, i, i, (x, 200.0, x + 10.0, 210.0)))
        attrs.append(_attr(rid, vid, i, i))
    bundle = _bundle(run_id=rid, video_id=vid, n_frames=8, dets=dets, attrs=attrs, windows=windows)
    bundle["fixture_kind"] = "frozen_eval"
    bundle["expect_shot_cut_split"] = True
    return bundle


def frozen_replay_nonplayable(*, run_id: str | None = None) -> dict[str, Any]:
    """12. Replay/non-playable/graphics terminate."""
    ctx = base_context(run_id=run_id, video_id="frozen_ball_replay_01", n_frames=10)
    rid, vid = ctx["run_id"], ctx["video_id"]
    dets, attrs = [], []
    for i in range(10):
        x = 40.0 + i * 5.0
        dets.append(_det(rid, vid, i, i, (x, 200.0, x + 10.0, 210.0)))
        attrs.append(_attr(rid, vid, i, i))
    # Use default base_context windows (second is non-playable graphics).
    return {
        **ctx,
        "detections": _cast("detections", dets),
        "detection_attributes": _cast("detection_attributes", attrs),
        "fixture_kind": "frozen_eval",
        "expect_non_playable_terminate": True,
    }


def frozen_vfr(*, run_id: str | None = None) -> dict[str, Any]:
    """13. VFR timestamp gaps."""
    rid = run_id or generate_run_id()
    vid = "frozen_ball_vfr_01"
    dets, attrs = [], []
    for i in range(5):
        x = 20.0 + i * 10.0
        dets.append(_det(rid, vid, i, i, (x, 200.0, x + 10.0, 210.0)))
        attrs.append(_attr(rid, vid, i, i))
    bundle = _bundle(run_id=rid, video_id=vid, n_frames=5, dets=dets, attrs=attrs, vfr=True)
    bundle["fixture_kind"] = "frozen_eval"
    return bundle


def frozen_zero_detection(*, run_id: str | None = None) -> dict[str, Any]:
    """14. Zero detection frames."""
    rid = run_id or generate_run_id()
    vid = "frozen_ball_zero_01"
    dets = [_det(rid, vid, 0, 0, (100.0, 200.0, 110.0, 210.0))]
    attrs = [_attr(rid, vid, 0, 0)]
    # frames 1-3 empty
    dets.append(_det(rid, vid, 4, 1, (140.0, 200.0, 150.0, 210.0)))
    attrs.append(_attr(rid, vid, 4, 1))
    ctx = base_context(run_id=rid, video_id=vid, n_frames=5)
    window = _playable_window(rid, vid, n_frames=5, times=ctx["times"])
    bundle = _bundle(run_id=rid, video_id=vid, n_frames=5, dets=dets, attrs=attrs, windows=[window])
    bundle["fixture_kind"] = "frozen_eval"
    return bundle


def frozen_reject_human(*, run_id: str | None = None) -> dict[str, Any]:
    """16. Human detection negatif kontrolü."""
    rid = run_id or generate_run_id()
    vid = "frozen_ball_reject_human_01"
    dets = [
        _det(rid, vid, 0, 0, (100.0, 200.0, 110.0, 210.0), class_name="ball"),
        _det(
            rid,
            vid,
            0,
            1,
            (50.0, 50.0, 80.0, 150.0),
            class_id=0,
            class_name="person",
            confidence=0.95,
        ),
    ]
    attrs = [
        _attr(rid, vid, 0, 0, entity_type="ball"),
        _attr(rid, vid, 0, 1, entity_type="human", role_label="player"),
    ]
    bundle = _bundle(run_id=rid, video_id=vid, n_frames=1, dets=dets, attrs=attrs)
    bundle["fixture_kind"] = "frozen_eval"
    bundle["expect_human_rejected"] = True
    return bundle


def frozen_tie_break(*, run_id: str | None = None) -> dict[str, Any]:
    """Deterministic tie-break across two ball tracks."""
    rid = run_id or generate_run_id()
    vid = "frozen_ball_tie_01"
    dets = [
        _det(rid, vid, 0, 0, (10.0, 200.0, 20.0, 210.0)),
        _det(rid, vid, 0, 1, (200.0, 200.0, 210.0, 210.0)),
        _det(rid, vid, 1, 2, (22.0, 200.0, 32.0, 210.0)),
        _det(rid, vid, 1, 3, (212.0, 200.0, 222.0, 210.0)),
    ]
    attrs = [_attr(rid, vid, d["frame_index"], d["detection_id"]) for d in dets]
    bundle = _bundle(run_id=rid, video_id=vid, n_frames=2, dets=dets, attrs=attrs)
    bundle["fixture_kind"] = "frozen_eval"
    return bundle


def all_frozen_fixtures() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("frozen_constant_velocity", frozen_constant_velocity()),
        ("frozen_accelerate", frozen_accelerate()),
        ("frozen_direction_change", frozen_direction_change()),
        ("frozen_fast_zero_iou", frozen_fast_zero_iou()),
        ("frozen_short_gap", frozen_short_gap()),
        ("frozen_long_gap", frozen_long_gap()),
        ("frozen_multi_fp", frozen_multi_fp()),
        ("frozen_ambiguity", frozen_ambiguity()),
        ("frozen_size_change", frozen_size_change()),
        ("frozen_frame_edge", frozen_frame_edge()),
        ("frozen_shot_cut", frozen_shot_cut()),
        ("frozen_replay_nonplayable", frozen_replay_nonplayable()),
        ("frozen_vfr", frozen_vfr()),
        ("frozen_zero_detection", frozen_zero_detection()),
        ("frozen_reject_human", frozen_reject_human()),
        ("frozen_tie_break", frozen_tie_break()),
    ]


__all__ = [
    "RUNTIME_ROOT",
    "assert_runtime_root",
    "dev_constant_velocity",
    "frozen_constant_velocity",
    "frozen_accelerate",
    "frozen_direction_change",
    "frozen_fast_zero_iou",
    "frozen_short_gap",
    "frozen_long_gap",
    "frozen_multi_fp",
    "frozen_ambiguity",
    "frozen_size_change",
    "frozen_frame_edge",
    "frozen_shot_cut",
    "frozen_replay_nonplayable",
    "frozen_vfr",
    "frozen_zero_detection",
    "frozen_reject_human",
    "frozen_tie_break",
    "all_frozen_fixtures",
]
