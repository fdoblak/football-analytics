"""Ball tracker frame loop (Stage 6C).

Motion-first association with Stage 6A lifecycle. Ball-only entity; role always
unknown. No ReID; terminated never reopen; no cross-shot/cross-cut prediction.
Primary ball candidate ≤1 per frame with ambiguity when margins are tight.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from football_analytics.perception.detection_evaluation import center_l2
from football_analytics.tracking.association_common import normalize_candidate_source
from football_analytics.tracking.ball_association import (
    greedy_ball_associate,
    primary_ball_score,
)
from football_analytics.tracking.human_motion import (
    BBox,
    bbox_center,
    bbox_wh,
    predict_bbox_constant_velocity,
    velocity_from_centers,
)
from football_analytics.tracking.lifecycle import assert_transition_allowed
from football_analytics.tracking.time_rules import gap_us
from football_analytics.tracking.track_ids import TrackIdAllocator
from football_analytics.tracking.types import (
    CONTRACT_VERSION,
    GapReason,
    LifecycleState,
    ObservationSource,
    TrackingContractError,
)


@dataclass
class _ActiveTrack:
    track_id: int
    lifecycle: LifecycleState
    bbox: BBox
    vx: float = 0.0
    vy: float = 0.0
    last_frame_index: int = 0
    last_time_us: int = 0
    last_observed_time_us: int = 0
    association_count: int = 0
    class_id: int = 32
    model_id: str = "ball_motion_cv_v1"
    quality_flags: list[str] = field(default_factory=list)
    event_index: int = 0
    shot_id: str | None = None
    window_id: str | None = None
    birth_frame: int = 0
    termination_reason: str | None = None
    review_required: bool = False
    candidate_source: str = "unknown"
    last_confidence: float | None = None
    last_association_cost: float | None = None
    prediction_uncertainty: float = 0.0


@dataclass
class TrackerResult:
    observations: list[dict[str, Any]]
    lifecycle: list[dict[str, Any]]
    summaries: list[dict[str, Any]]
    primary_sidecar: list[dict[str, Any]]
    findings: list[str]
    stats: dict[str, Any]


def _window_for_frame(
    frame_index: int, windows: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    for w in windows:
        start = w.get("start_frame_index")
        end = w.get("end_frame_index_exclusive")
        if start is None or end is None:
            continue
        if int(start) <= int(frame_index) < int(end):
            return w
    return None


def _extract_candidate_source(det: Mapping[str, Any], attr: Mapping[str, Any] | None) -> str:
    flags = det.get("quality_flags") or []
    if isinstance(flags, (list, tuple)):
        for f in flags:
            s = str(f)
            if s.startswith("src:") or "full_frame" in s or s.startswith("tile"):
                return normalize_candidate_source(s)
    if attr is not None:
        prov = attr.get("provenance_json")
        if isinstance(prov, str) and prov:
            for token in ("full_frame", "hybrid", "tile"):
                if token in prov.lower():
                    return normalize_candidate_source(token)
        qf = attr.get("quality_flags") or []
        if isinstance(qf, (list, tuple)):
            for f in qf:
                src = normalize_candidate_source(str(f))
                if src != "unknown":
                    return src
    return "unknown"


def _is_ball_detection(
    det: Mapping[str, Any],
    attr: Mapping[str, Any] | None,
    *,
    config: Mapping[str, Any],
) -> bool:
    entity = config["entity_filter"]
    reject_names = {str(x).lower() for x in entity["reject_class_names"]}
    ball_names = {str(x).lower() for x in entity["ball_class_names"]}
    class_name = str(det.get("class_name", "")).lower()
    if class_name in reject_names:
        return False
    if attr is not None:
        et = str(attr.get("entity_type", "")).lower()
        if et in {str(x).lower() for x in entity["reject_entity_types"]}:
            return False
        if et == "ball":
            return True
        if et and et != "ball":
            return False
    return class_name in ball_names


def _at_frame_edge(bbox: BBox, *, width: int, height: int, margin: float) -> bool:
    x1, y1, x2, y2 = bbox
    return (
        x1 <= margin or y1 <= margin or x2 >= float(width) - margin or y2 >= float(height) - margin
    )


def _life_event(
    track: _ActiveTrack,
    *,
    run_id: str,
    video_id: str,
    frame_index: int,
    video_time_us: int,
    state: LifecycleState,
    previous: LifecycleState | None,
    reason: str,
    observation_source: str | None,
    policy: Mapping[str, Any] | None,
) -> dict[str, Any]:
    assert_transition_allowed(previous, state, policy=policy)
    ev = {
        "run_id": run_id,
        "video_id": video_id,
        "track_id": track.track_id,
        "event_index": track.event_index,
        "frame_index": int(frame_index),
        "video_time_us": int(video_time_us),
        "lifecycle_state": state.value,
        "previous_state": None if previous is None else previous.value,
        "entity_type": "ball",
        "transition_reason": reason,
        "observation_source": observation_source,
        "manual_review_required": bool(track.review_required),
        "quality_flags": list(track.quality_flags),
        "provenance_json": None,
        "contract_version": CONTRACT_VERSION,
    }
    track.event_index += 1
    track.lifecycle = state
    return ev


def _obs_row(
    *,
    run_id: str,
    video_id: str,
    frame_index: int,
    track: _ActiveTrack,
    detection_id: int | None,
    bbox: BBox,
    observation_state: str,
    confidence: float | None,
    quality_flags: Sequence[str],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "frame_index": int(frame_index),
        "track_id": int(track.track_id),
        "detection_id": detection_id,
        "class_id": int(track.class_id),
        "confidence": confidence,
        "bbox_x1": float(bbox[0]),
        "bbox_y1": float(bbox[1]),
        "bbox_x2": float(bbox[2]),
        "bbox_y2": float(bbox[3]),
        "observation_state": observation_state,
        "model_id": track.model_id,
        "quality_flags": list(quality_flags),
    }


def _size_ratio(a: BBox, b: BBox) -> float:
    aw, ah = bbox_wh(a)
    bw, bh = bbox_wh(b)
    area_a = max(aw * ah, 1e-6)
    area_b = max(bw * bh, 1e-6)
    return max(area_a / area_b, area_b / area_a)


def run_ball_tracker(
    *,
    run_id: str,
    video_id: str,
    frames: Sequence[Mapping[str, Any]],
    detections: Sequence[Mapping[str, Any]],
    analysis_windows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    detection_attributes: Sequence[Mapping[str, Any]] | None = None,
    policy: Mapping[str, Any] | None = None,
    tracker_model_id: str = "ball_motion_cv_v1",
) -> TrackerResult:
    """Run deterministic ball MOT over a frame timeline (no video decode)."""
    if len(frames) > int(config["safety_limits"]["max_frames_per_run"]):
        raise TrackingContractError("max_frames_per_run exceeded")

    frames_sorted = sorted(frames, key=lambda r: (int(r["frame_index"]), int(r["video_time_us"])))
    dets_by_frame: dict[int, list[dict[str, Any]]] = {}
    for d in detections:
        dets_by_frame.setdefault(int(d["frame_index"]), []).append(dict(d))
    for fi in dets_by_frame:
        dets_by_frame[fi].sort(key=lambda r: int(r["detection_id"]))

    attr_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    if detection_attributes:
        for a in detection_attributes:
            attr_by_key[(int(a["frame_index"]), int(a["detection_id"]))] = dict(a)

    allocator = TrackIdAllocator(run_id=run_id, video_id=video_id, start=0)
    active: dict[int, _ActiveTrack] = {}
    observations: list[dict[str, Any]] = []
    lifecycle: list[dict[str, Any]] = []
    primary_sidecar: list[dict[str, Any]] = []
    findings: list[str] = []
    review_samples: list[dict[str, Any]] = []
    terminated_tracks: list[_ActiveTrack] = []

    assoc_cfg = config["association"]
    life_cfg = config["lifecycle"]
    bound_cfg = config["boundaries"]
    primary_cfg = config["primary_candidate"]
    pred_cfg = config["prediction"]
    review_cfg = config["review"]
    geom = config["frame_geometry"]
    min_conf = float(assoc_cfg["min_confidence"])
    confirm_n = int(life_cfg["confirmation_observation_threshold"])
    max_lost = int(life_cfg["max_lost_gap_us"])
    max_pred = int(life_cfg["max_prediction_gap_us"])
    tent_term = int(life_cfg["tentative_miss_terminate_us"])
    weak_term = int(life_cfg["weak_candidate_terminate_us"])
    emit_pred = bool(life_cfg["emit_predicted_observations"]) and bool(
        config["output_policy"]["emit_predicted"]
    )
    amb_margin = float(primary_cfg["ambiguity_margin"])
    prefer_confirmed = bool(primary_cfg["prefer_confirmed"])
    max_review = int(review_cfg["max_review_samples_per_run"])

    prev_shot: str | None = None
    prev_window: str | None = None
    ball_input = 0
    rejected_non_ball = 0
    recoveries = 0
    fragmentations = 0
    max_gap_seen = 0
    routing_gaps = 0
    invalid_jumps = 0
    invalid_size = 0
    primary_frames = 0
    ambiguous_frames = 0
    no_candidate_frames = 0
    source_dist: dict[str, int] = {"full_frame": 0, "tile": 0, "hybrid": 0, "unknown": 0}

    def _maybe_review(kind: str, payload: dict[str, Any]) -> None:
        if len(review_samples) >= max_review:
            return
        review_samples.append({"kind": kind, **payload})

    def terminate_track(
        track: _ActiveTrack,
        *,
        frame_index: int,
        video_time_us: int,
        reason: str,
    ) -> None:
        nonlocal fragmentations
        if track.lifecycle == LifecycleState.TERMINATED:
            return
        prev = track.lifecycle
        if prev == LifecycleState.LOST:
            fragmentations += 1
        lifecycle.append(
            _life_event(
                track,
                run_id=run_id,
                video_id=video_id,
                frame_index=frame_index,
                video_time_us=video_time_us,
                state=LifecycleState.TERMINATED,
                previous=prev,
                reason=reason,
                observation_source=None,
                policy=policy,
            )
        )
        track.termination_reason = reason
        terminated_tracks.append(track)
        active.pop(track.track_id, None)

    def terminate_all(*, frame_index: int, video_time_us: int, reason: str) -> None:
        for tid in list(active.keys()):
            terminate_track(
                active[tid],
                frame_index=frame_index,
                video_time_us=video_time_us,
                reason=reason,
            )

    for fr in frames_sorted:
        fi = int(fr["frame_index"])
        t_us = int(fr["video_time_us"])
        window = _window_for_frame(fi, analysis_windows)
        shot_id = None if window is None else window.get("shot_id")
        window_id = None if window is None else window.get("analysis_window_id")
        playability = None if window is None else str(window.get("playability", "unknown"))
        tracking_eligibility = (
            None if window is None else str(window.get("tracking_eligibility", "unknown"))
        )
        replay_status = None if window is None else str(window.get("replay_status", "")).lower()

        # Shot cut → terminate (no cross-cut prediction).
        if (
            bound_cfg["terminate_on_shot_cut"]
            and prev_shot is not None
            and shot_id is not None
            and str(shot_id) != str(prev_shot)
        ):
            terminate_all(frame_index=fi, video_time_us=t_us, reason="shot_cut")
            routing_gaps += 1
            findings.append(f"SHOT_CUT at frame={fi} prev={prev_shot} next={shot_id}")

        if (
            bound_cfg["terminate_on_window_boundary"]
            and prev_window is not None
            and window_id is not None
            and str(window_id) != str(prev_window)
            and (shot_id is None or prev_shot is None or str(shot_id) == str(prev_shot))
        ):
            terminate_all(
                frame_index=fi,
                video_time_us=t_us,
                reason=GapReason.ANALYSIS_WINDOW_BOUNDARY.value,
            )
            routing_gaps += 1

        ineligible = False
        reason = GapReason.ROUTING_INELIGIBLE.value
        if window is not None:
            if bound_cfg["terminate_on_non_playable"] and playability == "non_playable":
                ineligible = True
                reason = GapReason.GRAPHICS.value
                if replay_status == "replay":
                    reason = GapReason.REPLAY.value
            if replay_status == "replay":
                ineligible = True
                reason = GapReason.REPLAY.value
            if (
                bound_cfg["terminate_on_ineligible_tracking"]
                and tracking_eligibility == "ineligible"
            ):
                ineligible = True
                reason = GapReason.ROUTING_INELIGIBLE.value
        if ineligible:
            if active:
                terminate_all(frame_index=fi, video_time_us=t_us, reason=reason)
                routing_gaps += 1
            # No prediction across cut / non-playable.
            primary_sidecar.append(
                {
                    "frame_index": fi,
                    "video_time_us": t_us,
                    "status": "no_candidate",
                    "primary_track_id": None,
                    "primary_detection_id": None,
                    "scores": [],
                    "reason": reason,
                }
            )
            no_candidate_frames += 1
            prev_shot = None if shot_id is None else str(shot_id)
            prev_window = None if window_id is None else str(window_id)
            continue

        frame_dets_raw = dets_by_frame.get(fi, [])
        frame_balls: list[dict[str, Any]] = []
        for det in frame_dets_raw:
            attr = attr_by_key.get((fi, int(det["detection_id"])))
            if not _is_ball_detection(det, attr, config=config):
                rejected_non_ball += 1
                continue
            conf = det.get("confidence")
            if conf is not None and float(conf) < min_conf:
                rejected_non_ball += 1
                continue
            frame_balls.append(det)
        ball_input += len(frame_balls)

        predicted: dict[int, BBox] = {}
        dt_by_track: dict[int, int] = {}
        track_list: list[dict[str, Any]] = []
        for tid, tr in sorted(active.items(), key=lambda kv: kv[0]):
            dt = gap_us(tr.last_time_us, t_us) if t_us >= tr.last_time_us else 0
            dt_by_track[tid] = dt
            pred = predict_bbox_constant_velocity(tr.bbox, vx=tr.vx, vy=tr.vy, dt_us=dt)
            predicted[tid] = pred
            track_list.append({"track_id": tid})

        matches, unmatched_tracks, unmatched_dets = greedy_ball_associate(
            track_list,
            frame_balls,
            predicted_bboxes=predicted,
            dt_us_by_track=dt_by_track,
            motion_center_gate_px=float(assoc_cfg["motion_center_gate_px"]),
            motion_gate_scale_per_us=float(assoc_cfg["motion_gate_scale_per_us"]),
            size_ratio_gate=float(assoc_cfg["size_ratio_gate"]),
            iou_support_min=float(assoc_cfg["iou_support_min"]),
            require_motion_gate=bool(assoc_cfg["require_motion_gate"]),
            motion_weight=float(assoc_cfg["cost_motion_weight"]),
            size_weight=float(assoc_cfg["cost_size_weight"]),
            confidence_weight=float(assoc_cfg["cost_confidence_weight"]),
            iou_weight=float(assoc_cfg["cost_iou_weight"]),
        )
        unmatched_det_set = set(unmatched_dets)
        match_by_tid = {m.track_id: m for m in matches}
        frame_obs_tracks: list[tuple[int, int | None, float, float | None, float]] = []

        for tid, m in sorted(match_by_tid.items()):
            tr = active[tid]
            det = next(d for d in frame_balls if int(d["detection_id"]) == m.detection_id)
            attr = attr_by_key.get((fi, m.detection_id))
            new_bbox: BBox = (
                float(det["bbox_x1"]),
                float(det["bbox_y1"]),
                float(det["bbox_x2"]),
                float(det["bbox_y2"]),
            )
            jump = center_l2(tr.bbox, new_bbox)
            if jump > float(geom["invalid_jump_px"]):
                invalid_jumps += 1
                if review_cfg["sample_invalid_jump"]:
                    _maybe_review(
                        "invalid_jump",
                        {"frame_index": fi, "track_id": tid, "jump_px": jump},
                    )
                findings.append(f"INVALID_JUMP track_id={tid} frame={fi} jump_px={jump:.1f}")
                unmatched_det_set.add(m.detection_id)
                continue
            ratio = _size_ratio(tr.bbox, new_bbox)
            if ratio > float(assoc_cfg["size_ratio_gate"]):
                invalid_size += 1
                unmatched_det_set.add(m.detection_id)
                continue

            if tr.last_observed_time_us > 0 and t_us > tr.last_observed_time_us:
                tr.vx, tr.vy = velocity_from_centers(
                    bbox_center(tr.bbox),
                    tr.last_observed_time_us,
                    bbox_center(new_bbox),
                    t_us,
                )
            if tr.lifecycle == LifecycleState.LOST:
                miss_gap = gap_us(tr.last_observed_time_us, t_us)
                max_gap_seen = max(max_gap_seen, miss_gap)
                if miss_gap > max_lost:
                    terminate_track(
                        tr,
                        frame_index=fi,
                        video_time_us=t_us,
                        reason=GapReason.DETECTION_LOSS.value,
                    )
                    unmatched_det_set.add(m.detection_id)
                    continue
                lifecycle.append(
                    _life_event(
                        tr,
                        run_id=run_id,
                        video_id=video_id,
                        frame_index=fi,
                        video_time_us=t_us,
                        state=LifecycleState.CONFIRMED,
                        previous=LifecycleState.LOST,
                        reason="recover",
                        observation_source=ObservationSource.DETECTION_ASSOCIATED.value,
                        policy=policy,
                    )
                )
                recoveries += 1

            src = _extract_candidate_source(det, attr)
            source_dist[src] = source_dist.get(src, 0) + 1
            tr.candidate_source = src
            tr.bbox = new_bbox
            tr.last_frame_index = fi
            tr.last_time_us = t_us
            tr.last_observed_time_us = t_us
            tr.association_count += 1
            tr.class_id = int(det.get("class_id", tr.class_id))
            tr.shot_id = None if shot_id is None else str(shot_id)
            tr.window_id = None if window_id is None else str(window_id)
            tr.last_confidence = None if det.get("confidence") is None else float(det["confidence"])
            tr.last_association_cost = float(m.cost)
            tr.prediction_uncertainty = 0.0
            # Role always unknown — never convert from attributes.
            if "role_unknown" not in tr.quality_flags:
                tr.quality_flags.append("role_unknown")
            if src != "unknown" and f"src:{src}" not in tr.quality_flags:
                tr.quality_flags.append(f"src:{src}")

            if tr.lifecycle == LifecycleState.TENTATIVE and tr.association_count >= confirm_n:
                lifecycle.append(
                    _life_event(
                        tr,
                        run_id=run_id,
                        video_id=video_id,
                        frame_index=fi,
                        video_time_us=t_us,
                        state=LifecycleState.CONFIRMED,
                        previous=LifecycleState.TENTATIVE,
                        reason="confirmation_threshold",
                        observation_source=ObservationSource.DETECTION_ASSOCIATED.value,
                        policy=policy,
                    )
                )

            qflags = list(tr.quality_flags)
            if (
                _at_frame_edge(
                    new_bbox,
                    width=int(geom["frame_width"]),
                    height=int(geom["frame_height"]),
                    margin=float(geom["edge_margin_px"]),
                )
                and "frame_edge" not in qflags
            ):
                qflags.append("frame_edge")
            observations.append(
                _obs_row(
                    run_id=run_id,
                    video_id=video_id,
                    frame_index=fi,
                    track=tr,
                    detection_id=m.detection_id,
                    bbox=new_bbox,
                    observation_state="observed",
                    confidence=tr.last_confidence,
                    quality_flags=qflags,
                )
            )
            frame_obs_tracks.append((tid, m.detection_id, float(m.cost), tr.last_confidence, ratio))

        # Birth new tracks for unmatched detections.
        for did in sorted(unmatched_det_set):
            det = next(d for d in frame_balls if int(d["detection_id"]) == did)
            if len(active) + len(terminated_tracks) >= int(
                config["safety_limits"]["max_tracks_per_video"]
            ):
                findings.append("MAX_TRACKS_REACHED")
                continue
            tid = allocator.allocate()
            bbox: BBox = (
                float(det["bbox_x1"]),
                float(det["bbox_y1"]),
                float(det["bbox_x2"]),
                float(det["bbox_y2"]),
            )
            attr = attr_by_key.get((fi, did))
            src = _extract_candidate_source(det, attr)
            source_dist[src] = source_dist.get(src, 0) + 1
            tr = _ActiveTrack(
                track_id=tid,
                lifecycle=LifecycleState.TENTATIVE,
                bbox=bbox,
                last_frame_index=fi,
                last_time_us=t_us,
                last_observed_time_us=t_us,
                association_count=1,
                class_id=int(det.get("class_id", 32)),
                model_id=tracker_model_id,
                shot_id=None if shot_id is None else str(shot_id),
                window_id=None if window_id is None else str(window_id),
                birth_frame=fi,
                candidate_source=src,
                last_confidence=(
                    None if det.get("confidence") is None else float(det["confidence"])
                ),
                last_association_cost=0.0,
                quality_flags=(
                    ["role_unknown", f"src:{src}"] if src != "unknown" else ["role_unknown"]
                ),
            )
            lifecycle.append(
                _life_event(
                    tr,
                    run_id=run_id,
                    video_id=video_id,
                    frame_index=fi,
                    video_time_us=t_us,
                    state=LifecycleState.TENTATIVE,
                    previous=None,
                    reason="birth",
                    observation_source=ObservationSource.DETECTION_ASSOCIATED.value,
                    policy=policy,
                )
            )
            if tr.association_count >= confirm_n:
                lifecycle.append(
                    _life_event(
                        tr,
                        run_id=run_id,
                        video_id=video_id,
                        frame_index=fi,
                        video_time_us=t_us,
                        state=LifecycleState.CONFIRMED,
                        previous=LifecycleState.TENTATIVE,
                        reason="confirmation_threshold",
                        observation_source=ObservationSource.DETECTION_ASSOCIATED.value,
                        policy=policy,
                    )
                )
            active[tid] = tr
            observations.append(
                _obs_row(
                    run_id=run_id,
                    video_id=video_id,
                    frame_index=fi,
                    track=tr,
                    detection_id=did,
                    bbox=bbox,
                    observation_state="observed",
                    confidence=tr.last_confidence,
                    quality_flags=list(tr.quality_flags),
                )
            )
            frame_obs_tracks.append((tid, did, 0.0, tr.last_confidence, 1.0))

        # Primary / ambiguity selection among frame observations.
        score_rows: list[tuple[float, int, int | None]] = []
        for obs_tid, obs_did, cost, conf, ratio in frame_obs_tracks:
            if obs_tid not in active:
                continue
            tr = active[obs_tid]
            score = primary_ball_score(
                association_cost_value=cost,
                confidence=conf,
                lifecycle_confirmed=tr.lifecycle == LifecycleState.CONFIRMED,
                size_ratio=ratio,
                prefer_confirmed=prefer_confirmed,
            )
            score_rows.append((score, obs_tid, obs_did))
        score_rows.sort(key=lambda r: (-r[0], r[1], -1 if r[2] is None else r[2]))

        if not score_rows:
            status = "no_candidate"
            primary_tid = None
            primary_did = None
            no_candidate_frames += 1
            # Avoid review spam on empty frames.
            if not review_cfg["no_spam_empty_frames"]:
                _maybe_review("no_candidate", {"frame_index": fi})
        elif len(score_rows) == 1 or (score_rows[0][0] - score_rows[1][0]) >= amb_margin:
            status = "primary"
            primary_tid = score_rows[0][1]
            primary_did = score_rows[0][2]
            primary_frames += 1
        else:
            status = "ambiguous"
            primary_tid = None
            primary_did = None
            ambiguous_frames += 1
            if review_cfg["sample_ambiguous"]:
                _maybe_review(
                    "ambiguous",
                    {
                        "frame_index": fi,
                        "top_scores": [
                            {"track_id": s[1], "score": s[0], "detection_id": s[2]}
                            for s in score_rows[:3]
                        ],
                    },
                )
                findings.append(f"AMBIGUOUS_PRIMARY frame={fi}")

        primary_sidecar.append(
            {
                "frame_index": fi,
                "video_time_us": t_us,
                "status": status,
                "primary_track_id": primary_tid,
                "primary_detection_id": primary_did,
                "scores": [
                    {"track_id": s[1], "score": s[0], "detection_id": s[2]} for s in score_rows
                ],
                "reason": None,
            }
        )

        # Handle unmatched tracks (misses) — prediction only inside same shot.
        for tid in unmatched_tracks:
            if tid not in active:
                continue
            tr = active[tid]
            miss_gap = gap_us(tr.last_observed_time_us, t_us)
            max_gap_seen = max(max_gap_seen, miss_gap)

            if (
                emit_pred
                and miss_gap <= max_pred
                and tr.lifecycle in {LifecycleState.CONFIRMED, LifecycleState.LOST}
            ):
                pred_bbox = predicted.get(tid, tr.bbox)
                if pred_cfg["uncertainty_grows_with_gap"]:
                    tr.prediction_uncertainty = min(1.0, float(miss_gap) / float(max(max_pred, 1)))
                qflags = list(
                    dict.fromkeys(
                        [
                            *tr.quality_flags,
                            "physical_metric_ineligible",
                            "event_ineligible",
                            "predicted",
                            f"prediction_uncertainty:{tr.prediction_uncertainty:.3f}",
                        ]
                    )
                )
                observations.append(
                    _obs_row(
                        run_id=run_id,
                        video_id=video_id,
                        frame_index=fi,
                        track=tr,
                        detection_id=None,
                        bbox=pred_bbox,
                        observation_state="predicted",
                        confidence=None,
                        quality_flags=qflags,
                    )
                )
                tr.bbox = pred_bbox
                tr.last_frame_index = fi
                tr.last_time_us = t_us

            if tr.lifecycle == LifecycleState.TENTATIVE:
                if miss_gap > tent_term or miss_gap > weak_term:
                    terminate_track(
                        tr,
                        frame_index=fi,
                        video_time_us=t_us,
                        reason=GapReason.DETECTION_LOSS.value,
                    )
                continue

            if tr.lifecycle == LifecycleState.CONFIRMED:
                lifecycle.append(
                    _life_event(
                        tr,
                        run_id=run_id,
                        video_id=video_id,
                        frame_index=fi,
                        video_time_us=t_us,
                        state=LifecycleState.LOST,
                        previous=LifecycleState.CONFIRMED,
                        reason="miss",
                        observation_source=ObservationSource.NOT_OBSERVED.value,
                        policy=policy,
                    )
                )
                continue

            if tr.lifecycle == LifecycleState.LOST and miss_gap > max_lost:
                terminate_track(
                    tr,
                    frame_index=fi,
                    video_time_us=t_us,
                    reason=GapReason.DETECTION_LOSS.value,
                )
                findings.append(f"LONG_OCCLUSION_TERMINATE track_id={tid} gap_us={miss_gap}")

        prev_shot = None if shot_id is None else str(shot_id)
        prev_window = None if window_id is None else str(window_id)

    if frames_sorted:
        last = frames_sorted[-1]
        terminate_all(
            frame_index=int(last["frame_index"]),
            video_time_us=int(last["video_time_us"]),
            reason="end_of_clip",
        )

    all_tracks = {t.track_id: t for t in terminated_tracks}
    obs_by_tid: dict[int, list[dict[str, Any]]] = {}
    for o in observations:
        obs_by_tid.setdefault(int(o["track_id"]), []).append(o)

    summaries: list[dict[str, Any]] = []
    for tid in sorted(obs_by_tid):
        obs = sorted(obs_by_tid[tid], key=lambda r: int(r["frame_index"]))
        track_obj: _ActiveTrack | None = all_tracks.get(tid)
        confs = [float(o["confidence"]) for o in obs if o.get("confidence") is not None]
        observed = sum(1 for o in obs if o["observation_state"] == "observed")
        predicted_n = sum(1 for o in obs if o["observation_state"] == "predicted")
        qflags = list(track_obj.quality_flags) if track_obj is not None else []
        term_reason = "end_of_clip"
        if track_obj is not None and track_obj.termination_reason:
            term_reason = track_obj.termination_reason
        summaries.append(
            {
                "run_id": run_id,
                "video_id": video_id,
                "track_id": tid,
                "class_id": int(obs[0]["class_id"]),
                "first_frame_index": int(obs[0]["frame_index"]),
                "last_frame_index": int(obs[-1]["frame_index"]),
                "observation_count": len(obs),
                "observed_count": observed,
                "predicted_count": predicted_n,
                "mean_confidence": (sum(confs) / len(confs)) if confs else None,
                "max_confidence": max(confs) if confs else None,
                "termination_reason": term_reason,
                "quality_flags": qflags,
            }
        )

    final_state: dict[int, str] = {}
    for ev in sorted(lifecycle, key=lambda e: (int(e["track_id"]), int(e["event_index"]))):
        final_state[int(ev["track_id"])] = str(ev["lifecycle_state"])
    track_counts = {"tentative": 0, "confirmed": 0, "lost": 0, "terminated": 0}
    for st in final_state.values():
        track_counts[st] = track_counts.get(st, 0) + 1

    observed_n = sum(1 for o in observations if o["observation_state"] == "observed")
    predicted_n = sum(1 for o in observations if o["observation_state"] == "predicted")
    used = observed_n
    unassigned_final = max(0, ball_input - used)

    stats = {
        "ball_input_detections": ball_input,
        "rejected_non_ball": rejected_non_ball,
        "detections_used": used,
        "unassigned_detection_count": unassigned_final,
        "track_counts": track_counts,
        "observation_counts": {
            "detection_associated": observed_n,
            "predicted": predicted_n,
            "interpolated": 0,
            "observed": observed_n,
            "total": len(observations),
        },
        "recoveries": recoveries,
        "fragmentations": fragmentations,
        "max_gap_us": max_gap_seen,
        "routing_gap_count": routing_gaps,
        "invalid_jump_count": invalid_jumps,
        "invalid_size_count": invalid_size,
        "primary_frames": primary_frames,
        "ambiguous_frames": ambiguous_frames,
        "no_candidate_frames": no_candidate_frames,
        "candidate_source_distribution": dict(sorted(source_dist.items())),
        "review_required_count": sum(1 for t in terminated_tracks if t.review_required),
        "review_sample_count": len(review_samples),
        "review_samples": review_samples[:max_review],
        "findings": list(findings),
    }
    return TrackerResult(
        observations=observations,
        lifecycle=lifecycle,
        summaries=summaries,
        primary_sidecar=primary_sidecar,
        findings=findings,
        stats=stats,
    )


__all__ = ["TrackerResult", "run_ball_tracker"]
