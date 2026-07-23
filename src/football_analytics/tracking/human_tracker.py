"""Human multi-object tracker frame loop (Stage 6B).

IoU + constant-velocity association with Stage 6A lifecycle rules.
No ReID; terminated tracks never reopen; no cross-shot continuation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from football_analytics.tracking.human_association import greedy_associate
from football_analytics.tracking.human_motion import (
    BBox,
    bbox_center,
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
    class_id: int = 0
    model_id: str = "human_iou_cv_v1"
    roles: set[str] = field(default_factory=set)
    quality_flags: list[str] = field(default_factory=list)
    event_index: int = 0
    shot_id: str | None = None
    window_id: str | None = None
    birth_frame: int = 0
    termination_reason: str | None = None
    review_required: bool = False


@dataclass
class TrackerResult:
    observations: list[dict[str, Any]]
    lifecycle: list[dict[str, Any]]
    summaries: list[dict[str, Any]]
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


def _is_human_detection(
    det: Mapping[str, Any],
    attr: Mapping[str, Any] | None,
    *,
    config: Mapping[str, Any],
) -> bool:
    entity = config["entity_filter"]
    reject_names = {str(x).lower() for x in entity["reject_class_names"]}
    human_names = {str(x).lower() for x in entity["human_class_names"]}
    class_name = str(det.get("class_name", "")).lower()
    if class_name in reject_names:
        return False
    if attr is not None:
        et = str(attr.get("entity_type", "")).lower()
        if et in {str(x).lower() for x in entity["reject_entity_types"]}:
            return False
        if et == "human":
            return True
        if et and et != "human":
            return False
    return class_name in human_names


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
        "entity_type": "human",
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


def _apply_role(
    track: _ActiveTrack,
    attr: Mapping[str, Any] | None,
    *,
    config: Mapping[str, Any],
    findings: list[str],
) -> None:
    if attr is None or not config["role"]["soft_consistency"]:
        return
    role = str(attr.get("role_label", "unknown"))
    if role == "unknown":
        return  # unknown not punished / not converted
    track.roles.add(role)
    concrete = {r for r in track.roles if r != "unknown"}
    if len(concrete) > 1 and config["role"]["conflict_requires_review"]:
        track.review_required = True
        if "role_conflict" not in track.quality_flags:
            track.quality_flags.append("role_conflict")
        if "review_required" not in track.quality_flags:
            track.quality_flags.append("review_required")
        findings.append(f"ROLE_CONFLICT track_id={track.track_id} roles={sorted(concrete)}")


def run_human_tracker(
    *,
    run_id: str,
    video_id: str,
    frames: Sequence[Mapping[str, Any]],
    detections: Sequence[Mapping[str, Any]],
    analysis_windows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    detection_attributes: Sequence[Mapping[str, Any]] | None = None,
    policy: Mapping[str, Any] | None = None,
    tracker_model_id: str = "human_iou_cv_v1",
) -> TrackerResult:
    """Run deterministic human MOT over a frame timeline (no video decode)."""
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
    findings: list[str] = []
    terminated_tracks: list[_ActiveTrack] = []

    assoc_cfg = config["association"]
    life_cfg = config["lifecycle"]
    bound_cfg = config["boundaries"]
    min_conf = float(assoc_cfg["min_confidence"])
    confirm_n = int(life_cfg["confirmation_observation_threshold"])
    max_lost = int(life_cfg["max_lost_gap_us"])
    max_pred = int(life_cfg["max_prediction_gap_us"])
    tent_term = int(life_cfg["tentative_miss_terminate_us"])
    emit_pred = bool(life_cfg["emit_predicted_observations"]) and bool(
        config["output_policy"]["emit_predicted"]
    )

    prev_shot: str | None = None
    prev_window: str | None = None
    human_input = 0
    rejected_non_human = 0
    recoveries = 0
    fragmentations = 0
    max_gap_seen = 0
    routing_gaps = 0

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

        # Shot cut / window boundary → terminate (no cross-shot continuation).
        if (
            bound_cfg["terminate_on_shot_cut"]
            and prev_shot is not None
            and shot_id is not None
            and str(shot_id) != str(prev_shot)
        ):
            terminate_all(
                frame_index=fi,
                video_time_us=t_us,
                reason="shot_cut",
            )
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
        if window is not None:
            if bound_cfg["terminate_on_non_playable"] and playability == "non_playable":
                ineligible = True
                reason = GapReason.GRAPHICS.value
                if str(window.get("replay_status", "")).lower() == "replay":
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
            prev_shot = None if shot_id is None else str(shot_id)
            prev_window = None if window_id is None else str(window_id)
            continue

        # Filter human detections for this frame.
        frame_dets_raw = dets_by_frame.get(fi, [])
        frame_humans: list[dict[str, Any]] = []
        for det in frame_dets_raw:
            attr = attr_by_key.get((fi, int(det["detection_id"])))
            if not _is_human_detection(det, attr, config=config):
                rejected_non_human += 1
                continue
            conf = det.get("confidence")
            if conf is not None and float(conf) < min_conf:
                rejected_non_human += 1
                continue
            frame_humans.append(det)
        human_input += len(frame_humans)

        # Predict active tracks to current time.
        predicted: dict[int, BBox] = {}
        track_list: list[dict[str, Any]] = []
        for tid, tr in sorted(active.items(), key=lambda kv: kv[0]):
            dt = gap_us(tr.last_time_us, t_us) if t_us >= tr.last_time_us else 0
            pred = predict_bbox_constant_velocity(tr.bbox, vx=tr.vx, vy=tr.vy, dt_us=dt)
            predicted[tid] = pred
            track_list.append({"track_id": tid})

        matches, unmatched_tracks, unmatched_dets = greedy_associate(
            track_list,
            frame_humans,
            predicted_bboxes=predicted,
            iou_gate=float(assoc_cfg["iou_gate"]),
            motion_center_gate_px=float(assoc_cfg["motion_center_gate_px"]),
            iou_weight=float(assoc_cfg["cost_iou_weight"]),
            motion_weight=float(assoc_cfg["cost_motion_weight"]),
        )
        unmatched_det_set = set(unmatched_dets)
        match_by_tid = {m.track_id: m for m in matches}

        # Update matched tracks.
        for tid, m in sorted(match_by_tid.items()):
            tr = active[tid]
            det = next(d for d in frame_humans if int(d["detection_id"]) == m.detection_id)
            attr = attr_by_key.get((fi, m.detection_id))
            new_bbox: BBox = (
                float(det["bbox_x1"]),
                float(det["bbox_y1"]),
                float(det["bbox_x2"]),
                float(det["bbox_y2"]),
            )
            if tr.last_observed_time_us > 0 and t_us > tr.last_observed_time_us:
                tr.vx, tr.vy = velocity_from_centers(
                    bbox_center(tr.bbox),
                    tr.last_observed_time_us,
                    bbox_center(new_bbox),
                    t_us,
                )
            # Recover from lost.
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

            tr.bbox = new_bbox
            tr.last_frame_index = fi
            tr.last_time_us = t_us
            tr.last_observed_time_us = t_us
            tr.association_count += 1
            tr.class_id = int(det.get("class_id", tr.class_id))
            tr.shot_id = None if shot_id is None else str(shot_id)
            tr.window_id = None if window_id is None else str(window_id)
            _apply_role(tr, attr, config=config, findings=findings)

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
            observations.append(
                _obs_row(
                    run_id=run_id,
                    video_id=video_id,
                    frame_index=fi,
                    track=tr,
                    detection_id=m.detection_id,
                    bbox=new_bbox,
                    observation_state="observed",
                    confidence=None if det.get("confidence") is None else float(det["confidence"]),
                    quality_flags=qflags,
                )
            )

        # Birth new tracks for unmatched detections.
        for did in sorted(unmatched_det_set):
            det = next(d for d in frame_humans if int(d["detection_id"]) == did)
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
            tr = _ActiveTrack(
                track_id=tid,
                lifecycle=LifecycleState.TENTATIVE,
                bbox=bbox,
                last_frame_index=fi,
                last_time_us=t_us,
                last_observed_time_us=t_us,
                association_count=1,
                class_id=int(det.get("class_id", 0)),
                model_id=tracker_model_id,
                shot_id=None if shot_id is None else str(shot_id),
                window_id=None if window_id is None else str(window_id),
                birth_frame=fi,
            )
            attr = attr_by_key.get((fi, did))
            _apply_role(tr, attr, config=config, findings=findings)
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
                    confidence=None if det.get("confidence") is None else float(det["confidence"]),
                    quality_flags=list(tr.quality_flags),
                )
            )

        # Handle unmatched tracks (misses).
        for tid in unmatched_tracks:
            if tid not in active:
                continue
            tr = active[tid]
            miss_gap = gap_us(tr.last_observed_time_us, t_us)
            max_gap_seen = max(max_gap_seen, miss_gap)

            # Optional short-gap predicted observation (relative to last observed).
            obs_gap = gap_us(tr.last_observed_time_us, t_us)
            if (
                emit_pred
                and obs_gap <= max_pred
                and tr.lifecycle in {LifecycleState.CONFIRMED, LifecycleState.LOST}
            ):
                pred_bbox = predicted.get(tid, tr.bbox)
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
                        quality_flags=list(
                            dict.fromkeys(
                                [*tr.quality_flags, "physical_metric_ineligible", "predicted"]
                            )
                        ),
                    )
                )
                tr.bbox = pred_bbox
                tr.last_frame_index = fi
                tr.last_time_us = t_us

            if tr.lifecycle == LifecycleState.TENTATIVE:
                if miss_gap > tent_term:
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
                # Long occlusion: terminate; future detections birth NEW tracks (no ReID).
                terminate_track(
                    tr,
                    frame_index=fi,
                    video_time_us=t_us,
                    reason=GapReason.DETECTION_LOSS.value,
                )
                findings.append(f"LONG_OCCLUSION_TERMINATE track_id={tid} gap_us={miss_gap}")

        prev_shot = None if shot_id is None else str(shot_id)
        prev_window = None if window_id is None else str(window_id)

    # End of clip: terminate remaining.
    if frames_sorted:
        last = frames_sorted[-1]
        terminate_all(
            frame_index=int(last["frame_index"]),
            video_time_us=int(last["video_time_us"]),
            reason="end_of_clip",
        )

    # Build summaries from observations + terminated tracks.
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

    # Final track counts by last lifecycle state.
    final_state: dict[int, str] = {}
    for ev in sorted(lifecycle, key=lambda e: (int(e["track_id"]), int(e["event_index"]))):
        final_state[int(ev["track_id"])] = str(ev["lifecycle_state"])
    track_counts = {
        "tentative": 0,
        "confirmed": 0,
        "lost": 0,
        "terminated": 0,
    }
    for st in final_state.values():
        track_counts[st] = track_counts.get(st, 0) + 1

    observed_n = sum(1 for o in observations if o["observation_state"] == "observed")
    predicted_n = sum(1 for o in observations if o["observation_state"] == "predicted")
    # Unassigned human dets = human_input - observed associations
    used = observed_n
    unassigned_final = max(0, human_input - used)

    stats = {
        "human_input_detections": human_input,
        "rejected_non_human": rejected_non_human,
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
        "review_required_count": sum(1 for t in terminated_tracks if t.review_required),
        "findings": list(findings),
    }
    return TrackerResult(
        observations=observations,
        lifecycle=lifecycle,
        summaries=summaries,
        findings=findings,
        stats=stats,
    )


__all__ = ["TrackerResult", "run_human_tracker"]
