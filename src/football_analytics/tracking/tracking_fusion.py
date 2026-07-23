"""Stage 6D tracking fusion: merge human + ball tracks into one validated bundle.

Namespace policy (deterministic):
- Human track_ids kept as-is.
- Ball track_ids remapped to start after max(human track_id)+1 when enabled.
- Compound uniqueness: (run_id, video_id, track_id) unique across entity types.
- No human-ball relationship / possession table.
- Ambiguous primary ball is never upgraded to primary.
- Terminated tracks must not reopen; no cross-cut continuation.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from football_analytics.tracking.types import LifecycleState, TrackEntityType
from football_analytics.tracking.validation import validate_track_bundle

PHYSICAL_INELIGIBLE = "physical_metric_ineligible"
EVENT_INELIGIBLE = "event_ineligible"


class TrackingFusionError(ValueError):
    """Fusion alignment or merge failure with explicit error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class TrackingFusionResult:
    observations: list[dict[str, Any]]
    summaries: list[dict[str, Any]]
    lifecycle: list[dict[str, Any]]
    primary_sidecar: list[dict[str, Any]]
    track_id_remap: dict[int, int]
    run_id: str
    video_id: str
    human_track_count: int
    ball_track_count: int
    counts: dict[str, Any] = field(default_factory=dict)


def _rows(table_or_rows: Any) -> list[dict[str, Any]]:
    if table_or_rows is None:
        return []
    if hasattr(table_or_rows, "to_pylist"):
        return list(table_or_rows.to_pylist())
    return [dict(r) for r in table_or_rows]


def _receipt_get(receipt: Mapping[str, Any] | None, *keys: str) -> Any:
    if receipt is None:
        return None
    for k in keys:
        if k in receipt and receipt[k] is not None:
            return receipt[k]
    for nest_key in ("artifacts", "provenance", "input_hashes", "outputs"):
        nested = receipt.get(nest_key)
        if isinstance(nested, Mapping):
            for k in keys:
                if k in nested and nested[k] is not None:
                    return nested[k]
    return None


def _basename(path: str) -> str:
    return path.replace("\\", "/").rstrip("/").split("/")[-1]


def align_tracking_inputs(
    *,
    detection_receipt: Mapping[str, Any] | None,
    human_receipt: Mapping[str, Any] | None,
    ball_receipt: Mapping[str, Any] | None,
    config: Mapping[str, Any],
    expected_run_id: str | None = None,
    expected_video_id: str | None = None,
    expected_source_sha: str | None = None,
    expected_timeline_fp: str | None = None,
    expected_detection_fp: str | None = None,
    expected_analysis_window_fp: str | None = None,
) -> dict[str, Any]:
    """Align run/video/source/timeline/detection fingerprints or raise."""
    align = config["alignment"]
    if align["fail_on_missing_receipt"] and (
        detection_receipt is None or human_receipt is None or ball_receipt is None
    ):
        raise TrackingFusionError(
            "MISSING_RECEIPT", "detection/human/ball tracking receipts required"
        )

    receipts = [r for r in (detection_receipt, human_receipt, ball_receipt) if r is not None]

    run_ids = [str(r["run_id"]) for r in receipts if r.get("run_id") is not None]
    if align["require_run_id_match"] and run_ids:
        if len(set(run_ids)) != 1:
            raise TrackingFusionError("RUN_ID_MISMATCH", f"run_id mismatch: {run_ids}")
        if expected_run_id is not None and run_ids[0] != expected_run_id:
            raise TrackingFusionError(
                "RUN_ID_MISMATCH", f"run_id != expected ({run_ids[0]} vs {expected_run_id})"
            )

    video_ids: list[str] = []
    for r in receipts:
        vid = r.get("video_id") or _receipt_get(r, "video_id")
        if vid is not None:
            video_ids.append(str(vid))
    if align["require_video_id_match"] and video_ids:
        if len(set(video_ids)) != 1:
            raise TrackingFusionError("VIDEO_ID_MISMATCH", f"video_id mismatch: {video_ids}")
        if expected_video_id is not None and video_ids[0] != expected_video_id:
            raise TrackingFusionError(
                "VIDEO_ID_MISMATCH",
                f"video_id != expected ({video_ids[0]} vs {expected_video_id})",
            )

    shas: list[str] = []
    for r in receipts:
        sha = _receipt_get(r, "source_video_sha256", "source_sha256")
        if sha is not None:
            shas.append(str(sha).lower())
    if expected_source_sha is not None:
        shas.append(expected_source_sha.lower())
    if align["require_source_sha_match"] and len(shas) >= 2 and len(set(shas)) != 1:
        raise TrackingFusionError("SOURCE_SHA_MISMATCH", f"source sha mismatch: {shas}")

    timeline_fps: list[str] = []
    for r in receipts:
        tfp = _receipt_get(r, "timeline_fingerprint", "frames_fingerprint")
        if tfp is not None:
            timeline_fps.append(str(tfp).lower())
    if expected_timeline_fp is not None:
        timeline_fps.append(expected_timeline_fp.lower())
    if (
        align["require_timeline_fingerprint_match"]
        and len(timeline_fps) >= 2
        and len(set(timeline_fps)) != 1
    ):
        raise TrackingFusionError(
            "TIMELINE_FINGERPRINT_MISMATCH",
            f"timeline fingerprint mismatch: {timeline_fps}",
        )

    det_fps: list[str] = []
    for rec in (detection_receipt, human_receipt, ball_receipt):
        if rec is None:
            continue
        dfp = _receipt_get(
            rec,
            "detection_bundle_fingerprint",
            "detections_fingerprint",
            "input_detection_fingerprint",
        )
        if dfp is not None:
            det_fps.append(str(dfp).lower())
    if expected_detection_fp is not None:
        det_fps.append(expected_detection_fp.lower())
    if (
        align["require_detection_fingerprint_match"]
        and len(det_fps) >= 2
        and len(set(det_fps)) != 1
    ):
        raise TrackingFusionError(
            "DETECTION_FINGERPRINT_MISMATCH",
            f"detection fingerprint mismatch: {det_fps}",
        )

    aw_fps: list[str] = []
    for r in receipts:
        aw = _receipt_get(r, "analysis_window_fingerprint", "windows_fingerprint")
        if aw is not None:
            aw_fps.append(str(aw).lower())
    if expected_analysis_window_fp is not None:
        aw_fps.append(expected_analysis_window_fp.lower())
    if (
        align["require_analysis_window_fingerprint_match"]
        and len(aw_fps) >= 2
        and len(set(aw_fps)) != 1
    ):
        raise TrackingFusionError(
            "ANALYSIS_WINDOW_FINGERPRINT_MISMATCH",
            f"analysis window fingerprint mismatch: {aw_fps}",
        )

    return {
        "run_id": run_ids[0] if run_ids else expected_run_id,
        "video_id": video_ids[0] if video_ids else expected_video_id,
        "source_video_sha256": shas[0] if shas else expected_source_sha,
        "timeline_fingerprint": timeline_fps[0] if timeline_fps else expected_timeline_fp,
        "detection_bundle_fingerprint": det_fps[0] if det_fps else expected_detection_fp,
        "analysis_window_fingerprint": aw_fps[0] if aw_fps else expected_analysis_window_fp,
    }


def _validate_bbox_row(row: Mapping[str, Any]) -> None:
    for key in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"):
        v = row.get(key)
        if v is None or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise TrackingFusionError("INVALID_BBOX", f"non-finite {key} in observation")
    x1, y1, x2, y2 = (
        float(row["bbox_x1"]),
        float(row["bbox_y1"]),
        float(row["bbox_x2"]),
        float(row["bbox_y2"]),
    )
    if not (x1 < x2 and y1 < y2):
        raise TrackingFusionError("INVALID_BBOX", "zero/negative area bbox")
    conf = row.get("confidence")
    if conf is not None:
        if not isinstance(conf, (int, float)) or not math.isfinite(float(conf)):
            raise TrackingFusionError("INVALID_BBOX", "non-finite confidence")
        if not (0.0 <= float(conf) <= 1.0):
            raise TrackingFusionError("INVALID_BBOX", "confidence out of bounds")


def _ensure_predicted_flags(row: dict[str, Any], *, preserve: bool) -> dict[str, Any]:
    state = str(row["observation_state"])
    out = dict(row)
    if state not in {"predicted", "interpolated"}:
        return out
    flags = list(out.get("quality_flags") or [])
    if preserve:
        if PHYSICAL_INELIGIBLE not in flags:
            flags.append(PHYSICAL_INELIGIBLE)
        if EVENT_INELIGIBLE not in flags:
            flags.append(EVENT_INELIGIBLE)
        out["quality_flags"] = flags
        if out.get("detection_id") is not None:
            raise TrackingFusionError(
                "PREDICTED_HAS_DETECTION",
                f"{state} observation must have null detection_id",
            )
    return out


# ... in merge_track_tables, replace the clear/update block


def _remap_track_id(row: dict[str, Any], remap: Mapping[int, int], key: str = "track_id") -> None:
    old = int(row[key])
    if old in remap:
        row[key] = remap[old]


def compute_ball_track_remap(
    human_observations: Sequence[Mapping[str, Any]],
    ball_observations: Sequence[Mapping[str, Any]],
    *,
    enabled: bool,
) -> dict[int, int]:
    """Map ball track_ids into a namespace after max(human)+1."""
    if not enabled:
        return {}
    human_ids = {int(r["track_id"]) for r in human_observations}
    ball_ids = sorted({int(r["track_id"]) for r in ball_observations})
    if not ball_ids:
        return {}
    base = (max(human_ids) + 1) if human_ids else 0
    remap: dict[int, int] = {}
    next_id = base
    for bid in ball_ids:
        if bid in human_ids or bid in remap.values():
            while next_id in human_ids or next_id in remap.values():
                next_id += 1
            remap[bid] = next_id
            next_id += 1
        elif bid >= base and bid not in human_ids:
            # Already outside human namespace — keep if no collision among remapped.
            remap[bid] = bid
        else:
            while next_id in human_ids or next_id in remap.values():
                next_id += 1
            remap[bid] = next_id
            next_id += 1
    # Ensure all remapped values unique and disjoint from human.
    values = list(remap.values())
    if len(values) != len(set(values)):
        raise TrackingFusionError("TRACK_ID_COLLISION", "ball remap produced duplicates")
    if set(values) & human_ids:
        raise TrackingFusionError("TRACK_ID_COLLISION", "ball remap overlaps human namespace")
    return remap


def _window_for_frame(
    frames_index: int, windows: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    for w in windows:
        start = int(w["start_frame_index"])
        end = int(w["end_frame_index_exclusive"])
        if start <= frames_index < end:
            return w
    return None


def detect_cross_cut_violations(
    observations: Sequence[Mapping[str, Any]],
    lifecycle: Sequence[Mapping[str, Any]],
    analysis_windows: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Reject tracks that continue across shot/replay/non-playable boundaries."""
    _ = lifecycle
    by_track: dict[tuple[Any, Any, int], list[int]] = defaultdict(list)
    for o in observations:
        by_track[(o["run_id"], o["video_id"], int(o["track_id"]))].append(int(o["frame_index"]))
    violations: list[str] = []
    for tkey, frames in by_track.items():
        markers: list[tuple[Any, ...]] = []
        for fi in sorted(set(frames)):
            w = _window_for_frame(fi, analysis_windows)
            if w is None:
                continue
            markers.append(
                (
                    w.get("shot_id"),
                    w.get("replay_status"),
                    w.get("playability"),
                    w.get("tracking_eligibility"),
                )
            )
        if len(set(markers)) > 1:
            # Allow only if consecutive markers stay within eligible live playable same shot.
            unique = list(dict.fromkeys(markers))
            shots = {m[0] for m in unique}
            replays = {m[1] for m in unique}
            play = {m[2] for m in unique}
            if len(shots) > 1 or "replay" in replays or "non_playable" in play:
                violations.append(f"cross_cut_continuation:{tkey}")
    return violations


def detect_terminated_reopen(lifecycle: Sequence[Mapping[str, Any]]) -> int:
    by_track: dict[tuple[Any, Any, int], list[dict[str, Any]]] = defaultdict(list)
    for r in lifecycle:
        by_track[(r["run_id"], r["video_id"], int(r["track_id"]))].append(dict(r))
    count = 0
    for events in by_track.values():
        ordered = sorted(events, key=lambda e: int(e["event_index"]))
        seen_term = False
        for ev in ordered:
            if seen_term:
                count += 1
                break
            if str(ev["lifecycle_state"]) == LifecycleState.TERMINATED.value:
                seen_term = True
    return count


def recompute_summaries(
    observations: Sequence[Mapping[str, Any]],
    existing: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Rebuild summary counts from observations; preserve termination_reason when possible."""
    term_by_track: dict[tuple[Any, Any, int], str] = {}
    flags_by_track: dict[tuple[Any, Any, int], list[str]] = {}
    for s in existing or []:
        key = (s["run_id"], s["video_id"], int(s["track_id"]))
        term_by_track[key] = str(s.get("termination_reason") or "end_of_clip")
        flags_by_track[key] = list(s.get("quality_flags") or [])

    by_track: dict[tuple[Any, Any, int], list[dict[str, Any]]] = defaultdict(list)
    for o in observations:
        by_track[(o["run_id"], o["video_id"], int(o["track_id"]))].append(dict(o))

    out: list[dict[str, Any]] = []
    for key in sorted(by_track.keys(), key=lambda k: (str(k[0]), str(k[1]), int(k[2]))):
        obs = sorted(by_track[key], key=lambda r: int(r["frame_index"]))
        frames_idx = [int(o["frame_index"]) for o in obs]
        observed = sum(1 for o in obs if o["observation_state"] == "observed")
        predicted = sum(1 for o in obs if o["observation_state"] == "predicted")
        confs = [float(o["confidence"]) for o in obs if o.get("confidence") is not None]
        out.append(
            {
                "run_id": key[0],
                "video_id": key[1],
                "track_id": key[2],
                "class_id": int(obs[0]["class_id"]),
                "first_frame_index": min(frames_idx),
                "last_frame_index": max(frames_idx),
                "observation_count": len(obs),
                "observed_count": observed,
                "predicted_count": predicted,
                "mean_confidence": (sum(confs) / len(confs)) if confs else None,
                "max_confidence": max(confs) if confs else None,
                "termination_reason": term_by_track.get(key, "end_of_clip"),
                "quality_flags": flags_by_track.get(key, []),
            }
        )
    return out


def merge_track_tables(
    *,
    human_observations: Sequence[Mapping[str, Any]],
    human_summaries: Sequence[Mapping[str, Any]],
    human_lifecycle: Sequence[Mapping[str, Any]],
    ball_observations: Sequence[Mapping[str, Any]],
    ball_summaries: Sequence[Mapping[str, Any]],
    ball_lifecycle: Sequence[Mapping[str, Any]],
    primary_sidecar: Sequence[Mapping[str, Any]] | None,
    detection_attributes: Sequence[Mapping[str, Any]],
    analysis_windows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, int],
]:
    """Merge human+ball track tables with namespace remapping and integrity checks."""
    fusion = config["fusion"]
    if fusion["no_human_ball_relationship_table"] is not True:
        raise TrackingFusionError(
            "RELATIONSHIP_TABLE_FORBIDDEN", "no human-ball relationship table"
        )

    h_obs = [dict(r) for r in human_observations]
    b_obs = [dict(r) for r in ball_observations]
    h_life = [dict(r) for r in human_lifecycle]
    b_life = [dict(r) for r in ball_lifecycle]
    h_sum = [dict(r) for r in human_summaries]
    b_sum = [dict(r) for r in ball_summaries]
    primary = [dict(r) for r in (primary_sidecar or [])]

    attr_by_det = {
        (a["run_id"], a["video_id"], a["frame_index"], a["detection_id"]): a
        for a in detection_attributes
    }

    # Entity FK checks before remap.
    for side, rows, expected in (
        ("human", h_obs, TrackEntityType.HUMAN.value),
        ("ball", b_obs, TrackEntityType.BALL.value),
    ):
        for i, r in enumerate(rows):
            _validate_bbox_row(r)
            state = str(r["observation_state"])
            did = r.get("detection_id")
            if state == "observed":
                if did is None:
                    raise TrackingFusionError(
                        "DANGLING_FK", f"{side} observed missing detection_id"
                    )
                key = (r["run_id"], r["video_id"], r["frame_index"], did)
                attr = attr_by_det.get(key)
                if attr is None:
                    raise TrackingFusionError("DANGLING_FK", f"{side} detection FK missing {key}")
                if str(attr.get("entity_type")) != expected:
                    raise TrackingFusionError(
                        "ENTITY_FK_MISMATCH",
                        f"{side} observation bound to entity_type={attr.get('entity_type')}",
                    )
                if (
                    expected == TrackEntityType.HUMAN.value
                    and fusion["preserve_unknown_roles"]
                    and attr.get("role_label") is None
                ):
                    raise TrackingFusionError("ROLE_MISSING", f"human role missing for {key}")
            rows[i] = _ensure_predicted_flags(r, preserve=bool(fusion["preserve_predicted_flags"]))

    remap = compute_ball_track_remap(
        h_obs,
        b_obs,
        enabled=bool(fusion["remap_ball_track_ids"] and fusion["namespace_ball_tracks"]),
    )

    for rows in (b_obs, b_sum, b_life):
        for r in rows:
            _remap_track_id(r, remap)
    for frame in primary:
        tid = frame.get("primary_track_id")
        if tid is not None:
            old = int(tid)
            frame["primary_track_id"] = remap.get(old, old)
        status = str(frame.get("status") or frame.get("primary_status") or "")
        if fusion["do_not_upgrade_ambiguous_ball"] and status == "ambiguous":
            frame["primary_track_id"] = None
            frame["primary_detection_id"] = None

    for r in h_life:
        if str(r.get("entity_type")) != TrackEntityType.HUMAN.value:
            raise TrackingFusionError("ENTITY_MISMATCH", "human lifecycle entity_type invalid")
    for r in b_life:
        if str(r.get("entity_type")) != TrackEntityType.BALL.value:
            raise TrackingFusionError("ENTITY_MISMATCH", "ball lifecycle entity_type invalid")

    fused_obs = h_obs + b_obs
    # Duplicate detection assignment across fused bundle.
    det_to_tracks: dict[tuple[Any, ...], set[int]] = defaultdict(set)
    obs_pk: set[tuple[Any, ...]] = set()
    for r in fused_obs:
        pk = (r["run_id"], r["video_id"], r["frame_index"], r["track_id"])
        if pk in obs_pk:
            raise TrackingFusionError("DUPLICATE_FRAME_OBSERVATION", f"duplicate obs PK {pk}")
        obs_pk.add(pk)
        if str(r["observation_state"]) == "observed" and r.get("detection_id") is not None:
            dkey = (r["run_id"], r["video_id"], r["frame_index"], r["detection_id"])
            det_to_tracks[dkey].add(int(r["track_id"]))
    if fusion["reject_duplicate_detection_assignment"]:
        for dkey, tids in det_to_tracks.items():
            if len(tids) > 1:
                raise TrackingFusionError(
                    "DUPLICATE_DETECTION_ASSIGNMENT",
                    f"detection {dkey} assigned to tracks {sorted(tids)}",
                )

    fused_life = h_life + b_life
    if fusion["reject_terminated_reopen"]:
        n_reopen = detect_terminated_reopen(fused_life)
        if n_reopen:
            raise TrackingFusionError(
                "TERMINATED_REOPEN", f"{n_reopen} terminated track(s) reopened"
            )

    if fusion["reject_cross_cut_continuation"] and analysis_windows:
        viol = detect_cross_cut_violations(fused_obs, fused_life, analysis_windows)
        if viol:
            raise TrackingFusionError("CROSS_CUT_CONTINUATION", "; ".join(viol[:5]))

    fused_sum = recompute_summaries(fused_obs, existing=h_sum + b_sum)
    return fused_obs, fused_sum, fused_life, primary, remap


def _count_lifecycle_states(lifecycle: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {s.value: 0 for s in LifecycleState}
    for r in lifecycle:
        st = str(r["lifecycle_state"])
        if st in counts:
            counts[st] += 1
    # Per-track final state counts are more useful for receipt.
    by_track: dict[tuple[Any, Any, int], str] = {}
    for r in sorted(lifecycle, key=lambda x: int(x["event_index"])):
        by_track[(r["run_id"], r["video_id"], int(r["track_id"]))] = str(r["lifecycle_state"])
    final = {s.value: 0 for s in LifecycleState}
    for st in by_track.values():
        if st in final:
            final[st] += 1
    return final


def fuse_tracking_bundle(
    *,
    human_observations: Any,
    human_summaries: Any,
    human_lifecycle: Any,
    ball_observations: Any,
    ball_summaries: Any,
    ball_lifecycle: Any,
    primary_sidecar: Sequence[Mapping[str, Any]] | None,
    detections: Any,
    detection_attributes: Any,
    frames: Any | None,
    analysis_windows: Any | None,
    detection_receipt: Mapping[str, Any] | None,
    human_receipt: Mapping[str, Any] | None,
    ball_receipt: Mapping[str, Any] | None,
    config: Mapping[str, Any],
    expected_run_id: str | None = None,
    expected_video_id: str | None = None,
    expected_source_sha: str | None = None,
    expected_timeline_fp: str | None = None,
    expected_detection_fp: str | None = None,
    expected_analysis_window_fp: str | None = None,
    validate: bool = True,
) -> TrackingFusionResult:
    """Full fusion path: align → namespace merge → optional validate_track_bundle."""
    align_meta = align_tracking_inputs(
        detection_receipt=detection_receipt,
        human_receipt=human_receipt,
        ball_receipt=ball_receipt,
        config=config,
        expected_run_id=expected_run_id,
        expected_video_id=expected_video_id,
        expected_source_sha=expected_source_sha,
        expected_timeline_fp=expected_timeline_fp,
        expected_detection_fp=expected_detection_fp,
        expected_analysis_window_fp=expected_analysis_window_fp,
    )

    h_obs = _rows(human_observations)
    h_sum = _rows(human_summaries)
    h_life = _rows(human_lifecycle)
    b_obs = _rows(ball_observations)
    b_sum = _rows(ball_summaries)
    b_life = _rows(ball_lifecycle)
    attrs = _rows(detection_attributes)
    dets = _rows(detections)
    windows = _rows(analysis_windows)
    frame_rows = _rows(frames)

    all_rows = h_obs + h_sum + h_life + b_obs + b_sum + b_life + attrs + dets
    if all_rows:
        run_ids = {str(r["run_id"]) for r in all_rows if "run_id" in r}
        video_ids = {str(r["video_id"]) for r in all_rows if "video_id" in r}
        if len(run_ids) != 1:
            raise TrackingFusionError("RUN_ID_MISMATCH", f"table run_id mismatch: {run_ids}")
        if len(video_ids) != 1:
            raise TrackingFusionError("VIDEO_ID_MISMATCH", f"table video_id mismatch: {video_ids}")
        rid = next(iter(run_ids))
        vid = next(iter(video_ids))
        if expected_run_id is not None and rid != expected_run_id:
            raise TrackingFusionError("RUN_ID_MISMATCH", "table run_id != expected")
        if expected_video_id is not None and vid != expected_video_id:
            raise TrackingFusionError("VIDEO_ID_MISMATCH", "table video_id != expected")
    else:
        rid = str(align_meta.get("run_id") or expected_run_id or "")
        vid = str(align_meta.get("video_id") or expected_video_id or "")
        if not rid or not vid:
            raise TrackingFusionError("ID_INFERENCE_FAIL", "cannot infer run_id/video_id")

    fused_obs, fused_sum, fused_life, primary, remap = merge_track_tables(
        human_observations=h_obs,
        human_summaries=h_sum,
        human_lifecycle=h_life,
        ball_observations=b_obs,
        ball_summaries=b_sum,
        ball_lifecycle=b_life,
        primary_sidecar=primary_sidecar,
        detection_attributes=attrs,
        analysis_windows=windows,
        config=config,
    )

    human_track_ids = {int(r["track_id"]) for r in h_obs}
    ball_track_ids = {int(r["track_id"]) for r in fused_obs} - human_track_ids

    observed = sum(1 for o in fused_obs if o["observation_state"] == "observed")
    predicted = sum(1 for o in fused_obs if o["observation_state"] == "predicted")
    interpolated = sum(1 for o in fused_obs if o["observation_state"] == "interpolated")
    assigned_dets = {
        (o["run_id"], o["video_id"], o["frame_index"], o["detection_id"])
        for o in fused_obs
        if o["observation_state"] == "observed" and o.get("detection_id") is not None
    }
    human_det_n = sum(1 for a in attrs if a.get("entity_type") == TrackEntityType.HUMAN.value)
    ball_det_n = sum(1 for a in attrs if a.get("entity_type") == TrackEntityType.BALL.value)
    life_final = _count_lifecycle_states(fused_life)

    primary_n = sum(1 for f in primary if str(f.get("status") or "") == "primary")
    amb_n = sum(1 for f in primary if str(f.get("status") or "") == "ambiguous")
    no_cand = sum(
        1
        for f in primary
        if str(f.get("status") or "") in {"no_candidate", "none", "empty"}
        or (
            f.get("primary_track_id") is None
            and str(f.get("status") or "") not in {"ambiguous", "primary"}
        )
    )

    counts = {
        "human_input_detection_count": human_det_n,
        "ball_input_detection_count": ball_det_n,
        "assigned_detection_count": len(assigned_dets),
        "unassigned_detection_count": max(0, len(dets) - len(assigned_dets)),
        "human_track_count": len(human_track_ids),
        "ball_track_count": len(ball_track_ids),
        "total_track_count": len(human_track_ids) + len(ball_track_ids),
        "tentative_count": life_final.get("tentative", 0),
        "confirmed_count": life_final.get("confirmed", 0),
        "lost_count": life_final.get("lost", 0),
        "terminated_count": life_final.get("terminated", 0),
        "observed_count": observed,
        "predicted_count": predicted,
        "interpolated_count": interpolated,
        "primary_ball_frames": primary_n,
        "ambiguous_ball_frames": amb_n,
        "no_candidate_ball_frames": no_cand,
        "cross_cut_violation_count": 0,
        "invalid_fk_count": 0,
        "duplicate_count": 0,
        "frame_count": (
            len({(f["run_id"], f["video_id"], f["frame_index"]) for f in frame_rows})
            if frame_rows
            else len({(o["run_id"], o["video_id"], o["frame_index"]) for o in fused_obs})
        ),
    }

    if validate:
        import pyarrow as pa

        from football_analytics.data.compiler import compile_arrow_schema, get_contract

        def _tbl(name: str, rows: list[dict[str, Any]]) -> Any:
            schema = compile_arrow_schema(get_contract(name, 1))
            return pa.Table.from_pylist(rows, schema=schema) if rows else schema.empty_table()

        receipt_stub = {
            "observation_counts": {
                "observed": observed,
                "predicted": predicted,
                "interpolated": interpolated,
                "detection_associated": observed,
                "total": len(fused_obs),
            }
        }
        vr = validate_track_bundle(
            track_observations=_tbl("track_observations", fused_obs),
            track_summaries=_tbl("track_summaries", fused_sum),
            track_lifecycle=_tbl("track_lifecycle", fused_life),
            frames=_tbl("frames", frame_rows) if frame_rows else None,
            detections=_tbl("detections", dets) if dets else None,
            detection_attributes=_tbl("detection_attributes", attrs) if attrs else None,
            analysis_windows=_tbl("analysis_windows", windows) if windows else None,
            receipt=receipt_stub,
        )
        if vr.status == "FAIL":
            raise TrackingFusionError(
                "BUNDLE_INVALID",
                "; ".join(vr.errors[:5]) if vr.errors else "validate_track_bundle failed",
            )

    return TrackingFusionResult(
        observations=fused_obs,
        summaries=fused_sum,
        lifecycle=fused_life,
        primary_sidecar=primary,
        track_id_remap=dict(remap),
        run_id=rid,
        video_id=vid,
        human_track_count=len(human_track_ids),
        ball_track_count=len(ball_track_ids),
        counts=counts,
    )


__all__ = [
    "TrackingFusionError",
    "TrackingFusionResult",
    "align_tracking_inputs",
    "compute_ball_track_remap",
    "detect_cross_cut_violations",
    "detect_terminated_reopen",
    "fuse_tracking_bundle",
    "merge_track_tables",
    "recompute_summaries",
    "_basename",
]
