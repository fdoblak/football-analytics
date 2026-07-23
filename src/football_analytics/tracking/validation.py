"""Track bundle validation (observations + summaries + lifecycle + FKs)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from football_analytics.data.types import ValidationResult
from football_analytics.data.validation import validate_table
from football_analytics.tracking.bbox_rules import validate_track_bbox
from football_analytics.tracking.lifecycle import validate_lifecycle_sequence
from football_analytics.tracking.time_rules import gap_us, require_monotonic_times
from football_analytics.tracking.types import (
    LifecycleState,
    ObservationSource,
    TrackEntityType,
    observation_state_for_source,
)

_LIFECYCLE_SET = {s.value for s in LifecycleState}
_SOURCE_SET = {s.value for s in ObservationSource}
_ENTITY_SET = {e.value for e in TrackEntityType}


def _keys(table: Any, cols: list[str]) -> set[tuple[Any, ...]]:
    if table is None or table.num_rows == 0:
        return set()
    arrays = [table.column(c).to_pylist() for c in cols]
    out = set()
    for i in range(table.num_rows):
        out.add(tuple(a[i] for a in arrays))
    return out


def _rows(table: Any | None) -> list[dict[str, Any]]:
    if table is None:
        return []
    return table.to_pylist()


def validate_track_bundle(
    *,
    track_observations: Any | None,
    track_summaries: Any | None,
    track_lifecycle: Any | None,
    frames: Any | None = None,
    detections: Any | None = None,
    detection_attributes: Any | None = None,
    videos: Any | None = None,
    analysis_windows: Any | None = None,
    specs: Mapping[str, Any] | None = None,
    policy: Mapping[str, Any] | None = None,
    receipt: Mapping[str, Any] | None = None,
    expected_input_fingerprint: str | None = None,
    actual_input_fingerprint: str | None = None,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> ValidationResult:
    """Validate tracking contract bundle consistency (no tracker algorithm)."""
    result = ValidationResult(contract="track_bundle", version=1)

    if (
        expected_input_fingerprint is not None
        and actual_input_fingerprint is not None
        and expected_input_fingerprint != actual_input_fingerprint
    ):
        result.err("input hash/fingerprint mismatch")

    if specs is not None:
        for name, table in (
            ("track_observations", track_observations),
            ("track_summaries", track_summaries),
            ("track_lifecycle", track_lifecycle),
            ("frames", frames),
            ("detections", detections),
            ("detection_attributes", detection_attributes),
            ("videos", videos),
            ("analysis_windows", analysis_windows),
        ):
            if table is None or name not in specs:
                continue
            vr = validate_table(table, specs[name])
            if vr.status == "FAIL":
                for e in vr.errors[:10]:
                    result.err(f"{name}: {e}")

    frame_keys = (
        _keys(frames, ["run_id", "video_id", "frame_index"]) if frames is not None else None
    )
    det_keys = (
        _keys(detections, ["run_id", "video_id", "frame_index", "detection_id"])
        if detections is not None
        else None
    )
    video_keys = _keys(videos, ["run_id", "video_id"]) if videos is not None else None

    frames_by_rv: dict[tuple[Any, Any], dict[int, dict[str, Any]]] = defaultdict(dict)
    for r in _rows(frames):
        frames_by_rv[(r["run_id"], r["video_id"])][int(r["frame_index"])] = r

    attr_by_det: dict[tuple[Any, ...], dict[str, Any]] = {}
    for a in _rows(detection_attributes):
        key = (a["run_id"], a["video_id"], a["frame_index"], a["detection_id"])
        attr_by_det[key] = a

    obs_rows = _rows(track_observations)
    obs_pk = {(r["run_id"], r["video_id"], r["frame_index"], r["track_id"]) for r in obs_rows}
    if track_observations is not None and len(obs_pk) != track_observations.num_rows:
        result.err("duplicate track_observations primary keys")

    # One observation per track per frame already covered by PK; reinforce.
    # Detection assignment uniqueness: one detection → at most one track.
    det_to_tracks: dict[tuple[Any, ...], set[int]] = defaultdict(set)
    track_entity: dict[tuple[Any, Any, int], str] = {}
    track_roles: dict[tuple[Any, Any, int], set[str]] = defaultdict(set)
    track_times: dict[tuple[Any, Any, int], list[tuple[int, int]]] = defaultdict(list)

    for r in obs_rows:
        run_id, video_id = r["run_id"], r["video_id"]
        if video_keys is not None and (run_id, video_id) not in video_keys:
            result.err(f"cross-video / missing video FK for observation {run_id}/{video_id}")
            break
        fkey = (run_id, video_id, r["frame_index"])
        if frame_keys is not None and fkey not in frame_keys:
            result.err(f"dangling frame FK for observation {fkey}")
            break

        state = str(r["observation_state"])
        did = r.get("detection_id")
        if state == "observed":
            if did is None:
                result.err(f"observed observation requires detection_id at {fkey}")
                break
            dkey = (run_id, video_id, r["frame_index"], did)
            if det_keys is not None and dkey not in det_keys:
                result.err(f"dangling detection FK {dkey}")
                break
            det_to_tracks[dkey].add(int(r["track_id"]))
            attr = attr_by_det.get(dkey)
            if attr is not None:
                et = str(attr["entity_type"])
                tkey = (run_id, video_id, int(r["track_id"]))
                if et in _ENTITY_SET:
                    if tkey in track_entity and track_entity[tkey] != et:
                        result.err(f"human-ball merge rejected for track {tkey}")
                        break
                    track_entity[tkey] = et
                track_roles[tkey].add(str(attr.get("role_label", "unknown")))
        elif state in {"predicted", "interpolated"}:
            if did is not None:
                result.err(f"{state} observation must have null detection_id at {fkey}")
                break
            flags = list(r.get("quality_flags") or [])
            if "physical_metric_ineligible" not in flags and policy is not None:
                # Soft signal: predicted/interpolated must not claim physical eligibility.
                result.err(f"{state} missing physical_metric_ineligible quality_flag at {fkey}")
                break
        else:
            result.err(f"invalid observation_state {state}")
            break

        try:
            validate_track_bbox(
                (r["bbox_x1"], r["bbox_y1"], r["bbox_x2"], r["bbox_y2"]),
                frame_width=frame_width,
                frame_height=frame_height,
            )
        except Exception as exc:  # noqa: BLE001
            result.err(str(exc))
            break

        fmap = frames_by_rv.get((run_id, video_id), {})
        fr = fmap.get(int(r["frame_index"]))
        if fr is not None:
            track_times[(run_id, video_id, int(r["track_id"]))].append(
                (int(r["frame_index"]), int(fr["video_time_us"]))
            )

    for dkey, tids in det_to_tracks.items():
        if len(tids) > 1:
            result.err(f"duplicate detection assignment {dkey} -> {sorted(tids)}")
            break

    for tkey, pairs in track_times.items():
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        try:
            require_monotonic_times(pairs_sorted, label=f"track {tkey}")
        except Exception as exc:  # noqa: BLE001
            result.err(str(exc))
            break

    for tkey, roles in track_roles.items():
        concrete = {r for r in roles if r != "unknown"}
        if len(concrete) > 1:
            result.err(f"role conflict review required for track {tkey}: {sorted(concrete)}")
            break

    # Lifecycle events
    life_rows = _rows(track_lifecycle)
    life_pk = {(r["run_id"], r["video_id"], r["track_id"], r["event_index"]) for r in life_rows}
    if track_lifecycle is not None and len(life_pk) != track_lifecycle.num_rows:
        result.err("duplicate track_lifecycle primary keys")

    by_track: dict[tuple[Any, Any, int], list[dict[str, Any]]] = defaultdict(list)
    for r in life_rows:
        run_id, video_id = r["run_id"], r["video_id"]
        if video_keys is not None and (run_id, video_id) not in video_keys:
            result.err(f"cross-video FK in lifecycle {run_id}/{video_id}")
            break
        fkey = (run_id, video_id, r["frame_index"])
        if frame_keys is not None and fkey not in frame_keys:
            result.err(f"dangling frame FK in lifecycle {fkey}")
            break
        state = str(r["lifecycle_state"])
        if state not in _LIFECYCLE_SET:
            result.err(f"invalid lifecycle_state {state}")
            break
        prev = r.get("previous_state")
        if prev is not None and prev not in _LIFECYCLE_SET:
            result.err(f"invalid previous_state {prev}")
            break
        src = r.get("observation_source")
        if src is not None and src not in _SOURCE_SET:
            result.err(f"invalid observation_source {src}")
            break
        et = str(r["entity_type"])
        if et not in _ENTITY_SET:
            result.err(f"invalid entity_type {et}")
            break
        tkey = (run_id, video_id, int(r["track_id"]))
        if tkey in track_entity and track_entity[tkey] != et:
            result.err(f"lifecycle entity conflicts with observation entity for {tkey}")
            break
        by_track[tkey].append(r)

    max_lost = None
    if policy is not None:
        life = policy.get("lifecycle")
        if isinstance(life, Mapping):
            max_lost = int(life["max_lost_gap_us"])

    for tkey, events in by_track.items():
        errs = validate_lifecycle_sequence(events, policy=policy)
        for e in errs:
            result.err(f"track {tkey}: {e}")
            break
        if result.errors:
            break
        # Lost recovery gap check
        ordered = sorted(events, key=lambda e: int(e["event_index"]))
        for i in range(1, len(ordered)):
            prev_e, cur_e = ordered[i - 1], ordered[i]
            if (
                str(prev_e["lifecycle_state"]) == LifecycleState.LOST.value
                and str(cur_e["lifecycle_state"]) == LifecycleState.CONFIRMED.value
                and max_lost is not None
            ):
                try:
                    g = gap_us(int(prev_e["video_time_us"]), int(cur_e["video_time_us"]))
                except Exception as exc:  # noqa: BLE001
                    result.err(str(exc))
                    break
                if g > max_lost:
                    result.err(f"lost recovery exceeds max_lost_gap_us for {tkey}")
                    break
        if result.errors:
            break

        # Terminated reopen: any event after terminated
        seen_term = False
        for ev in ordered:
            if seen_term:
                result.err(f"terminated track reopen rejected for {tkey}")
                break
            if str(ev["lifecycle_state"]) == LifecycleState.TERMINATED.value:
                seen_term = True
        if result.errors:
            break

    # Summaries consistency
    sum_rows = _rows(track_summaries)
    sum_pk = {(r["run_id"], r["video_id"], r["track_id"]) for r in sum_rows}
    if track_summaries is not None and len(sum_pk) != track_summaries.num_rows:
        result.err("duplicate track_summaries primary keys")

    obs_by_track: dict[tuple[Any, Any, int], list[dict[str, Any]]] = defaultdict(list)
    for r in obs_rows:
        obs_by_track[(r["run_id"], r["video_id"], int(r["track_id"]))].append(r)

    for r in sum_rows:
        tkey = (r["run_id"], r["video_id"], int(r["track_id"]))
        obs = obs_by_track.get(tkey, [])
        if not obs:
            result.err(f"summary without observations {tkey}")
            break
        observed = sum(1 for o in obs if o["observation_state"] == "observed")
        predicted = sum(1 for o in obs if o["observation_state"] == "predicted")
        if int(r["observation_count"]) != len(obs):
            result.err(f"summary observation_count mismatch for {tkey}")
            break
        if int(r["observed_count"]) != observed:
            result.err(f"summary observed_count mismatch for {tkey}")
            break
        if int(r["predicted_count"]) != predicted:
            result.err(f"summary predicted_count mismatch for {tkey}")
            break
        frames_idx = [int(o["frame_index"]) for o in obs]
        if int(r["first_frame_index"]) != min(frames_idx) or int(r["last_frame_index"]) != max(
            frames_idx
        ):
            result.err(f"summary first/last frame mismatch for {tkey}")
            break

    if receipt is not None:
        _validate_receipt_counts(
            result,
            receipt=receipt,
            obs_rows=obs_rows,
            life_rows=life_rows,
        )

    # Document mapping helper available for callers
    _ = observation_state_for_source

    return result.finalize()


def _validate_receipt_counts(
    result: ValidationResult,
    *,
    receipt: Mapping[str, Any],
    obs_rows: Sequence[Mapping[str, Any]],
    life_rows: Sequence[Mapping[str, Any]],
) -> None:
    oc = receipt.get("observation_counts")
    if not isinstance(oc, Mapping):
        result.err("receipt missing observation_counts")
        return
    observed = sum(1 for o in obs_rows if o["observation_state"] == "observed")
    predicted = sum(1 for o in obs_rows if o["observation_state"] == "predicted")
    interpolated = sum(1 for o in obs_rows if o["observation_state"] == "interpolated")
    if int(oc.get("observed", -1)) != observed:
        result.err("receipt observed count mismatch")
    if int(oc.get("predicted", -1)) != predicted:
        result.err("receipt predicted count mismatch")
    if int(oc.get("interpolated", -1)) != interpolated:
        result.err("receipt interpolated count mismatch")
    if int(oc.get("detection_associated", -1)) != observed:
        result.err("receipt detection_associated count mismatch")
    if int(oc.get("total", -1)) != len(obs_rows):
        result.err("receipt total observation count mismatch")

    tc = receipt.get("track_counts")
    if isinstance(tc, Mapping) and life_rows:
        # Final state per track
        final: dict[tuple[Any, Any, int], str] = {}
        for ev in sorted(life_rows, key=lambda e: int(e["event_index"])):
            final[(ev["run_id"], ev["video_id"], int(ev["track_id"]))] = str(ev["lifecycle_state"])
        counts = {s.value: 0 for s in LifecycleState}
        for st in final.values():
            counts[st] = counts.get(st, 0) + 1
        for s in LifecycleState:
            if int(tc.get(s.value, -1)) != counts[s.value]:
                result.err(f"receipt track_counts.{s.value} mismatch")
                break


__all__ = ["validate_track_bundle"]
