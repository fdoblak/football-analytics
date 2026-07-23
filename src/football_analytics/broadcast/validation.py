"""Cross-table broadcast bundle validation (Stage 4A contracts only)."""

from __future__ import annotations

import math
from typing import Any

from football_analytics.broadcast.contracts import CONTRACT_NAMES, load_broadcast_contract
from football_analytics.broadcast.types import (
    CONTRACT_VERSION,
    NON_PLAYABLE_VIEW_FAMILIES,
    FramingScale,
    GraphicsStatus,
    Playability,
    ReplayStatus,
    SegmentStatus,
    Suitability,
    ViewFamily,
)
from football_analytics.data.types import ValidationResult
from football_analytics.data.validation import validate_table
from football_analytics.video.types import MappingQuality

PLAYABLE_STATUSES = frozenset({Playability.PLAYABLE.value, Playability.PARTIALLY_PLAYABLE.value})
ACTIVE_PRODUCTION = frozenset({SegmentStatus.ACTIVE.value})


def _keys(table: Any, cols: list[str]) -> set[tuple[Any, ...]]:
    if table is None or table.num_rows == 0:
        return set()
    arrays = [table.column(c).to_pylist() for c in cols]
    out = set()
    for i in range(table.num_rows):
        out.add(tuple(a[i] for a in arrays))
    return out


def _finite_unit(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    f = float(v)
    return math.isfinite(f) and 0.0 <= f <= 1.0


def _finite_unit_or_null(v: Any) -> bool:
    if v is None:
        return True
    return _finite_unit(v)


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    """Half-open [start, end) overlap test."""
    return a0 < b1 and b0 < a1


def _frame_time_map(frames: Any) -> dict[tuple[str, str], dict[int, int]]:
    """(run_id, video_id) -> {frame_index: video_time_us}."""
    out: dict[tuple[str, str], dict[int, int]] = {}
    if frames is None:
        return out
    for r in frames.to_pylist():
        key = (r["run_id"], r["video_id"])
        out.setdefault(key, {})[int(r["frame_index"])] = int(r["video_time_us"])
    return out


def _check_frame_time_consistency(
    row: dict[str, Any],
    *,
    frame_map: dict[tuple[str, str], dict[int, int]],
    start_key: str,
    end_key: str,
    result: ValidationResult,
    label: str,
) -> None:
    key = (row["run_id"], row["video_id"])
    fmap = frame_map.get(key)
    if fmap is None:
        return
    sf = row.get("start_frame_index") if start_key == "start_frame_index" else row.get(start_key)
    ef = row.get(end_key)
    if sf is not None:
        if int(sf) not in fmap:
            result.err(f"{label} start_frame_index missing from frames for {key}")
            return
        if fmap[int(sf)] != int(row["start_time_us"]):
            result.err(f"{label} start frame/time mismatch for {key}")
            return
    if ef is not None:
        # end is exclusive: last included frame is ef-1 when ef > 0
        if int(ef) < 0:
            result.err(f"{label} invalid exclusive end frame for {key}")
            return
        # Exclusive end may equal next frame index or one past last known frame.
        if int(ef) > 0 and int(ef) not in fmap and (int(ef) - 1) not in fmap:
            result.warn(f"{label} end_frame_index_exclusive not anchored in frames for {key}")
        if sf is not None and int(ef) <= int(sf):
            result.err(f"{label} end_frame_index_exclusive must be > start_frame_index for {key}")


def validate_broadcast_bundle(
    boundaries: Any | None,
    shots: Any | None,
    cameras: Any | None,
    *,
    videos: Any | None = None,
    frames: Any | None = None,
    check_table_semantics: bool = True,
) -> ValidationResult:
    """Validate shot/camera tables and cross-table temporal/FK/suitability rules."""
    result = ValidationResult(contract="broadcast_bundle", version=1)
    tables = {
        "shot_boundaries": boundaries,
        "shot_segments": shots,
        "camera_view_segments": cameras,
    }
    specs = {name: load_broadcast_contract(name) for name in CONTRACT_NAMES}

    for name, table in tables.items():
        if table is None:
            continue
        vr = validate_table(table, specs[name], check_semantics=check_table_semantics)
        if vr.status == "FAIL":
            for e in vr.errors[:10]:
                result.err(f"{name}: {e}")
        for w in vr.warnings[:5]:
            result.warn(f"{name}: {w}")
        result.statistics[name] = {"rows": table.num_rows, "status": vr.status}

    video_keys = _keys(videos, ["run_id", "video_id"]) if videos is not None else None
    frame_map = _frame_time_map(frames)

    # FK to videos when provided
    for name, table in tables.items():
        if table is None or video_keys is None:
            continue
        for key in _keys(table, ["run_id", "video_id"]):
            if key not in video_keys:
                result.err(f"{name} FK missing parent video {key}")
                break

    if videos is None and any(t is not None for t in tables.values()):
        result.warn("broadcast tables present without videos")

    # --- Boundaries ---
    boundary_ids: set[tuple[str, str, str]] = set()
    if boundaries is not None:
        rows = boundaries.to_pylist()
        boundary_groups: dict[tuple[str, str], list[tuple[int, str, int]]] = {}
        for i, r in enumerate(rows):
            if int(r["contract_version"]) != CONTRACT_VERSION:
                result.err(f"shot_boundaries contract_version != {CONTRACT_VERSION} at row {i}")
            if not _finite_unit_or_null(r.get("confidence")):
                result.err(f"shot_boundaries confidence out of range at row {i}")
            bkey = (r["run_id"], r["video_id"], r["boundary_id"])
            boundary_ids.add(bkey)
            boundary_groups.setdefault((r["run_id"], r["video_id"]), []).append(
                (int(r["boundary_time_us"]), str(r["boundary_id"]), i)
            )
            lf, rf = r.get("left_frame_index"), r.get("right_frame_index")
            if lf is not None and rf is not None and int(lf) > int(rf):
                result.err(f"shot_boundaries left_frame_index > right_frame_index at row {i}")
            # frame existence when frames present
            vkey = (r["run_id"], r["video_id"])
            fmap = frame_map.get(vkey)
            if fmap is not None:
                for label, fi in (("left_frame_index", lf), ("right_frame_index", rf)):
                    if fi is None:
                        continue
                    if int(fi) not in fmap:
                        result.err(f"shot_boundaries {label} missing from frames at row {i}")
                    elif (
                        fmap[int(fi)] != int(r["boundary_time_us"]) and label == "right_frame_index"
                    ):
                        # boundary time may sit between left/right; warn only if both sides miss
                        pass
        for vkey, b_items in boundary_groups.items():
            items_sorted = sorted(b_items, key=lambda x: (x[0], x[1]))
            prev_t = -1
            for t, _bid, _i in items_sorted:
                if t < prev_t:
                    result.err(f"shot_boundaries non-ascending times for {vkey}")
                    break
                prev_t = t

    # --- Shots ---
    shot_lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    if shots is not None:
        rows = shots.to_pylist()
        shot_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for i, r in enumerate(rows):
            if int(r["contract_version"]) != CONTRACT_VERSION:
                result.err(f"shot_segments contract_version != {CONTRACT_VERSION} at row {i}")
            start, end = int(r["start_time_us"]), int(r["end_time_us"])
            if end <= start:
                result.err(f"shot_segments end_time_us must be > start_time_us at row {i}")
            if int(r["duration_us"]) != end - start:
                result.err(f"shot_segments duration_us mismatch at row {i}")
            if int(r["duration_us"]) <= 0:
                result.err(f"shot_segments duration_us must be > 0 at row {i}")
            try:
                MappingQuality(r["timeline_mapping_quality"])
            except ValueError:
                result.err(f"shot_segments invalid timeline_mapping_quality at row {i}")
            skey = (r["run_id"], r["video_id"], r["shot_id"])
            shot_lookup[skey] = r
            shot_groups.setdefault((r["run_id"], r["video_id"]), []).append(r)

            for field, bid in (
                ("start_boundary_id", r.get("start_boundary_id")),
                ("end_boundary_id", r.get("end_boundary_id")),
            ):
                if bid is None:
                    continue
                if (r["run_id"], r["video_id"], bid) not in boundary_ids and boundaries is not None:
                    result.err(f"shot_segments {field} missing boundary FK at row {i}")

            sf, ef = r.get("start_frame_index"), r.get("end_frame_index_exclusive")
            if sf is not None and ef is not None and int(ef) <= int(sf):
                result.err(f"shot_segments frame interval invalid at row {i}")
            _check_frame_time_consistency(
                r,
                frame_map=frame_map,
                start_key="start_frame_index",
                end_key="end_frame_index_exclusive",
                result=result,
                label="shot_segments",
            )

        for vkey, s_items in shot_groups.items():
            ordered = sorted(
                s_items, key=lambda row: (int(row["start_time_us"]), str(row["shot_id"]))
            )
            # active production: ordered, no overlap; gaps only with gap_coverage/incomplete
            prev_end: int | None = None
            prev_status: str | None = None
            for row in ordered:
                start, end = int(row["start_time_us"]), int(row["end_time_us"])
                status = str(row["segment_status"])
                if status in ACTIVE_PRODUCTION and prev_end is not None:
                    if start < prev_end and prev_status in ACTIVE_PRODUCTION:
                        result.err(f"shot_segments active overlap for {vkey}")
                        break
                    if start > prev_end and status in ACTIVE_PRODUCTION:
                        # gap between active shots — require an explaining gap/incomplete segment
                        covered = any(
                            str(g["segment_status"])
                            in {
                                SegmentStatus.GAP_COVERAGE.value,
                                SegmentStatus.INCOMPLETE.value,
                            }
                            and int(g["start_time_us"]) <= prev_end
                            and int(g["end_time_us"]) >= start
                            for g in ordered
                        )
                        if not covered:
                            result.warn(
                                f"shot_segments gap without gap_coverage/incomplete for {vkey}"
                            )
                if status in ACTIVE_PRODUCTION:
                    prev_end = end if prev_end is None else max(prev_end, end)
                    prev_status = status
                elif prev_end is None:
                    prev_status = status

    # --- Cameras ---
    if cameras is not None:
        rows = cameras.to_pylist()
        camera_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for i, r in enumerate(rows):
            if int(r["contract_version"]) != CONTRACT_VERSION:
                result.err(
                    f"camera_view_segments contract_version != {CONTRACT_VERSION} at row {i}"
                )
            start, end = int(r["start_time_us"]), int(r["end_time_us"])
            if end <= start:
                result.err(f"camera_view_segments end_time_us must be > start_time_us at row {i}")
            if not _finite_unit_or_null(r.get("confidence")):
                result.err(f"camera_view_segments confidence out of range at row {i}")
            if not _finite_unit(r.get("coverage")):
                result.err(f"camera_view_segments coverage out of range at row {i}")
            refs = r.get("evidence_refs")
            if refs is None or any(x is None for x in refs):
                result.err(f"camera_view_segments evidence_refs invalid at row {i}")

            view = str(r["view_family"])
            play = str(r["playability"])
            replay = str(r["replay_status"])
            graphics = str(r["graphics_status"])
            calib = str(r["calibration_suitability"])
            track = str(r["tracking_suitability"])
            identity = str(r["target_identity_suitability"])
            framing = str(r["framing_scale"])

            # Hard suitability / playability rules
            if (
                view in {v.value for v in NON_PLAYABLE_VIEW_FAMILIES}
                and play != Playability.NON_PLAYABLE.value
            ):
                result.err(
                    f"camera_view_segments view_family {view} must be non_playable at row {i}"
                )
            if graphics == GraphicsStatus.FULL_SCREEN.value:
                if play != Playability.NON_PLAYABLE.value:
                    result.err(
                        "camera_view_segments full_screen graphics must be non_playable "
                        f"at row {i}"
                    )
                if track != Suitability.UNSUITABLE.value:
                    result.err(
                        "camera_view_segments full_screen graphics tracking must be unsuitable "
                        f"at row {i}"
                    )
                if calib != Suitability.UNSUITABLE.value:
                    result.err(
                        "camera_view_segments full_screen graphics calibration must be unsuitable "
                        f"at row {i}"
                    )
            if replay in {ReplayStatus.REPLAY.value, ReplayStatus.REPLAY_TRANSITION.value}:
                if play == Playability.PLAYABLE.value:
                    result.err(
                        "camera_view_segments replay cannot be fully playable as live at row {i}"
                    )
                if identity == Suitability.SUITABLE.value:
                    result.warn(
                        "camera_view_segments replay marked identity-suitable at row {i}; "
                        "replay is not live match time"
                    )

            # Soft single-player semantics
            if framing in {FramingScale.CLOSE_UP.value, FramingScale.EXTREME_CLOSE_UP.value}:
                if calib == Suitability.SUITABLE.value:
                    result.warn(
                        "camera_view_segments close-up marked calibration-suitable at row {i}"
                    )
                if track == Suitability.SUITABLE.value and identity == Suitability.UNSUITABLE.value:
                    result.warn(
                        "camera_view_segments close-up tracking-suitable but identity-unsuitable "
                        f"at row {i}"
                    )
            if view == ViewFamily.UNKNOWN.value and play == Playability.PLAYABLE.value:
                result.warn(
                    "camera_view_segments unknown view marked playable at row {i}; "
                    "prefer coverage reduction over invented metrics"
                )

            shot_id = r.get("shot_id")
            if shot_id is not None:
                skey = (r["run_id"], r["video_id"], shot_id)
                if shots is not None and skey not in shot_lookup:
                    result.err(f"camera_view_segments shot_id FK missing at row {i}")
                elif skey in shot_lookup:
                    shot = shot_lookup[skey]
                    if start < int(shot["start_time_us"]) or end > int(shot["end_time_us"]):
                        result.err(
                            f"camera_view_segments not contained in shot {shot_id} at row {i}"
                        )

            sf, ef = r.get("start_frame_index"), r.get("end_frame_index_exclusive")
            if sf is not None and ef is not None and int(ef) <= int(sf):
                result.err(f"camera_view_segments frame interval invalid at row {i}")
            _check_frame_time_consistency(
                r,
                frame_map=frame_map,
                start_key="start_frame_index",
                end_key="end_frame_index_exclusive",
                result=result,
                label="camera_view_segments",
            )
            camera_groups.setdefault((r["run_id"], r["video_id"]), []).append(r)

        # no overlap among playable camera segments per video
        for vkey, c_items in camera_groups.items():
            playable = [row for row in c_items if str(row["playability"]) in PLAYABLE_STATUSES]
            playable.sort(
                key=lambda row: (int(row["start_time_us"]), str(row["camera_segment_id"]))
            )
            for a, b in zip(playable, playable[1:], strict=False):
                if _intervals_overlap(
                    int(a["start_time_us"]),
                    int(a["end_time_us"]),
                    int(b["start_time_us"]),
                    int(b["end_time_us"]),
                ):
                    result.err(f"camera_view_segments playable overlap for {vkey}")
                    break

    return result.finalize()


def validate_analysis_windows_bundle(
    windows: Any | None,
    *,
    shots: Any | None = None,
    cameras: Any | None = None,
    videos: Any | None = None,
    frames: Any | None = None,
    check_table_semantics: bool = True,
) -> ValidationResult:
    """Validate analysis_windows against shot/camera temporal + FK rules."""
    from football_analytics.data.compiler import get_contract

    result = ValidationResult(contract="analysis_windows_bundle", version=1)
    if windows is None:
        result.warn("analysis_windows table missing")
        return result.finalize()

    spec = get_contract("analysis_windows", 1)
    vr = validate_table(windows, spec, check_semantics=check_table_semantics)
    if vr.status == "FAIL":
        for e in vr.errors[:10]:
            result.err(f"analysis_windows: {e}")
    for w in vr.warnings[:5]:
        result.warn(f"analysis_windows: {w}")
    result.statistics["analysis_windows"] = {"rows": windows.num_rows, "status": vr.status}

    video_keys = _keys(videos, ["run_id", "video_id"]) if videos is not None else None
    if video_keys is not None:
        for key in _keys(windows, ["run_id", "video_id"]):
            if key not in video_keys:
                result.err(f"analysis_windows FK missing parent video {key}")
                break

    shot_lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    if shots is not None:
        for r in shots.to_pylist():
            shot_lookup[(r["run_id"], r["video_id"], r["shot_id"])] = r

    camera_ids: set[tuple[str, str, str]] = set()
    if cameras is not None:
        for r in cameras.to_pylist():
            camera_ids.add((r["run_id"], r["video_id"], r["camera_segment_id"]))

    frame_map = _frame_time_map(frames)
    rows = windows.to_pylist()
    by_video: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for i, r in enumerate(rows):
        if int(r["contract_version"]) != CONTRACT_VERSION:
            result.err(f"analysis_windows contract_version != {CONTRACT_VERSION} at row {i}")
        start, end = int(r["start_time_us"]), int(r["end_time_us"])
        if end <= start:
            result.err(f"analysis_windows end_time_us must be > start_time_us at row {i}")
        if not _finite_unit(r.get("coverage")):
            result.err(f"analysis_windows coverage out of range at row {i}")
        if not _finite_unit_or_null(r.get("confidence")):
            result.err(f"analysis_windows confidence out of range at row {i}")
        if not isinstance(r.get("manual_review_required"), bool):
            result.err(f"analysis_windows manual_review_required must be bool at row {i}")
        if not isinstance(r.get("decision_codes"), list) or any(
            x is None for x in (r.get("decision_codes") or [])
        ):
            result.err(f"analysis_windows decision_codes invalid at row {i}")
        if not isinstance(r.get("camera_segment_ids"), list) or any(
            x is None for x in (r.get("camera_segment_ids") or [])
        ):
            result.err(f"analysis_windows camera_segment_ids invalid at row {i}")
        if not isinstance(r.get("source_refs"), list) or any(
            x is None for x in (r.get("source_refs") or [])
        ):
            result.err(f"analysis_windows source_refs invalid at row {i}")

        try:
            MappingQuality(r["timeline_mapping_quality"])
        except ValueError:
            result.err(f"analysis_windows invalid timeline_mapping_quality at row {i}")

        # Replay unknown must never be live_event eligible.
        if (
            str(r["replay_status"]) == ReplayStatus.UNKNOWN.value
            and str(r["live_event_eligibility"]) == "eligible"
        ):
            result.err(
                "analysis_windows replay unknown cannot have live_event eligible " f"at row {i}"
            )
        if str(r["replay_status"]) in {
            ReplayStatus.REPLAY.value,
            ReplayStatus.REPLAY_TRANSITION.value,
        }:
            if str(r["live_event_eligibility"]) == "eligible":
                result.err(
                    "analysis_windows confirmed replay cannot have live_event eligible "
                    f"at row {i}"
                )
            if str(r["physical_metric_eligibility"]) == "eligible":
                result.err(
                    "analysis_windows confirmed replay cannot have physical_metric eligible "
                    f"at row {i}"
                )

        if str(r["playability"]) == Playability.NON_PLAYABLE.value:
            for axis in (
                "tracking_eligibility",
                "calibration_eligibility",
                "ball_analysis_eligibility",
                "live_event_eligibility",
                "physical_metric_eligibility",
            ):
                if str(r[axis]) == "eligible":
                    result.err(
                        f"analysis_windows non_playable cannot have {axis}=eligible at row {i}"
                    )

        shot_id = r.get("shot_id")
        if shot_id is not None:
            skey = (r["run_id"], r["video_id"], shot_id)
            if shots is not None and skey not in shot_lookup:
                result.err(f"analysis_windows shot_id FK missing at row {i}")
            elif skey in shot_lookup:
                shot = shot_lookup[skey]
                if start < int(shot["start_time_us"]) or end > int(shot["end_time_us"]):
                    result.err(f"analysis_windows not contained in shot {shot_id} at row {i}")

        for cid in r.get("camera_segment_ids") or []:
            ckey = (r["run_id"], r["video_id"], cid)
            if cameras is not None and ckey not in camera_ids:
                result.err(f"analysis_windows camera_segment_id FK missing {cid} at row {i}")
                break

        sf, ef = r.get("start_frame_index"), r.get("end_frame_index_exclusive")
        if sf is not None and ef is not None and int(ef) <= int(sf):
            result.err(f"analysis_windows frame interval invalid at row {i}")
        _check_frame_time_consistency(
            r,
            frame_map=frame_map,
            start_key="start_frame_index",
            end_key="end_frame_index_exclusive",
            result=result,
            label="analysis_windows",
        )
        by_video.setdefault((r["run_id"], r["video_id"]), []).append(r)

    for vkey, items in by_video.items():
        ordered = sorted(
            items, key=lambda row: (int(row["start_time_us"]), str(row["analysis_window_id"]))
        )
        for a, b in zip(ordered, ordered[1:], strict=False):
            if _intervals_overlap(
                int(a["start_time_us"]),
                int(a["end_time_us"]),
                int(b["start_time_us"]),
                int(b["end_time_us"]),
            ):
                result.err(f"analysis_windows overlap for {vkey}")
                break

    return result.finalize()


__all__ = ["validate_broadcast_bundle", "validate_analysis_windows_bundle"]
