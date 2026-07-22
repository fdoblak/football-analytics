"""Cross-table contract bundle validation and synthetic fixtures."""

from __future__ import annotations

from typing import Any

from football_analytics.data.types import ValidationResult
from football_analytics.data.validation import validate_table


def _keys(table: Any, cols: list[str]) -> set[tuple[Any, ...]]:
    if table is None or table.num_rows == 0:
        return set()
    arrays = [table.column(c).to_pylist() for c in cols]
    out = set()
    for i in range(table.num_rows):
        out.add(tuple(a[i] for a in arrays))
    return out


def validate_contract_bundle(tables: dict[str, Any], specs: dict[str, Any]) -> ValidationResult:
    """Validate provided tables structurally and enforce FK relationships.

    Optional missing tables produce warnings; present tables with FK violations error.
    """
    result = ValidationResult(contract="bundle", version=1)
    # per-table validate
    for name, table in tables.items():
        if name not in specs:
            result.err(f"unknown table in bundle: {name}")
            continue
        vr = validate_table(table, specs[name])
        if vr.status == "FAIL":
            for e in vr.errors[:10]:
                result.err(f"{name}: {e}")
        result.statistics[name] = {"rows": table.num_rows, "status": vr.status}

    run_ids = set()
    for _name, table in tables.items():
        if "run_id" in table.column_names:
            run_ids.update(table.column("run_id").to_pylist())
    if len(run_ids) > 1:
        result.err("multiple run_id values in bundle")

    videos = tables.get("videos")
    frames = tables.get("frames")
    detections = tables.get("detections")
    tracks = tables.get("track_observations")
    summaries = tables.get("track_summaries")
    teams = tables.get("team_assignments")
    jerseys = tables.get("jersey_observations")
    calibrations = tables.get("calibrations")
    events = tables.get("events")

    if frames is not None and videos is None:
        result.warn("frames present without videos")
    if frames is not None and videos is not None:
        vkeys = _keys(videos, ["run_id", "video_id"])
        for key in _keys(frames, ["run_id", "video_id"]):
            if key not in vkeys:
                result.err(f"frames FK missing parent video {key}")
                break

    if detections is not None:
        if frames is None:
            result.warn("detections present without frames")
        else:
            fkeys = _keys(frames, ["run_id", "video_id", "frame_index"])
            for key in _keys(detections, ["run_id", "video_id", "frame_index"]):
                if key not in fkeys:
                    result.err(f"detections FK missing parent frame {key}")
                    break

    if tracks is not None:
        if frames is None:
            result.warn("track_observations present without frames")
        else:
            fkeys = _keys(frames, ["run_id", "video_id", "frame_index"])
            for key in _keys(tracks, ["run_id", "video_id", "frame_index"]):
                if key not in fkeys:
                    result.err(f"track_observations FK missing frame {key}")
                    break
        if detections is not None:
            dkeys = _keys(detections, ["run_id", "video_id", "frame_index", "detection_id"])
            for r in tracks.to_pylist():
                did = r["detection_id"]
                if did is None:
                    continue
                key = (r["run_id"], r["video_id"], r["frame_index"], did)
                if key not in dkeys:
                    result.err(f"track_observations FK missing detection {key}")
                    break

    if summaries is not None and tracks is not None:
        # optional consistency: summary track ids should appear in observations
        obs_ids = _keys(tracks, ["run_id", "video_id", "track_id"])
        for key in _keys(summaries, ["run_id", "video_id", "track_id"]):
            if key not in obs_ids:
                result.err(f"track_summary without observations {key}")
                break

    if teams is not None:
        if summaries is None:
            result.warn("team_assignments without track_summaries")
        else:
            skeys = _keys(summaries, ["run_id", "video_id", "track_id"])
            for key in _keys(teams, ["run_id", "video_id", "track_id"]):
                if key not in skeys:
                    result.err(f"team_assignments FK missing track {key}")
                    break

    if jerseys is not None:
        if summaries is None:
            result.warn("jersey_observations without track_summaries")
        else:
            skeys = _keys(summaries, ["run_id", "video_id", "track_id"])
            for key in _keys(jerseys, ["run_id", "video_id", "track_id"]):
                if key not in skeys:
                    result.err(f"jersey_observations FK missing track {key}")
                    break

    if calibrations is not None and frames is not None:
        fkeys = _keys(frames, ["run_id", "video_id", "frame_index"])
        for key in _keys(calibrations, ["run_id", "video_id", "frame_index"]):
            if key not in fkeys:
                result.err(f"calibrations FK missing frame {key}")
                break

    if events is not None and summaries is not None:
        skeys = _keys(summaries, ["run_id", "video_id", "track_id"])
        for r in events.to_pylist():
            for tid in r["actor_track_ids"]:
                key = (r["run_id"], r["video_id"], tid)
                if key not in skeys:
                    result.err(f"event actor missing track {key}")
                    break

    # bbox vs video dimensions
    if detections is not None and videos is not None:
        dims = {
            (r["run_id"], r["video_id"]): (r["width_px"], r["height_px"])
            for r in videos.to_pylist()
        }
        for r in detections.to_pylist():
            key = (r["run_id"], r["video_id"])
            if key not in dims:
                continue
            w, h = dims[key]
            if r["bbox_x2"] > w or r["bbox_y2"] > h:
                result.err(f"detection bbox exceeds video dims for {key}")
                break

    return result.finalize()


def build_synthetic_bundle(run_id: str) -> dict[str, Any]:
    """Deterministic tiny FK-consistent synthetic tables (no real data)."""
    pa = __import__("pyarrow")
    from football_analytics.data.compiler import compile_arrow_schema, get_contract

    def cast(name: str, rows: list[dict[str, Any]]) -> Any:
        spec = get_contract(name, 1)
        schema = compile_arrow_schema(spec)
        return pa.Table.from_pylist(rows, schema=schema)

    video_id = "clip_demo_01"
    sha = "a" * 64
    videos = cast(
        "videos",
        [
            {
                "run_id": run_id,
                "video_id": video_id,
                "source_sha256": sha,
                "container": "mp4",
                "codec": "h264",
                "width_px": 1280,
                "height_px": 720,
                "fps_numerator": 25,
                "fps_denominator": 1,
                "time_base_numerator": 1,
                "time_base_denominator": 25,
                "frame_count": 8,
                "duration_us": 320000,
                "has_audio": False,
                "source_ref": "logical_clip_demo_01",
            }
        ],
    )
    frames_rows = []
    for i in range(8):
        frames_rows.append(
            {
                "run_id": run_id,
                "video_id": video_id,
                "frame_index": i,
                "pts": i,
                "video_time_us": i * 40000,
                "duration_us": 40000,
                "is_key_frame": i % 4 == 0,
                "decode_status": "ok",
            }
        )
    frames = cast("frames", frames_rows)
    detections = cast(
        "detections",
        [
            {
                "run_id": run_id,
                "video_id": video_id,
                "frame_index": 0,
                "detection_id": 0,
                "class_id": 1,
                "class_name": "player",
                "confidence": 0.9,
                "bbox_x1": 10.0,
                "bbox_y1": 20.0,
                "bbox_x2": 40.0,
                "bbox_y2": 80.0,
                "model_id": "det_dummy_v1",
                "is_interpolated": False,
                "quality_flags": [],
            },
            {
                "run_id": run_id,
                "video_id": video_id,
                "frame_index": 1,
                "detection_id": 0,
                "class_id": 0,
                "class_name": "ball",
                "confidence": 0.8,
                "bbox_x1": 100.0,
                "bbox_y1": 100.0,
                "bbox_x2": 110.0,
                "bbox_y2": 110.0,
                "model_id": "det_dummy_v1",
                "is_interpolated": False,
                "quality_flags": ["low_res"],
            },
        ],
    )
    track_observations = cast(
        "track_observations",
        [
            {
                "run_id": run_id,
                "video_id": video_id,
                "frame_index": 0,
                "track_id": 1,
                "detection_id": 0,
                "class_id": 1,
                "confidence": 0.9,
                "bbox_x1": 10.0,
                "bbox_y1": 20.0,
                "bbox_x2": 40.0,
                "bbox_y2": 80.0,
                "observation_state": "observed",
                "model_id": "trk_dummy_v1",
                "quality_flags": [],
            },
            {
                "run_id": run_id,
                "video_id": video_id,
                "frame_index": 1,
                "track_id": 1,
                "detection_id": None,
                "class_id": 1,
                "confidence": 0.7,
                "bbox_x1": 12.0,
                "bbox_y1": 22.0,
                "bbox_x2": 42.0,
                "bbox_y2": 82.0,
                "observation_state": "predicted",
                "model_id": "trk_dummy_v1",
                "quality_flags": [],
            },
            {
                "run_id": run_id,
                "video_id": video_id,
                "frame_index": 0,
                "track_id": 2,
                "detection_id": None,
                "class_id": 0,
                "confidence": None,
                "bbox_x1": 100.0,
                "bbox_y1": 100.0,
                "bbox_x2": 110.0,
                "bbox_y2": 110.0,
                "observation_state": "interpolated",
                "model_id": "trk_dummy_v1",
                "quality_flags": [],
            },
        ],
    )
    track_summaries = cast(
        "track_summaries",
        [
            {
                "run_id": run_id,
                "video_id": video_id,
                "track_id": 1,
                "class_id": 1,
                "first_frame_index": 0,
                "last_frame_index": 1,
                "observation_count": 2,
                "observed_count": 1,
                "predicted_count": 1,
                "mean_confidence": 0.8,
                "max_confidence": 0.9,
                "termination_reason": "active",
                "quality_flags": [],
            },
            {
                "run_id": run_id,
                "video_id": video_id,
                "track_id": 2,
                "class_id": 0,
                "first_frame_index": 0,
                "last_frame_index": 0,
                "observation_count": 1,
                "observed_count": 0,
                "predicted_count": 0,
                "mean_confidence": None,
                "max_confidence": None,
                "termination_reason": "lost",
                "quality_flags": [],
            },
        ],
    )
    calibrations = cast(
        "calibrations",
        [
            {
                "run_id": run_id,
                "video_id": video_id,
                "frame_index": 0,
                "calibration_id": 0,
                "method": "dummy_h",
                "is_valid": True,
                "confidence": 0.95,
                "homography_image_to_pitch": [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0],
                "pitch_length_m": 105.0,
                "pitch_width_m": 68.0,
                "reprojection_error_px": 1.2,
                "quality_flags": [],
            }
        ],
    )
    team_assignments = cast(
        "team_assignments",
        [
            {
                "run_id": run_id,
                "video_id": video_id,
                "assignment_id": 0,
                "track_id": 1,
                "start_frame_index": 0,
                "end_frame_index": 1,
                "team_id": "home",
                "team_role": "home",
                "confidence": 0.9,
                "source": "rule",
                "quality_flags": [],
            },
            {
                "run_id": run_id,
                "video_id": video_id,
                "assignment_id": 1,
                "track_id": 2,
                "start_frame_index": 0,
                "end_frame_index": 0,
                "team_id": "unknown",
                "team_role": "unknown",
                "confidence": None,
                "source": "model",
                "quality_flags": [],
            },
        ],
    )
    jersey_observations = cast(
        "jersey_observations",
        [
            {
                "run_id": run_id,
                "video_id": video_id,
                "frame_index": 0,
                "observation_id": 0,
                "track_id": 1,
                "raw_text": "10",
                "normalized_number": 10,
                "digit_count": 2,
                "visibility": "visible",
                "readability": "clear",
                "confidence": 0.88,
                "source": "model",
                "review_status": "unreviewed",
                "crop_artifact_id": None,
                "quality_flags": [],
            }
        ],
    )
    events = cast(
        "events",
        [
            {
                "run_id": run_id,
                "video_id": video_id,
                "event_id": "evt_pass_001",
                "event_type": "pass",
                "start_frame_index": 0,
                "end_frame_index": 1,
                "start_time_us": 0,
                "end_time_us": 40000,
                "confidence": 0.7,
                "team_id": "home",
                "actor_track_ids": [1],
                "source": "rule",
                "attributes_json": '{"direction":"forward"}',
                "quality_flags": [],
            }
        ],
    )
    return {
        "videos": videos,
        "frames": frames,
        "detections": detections,
        "track_observations": track_observations,
        "track_summaries": track_summaries,
        "calibrations": calibrations,
        "team_assignments": team_assignments,
        "jersey_observations": jersey_observations,
        "events": events,
    }
