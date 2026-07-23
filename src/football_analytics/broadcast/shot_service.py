"""Shot boundary detection service: stream detect → contract parquet + receipt."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa

from football_analytics.broadcast.contracts import load_broadcast_contract
from football_analytics.broadcast.shot_config import shot_config_fingerprint
from football_analytics.broadcast.shot_detection import detect_shots
from football_analytics.broadcast.shot_features import (
    ShotFeatureError,
    build_cfr_timeline,
    extract_feature_frames,
    load_timeline_times,
)
from football_analytics.broadcast.types import ShotBoundary, ShotSegment
from football_analytics.broadcast.validation import validate_broadcast_bundle
from football_analytics.core.records import write_json_record
from football_analytics.core.run_id import generate_run_id, validate_run_id
from football_analytics.data.compiler import compile_arrow_schema, get_contract
from football_analytics.data.parquet import read_contract_parquet, write_contract_parquet
from football_analytics.data.types import SAFE_ID_RE
from football_analytics.utils.archive_safety import assert_contained, resolve_strict
from football_analytics.video.probe_service import assert_snapshots_equal, snapshot_source
from football_analytics.video.types import MappingQuality, VideoSourceError
from football_analytics.video.validation import (
    assert_safe_output_root,
    assert_safe_source_path,
    reject_unsafe_path_string,
    require_absolute_path,
)


class ShotServiceError(ValueError):
    """Shot boundary service failure."""


@dataclass
class ShotServiceResult:
    accepted: bool
    exit_code: int
    boundaries_parquet: str | None
    segments_parquet: str | None
    detection_receipt: str | None
    scores_jsonl: str | None
    error_code: str | None
    boundary_count: int
    segment_count: int
    config_fingerprint: str | None
    source_sha256: str | None

    def to_summary(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "exit_code": self.exit_code,
            "boundaries_parquet": self.boundaries_parquet,
            "segments_parquet": self.segments_parquet,
            "detection_receipt": self.detection_receipt,
            "scores_jsonl": self.scores_jsonl,
            "error_code": self.error_code,
            "boundary_count": self.boundary_count,
            "segment_count": self.segment_count,
            "config_fingerprint": self.config_fingerprint,
            "source_sha256": self.source_sha256,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _fail(
    *,
    error_code: str,
    exit_code: int = 1,
    config_fingerprint: str | None = None,
    source_sha256: str | None = None,
) -> ShotServiceResult:
    return ShotServiceResult(
        accepted=False,
        exit_code=exit_code,
        boundaries_parquet=None,
        segments_parquet=None,
        detection_receipt=None,
        scores_jsonl=None,
        error_code=error_code,
        boundary_count=0,
        segment_count=0,
        config_fingerprint=config_fingerprint,
        source_sha256=source_sha256,
    )


def _rows_to_table(rows: list[dict[str, Any]], contract_name: str) -> Any:
    contract = get_contract(contract_name, 1)
    schema = compile_arrow_schema(contract)
    return pa.Table.from_pylist(rows, schema=schema)


def _empty_contract_table(contract_name: str) -> Any:
    schema = compile_arrow_schema(get_contract(contract_name, 1))
    return schema.empty_table()


def _duration_from_timeline(timeline: list[tuple[int, int]], fps: int | None = None) -> int:
    if not timeline:
        raise ShotServiceError("empty timeline")
    _last_idx, last_t = timeline[-1]
    if fps and fps > 0:
        return int(last_t + round(1_000_000 / fps))
    if len(timeline) >= 2:
        step = timeline[1][1] - timeline[0][1]
        if step > 0:
            return int(last_t + step)
    return int(last_t + 1)


def run_shot_boundary_detection(
    *,
    source: str,
    timeline: str,
    output_dir: str,
    config: Mapping[str, Any],
    contain_root: Path | str | None = None,
    run_id: str | None = None,
    video_id: str | None = None,
    write_scores: bool = True,
    expected_source_sha256: str | None = None,
    mapping_quality: MappingQuality = MappingQuality.NOT_AVAILABLE,
) -> ShotServiceResult:
    """Detect shot boundaries for a local source using frames.parquet timeline."""
    cfg_fp = shot_config_fingerprint(config)
    try:
        reject_unsafe_path_string(source, label="source")
        reject_unsafe_path_string(timeline, label="timeline")
        reject_unsafe_path_string(output_dir, label="output_dir")
        src = require_absolute_path(source, label="source")
        tl_path = require_absolute_path(timeline, label="timeline")
        out = require_absolute_path(output_dir, label="output_dir")
    except (VideoSourceError, Exception):  # noqa: BLE001
        code = "UNSAFE_PATH"
        if "://" in str(source):
            code = "NETWORK_SOURCE_FORBIDDEN"
        return _fail(error_code=code, exit_code=3, config_fingerprint=cfg_fp)

    root = Path(contain_root) if contain_root is not None else Path(str(config["runtime_root"]))
    policy = {
        "symlinks_allowed": False,
        "network_sources_allowed": False,
        "allowed_file_extensions": [".mp4", ".mkv", ".mov", ".webm", ".avi"],
    }
    try:
        from football_analytics.video.validation import assert_extension_allowed

        root = require_absolute_path(str(root), label="contain_root")
        src = assert_safe_source_path(str(src), contain_root=root, policy=policy)
        assert_extension_allowed(src, policy)
        assert_safe_output_root(
            str(out), contain_root=root, source_path=str(src), overwrite_allowed=False
        )
        if tl_path.is_symlink():
            raise VideoSourceError("timeline must not be a symlink")
        if not tl_path.is_file():
            raise VideoSourceError("timeline missing")
        assert_contained(resolve_strict(tl_path), resolve_strict(root), label="timeline")
    except Exception:  # noqa: BLE001
        return _fail(error_code="PATH_SECURITY", exit_code=3, config_fingerprint=cfg_fp)

    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    boundaries_out = out / "shot_boundaries.parquet"
    segments_out = out / "shot_segments.parquet"
    receipt_out = out / "detection_receipt.json"
    scores_out = out / "scores.jsonl"
    for p in (boundaries_out, segments_out, receipt_out, scores_out):
        if p.exists():
            return _fail(error_code="OVERWRITE_FORBIDDEN", exit_code=3, config_fingerprint=cfg_fp)

    try:
        before = snapshot_source(src)
    except Exception:  # noqa: BLE001
        return _fail(error_code="SOURCE_SNAPSHOT_FAILED", exit_code=3, config_fingerprint=cfg_fp)

    if expected_source_sha256 and before.sha256 != expected_source_sha256:
        return _fail(
            error_code="SOURCE_HASH_MISMATCH",
            exit_code=3,
            config_fingerprint=cfg_fp,
            source_sha256=before.sha256,
        )

    rid = run_id or generate_run_id()
    try:
        validate_run_id(rid)
    except Exception:  # noqa: BLE001
        return _fail(
            error_code="INVALID_RUN_ID",
            exit_code=2,
            config_fingerprint=cfg_fp,
            source_sha256=before.sha256,
        )

    vid = video_id or "video_shot_001"
    if not SAFE_ID_RE.fullmatch(vid):
        return _fail(
            error_code="INVALID_VIDEO_ID",
            exit_code=2,
            config_fingerprint=cfg_fp,
            source_sha256=before.sha256,
        )

    try:
        frames_tbl = read_contract_parquet(
            tl_path,
            get_contract("frames", 1),
            contain_root=root,
        )
        timeline_pairs = load_timeline_times(frames_tbl)
    except Exception:  # noqa: BLE001
        return _fail(
            error_code="TIMELINE_LOAD_FAILED",
            exit_code=1,
            config_fingerprint=cfg_fp,
            source_sha256=before.sha256,
        )

    try:
        features = extract_feature_frames(src, timeline_pairs, config)
        duration_us = _duration_from_timeline(timeline_pairs)
        boundaries, segments, scored = detect_shots(
            features,
            run_id=rid,
            video_id=vid,
            duration_us=duration_us,
            timeline=timeline_pairs,
            config=config,
            mapping_quality=mapping_quality,
            config_fingerprint=cfg_fp,
        )
    except ShotFeatureError:
        return _fail(
            error_code="FEATURE_EXTRACT_FAILED",
            exit_code=1,
            config_fingerprint=cfg_fp,
            source_sha256=before.sha256,
        )
    except Exception:  # noqa: BLE001
        return _fail(
            error_code="DETECTION_FAILED",
            exit_code=1,
            config_fingerprint=cfg_fp,
            source_sha256=before.sha256,
        )

    try:
        after = snapshot_source(src)
        assert_snapshots_equal(before, after)
    except Exception:  # noqa: BLE001
        return _fail(
            error_code="SOURCE_MUTATED",
            exit_code=3,
            config_fingerprint=cfg_fp,
            source_sha256=before.sha256,
        )

    b_rows = [b.to_dict() for b in boundaries]
    s_rows = [s.to_dict() for s in segments]
    try:
        b_table = (
            _rows_to_table(b_rows, "shot_boundaries")
            if b_rows
            else _empty_contract_table("shot_boundaries")
        )
        s_table = (
            _rows_to_table(s_rows, "shot_segments")
            if s_rows
            else _empty_contract_table("shot_segments")
        )
        write_contract_parquet(
            b_table,
            boundaries_out,
            load_broadcast_contract("shot_boundaries"),
            contain_root=root,
        )
        write_contract_parquet(
            s_table,
            segments_out,
            load_broadcast_contract("shot_segments"),
            contain_root=root,
        )
        vr = validate_broadcast_bundle(b_table, s_table, None, frames=frames_tbl)
        if vr.status == "FAIL":
            return _fail(
                error_code="BUNDLE_VALIDATION_FAILED",
                exit_code=1,
                config_fingerprint=cfg_fp,
                source_sha256=before.sha256,
            )
    except Exception:  # noqa: BLE001
        return _fail(
            error_code="WRITE_FAILED",
            exit_code=1,
            config_fingerprint=cfg_fp,
            source_sha256=before.sha256,
        )

    scores_path = None
    if write_scores:
        try:
            with scores_out.open("w", encoding="utf-8") as fh:
                for s in scored:
                    fh.write(
                        json.dumps(
                            {
                                "frame_index": s.frame_index,
                                "video_time_us": s.video_time_us,
                                "score": round(s.score, 8),
                                "luma_mae": round(s.luma_mae, 8),
                                "hist_distance": round(s.hist_distance, 8),
                                "edge_change_ratio": round(s.edge_change_ratio, 8),
                                "mean_luma": round(s.mean_luma, 8),
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
            scores_path = str(scores_out)
        except Exception:  # noqa: BLE001
            return _fail(
                error_code="SCORES_WRITE_FAILED",
                exit_code=1,
                config_fingerprint=cfg_fp,
                source_sha256=before.sha256,
            )

    receipt = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "run_id": rid,
        "video_id": vid,
        "source_path": str(src),
        "source_sha256": before.sha256,
        "timeline_path": str(tl_path),
        "config_fingerprint": cfg_fp,
        "boundary_count": len(boundaries),
        "segment_count": len(segments),
        "duration_us": duration_us,
        "mapping_quality": mapping_quality.value,
        "boundaries_parquet": str(boundaries_out),
        "segments_parquet": str(segments_out),
        "scores_jsonl": scores_path,
        "detector": "shot_boundary_baseline",
        "streaming_decode": True,
    }
    try:
        write_json_record(receipt_out, receipt, contain_root=root, overwrite=False)
    except Exception:  # noqa: BLE001
        return _fail(
            error_code="RECEIPT_WRITE_FAILED",
            exit_code=1,
            config_fingerprint=cfg_fp,
            source_sha256=before.sha256,
        )

    return ShotServiceResult(
        accepted=True,
        exit_code=0,
        boundaries_parquet=str(boundaries_out),
        segments_parquet=str(segments_out),
        detection_receipt=str(receipt_out),
        scores_jsonl=scores_path,
        error_code=None,
        boundary_count=len(boundaries),
        segment_count=len(segments),
        config_fingerprint=cfg_fp,
        source_sha256=before.sha256,
    )


def write_minimal_frames_parquet(
    path: Path,
    *,
    run_id: str,
    video_id: str,
    timeline: list[tuple[int, int]],
    contain_root: Path | None = None,
    fps: int = 25,
) -> Path:
    """Write a minimal frames.parquet for synthetic fixtures (CFR times)."""
    contract = get_contract("frames", 1)
    schema = compile_arrow_schema(contract)
    step = int(round(1_000_000 / fps)) if fps > 0 else 40000
    rows = []
    for idx, t_us in timeline:
        rows.append(
            {
                "run_id": run_id,
                "video_id": video_id,
                "frame_index": idx,
                "pts": idx,
                "video_time_us": t_us,
                "duration_us": step,
                "is_key_frame": idx == 0,
                "decode_status": "ok",
            }
        )
    table = pa.Table.from_pylist(rows, schema=schema)
    return write_contract_parquet(table, path, contract, contain_root=contain_root)


def count_decoded_frames(video_path: Path) -> int:
    """Count frames via OpenCV decode (authoritative for feature extraction)."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ShotServiceError("cannot open video to count frames")
    n = 0
    try:
        while True:
            ok, _frame = cap.read()
            if not ok:
                break
            n += 1
    finally:
        cap.release()
    return n


def prepare_cfr_timeline_for_video(
    video_path: Path,
    *,
    frames_out: Path,
    run_id: str,
    video_id: str,
    fps: int,
    frame_count: int | None = None,
    contain_root: Path | None = None,
) -> list[tuple[int, int]]:
    n = frame_count if frame_count is not None else count_decoded_frames(video_path)
    timeline = build_cfr_timeline(frame_count=n, fps_num=fps, fps_den=1)
    write_minimal_frames_parquet(
        frames_out,
        run_id=run_id,
        video_id=video_id,
        timeline=timeline,
        contain_root=contain_root,
        fps=fps,
    )
    return timeline


__all__ = [
    "ShotServiceError",
    "ShotServiceResult",
    "run_shot_boundary_detection",
    "write_minimal_frames_parquet",
    "count_decoded_frames",
    "prepare_cfr_timeline_for_video",
    "ShotBoundary",
    "ShotSegment",
]
