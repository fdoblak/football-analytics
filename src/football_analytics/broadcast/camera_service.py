"""Camera-view classification service: samples → contract parquet + receipt."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa

from football_analytics.broadcast.camera_classification import (
    CameraClassificationError,
    aggregate_shot_classification,
    classify_sample,
)
from football_analytics.broadcast.camera_config import camera_config_fingerprint
from football_analytics.broadcast.camera_features import (
    CameraFeatureError,
    extract_features_for_samples,
)
from football_analytics.broadcast.camera_sampling import CameraSamplingError, plan_sample_points
from football_analytics.broadcast.contracts import load_broadcast_contract
from football_analytics.broadcast.shot_features import load_timeline_times
from football_analytics.broadcast.shot_service import (
    prepare_cfr_timeline_for_video,
    write_minimal_frames_parquet,
)
from football_analytics.broadcast.types import ShotSegment
from football_analytics.broadcast.validation import validate_broadcast_bundle
from football_analytics.core.records import write_json_record
from football_analytics.core.run_id import generate_run_id, validate_run_id
from football_analytics.data.compiler import compile_arrow_schema, get_contract
from football_analytics.data.parquet import read_contract_parquet, write_contract_parquet
from football_analytics.data.types import SAFE_ID_RE
from football_analytics.utils.archive_safety import assert_contained, resolve_strict
from football_analytics.video.probe_service import assert_snapshots_equal, snapshot_source
from football_analytics.video.types import VideoSourceError
from football_analytics.video.validation import (
    assert_safe_output_root,
    assert_safe_source_path,
    reject_unsafe_path_string,
    require_absolute_path,
)


class CameraServiceError(ValueError):
    """Camera-view service failure."""


@dataclass
class CameraServiceResult:
    accepted: bool
    exit_code: int
    cameras_parquet: str | None
    classification_receipt: str | None
    error_code: str | None
    segment_count: int
    config_fingerprint: str | None
    source_sha256: str | None

    def to_summary(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "exit_code": self.exit_code,
            "cameras_parquet": self.cameras_parquet,
            "classification_receipt": self.classification_receipt,
            "error_code": self.error_code,
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
) -> CameraServiceResult:
    return CameraServiceResult(
        accepted=False,
        exit_code=exit_code,
        cameras_parquet=None,
        classification_receipt=None,
        error_code=error_code,
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


def _load_shots(shots_path: Path, *, contain_root: Path) -> list[ShotSegment]:
    table = read_contract_parquet(
        shots_path,
        get_contract("shot_segments", 1),
        contain_root=contain_root,
    )
    rows = table.to_pylist()
    return [ShotSegment.from_dict(r) for r in rows]


def run_camera_view_classification(
    *,
    source: str,
    timeline: str,
    shots: str,
    output_dir: str,
    config: Mapping[str, Any],
    contain_root: Path | str | None = None,
    run_id: str | None = None,
    video_id: str | None = None,
    expected_source_sha256: str | None = None,
) -> CameraServiceResult:
    """Classify camera views for local source using frames + shot_segments parquet."""
    cfg_fp = camera_config_fingerprint(config)
    try:
        reject_unsafe_path_string(source, label="source")
        reject_unsafe_path_string(timeline, label="timeline")
        reject_unsafe_path_string(shots, label="shots")
        reject_unsafe_path_string(output_dir, label="output_dir")
        src = require_absolute_path(source, label="source")
        tl_path = require_absolute_path(timeline, label="timeline")
        shots_path = require_absolute_path(shots, label="shots")
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
        for p, label in ((tl_path, "timeline"), (shots_path, "shots")):
            if p.is_symlink():
                raise VideoSourceError(f"{label} must not be a symlink")
            if not p.is_file():
                raise VideoSourceError(f"{label} missing")
            assert_contained(resolve_strict(p), resolve_strict(root), label=label)
    except Exception:  # noqa: BLE001
        return _fail(error_code="PATH_SECURITY", exit_code=3, config_fingerprint=cfg_fp)

    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    cameras_out = out / "camera_view_segments.parquet"
    receipt_out = out / "classification_receipt.json"
    for p in (cameras_out, receipt_out):
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

    vid = video_id or "video_camera_001"
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
        shot_segs = _load_shots(shots_path, contain_root=root)
    except Exception:  # noqa: BLE001
        return _fail(
            error_code="INPUT_LOAD_FAILED",
            exit_code=1,
            config_fingerprint=cfg_fp,
            source_sha256=before.sha256,
        )

    if not shot_segs:
        return _fail(
            error_code="NO_SHOTS",
            exit_code=1,
            config_fingerprint=cfg_fp,
            source_sha256=before.sha256,
        )

    try:
        segments = []
        max_shots = int(config["resource_limits"]["max_shots"])
        if len(shot_segs) > max_shots:
            raise CameraServiceError("max_shots exceeded")
        for i, shot in enumerate(shot_segs):
            samples = plan_sample_points(
                timeline_pairs,
                start_time_us=shot.start_time_us,
                end_time_us=shot.end_time_us,
                config=config,
            )
            feats = extract_features_for_samples(src, samples, config)
            decisions = [classify_sample(f, config) for f in feats]
            cam_id = f"cam_{i:04d}"
            seg = aggregate_shot_classification(
                decisions,
                run_id=rid,
                video_id=vid,
                shot_id=shot.shot_id,
                camera_segment_id=cam_id,
                start_time_us=shot.start_time_us,
                end_time_us=shot.end_time_us,
                start_frame_index=shot.start_frame_index,
                end_frame_index_exclusive=shot.end_frame_index_exclusive,
                config=config,
                config_fingerprint=cfg_fp,
            )
            segments.append(seg)
    except (CameraFeatureError, CameraSamplingError, CameraClassificationError, CameraServiceError):
        return _fail(
            error_code="CLASSIFICATION_FAILED",
            exit_code=1,
            config_fingerprint=cfg_fp,
            source_sha256=before.sha256,
        )
    except Exception:  # noqa: BLE001
        return _fail(
            error_code="CLASSIFICATION_FAILED",
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

    rows = [s.to_dict() for s in segments]
    try:
        shots_tbl = read_contract_parquet(
            shots_path, get_contract("shot_segments", 1), contain_root=root
        )
        cam_table = (
            _rows_to_table(rows, "camera_view_segments")
            if rows
            else _empty_contract_table("camera_view_segments")
        )
        write_contract_parquet(
            cam_table,
            cameras_out,
            load_broadcast_contract("camera_view_segments"),
            contain_root=root,
        )
        vr = validate_broadcast_bundle(None, shots_tbl, cam_table, frames=frames_tbl)
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

    receipt = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "run_id": rid,
        "video_id": vid,
        "source_path": str(src),
        "source_sha256": before.sha256,
        "timeline_path": str(tl_path),
        "shots_path": str(shots_path),
        "config_fingerprint": cfg_fp,
        "segment_count": len(segments),
        "cameras_parquet": str(cameras_out),
        "classifier": "camera_view_baseline",
        "limitation": "one_camera_view_segment_per_shot",
        "confidence_policy": "null_contract_confidence_heuristic_in_provenance",
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

    return CameraServiceResult(
        accepted=True,
        exit_code=0,
        cameras_parquet=str(cameras_out),
        classification_receipt=str(receipt_out),
        error_code=None,
        segment_count=len(segments),
        config_fingerprint=cfg_fp,
        source_sha256=before.sha256,
    )


def write_single_shot_parquet(
    path: Path,
    *,
    run_id: str,
    video_id: str,
    shot_id: str,
    start_time_us: int,
    end_time_us: int,
    start_frame_index: int | None,
    end_frame_index_exclusive: int | None,
    contain_root: Path | None = None,
) -> Path:
    """Write a one-row shot_segments.parquet covering the full clip."""
    from football_analytics.broadcast.types import SegmentStatus
    from football_analytics.video.types import MappingQuality

    contract = get_contract("shot_segments", 1)
    schema = compile_arrow_schema(contract)
    row = ShotSegment(
        run_id=run_id,
        video_id=video_id,
        shot_id=shot_id,
        start_time_us=start_time_us,
        end_time_us=end_time_us,
        start_frame_index=start_frame_index,
        end_frame_index_exclusive=end_frame_index_exclusive,
        start_boundary_id=None,
        end_boundary_id=None,
        duration_us=end_time_us - start_time_us,
        frame_count=(
            (end_frame_index_exclusive - start_frame_index)
            if start_frame_index is not None and end_frame_index_exclusive is not None
            else None
        ),
        timeline_mapping_quality=MappingQuality.DERIVED_WITH_CONSTANT_OFFSET,
        segment_status=SegmentStatus.ACTIVE,
        provenance_json=None,
    ).to_dict()
    table = pa.Table.from_pylist([row], schema=schema)
    return write_contract_parquet(table, path, contract, contain_root=contain_root)


__all__ = [
    "CameraServiceError",
    "CameraServiceResult",
    "run_camera_view_classification",
    "write_single_shot_parquet",
    "prepare_cfr_timeline_for_video",
    "write_minimal_frames_parquet",
]
