"""Stage 4D broadcast integrate pipeline: fuse → route → write artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa

from football_analytics.broadcast.playability import (
    RoutingPolicyError,
    build_review_queue,
    route_fused_windows,
    routing_policy_fingerprint,
)
from football_analytics.broadcast.segment_fusion import FusionError, fuse_shot_camera_intervals
from football_analytics.broadcast.validation import (
    validate_analysis_windows_bundle,
    validate_broadcast_bundle,
)
from football_analytics.core.records import RecordError, write_json_record
from football_analytics.core.run_id import generate_run_id, validate_run_id
from football_analytics.data.compiler import compile_arrow_schema, get_contract
from football_analytics.data.parquet import read_contract_parquet, write_contract_parquet
from football_analytics.data.types import SAFE_ID_RE
from football_analytics.utils.archive_safety import (
    assert_contained,
    assert_not_dangerous_operation_root,
    resolve_strict,
)
from football_analytics.video.types import VideoSourceError
from football_analytics.video.validation import (
    reject_unsafe_path_string,
    require_absolute_path,
)


class BroadcastPipelineError(ValueError):
    """Broadcast integrate pipeline failure."""


@dataclass
class BroadcastPipelineResult:
    accepted: bool
    exit_code: int
    analysis_windows_parquet: str | None
    review_queue_json: str | None
    pipeline_receipt_json: str | None
    error_code: str | None
    window_count: int
    review_count: int
    policy_fingerprint: str | None

    def to_summary(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "exit_code": self.exit_code,
            "analysis_windows_parquet": self.analysis_windows_parquet,
            "review_queue_json": self.review_queue_json,
            "pipeline_receipt_json": self.pipeline_receipt_json,
            "error_code": self.error_code,
            "window_count": self.window_count,
            "review_count": self.review_count,
            "policy_fingerprint": self.policy_fingerprint,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _fail(
    *,
    error_code: str,
    exit_code: int = 1,
    policy_fingerprint: str | None = None,
) -> BroadcastPipelineResult:
    return BroadcastPipelineResult(
        accepted=False,
        exit_code=exit_code,
        analysis_windows_parquet=None,
        review_queue_json=None,
        pipeline_receipt_json=None,
        error_code=error_code,
        window_count=0,
        review_count=0,
        policy_fingerprint=policy_fingerprint,
    )


def _rows_to_table(rows: list[dict[str, Any]], contract_name: str) -> Any:
    contract = get_contract(contract_name, 1)
    schema = compile_arrow_schema(contract)
    return pa.Table.from_pylist(rows, schema=schema)


def _empty_contract_table(contract_name: str) -> Any:
    return compile_arrow_schema(get_contract(contract_name, 1)).empty_table()


def _infer_ids(
    shots: list[dict[str, Any]],
    cameras: list[dict[str, Any]],
    *,
    run_id: str | None,
    video_id: str | None,
) -> tuple[str, str]:
    rows = shots or cameras
    if not rows:
        raise BroadcastPipelineError("no shot/camera rows to infer run/video ids")
    rid = run_id or str(rows[0]["run_id"])
    vid = video_id or str(rows[0]["video_id"])
    validate_run_id(rid)
    if not SAFE_ID_RE.fullmatch(vid):
        raise BroadcastPipelineError("invalid video_id")
    for r in rows:
        if str(r["run_id"]) != rid or str(r["video_id"]) != vid:
            raise BroadcastPipelineError("mixed run_id/video_id in inputs")
    return rid, vid


def run_broadcast_integrate(
    *,
    timeline: str,
    boundaries: str,
    shots: str,
    camera_views: str,
    output_dir: str,
    policy: Mapping[str, Any],
    contain_root: Path | str | None = None,
    run_id: str | None = None,
    video_id: str | None = None,
) -> BroadcastPipelineResult:
    """Fuse shot/camera segments, apply routing policy, write analysis_windows bundle."""
    pol_fp = routing_policy_fingerprint(policy)
    try:
        reject_unsafe_path_string(timeline, label="timeline")
        reject_unsafe_path_string(boundaries, label="boundaries")
        reject_unsafe_path_string(shots, label="shots")
        reject_unsafe_path_string(camera_views, label="camera_views")
        reject_unsafe_path_string(output_dir, label="output_dir")
        tl_path = require_absolute_path(timeline, label="timeline")
        bnd_path = require_absolute_path(boundaries, label="boundaries")
        shots_path = require_absolute_path(shots, label="shots")
        cam_path = require_absolute_path(camera_views, label="camera_views")
        out = require_absolute_path(output_dir, label="output_dir")
    except (VideoSourceError, Exception):  # noqa: BLE001
        return _fail(error_code="UNSAFE_PATH", exit_code=3, policy_fingerprint=pol_fp)

    root = Path(contain_root) if contain_root is not None else Path(str(policy["runtime_root"]))
    try:
        root = require_absolute_path(str(root), label="contain_root")
        assert_not_dangerous_operation_root(root)
        out_resolved = resolve_strict(out) if out.exists() else out.resolve()
        assert_contained(out_resolved, resolve_strict(root), label="output_dir")
        for p, label in (
            (tl_path, "timeline"),
            (bnd_path, "boundaries"),
            (shots_path, "shots"),
            (cam_path, "camera_views"),
        ):
            if p.is_symlink():
                raise VideoSourceError(f"{label} must not be a symlink")
            if not p.is_file():
                raise VideoSourceError(f"{label} missing")
            assert_contained(resolve_strict(p), resolve_strict(root), label=label)
    except Exception:  # noqa: BLE001
        return _fail(error_code="CONTAINMENT_FAILURE", exit_code=3, policy_fingerprint=pol_fp)

    if policy.get("overwrite_allowed") is not False:
        return _fail(error_code="OVERWRITE_POLICY", exit_code=2, policy_fingerprint=pol_fp)

    try:
        frames_table = read_contract_parquet(tl_path, get_contract("frames", 1), contain_root=root)
        boundaries_table = read_contract_parquet(
            bnd_path, get_contract("shot_boundaries", 1), contain_root=root
        )
        shots_table = read_contract_parquet(
            shots_path, get_contract("shot_segments", 1), contain_root=root
        )
        cameras_table = read_contract_parquet(
            cam_path, get_contract("camera_view_segments", 1), contain_root=root
        )
    except Exception:  # noqa: BLE001
        return _fail(error_code="INPUT_READ_FAIL", exit_code=1, policy_fingerprint=pol_fp)

    bundle_vr = validate_broadcast_bundle(
        boundaries_table,
        shots_table,
        cameras_table,
        frames=frames_table,
        check_table_semantics=True,
    )
    if bundle_vr.status == "FAIL":
        return _fail(error_code="INPUT_BUNDLE_INVALID", exit_code=1, policy_fingerprint=pol_fp)

    shot_rows = shots_table.to_pylist()
    cam_rows = cameras_table.to_pylist()
    try:
        rid, vid = _infer_ids(shot_rows, cam_rows, run_id=run_id, video_id=video_id)
    except Exception:  # noqa: BLE001
        return _fail(error_code="ID_INFERENCE_FAIL", exit_code=1, policy_fingerprint=pol_fp)

    merge = bool(policy["thresholds"]["merge_identical_adjacent"])
    try:
        fused = fuse_shot_camera_intervals(shot_rows, cam_rows, merge_identical_adjacent=merge)
    except FusionError:
        return _fail(error_code="FUSION_ERROR", exit_code=1, policy_fingerprint=pol_fp)

    try:
        routed = route_fused_windows(fused, policy)
    except (RoutingPolicyError, Exception):  # noqa: BLE001
        return _fail(error_code="ROUTING_ERROR", exit_code=1, policy_fingerprint=pol_fp)

    # Attach provenance (safe ids only).
    for row in routed:
        row["provenance_json"] = json.dumps(
            {
                "stage": "4D",
                "policy_fingerprint": pol_fp,
                "fusion": "interval_sweep_v1",
            },
            separators=(",", ":"),
            sort_keys=True,
        )

    windows_table = (
        _rows_to_table(routed, "analysis_windows")
        if routed
        else _empty_contract_table("analysis_windows")
    )
    aw_vr = validate_analysis_windows_bundle(
        windows_table,
        shots=shots_table,
        cameras=cameras_table,
        frames=frames_table,
    )
    if aw_vr.status == "FAIL":
        return _fail(error_code="WINDOWS_INVALID", exit_code=1, policy_fingerprint=pol_fp)

    review = build_review_queue(
        routed,
        policy_version=str(policy["policy_version"]),
        run_id=rid,
        video_id=vid,
    )

    out.mkdir(parents=True, exist_ok=True)
    windows_path = out / "analysis_windows.parquet"
    review_path = out / "review_queue.json"
    receipt_path = out / "pipeline_receipt.json"

    try:
        write_contract_parquet(
            windows_table,
            windows_path,
            get_contract("analysis_windows", 1),
            contain_root=root,
            overwrite=False,
        )
        write_json_record(review_path, review, contain_root=root, overwrite=False)
        receipt = {
            "schema_version": 1,
            "stage": "4D",
            "created_at": _utc_now(),
            "run_id": rid,
            "video_id": vid,
            "policy_version": str(policy["policy_version"]),
            "policy_fingerprint": pol_fp,
            "window_count": len(routed),
            "review_count": len(review["items"]),
            "inputs": {
                "timeline": tl_path.name,
                "boundaries": bnd_path.name,
                "shots": shots_path.name,
                "camera_views": cam_path.name,
            },
            "outputs": {
                "analysis_windows": windows_path.name,
                "review_queue": review_path.name,
            },
            "validation": {
                "input_bundle": bundle_vr.status,
                "analysis_windows": aw_vr.status,
            },
        }
        write_json_record(receipt_path, receipt, contain_root=root, overwrite=False)
    except (RecordError, Exception):  # noqa: BLE001
        return _fail(error_code="WRITE_FAIL", exit_code=1, policy_fingerprint=pol_fp)

    return BroadcastPipelineResult(
        accepted=True,
        exit_code=0,
        analysis_windows_parquet=str(windows_path),
        review_queue_json=str(review_path),
        pipeline_receipt_json=str(receipt_path),
        error_code=None,
        window_count=len(routed),
        review_count=len(review["items"]),
        policy_fingerprint=pol_fp,
    )


def ensure_run_id(run_id: str | None) -> str:
    if run_id is None:
        return generate_run_id()
    validate_run_id(run_id)
    return run_id


__all__ = [
    "BroadcastPipelineError",
    "BroadcastPipelineResult",
    "run_broadcast_integrate",
    "ensure_run_id",
]
