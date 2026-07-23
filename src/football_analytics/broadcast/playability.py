"""Broadcast routing policy loader and playability eligibility routing."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from football_analytics.broadcast.segment_fusion import FusedWindow
from football_analytics.broadcast.types import (
    CONTRACT_VERSION,
    Eligibility,
    FramingScale,
    GraphicsStatus,
    Playability,
    ReplayStatus,
    ViewFamily,
)
from football_analytics.core.hashing import hash_canonical_json
from football_analytics.data.types import SAFE_ID_RE
from football_analytics.video.types import MappingQuality

CONFIG_VERSION = 1
MAX_CONFIG_BYTES = 256 * 1024

TASK_AXES = (
    "tracking",
    "calibration",
    "identity",
    "ball_analysis",
    "live_event",
    "physical_metric",
)

ELIGIBILITY_VALUES = frozenset(e.value for e in Eligibility)

REQUIRED_TOP = frozenset(
    {
        "policy_version",
        "config_version",
        "task_axes",
        "eligibility_values",
        "decision_codes",
        "thresholds",
        "unsafe_mapping_qualities",
        "blocking_graphics",
        "non_playable_view_families",
        "identity_framing",
        "identity_view_families",
        "wide_tracking_framing",
        "wide_tracking_views",
        "replay_confirmed",
        "evaluation",
        "runtime_root",
        "overwrite_allowed",
        "symlinks_allowed",
        "network_sources_allowed",
    }
)


class RoutingPolicyError(ValueError):
    """Routing policy load/validation failure."""


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RoutingPolicyError(f"{label} must be a mapping")
    return value


def _require_int(value: Any, *, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RoutingPolicyError(f"{label} must be an int")
    if minimum is not None and value < minimum:
        raise RoutingPolicyError(f"{label} must be >= {minimum}")
    return value


def _require_float(
    value: Any, *, label: str, minimum: float | None = None, maximum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RoutingPolicyError(f"{label} must be a number")
    f = float(value)
    if not math.isfinite(f):
        raise RoutingPolicyError(f"{label} must be finite")
    if minimum is not None and f < minimum:
        raise RoutingPolicyError(f"{label} must be >= {minimum}")
    if maximum is not None and f > maximum:
        raise RoutingPolicyError(f"{label} must be <= {maximum}")
    return f


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise RoutingPolicyError(f"{label} must be a bool")
    return value


def _require_str(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RoutingPolicyError(f"{label} must be a non-empty string")
    return value


def _require_str_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise RoutingPolicyError(f"{label} must be a non-empty list")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise RoutingPolicyError(f"{label} entries must be non-empty strings")
        out.append(item)
    return out


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({k: _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(v) for v in value)
    return value


def _validate_policy(raw: Mapping[str, Any]) -> dict[str, Any]:
    missing = REQUIRED_TOP - set(raw.keys())
    if missing:
        raise RoutingPolicyError(f"missing keys: {sorted(missing)}")
    unknown = set(raw.keys()) - REQUIRED_TOP
    if unknown:
        raise RoutingPolicyError(f"unknown top-level keys: {sorted(unknown)}")

    if int(raw["config_version"]) != CONFIG_VERSION:
        raise RoutingPolicyError(f"config_version must be {CONFIG_VERSION}")

    policy_version = _require_str(raw["policy_version"], label="policy_version")
    axes = _require_str_list(raw["task_axes"], label="task_axes")
    if tuple(axes) != TASK_AXES:
        raise RoutingPolicyError(f"task_axes must be exactly {list(TASK_AXES)}")

    elig = _require_str_list(raw["eligibility_values"], label="eligibility_values")
    if set(elig) != ELIGIBILITY_VALUES:
        raise RoutingPolicyError("eligibility_values mismatch")

    codes = _require_str_list(raw["decision_codes"], label="decision_codes")
    for code in codes:
        if not SAFE_ID_RE.fullmatch(code):
            raise RoutingPolicyError(f"invalid decision_code: {code}")

    thresholds = dict(_require_mapping(raw["thresholds"], label="thresholds"))
    low_cov = _require_float(
        thresholds.get("low_coverage_max"),
        label="thresholds.low_coverage_max",
        minimum=0.0,
        maximum=1.0,
    )
    merge = _require_bool(
        thresholds.get("merge_identical_adjacent"),
        label="thresholds.merge_identical_adjacent",
    )

    unsafe_map = _require_str_list(
        raw["unsafe_mapping_qualities"], label="unsafe_mapping_qualities"
    )
    for m in unsafe_map:
        try:
            MappingQuality(m)
        except ValueError as exc:
            raise RoutingPolicyError(f"invalid mapping quality: {m}") from exc

    blocking_graphics = _require_str_list(raw["blocking_graphics"], label="blocking_graphics")
    for g in blocking_graphics:
        try:
            GraphicsStatus(g)
        except ValueError as exc:
            raise RoutingPolicyError(f"invalid graphics status: {g}") from exc

    for label, key in (
        ("non_playable_view_families", "non_playable_view_families"),
        ("identity_framing", "identity_framing"),
        ("identity_view_families", "identity_view_families"),
        ("wide_tracking_framing", "wide_tracking_framing"),
        ("wide_tracking_views", "wide_tracking_views"),
        ("replay_confirmed", "replay_confirmed"),
    ):
        _require_str_list(raw[key], label=label)

    evaluation = dict(_require_mapping(raw["evaluation"], label="evaluation"))
    for ek in (
        "unsafe_live_event_fp_max",
        "unsafe_physical_metric_fp_max",
        "unsafe_calibration_fp_max",
        "unsafe_tracking_fp_max",
        "non_playable_eligible_fp_max",
        "manual_review_recall_min",
        "overlap_rate_max",
        "unexplained_gap_rate_max",
    ):
        if ek not in evaluation:
            raise RoutingPolicyError(f"evaluation.{ek} required")
        _require_float(evaluation[ek], label=f"evaluation.{ek}", minimum=0.0, maximum=1.0)

    runtime_root = _require_str(raw["runtime_root"], label="runtime_root")
    overwrite = _require_bool(raw["overwrite_allowed"], label="overwrite_allowed")
    symlinks = _require_bool(raw["symlinks_allowed"], label="symlinks_allowed")
    network = _require_bool(raw["network_sources_allowed"], label="network_sources_allowed")

    return {
        "policy_version": policy_version,
        "config_version": CONFIG_VERSION,
        "task_axes": list(axes),
        "eligibility_values": list(elig),
        "decision_codes": list(codes),
        "thresholds": {"low_coverage_max": low_cov, "merge_identical_adjacent": merge},
        "unsafe_mapping_qualities": list(unsafe_map),
        "blocking_graphics": list(blocking_graphics),
        "non_playable_view_families": list(raw["non_playable_view_families"]),
        "identity_framing": list(raw["identity_framing"]),
        "identity_view_families": list(raw["identity_view_families"]),
        "wide_tracking_framing": list(raw["wide_tracking_framing"]),
        "wide_tracking_views": list(raw["wide_tracking_views"]),
        "replay_confirmed": list(raw["replay_confirmed"]),
        "evaluation": {k: float(evaluation[k]) for k in evaluation},
        "runtime_root": runtime_root,
        "overwrite_allowed": overwrite,
        "symlinks_allowed": symlinks,
        "network_sources_allowed": network,
    }


def load_routing_policy(path: Path | str) -> Mapping[str, Any]:
    p = Path(path)
    if not p.is_file() or p.is_symlink():
        raise RoutingPolicyError(f"policy path must be a regular file: {p}")
    raw_bytes = p.read_bytes()
    if len(raw_bytes) > MAX_CONFIG_BYTES:
        raise RoutingPolicyError("policy file too large")
    try:
        data = yaml.safe_load(raw_bytes.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise RoutingPolicyError("policy YAML parse failed") from exc
    if not isinstance(data, dict):
        raise RoutingPolicyError("policy root must be a mapping")
    validated = _validate_policy(data)
    return _deep_freeze(validated)


def routing_policy_fingerprint(policy: Mapping[str, Any]) -> str:
    def _thaw(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {k: _thaw(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_thaw(v) for v in value]
        return value

    return hash_canonical_json(_thaw(policy))


def default_routing_policy_path(*, repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[3]
    return root / "configs" / "broadcast" / "broadcast_routing_policy.yaml"


def _add_code(codes: list[str], code: str) -> None:
    if code not in codes:
        codes.append(code)


def route_fused_window(
    window: FusedWindow,
    policy: Mapping[str, Any],
    *,
    analysis_window_id: str,
) -> dict[str, Any]:
    """Apply conservative routing policy → analysis_windows row dict."""
    codes: list[str] = []
    review = False

    tracking = Eligibility.UNKNOWN
    calibration = Eligibility.UNKNOWN
    identity = Eligibility.UNKNOWN
    ball = Eligibility.UNKNOWN
    live_event = Eligibility.UNKNOWN
    physical = Eligibility.UNKNOWN

    view = window.view_family
    framing = window.framing_scale
    replay = window.replay_status
    graphics = window.graphics_status
    play = window.playability
    coverage = float(window.coverage)
    mapping = window.timeline_mapping_quality
    low_cov_max = float(policy["thresholds"]["low_coverage_max"])
    blocking_graphics = set(policy["blocking_graphics"])
    non_playable_views = set(policy["non_playable_view_families"])
    identity_framing = set(policy["identity_framing"])
    identity_views = set(policy["identity_view_families"])
    wide_framing = set(policy["wide_tracking_framing"])
    wide_views = set(policy["wide_tracking_views"])
    replay_confirmed = set(policy["replay_confirmed"])
    unsafe_mapping = set(policy["unsafe_mapping_qualities"])

    # --- Gap / conflict (never auto-eligible) ---
    if window.is_gap:
        review = True
        _add_code(codes, "CAMERA_GAP")
        tracking = calibration = identity = ball = live_event = physical = Eligibility.UNKNOWN
    elif window.is_conflict:
        review = True
        _add_code(codes, "CONFLICTING_CAMERA_LABELS")
        tracking = calibration = identity = ball = live_event = physical = Eligibility.UNKNOWN
    else:
        graphics_block = graphics in blocking_graphics or view in non_playable_views
        non_playable = play == Playability.NON_PLAYABLE.value or graphics_block

        if graphics_block:
            _add_code(codes, "GRAPHICS_NON_PLAYABLE")
        if play == Playability.NON_PLAYABLE.value:
            _add_code(codes, "NON_PLAYABLE_BLOCK")

        if non_playable:
            tracking = calibration = identity = ball = live_event = physical = (
                Eligibility.INELIGIBLE
            )
        elif coverage < low_cov_max:
            review = True
            _add_code(codes, "LOW_COVERAGE")
            tracking = calibration = identity = ball = live_event = physical = Eligibility.UNKNOWN
        elif view == ViewFamily.UNKNOWN.value or play == Playability.UNCERTAIN.value:
            review = True
            _add_code(codes, "UNKNOWN_VIEW_REVIEW_REQUIRED")
            tracking = calibration = identity = ball = live_event = physical = Eligibility.UNKNOWN
        elif play == Playability.PARTIALLY_PLAYABLE.value:
            review = True
            _add_code(codes, "PARTIALLY_PLAYABLE_REVIEW")
            # Conservative: only identity may be conditionally eligible for close-ups.
            if framing in identity_framing or view in identity_views:
                identity = Eligibility.CONDITIONALLY_ELIGIBLE
                tracking = calibration = ball = physical = Eligibility.INELIGIBLE
                live_event = Eligibility.UNKNOWN
                _add_code(codes, "IDENTITY_ONLY_CLOSEUP")
            elif framing in wide_framing and view in wide_views:
                tracking = Eligibility.CONDITIONALLY_ELIGIBLE
                calibration = Eligibility.CONDITIONALLY_ELIGIBLE
                identity = Eligibility.UNKNOWN
                ball = Eligibility.CONDITIONALLY_ELIGIBLE
                live_event = Eligibility.UNKNOWN
                physical = Eligibility.UNKNOWN
            else:
                tracking = calibration = identity = ball = live_event = physical = (
                    Eligibility.UNKNOWN
                )
        elif framing in identity_framing or view in identity_views:
            identity = Eligibility.ELIGIBLE
            tracking = calibration = ball = physical = Eligibility.INELIGIBLE
            live_event = Eligibility.UNKNOWN
            _add_code(codes, "IDENTITY_ONLY_CLOSEUP")
            _add_code(codes, "CALIBRATION_UNSUITABLE")
            _add_code(codes, "TRACKING_UNSUITABLE")
            _add_code(codes, "PHYSICAL_METRIC_UNSAFE")
        elif (
            play == Playability.PLAYABLE.value
            and framing in wide_framing
            and view in wide_views
            and graphics == GraphicsStatus.NONE.value
        ):
            tracking = Eligibility.ELIGIBLE
            calibration = Eligibility.ELIGIBLE
            identity = Eligibility.CONDITIONALLY_ELIGIBLE
            ball = Eligibility.ELIGIBLE
            live_event = Eligibility.ELIGIBLE
            physical = Eligibility.ELIGIBLE
            _add_code(codes, "PLAYABLE_WIDE_VIEW")
        elif play == Playability.PLAYABLE.value and framing == FramingScale.MEDIUM.value:
            tracking = Eligibility.CONDITIONALLY_ELIGIBLE
            calibration = Eligibility.CONDITIONALLY_ELIGIBLE
            identity = Eligibility.CONDITIONALLY_ELIGIBLE
            ball = Eligibility.CONDITIONALLY_ELIGIBLE
            live_event = Eligibility.CONDITIONALLY_ELIGIBLE
            physical = Eligibility.CONDITIONALLY_ELIGIBLE
            review = True
            _add_code(codes, "PARTIALLY_PLAYABLE_REVIEW")
        else:
            review = True
            _add_code(codes, "UNKNOWN_VIEW_REVIEW_REQUIRED")
            tracking = calibration = identity = ball = live_event = physical = Eligibility.UNKNOWN

    # --- Replay overlays (never auto live counting when unknown/confirmed) ---
    if replay in replay_confirmed:
        live_event = Eligibility.INELIGIBLE
        if physical in {Eligibility.ELIGIBLE, Eligibility.CONDITIONALLY_ELIGIBLE}:
            physical = Eligibility.INELIGIBLE
        _add_code(codes, "REPLAY_CONFIRMED_EXCLUDE_LIVE")
        _add_code(codes, "PHYSICAL_METRIC_UNSAFE")
    elif replay == ReplayStatus.UNKNOWN.value:
        # Visual tracking/identity may remain; live counting never auto-eligible.
        if live_event == Eligibility.ELIGIBLE:
            live_event = Eligibility.CONDITIONALLY_ELIGIBLE
        elif live_event not in {
            Eligibility.INELIGIBLE,
            Eligibility.UNKNOWN,
            Eligibility.CONDITIONALLY_ELIGIBLE,
        }:
            live_event = Eligibility.UNKNOWN
        review = True
        _add_code(codes, "REPLAY_UNKNOWN_BLOCK_LIVE_COUNTING")

    # --- Timeline mapping quality ---
    if mapping in unsafe_mapping:
        if physical in {
            Eligibility.ELIGIBLE,
            Eligibility.CONDITIONALLY_ELIGIBLE,
        }:
            physical = Eligibility.INELIGIBLE
        elif physical == Eligibility.UNKNOWN:
            physical = Eligibility.UNKNOWN
        _add_code(codes, "TIMELINE_MAPPING_INSUFFICIENT")
        _add_code(codes, "PHYSICAL_METRIC_UNSAFE")

    # Final safety clamps.
    if replay == ReplayStatus.UNKNOWN.value and live_event == Eligibility.ELIGIBLE:
        live_event = Eligibility.CONDITIONALLY_ELIGIBLE
        review = True
        _add_code(codes, "REPLAY_UNKNOWN_BLOCK_LIVE_COUNTING")

    # Allowed decision codes only from policy list (stable order as appended).
    allowed = set(policy["decision_codes"])
    codes = [c for c in codes if c in allowed]

    return {
        "run_id": window.run_id,
        "video_id": window.video_id,
        "analysis_window_id": analysis_window_id,
        "start_time_us": window.start_time_us,
        "end_time_us": window.end_time_us,
        "start_frame_index": window.start_frame_index,
        "end_frame_index_exclusive": window.end_frame_index_exclusive,
        "shot_id": window.shot_id,
        "camera_segment_ids": list(window.camera_segment_ids),
        "view_family": window.view_family,
        "framing_scale": window.framing_scale,
        "replay_status": window.replay_status,
        "graphics_status": window.graphics_status,
        "playability": window.playability,
        "tracking_eligibility": tracking.value,
        "calibration_eligibility": calibration.value,
        "identity_eligibility": identity.value,
        "ball_analysis_eligibility": ball.value,
        "live_event_eligibility": live_event.value,
        "physical_metric_eligibility": physical.value,
        "decision_codes": codes,
        "manual_review_required": bool(review),
        "coverage": float(window.coverage),
        "confidence": window.confidence,
        "timeline_mapping_quality": window.timeline_mapping_quality,
        "source_refs": list(window.source_refs),
        "policy_version": str(policy["policy_version"]),
        "provenance_json": None,
        "contract_version": CONTRACT_VERSION,
    }


def route_fused_windows(
    windows: Sequence[FusedWindow],
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, w in enumerate(sorted(windows, key=lambda x: (x.start_time_us, x.shot_id or ""))):
        wid = f"aw_{i:04d}"
        rows.append(route_fused_window(w, policy, analysis_window_id=wid))
    return rows


def build_review_queue(
    windows: Sequence[Mapping[str, Any]],
    *,
    policy_version: str,
    run_id: str | None = None,
    video_id: str | None = None,
) -> dict[str, Any]:
    """Build manual review queue from routed windows (deduped by window_id)."""
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in windows:
        if not row.get("manual_review_required"):
            continue
        wid = str(row["analysis_window_id"])
        if wid in seen:
            continue
        seen.add(wid)
        reasons = list(row.get("decision_codes") or [])
        if not reasons:
            reasons = ["UNKNOWN_VIEW_REVIEW_REQUIRED"]
        priority = "high"
        if any(
            c in reasons
            for c in (
                "CONFLICTING_CAMERA_LABELS",
                "CAMERA_GAP",
                "REPLAY_UNKNOWN_BLOCK_LIVE_COUNTING",
                "GRAPHICS_NON_PLAYABLE",
            )
        ):
            priority = "high"
        elif "LOW_COVERAGE" in reasons or "PARTIALLY_PLAYABLE_REVIEW" in reasons:
            priority = "medium"
        else:
            priority = "low"
        mid = (int(row["start_time_us"]) + int(row["end_time_us"])) // 2
        items.append(
            {
                "window_id": wid,
                "reason_codes": reasons,
                "priority": priority,
                "suggested_evidence_times": [
                    int(row["start_time_us"]),
                    mid,
                    int(row["end_time_us"]),
                ],
                "source_refs": list(row.get("source_refs") or []),
                "status": "pending",
            }
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "policy_version": policy_version,
        "items": items,
    }
    if run_id is not None:
        payload["run_id"] = run_id
    if video_id is not None:
        payload["video_id"] = video_id
    return payload


__all__ = [
    "CONFIG_VERSION",
    "TASK_AXES",
    "RoutingPolicyError",
    "load_routing_policy",
    "routing_policy_fingerprint",
    "default_routing_policy_path",
    "route_fused_window",
    "route_fused_windows",
    "build_review_queue",
]
