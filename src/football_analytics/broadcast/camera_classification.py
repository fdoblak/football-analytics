"""Per-sample multi-axis camera-view classification and shot aggregation."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from football_analytics.broadcast.camera_features import CameraSampleFeatures
from football_analytics.broadcast.types import (
    CameraMotion,
    CameraPosition,
    CameraViewSegment,
    ClassificationSource,
    FramingScale,
    GraphicsStatus,
    Playability,
    ReplayStatus,
    ReviewStatus,
    Suitability,
    ViewFamily,
)


class CameraClassificationError(ValueError):
    """Classification failure."""


@dataclass(frozen=True)
class AxisDecision:
    label: str
    heuristic_score: float
    abstained: bool
    scores: Mapping[str, float]


@dataclass(frozen=True)
class SampleDecision:
    frame_index: int
    time_us: int
    view_family: AxisDecision
    framing_scale: AxisDecision
    camera_motion: AxisDecision
    graphics_status: AxisDecision
    playability: AxisDecision
    ood_like: bool


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _pick_label(
    scores: Mapping[str, float],
    *,
    abstain_margin: float,
    min_score: float,
    unknown_label: str = "unknown",
) -> AxisDecision:
    if not scores:
        return AxisDecision(unknown_label, 0.0, True, {})
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    best_label, best = ordered[0]
    second = ordered[1][1] if len(ordered) > 1 else 0.0
    if best < min_score or (best - second) < abstain_margin:
        return AxisDecision(unknown_label, _clamp01(best), True, dict(scores))
    return AxisDecision(best_label, _clamp01(best), False, dict(scores))


def _classify_view(f: CameraSampleFeatures, cfg: Mapping[str, Any]) -> AxisDecision:
    th = cfg["thresholds"]["view"]
    ab = float(cfg["thresholds"]["abstention"]["min_heuristic_score"])
    pitch = f.pitch_green_fraction
    spread = f.pitch_spatial_spread
    overlay = f.overlay_high_contrast_fraction
    skin = f.skin_like_fraction

    main = 0.0
    if pitch >= float(th["main_broadcast_pitch_min"]):
        main = 0.55 + 0.45 * _clamp01((pitch - float(th["main_broadcast_pitch_min"])) / 0.4)
        if spread >= float(th["main_broadcast_spread_min"]):
            main = min(1.0, main + 0.15)
        main *= max(0.0, 1.0 - overlay)

    iso = 0.0
    if (
        skin >= float(th["player_isolation_skin_min"])
        and pitch <= float(th["player_isolation_pitch_max"])
        and overlay < 0.35
    ):
        iso = 0.55 + 0.45 * _clamp01(skin / 0.2)
        iso *= max(0.2, 1.0 - pitch * 3.0)

    graphics = 0.0
    if overlay >= float(th["graphics_overlay_min"]) and pitch <= float(th["graphics_pitch_max"]):
        graphics = 0.55 + 0.45 * _clamp01((overlay - float(th["graphics_overlay_min"])) / 0.4)

    return _pick_label(
        {
            "main_broadcast": main,
            "player_isolation": iso,
            "graphics": graphics,
        },
        abstain_margin=float(th["abstain_margin"]),
        min_score=ab,
    )


def _classify_framing(f: CameraSampleFeatures, cfg: Mapping[str, Any]) -> AxisDecision:
    th = cfg["thresholds"]["framing"]
    ab = float(cfg["thresholds"]["abstention"]["min_heuristic_score"])
    pitch = f.pitch_green_fraction
    skin = f.skin_like_fraction

    wide = 0.0
    medium = 0.0
    close_up = 0.0
    if pitch >= float(th["wide_pitch_min"]):
        wide = 0.6 + 0.4 * _clamp01((pitch - float(th["wide_pitch_min"])) / 0.35)
    if float(th["medium_pitch_min"]) <= pitch < float(th["wide_pitch_min"]):
        medium = 0.6 + 0.4 * _clamp01(
            (pitch - float(th["medium_pitch_min"]))
            / max(1e-6, float(th["wide_pitch_min"]) - float(th["medium_pitch_min"]))
        )
    if pitch <= float(th["close_up_pitch_max"]) and skin >= float(th["close_up_skin_min"]):
        close_up = 0.6 + 0.4 * _clamp01(skin / 0.2)
    # Low pitch without skin evidence → abstain (do not invent close_up)
    return _pick_label(
        {"wide": wide, "medium": medium, "close_up": close_up},
        abstain_margin=float(th["abstain_margin"]),
        min_score=ab,
    )


def _classify_graphics(f: CameraSampleFeatures, cfg: Mapping[str, Any]) -> AxisDecision:
    th = cfg["thresholds"]["graphics"]
    ab = float(cfg["thresholds"]["abstention"]["min_heuristic_score"])
    overlay = f.overlay_high_contrast_fraction
    pitch = f.pitch_green_fraction
    skin = f.skin_like_fraction

    none_s = 0.0
    partial = 0.0
    dominant = 0.0
    full = 0.0

    # Player-isolation skin blobs can look like mild overlay — prefer none.
    if skin >= 0.04 and pitch <= 0.12 and overlay < float(th["partial_overlay_max"]):
        none_s = 0.75
        return _pick_label(
            {
                "none": none_s,
                "partial_overlay": 0.0,
                "dominant_overlay": 0.0,
                "full_screen": 0.0,
            },
            abstain_margin=float(th["abstain_margin"]),
            min_score=ab,
        )

    if overlay <= float(th["none_overlay_max"]):
        none_s = 0.7 + 0.3 * _clamp01(1.0 - overlay / max(1e-6, float(th["none_overlay_max"])))
    elif overlay <= float(th["partial_overlay_max"]):
        partial = 0.65
    elif overlay <= float(th["dominant_overlay_max"]):
        dominant = 0.65 + 0.2 * _clamp01(
            (overlay - float(th["partial_overlay_max"]))
            / max(1e-6, float(th["dominant_overlay_max"]) - float(th["partial_overlay_max"]))
        )
    else:
        # overlay above dominant_max
        dominant = 0.55

    # Full-screen: very high overlay with little pitch (allow small false-green leak)
    if overlay >= float(th["full_screen_overlay_min"]) and pitch <= float(
        th["full_screen_pitch_max"]
    ):
        full = 0.7 + 0.3 * _clamp01((overlay - float(th["full_screen_overlay_min"])) / 0.4)
        dominant = 0.0

    return _pick_label(
        {
            "none": none_s,
            "partial_overlay": partial,
            "dominant_overlay": dominant,
            "full_screen": full,
        },
        abstain_margin=float(th["abstain_margin"]),
        min_score=ab,
    )


def _classify_motion(f: CameraSampleFeatures, cfg: Mapping[str, Any]) -> AxisDecision:
    th = cfg["thresholds"]["motion"]
    ab = float(cfg["thresholds"]["abstention"]["min_heuristic_score"])
    diff = f.frame_diff_mean
    flow = f.flow_mag_mean
    flow_std = f.flow_mag_std
    horiz = f.flow_horizontal_ratio
    radial = f.flow_radial_consistency

    static = 0.0
    pan = 0.0
    zoom = 0.0
    compound = 0.0
    unstable = 0.0

    if diff <= float(th["static_diff_max"]) and flow <= float(th["static_flow_max"]):
        static = 0.75 + 0.25 * _clamp01(1.0 - flow / max(1e-6, float(th["static_flow_max"])))

    if flow >= float(th["pan_flow_min"]) and horiz >= float(th["pan_horizontal_ratio_min"]):
        pan = 0.6 + 0.4 * _clamp01((horiz - 0.5) / 0.5)

    if flow >= float(th["zoom_flow_min"]) and radial >= float(th["zoom_radial_min"]):
        zoom = 0.6 + 0.4 * _clamp01(radial)

    if (
        flow >= float(th["compound_flow_min"])
        and pan < 0.7
        and zoom < 0.7
        and flow_std > 1.0
        and diff > float(th["static_diff_max"])
    ):
        compound = 0.55 + 0.3 * _clamp01(flow / 5.0)

    if diff >= float(th["unstable_diff_min"]) or flow_std >= float(th["unstable_flow_std_min"]):
        unstable = 0.6 + 0.4 * _clamp01(max(diff, flow_std / 5.0))

    return _pick_label(
        {
            "static": static,
            "pan": pan,
            "zoom": zoom,
            "compound": compound,
            "unstable": unstable,
        },
        abstain_margin=float(th["abstain_margin"]),
        min_score=ab,
    )


def _ood_like(f: CameraSampleFeatures, cfg: Mapping[str, Any]) -> bool:
    th = cfg["thresholds"]["abstention"]
    # Crowd/noise OOD: little pitch, textured, not a clear full UI, not a skin blob.
    return (
        f.pitch_green_fraction <= float(th["ood_pitch_max"])
        and f.hist_entropy >= float(th["ood_entropy_min"])
        and f.edge_density >= float(th["ood_edge_min"])
        and f.skin_like_fraction < 0.08
        and f.overlay_high_contrast_fraction < 0.50
        and f.mean_luma > 0.15
        and f.mean_luma < 0.75
    )


def _playability_from_axes(
    view: AxisDecision,
    graphics: AxisDecision,
    framing: AxisDecision,
    *,
    ood: bool,
    cfg: Mapping[str, Any],
) -> AxisDecision:
    ab = float(cfg["thresholds"]["abstention"]["min_heuristic_score"])
    del ab  # reserved for future per-axis playability abstention
    if ood or view.label == "unknown":
        return AxisDecision("uncertain", 0.5, True, {"uncertain": 0.5})
    if graphics.label in {"full_screen", "dominant_overlay"} or view.label == "graphics":
        return AxisDecision(
            "non_playable",
            max(graphics.heuristic_score, 0.7),
            False,
            {"non_playable": 0.9},
        )
    if view.label == "player_isolation" or framing.label == "close_up":
        return AxisDecision(
            "partially_playable",
            0.7,
            False,
            {"partially_playable": 0.7},
        )
    if graphics.label == "partial_overlay" and view.label == "main_broadcast":
        return AxisDecision(
            "partially_playable",
            0.65,
            False,
            {"partially_playable": 0.65},
        )
    if view.label == "main_broadcast" and framing.label in {"wide", "medium"}:
        return AxisDecision("playable", 0.75, False, {"playable": 0.75})
    return AxisDecision("uncertain", 0.4, True, {"uncertain": 0.4})


def classify_sample(features: CameraSampleFeatures, config: Mapping[str, Any]) -> SampleDecision:
    view = _classify_view(features, config)
    framing = _classify_framing(features, config)
    graphics = _classify_graphics(features, config)
    motion = _classify_motion(features, config)
    ood = _ood_like(features, config)
    # Graphics-dominant frames: do not invent framing / force graphics view
    if (not ood) and graphics.label in {"full_screen", "dominant_overlay"}:
        view = AxisDecision(
            "graphics",
            max(view.heuristic_score, graphics.heuristic_score),
            False,
            dict(view.scores),
        )
        framing = AxisDecision("unknown", framing.heuristic_score, True, dict(framing.scores))
    elif (not ood) and view.label == "graphics":
        framing = AxisDecision("unknown", framing.heuristic_score, True, dict(framing.scores))
    if ood:
        view = AxisDecision("unknown", view.heuristic_score, True, dict(view.scores))
        framing = AxisDecision("unknown", framing.heuristic_score, True, dict(framing.scores))
        graphics = AxisDecision("unknown", graphics.heuristic_score, True, dict(graphics.scores))
        motion = AxisDecision("unknown", motion.heuristic_score, True, dict(motion.scores))
    playability = _playability_from_axes(view, graphics, framing, ood=ood, cfg=config)
    return SampleDecision(
        frame_index=features.frame_index,
        time_us=features.time_us,
        view_family=view,
        framing_scale=framing,
        camera_motion=motion,
        graphics_status=graphics,
        playability=playability,
        ood_like=ood,
    )


def _aggregate_axis(
    decisions: Sequence[AxisDecision],
    *,
    config: Mapping[str, Any],
    unknown_label: str = "unknown",
) -> tuple[str, float, float, dict[str, Any]]:
    """Aggregate via coverage + disagreement (not naive majority alone)."""
    agg = config["thresholds"]["aggregation"]
    min_frac = float(agg["min_label_fraction"])
    max_dis = float(agg["max_disagreement"])
    labels = [d.label for d in decisions]
    n = len(labels)
    if n == 0:
        return unknown_label, 0.0, 0.0, {"reason": "empty"}
    counts = Counter(labels)
    # Prefer non-unknown when competing
    ranked = sorted(
        counts.items(),
        key=lambda kv: (
            0 if kv[0] == unknown_label else 1,
            kv[1],
            -len(kv[0]),
            kv[0],
        ),
        reverse=True,
    )
    best_label, best_count = ranked[0]
    frac = best_count / n
    disagreement = 1.0 - frac
    scores = [d.heuristic_score for d in decisions if d.label == best_label]
    mean_score = sum(scores) / len(scores) if scores else 0.0
    coverage = frac
    meta = {
        "counts": dict(counts),
        "fraction": frac,
        "disagreement": disagreement,
        "n": n,
    }
    if best_label == unknown_label or frac < min_frac or disagreement > max_dis:
        return unknown_label, mean_score, coverage, {**meta, "abstained": True}
    return best_label, mean_score, coverage, {**meta, "abstained": False}


def _derive_suitability(
    *,
    view: str,
    framing: str,
    graphics: str,
    playability: str,
    coverage: float,
    config: Mapping[str, Any],
) -> tuple[Suitability, Suitability, Suitability, list[str]]:
    rules: list[str] = []
    cal = Suitability.UNKNOWN
    trk = Suitability.UNKNOWN
    ident = Suitability.UNKNOWN
    min_cov = float(config["min_coverage"])

    if graphics in {"full_screen", "dominant_overlay"} or view == "graphics":
        cal = Suitability.UNSUITABLE
        trk = Suitability.UNSUITABLE
        ident = Suitability.UNSUITABLE
        rules.append("S4C-UNSUIT-GRAPHICS")
        return cal, trk, ident, rules

    if coverage < min_cov or view == "unknown" or playability in {"uncertain", "non_playable"}:
        if playability == "non_playable":
            cal = Suitability.UNSUITABLE
            trk = Suitability.UNSUITABLE
            ident = Suitability.UNSUITABLE
        else:
            cal = Suitability.UNKNOWN
            trk = Suitability.UNKNOWN
            ident = Suitability.UNKNOWN
        rules.append("S4C-UNSUIT-UNKNOWN")
        return cal, trk, ident, rules

    if view == "player_isolation" or framing == "close_up":
        cal = Suitability.UNSUITABLE
        trk = Suitability.UNSUITABLE
        ident = Suitability.CONDITIONALLY_SUITABLE
        rules.append("S4C-ID-CLOSE-ISO")
        rules.append("S4C-UNSUIT-ISO-TRACK-CAL")
        return cal, trk, ident, rules

    if (
        view == "main_broadcast"
        and framing == "wide"
        and playability in {"playable", "partially_playable"}
        and graphics in {"none", "partial_overlay", "unknown"}
    ):
        cal = Suitability.SUITABLE
        trk = Suitability.SUITABLE
        ident = Suitability.CONDITIONALLY_SUITABLE
        rules.append("S4C-CAL-WIDE-MAIN")
        rules.append("S4C-TRK-WIDE-MAIN")
        return cal, trk, ident, rules

    if view == "main_broadcast" and framing == "medium":
        cal = Suitability.CONDITIONALLY_SUITABLE
        trk = Suitability.CONDITIONALLY_SUITABLE
        ident = Suitability.CONDITIONALLY_SUITABLE
        rules.append("S4C-CAL-WIDE-MAIN")
        return cal, trk, ident, rules

    rules.append("S4C-UNSUIT-UNKNOWN")
    return Suitability.UNKNOWN, Suitability.UNKNOWN, Suitability.UNKNOWN, rules


def _playability_enum(label: str) -> Playability:
    return Playability(label if label in {e.value for e in Playability} else "uncertain")


def aggregate_shot_classification(
    sample_decisions: Sequence[SampleDecision],
    *,
    run_id: str,
    video_id: str,
    shot_id: str,
    camera_segment_id: str,
    start_time_us: int,
    end_time_us: int,
    start_frame_index: int | None,
    end_frame_index_exclusive: int | None,
    config: Mapping[str, Any],
    config_fingerprint: str,
) -> CameraViewSegment:
    """One camera_view_segment per shot (documented Stage 4C limitation)."""
    if not sample_decisions:
        raise CameraClassificationError("no sample decisions")

    view_l, view_s, view_c, view_m = _aggregate_axis(
        [d.view_family for d in sample_decisions], config=config
    )
    fram_l, fram_s, fram_c, fram_m = _aggregate_axis(
        [d.framing_scale for d in sample_decisions], config=config
    )
    mot_l, mot_s, mot_c, mot_m = _aggregate_axis(
        [d.camera_motion for d in sample_decisions], config=config
    )
    gfx_l, gfx_s, gfx_c, gfx_m = _aggregate_axis(
        [d.graphics_status for d in sample_decisions], config=config
    )
    play_l, play_s, play_c, play_m = _aggregate_axis(
        [d.playability for d in sample_decisions],
        config=config,
        unknown_label="uncertain",
    )

    # Overall coverage: mean of axis coverages (classified sample fraction)
    coverage = float(min(1.0, (view_c + fram_c + mot_c + gfx_c + play_c) / 5.0))
    if coverage < float(config["min_coverage"]):
        # Force conservative unknowns / uncertain
        if view_l != "unknown":
            view_l = "unknown"
        if fram_l != "unknown":
            fram_l = "unknown"
        play_l = "uncertain"

    ood_rate = sum(1 for d in sample_decisions if d.ood_like) / len(sample_decisions)
    if ood_rate >= 0.5:
        view_l = "unknown"
        fram_l = "unknown"
        play_l = "uncertain"

    cal, trk, ident, rule_ids = _derive_suitability(
        view=view_l,
        framing=fram_l,
        graphics=gfx_l,
        playability=play_l,
        coverage=coverage,
        config=config,
    )

    heuristic = float(
        min(
            1.0,
            (view_s + fram_s + mot_s + gfx_s + play_s) / 5.0,
        )
    )
    provenance = {
        "classifier": "camera_view_baseline",
        "feature_version": int(config["feature_version"]),
        "config_fingerprint": config_fingerprint,
        "limitation": "one_camera_view_segment_per_shot",
        "heuristic_score": round(heuristic, 6),
        "sample_count": len(sample_decisions),
        "ood_rate": round(ood_rate, 6),
        "axes": {
            "view_family": {"label": view_l, "meta": view_m, "heuristic_score": round(view_s, 6)},
            "framing_scale": {"label": fram_l, "meta": fram_m, "heuristic_score": round(fram_s, 6)},
            "camera_motion": {"label": mot_l, "meta": mot_m, "heuristic_score": round(mot_s, 6)},
            "graphics_status": {"label": gfx_l, "meta": gfx_m, "heuristic_score": round(gfx_s, 6)},
            "playability": {"label": play_l, "meta": play_m, "heuristic_score": round(play_s, 6)},
        },
        "always_unknown": ["camera_position", "replay_status"],
        "suitability_rule_ids": rule_ids,
        "sample_frame_indices": [d.frame_index for d in sample_decisions],
    }

    return CameraViewSegment(
        run_id=run_id,
        video_id=video_id,
        camera_segment_id=camera_segment_id,
        shot_id=shot_id,
        start_time_us=start_time_us,
        end_time_us=end_time_us,
        start_frame_index=start_frame_index,
        end_frame_index_exclusive=end_frame_index_exclusive,
        view_family=ViewFamily(view_l),
        framing_scale=FramingScale(fram_l),
        camera_position=CameraPosition.UNKNOWN,
        camera_motion=CameraMotion(mot_l),
        replay_status=ReplayStatus.UNKNOWN,
        graphics_status=GraphicsStatus(gfx_l),
        playability=_playability_enum(play_l),
        calibration_suitability=cal,
        tracking_suitability=trk,
        target_identity_suitability=ident,
        classification_source=ClassificationSource.RULE,
        confidence=None,
        coverage=coverage,
        review_status=ReviewStatus.UNREVIEWED,
        evidence_refs=tuple(f"sample_{d.frame_index}" for d in sample_decisions),
        provenance_json=json.dumps(provenance, sort_keys=True, separators=(",", ":")),
    )


__all__ = [
    "CameraClassificationError",
    "AxisDecision",
    "SampleDecision",
    "classify_sample",
    "aggregate_shot_classification",
]
