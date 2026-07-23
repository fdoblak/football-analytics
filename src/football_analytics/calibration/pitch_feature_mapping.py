"""Channel → NBJW name → optional canonical pitch feature mapping (Stage 8B).

Unknown / unverified mappings stay null canonical ids. Do not invent class labels.
NBJW `lines_list` index 7 (`Goal left post left `) retains a trailing space in the
source name; sanitized ids strip trailing whitespace for safe identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Exact NBJW lines_list order (23). Channel i (0-based after dropping bg) → name.
# Note trailing space on "Goal left post left ".
NBJW_LINES_LIST: tuple[str, ...] = (
    "Big rect. left bottom",
    "Big rect. left main",
    "Big rect. left top",
    "Big rect. right bottom",
    "Big rect. right main",
    "Big rect. right top",
    "Goal left crossbar",
    "Goal left post left ",
    "Goal left post right",
    "Goal right crossbar",
    "Goal right post left",
    "Goal right post right",
    "Middle line",
    "Side line bottom",
    "Side line left",
    "Side line right",
    "Side line top",
    "Small rect. left bottom",
    "Small rect. left main",
    "Small rect. left top",
    "Small rect. right bottom",
    "Small rect. right main",
    "Small rect. right top",
)

# Verified planar line → Stage 8A pitch_template line ids (105×68 default semantics).
# Goal posts/crossbars are non-planar / vertical — left unmatched (null canonical).
LINE_CANONICAL_MAP: dict[str, str] = {
    "Side line top": "touchline_left",
    "Side line bottom": "touchline_right",
    "Side line left": "goalline_a",
    "Side line right": "goalline_b",
    "Middle line": "halfway_line",
    "Big rect. left main": "pa_a_far",
    "Big rect. right main": "pa_b_far",
    "Small rect. left main": "ga_a_far",
    "Small rect. right main": "ga_b_far",
}

# SoccerNet / NBJW keypoint world coords (metres) for channels 1..57 (index 0 unused).
# Used only to match against Stage 8A template points when unambiguous.
NBJW_KP_WORLD_XY: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (52.5, 0.0),
    (105.0, 0.0),
    (0.0, 13.84),
    (16.5, 13.84),
    (88.5, 13.84),
    (105.0, 13.84),
    (0.0, 24.84),
    (5.5, 24.84),
    (99.5, 24.84),
    (105.0, 24.84),
    (0.0, 30.34),
    (0.0, 30.34),
    (105.0, 30.34),
    (105.0, 30.34),
    (0.0, 37.66),
    (0.0, 37.66),
    (105.0, 37.66),
    (105.0, 37.66),
    (0.0, 43.16),
    (5.5, 43.16),
    (99.5, 43.16),
    (105.0, 43.16),
    (0.0, 54.16),
    (16.5, 54.16),
    (88.5, 54.16),
    (105.0, 54.16),
    (0.0, 68.0),
    (52.5, 68.0),
    (105.0, 68.0),
    (16.5, 26.68),
    (52.5, 24.85),
    (88.5, 26.68),
    (16.5, 41.31),
    (52.5, 43.15),
    (88.5, 41.31),
    (19.99, 32.29),
    (43.68, 31.53),
    (61.31, 31.53),
    (85.0, 32.29),
    (19.99, 35.7),
    (43.68, 36.46),
    (61.31, 36.46),
    (85.0, 35.7),
    (11.0, 34.0),
    (16.5, 34.0),
    (20.15, 34.0),
    (46.03, 27.53),
    (58.97, 27.53),
    (43.35, 34.0),
    (52.5, 34.0),
    (61.5, 34.0),
    (46.03, 40.47),
    (58.97, 40.47),
    (84.85, 34.0),
    (88.5, 34.0),
    (94.0, 34.0),
)

# Ambiguous / circle / goal-post-height channels stay unmatched.
_KP_CANONICAL_BY_WORLD: dict[tuple[float, float], str] = {
    (0.0, 0.0): "corner_a_left",
    (105.0, 0.0): "corner_b_left",
    (0.0, 68.0): "corner_a_right",
    (105.0, 68.0): "corner_b_right",
    (52.5, 0.0): "halfway_left",
    (52.5, 68.0): "halfway_right",
    (52.5, 34.0): "centre_spot",
    (11.0, 34.0): "penalty_spot_a",
    (94.0, 34.0): "penalty_spot_b",
    (0.0, 13.84): "pa_a_near_left",
    (0.0, 54.16): "pa_a_near_right",
    (16.5, 13.84): "pa_a_far_left",
    (16.5, 54.16): "pa_a_far_right",
    (105.0, 13.84): "pa_b_near_left",
    (105.0, 54.16): "pa_b_near_right",
    (88.5, 13.84): "pa_b_far_left",
    (88.5, 54.16): "pa_b_far_right",
    (5.5, 24.84): "ga_a_far_left",
    (5.5, 43.16): "ga_a_far_right",
    (99.5, 24.84): "ga_b_far_left",
    (99.5, 43.16): "ga_b_far_right",
    (0.0, 30.34): "goal_a_left_post",
    (0.0, 37.66): "goal_a_right_post",
    (105.0, 30.34): "goal_b_left_post",
    (105.0, 37.66): "goal_b_right_post",
}


@dataclass(frozen=True)
class FeatureMapping:
    channel_index: int  # 0-based after background drop
    source_name: str
    sanitized_id: str
    canonical_pitch_feature_id: str | None
    mapped: bool


def sanitize_feature_token(name: str) -> str:
    """Map arbitrary NBJW names to SAFE_ID_RE tokens."""
    cleaned = name.strip().lower()
    out: list[str] = []
    for ch in cleaned:
        if ch.isalnum() or ch in {".", "_", "-"}:
            out.append(ch)
        elif ch in {" ", "/"}:
            out.append("_")
        else:
            out.append("_")
    token = "".join(out).strip("._-") or "unknown"
    while "__" in token:
        token = token.replace("__", "_")
    return token[:120]


def line_mapping(channel_index: int) -> FeatureMapping:
    if channel_index < 0 or channel_index >= len(NBJW_LINES_LIST):
        return FeatureMapping(
            channel_index=channel_index,
            source_name="unknown",
            sanitized_id=f"sv_line_unknown_{channel_index}",
            canonical_pitch_feature_id=None,
            mapped=False,
        )
    name = NBJW_LINES_LIST[channel_index]
    canonical = LINE_CANONICAL_MAP.get(name)  # exact key; trailing-space name misses map
    # Also try stripped key for goal post (still null — intentional).
    if canonical is None:
        canonical = LINE_CANONICAL_MAP.get(name.strip())
    return FeatureMapping(
        channel_index=channel_index,
        source_name=name,
        sanitized_id=f"sv_line_{channel_index + 1:02d}_{sanitize_feature_token(name)}",
        canonical_pitch_feature_id=canonical,
        mapped=canonical is not None,
    )


def keypoint_mapping(channel_index: int) -> FeatureMapping:
    """channel_index 0..56 → NBJW joints 1..57."""
    joint_id = channel_index + 1
    if channel_index < 0 or channel_index >= len(NBJW_KP_WORLD_XY):
        return FeatureMapping(
            channel_index=channel_index,
            source_name=f"sv_kp_{joint_id:02d}",
            sanitized_id=f"sv_kp_{joint_id:02d}",
            canonical_pitch_feature_id=None,
            mapped=False,
        )
    xy = NBJW_KP_WORLD_XY[channel_index]
    # Ambiguous duplicate world coords (goal posts appear twice) → leave unmatched
    # unless unique among all coords.
    count = sum(1 for p in NBJW_KP_WORLD_XY if p == xy)
    canonical = _KP_CANONICAL_BY_WORLD.get(xy) if count == 1 else None
    if count > 1:
        canonical = None
    return FeatureMapping(
        channel_index=channel_index,
        source_name=f"sv_kp_{joint_id:02d}",
        sanitized_id=f"sv_kp_{joint_id:02d}",
        canonical_pitch_feature_id=canonical,
        mapped=canonical is not None,
    )


def mapping_table_summary() -> dict[str, Any]:
    lines = [line_mapping(i) for i in range(len(NBJW_LINES_LIST))]
    kps = [keypoint_mapping(i) for i in range(len(NBJW_KP_WORLD_XY))]
    return {
        "lines_total": len(lines),
        "lines_mapped": sum(1 for x in lines if x.mapped),
        "keypoints_total": len(kps),
        "keypoints_mapped": sum(1 for x in kps if x.mapped),
        "trailing_space_line_name": "Goal left post left ",
        "note": "Unmapped channels keep canonical_pitch_feature_id=null",
    }


__all__ = [
    "NBJW_LINES_LIST",
    "LINE_CANONICAL_MAP",
    "NBJW_KP_WORLD_XY",
    "FeatureMapping",
    "sanitize_feature_token",
    "line_mapping",
    "keypoint_mapping",
    "mapping_table_summary",
]
