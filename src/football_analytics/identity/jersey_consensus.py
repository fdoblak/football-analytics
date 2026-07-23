"""Tracklet-level jersey number consensus (Stage 7D)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class JerseyObservationVote:
    track_id: int
    frame_index: int
    observation_id: int
    raw_text: str
    normalized_number: int | None
    quality: float
    score: float | None
    status: str


@dataclass(frozen=True)
class TrackJerseyConsensus:
    track_id: int
    status: str
    raw_text: str | None
    normalized_number: int | None
    digit_count: int | None
    vote_weight: float
    margin: float
    observation_count: int
    temporal_spread: int
    observation_ids: tuple[int, ...]
    reason_codes: tuple[str, ...]
    quality_flags: tuple[str, ...]
    review_required: bool


def _vote_key(raw_text: str, normalized: int | None) -> str:
    # Prefer raw_text so leading zeros stay distinct from single-digit forms.
    return f"raw:{raw_text}|norm:{normalized if normalized is not None else 'null'}"


def consensus_for_track(
    votes: Sequence[JerseyObservationVote],
    *,
    config: Mapping[str, Any],
) -> TrackJerseyConsensus:
    """Quality-weighted consensus; conflict/insufficient → ambiguous/unknown."""
    cons = config["consensus"]
    if not votes:
        return TrackJerseyConsensus(
            track_id=-1,
            status="unknown",
            raw_text=None,
            normalized_number=None,
            digit_count=None,
            vote_weight=0.0,
            margin=0.0,
            observation_count=0,
            temporal_spread=0,
            observation_ids=(),
            reason_codes=("NO_VOTES",),
            quality_flags=("no_digits",),
            review_required=False,
        )
    track_id = int(votes[0].track_id)
    observed = [v for v in votes if v.status == "observed" and v.raw_text]
    frames = sorted({int(v.frame_index) for v in observed})
    spread = (frames[-1] - frames[0]) if len(frames) >= 2 else (1 if frames else 0)
    obs_ids = tuple(sorted(int(v.observation_id) for v in observed))

    if len(observed) < int(cons["min_observations"]):
        return TrackJerseyConsensus(
            track_id=track_id,
            status="ambiguous",
            raw_text=None,
            normalized_number=None,
            digit_count=None,
            vote_weight=0.0,
            margin=0.0,
            observation_count=len(observed),
            temporal_spread=spread,
            observation_ids=obs_ids,
            reason_codes=("INSUFFICIENT_OBSERVATIONS",),
            quality_flags=("ambiguous", "weak_single_observation"),
            review_required=True,
        )
    min_spread = int(cons["min_temporal_spread_frames"])
    if spread < min_spread and len(frames) >= 2:
        # Prefer abstain when temporal spread is below config minimum.
        return TrackJerseyConsensus(
            track_id=track_id,
            status="ambiguous",
            raw_text=None,
            normalized_number=None,
            digit_count=None,
            vote_weight=0.0,
            margin=0.0,
            observation_count=len(observed),
            temporal_spread=spread,
            observation_ids=obs_ids,
            reason_codes=("INSUFFICIENT_TEMPORAL_SPREAD",),
            quality_flags=("ambiguous",),
            review_required=True,
        )

    power = float(cons["quality_weight_power"])
    weights: dict[str, float] = defaultdict(float)
    exemplars: dict[str, JerseyObservationVote] = {}
    for v in observed:
        key = _vote_key(v.raw_text, v.normalized_number)
        score = float(v.score) if v.score is not None else 0.5
        w = (max(0.0, float(v.quality)) ** power) * max(0.0, score)
        weights[key] += w
        if key not in exemplars:
            exemplars[key] = v
            continue
        prev = exemplars[key]
        prev_score = float(prev.score) if prev.score is not None else 0.5
        prev_w = (float(prev.quality) ** power) * prev_score
        if w > prev_w:
            exemplars[key] = v

    ranked = sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))
    best_key, best_w = ranked[0]
    second_w = ranked[1][1] if len(ranked) > 1 else 0.0
    total = sum(weights.values()) or 1.0
    margin = (best_w - second_w) / total

    # Conflict: multiple distinct numbers with non-trivial weight.
    distinct_raws = {v.raw_text for v in observed}
    if len(ranked) >= 2 and margin < float(cons["min_winning_margin"]):
        return TrackJerseyConsensus(
            track_id=track_id,
            status="ambiguous",
            raw_text=None,
            normalized_number=None,
            digit_count=None,
            vote_weight=float(best_w),
            margin=float(margin),
            observation_count=len(observed),
            temporal_spread=spread,
            observation_ids=obs_ids,
            reason_codes=("CONSENSUS_MARGIN_LOW", "NUMBER_CONFLICT"),
            quality_flags=("ambiguous", "conflict"),
            review_required=True,
        )
    if (
        cons["conflict_on_switch"]
        and len(distinct_raws) > 1
        and margin < float(cons["min_winning_margin"]) * 2
    ):
        return TrackJerseyConsensus(
            track_id=track_id,
            status="ambiguous",
            raw_text=None,
            normalized_number=None,
            digit_count=None,
            vote_weight=float(best_w),
            margin=float(margin),
            observation_count=len(observed),
            temporal_spread=spread,
            observation_ids=obs_ids,
            reason_codes=("TRACK_NUMBER_SWITCH",),
            quality_flags=("ambiguous", "conflict"),
            review_required=True,
        )

    winner = exemplars[best_key]
    digit_count = len(winner.raw_text) if winner.raw_text else None
    return TrackJerseyConsensus(
        track_id=track_id,
        status="observed",
        raw_text=winner.raw_text,
        normalized_number=winner.normalized_number,
        digit_count=digit_count,
        vote_weight=float(best_w),
        margin=float(margin),
        observation_count=len(observed),
        temporal_spread=spread,
        observation_ids=obs_ids,
        reason_codes=("CONSENSUS_OK",),
        quality_flags=("consensus", f"vote_weight:{best_w:.4f}", f"margin:{margin:.4f}"),
        review_required=False,
    )


def build_track_consensus(
    votes: Sequence[JerseyObservationVote],
    *,
    config: Mapping[str, Any],
) -> list[TrackJerseyConsensus]:
    by_track: dict[int, list[JerseyObservationVote]] = defaultdict(list)
    for v in votes:
        by_track[int(v.track_id)].append(v)
    out: list[TrackJerseyConsensus] = []
    for tid in sorted(by_track):
        out.append(consensus_for_track(by_track[tid], config=config))
    return out


__all__ = [
    "JerseyObservationVote",
    "TrackJerseyConsensus",
    "consensus_for_track",
    "build_track_consensus",
]
