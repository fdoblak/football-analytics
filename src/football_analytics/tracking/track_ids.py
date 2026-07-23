"""Deterministic track ID allocation (run_id + video_id namespace; no reuse)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from football_analytics.core.run_id import validate_run_id
from football_analytics.data.types import assert_safe_identifier
from football_analytics.tracking.types import TrackingContractError


def allocate_hash_track_id(
    *,
    run_id: str,
    video_id: str,
    seed: str,
    namespace_version: int = 1,
) -> int:
    """Canonical hash-based non-negative int64 ID (deterministic; injectable seed)."""
    validate_run_id(run_id)
    assert_safe_identifier(video_id)
    if not seed:
        raise TrackingContractError("seed must be non-empty")
    payload = f"v{namespace_version}|{run_id}|{video_id}|{seed}".encode()
    digest = hashlib.sha256(payload).digest()
    # Keep in signed int64 non-negative range
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


@dataclass
class TrackIdAllocator:
    """Sequential IDs scoped to run_id+video_id; reuse forbidden within namespace."""

    run_id: str
    video_id: str
    namespace_version: int = 1
    start: int = 0
    _next: int = field(init=False, default=0)
    _issued: set[int] = field(init=False, default_factory=set)

    def __post_init__(self) -> None:
        validate_run_id(self.run_id)
        assert_safe_identifier(self.video_id)
        if self.start < 0:
            raise TrackingContractError("start must be >= 0")
        self._next = int(self.start)

    def allocate(self) -> int:
        tid = self._next
        if tid in self._issued:
            raise TrackingContractError(f"track_id reuse forbidden: {tid}")
        self._issued.add(tid)
        self._next = tid + 1
        return tid

    def register_external(self, track_id: int) -> None:
        if track_id < 0:
            raise TrackingContractError("track_id must be >= 0")
        if track_id in self._issued:
            raise TrackingContractError(f"track_id reuse forbidden: {track_id}")
        self._issued.add(track_id)
        if track_id >= self._next:
            self._next = track_id + 1

    def contains(self, track_id: int) -> bool:
        return track_id in self._issued

    @property
    def issued(self) -> frozenset[int]:
        return frozenset(self._issued)


__all__ = ["TrackIdAllocator", "allocate_hash_track_id"]
