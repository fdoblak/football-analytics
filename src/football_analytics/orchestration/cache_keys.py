"""Deterministic cache keys for Stage 14 orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from football_analytics.core.hashing import hash_canonical_json


def stage_cache_key(
    *,
    stage_name: str,
    run_id: str,
    video_id: str,
    target_player_id: str,
    config_fingerprint: str,
    upstream_fingerprints: Mapping[str, str] | None = None,
    mode: str = "synthetic_fixture",
) -> str:
    payload: dict[str, Any] = {
        "stage": stage_name,
        "run_id": run_id,
        "video_id": video_id,
        "target_player_id": target_player_id,
        "config_fingerprint": config_fingerprint,
        "mode": mode,
        "upstream": dict(sorted((upstream_fingerprints or {}).items())),
    }
    return hash_canonical_json(payload)


def plan_fingerprint(stages: Sequence[Mapping[str, Any]], *, request_id: str, run_id: str) -> str:
    return hash_canonical_json(
        {
            "request_id": request_id,
            "run_id": run_id,
            "stages": [
                {
                    "name": s["name"],
                    "order": s["order"],
                    "cache_key": s["cache_key"],
                    "depends_on": list(s.get("depends_on") or []),
                    "mode": s.get("mode"),
                }
                for s in stages
            ],
        }
    )


__all__ = ["stage_cache_key", "plan_fingerprint"]
