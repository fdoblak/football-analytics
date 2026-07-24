"""Stage 12E synthetic fused pipeline fixtures."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from football_analytics.core.run_id import generate_run_id
from football_analytics.duels.aerial_fixtures import (
    clearance_with_evidence_fixture,
    long_ball_alone_fixture,
    monocular_aerial_fixture,
)
from football_analytics.duels.ground_fixtures import (
    contested_ground_fixture,
    nearest_switch_alone_fixture,
)
from football_analytics.duels.take_on_fixtures import (
    nearby_opponent_alone_fixture,
    successful_take_on_fixture,
)


def _merge(
    *,
    run_id: str | None,
    take_name: str | None,
    ground_name: str | None,
    aerial_name: str | None,
) -> dict[str, Any]:
    rid = run_id or generate_run_id()
    take_ctx: list[dict[str, Any]] = []
    ground_ctx: list[dict[str, Any]] = []
    aerial_ctx: list[dict[str, Any]] = []
    if take_name == "successful_take_on":
        take_ctx = list(successful_take_on_fixture(run_id=rid)["contexts"])
    elif take_name == "nearby_opponent_alone":
        take_ctx = list(nearby_opponent_alone_fixture(run_id=rid)["contexts"])
    if ground_name == "contested_ground":
        ground_ctx = list(contested_ground_fixture(run_id=rid)["contexts"])
    elif ground_name == "nearest_switch_alone":
        ground_ctx = list(nearest_switch_alone_fixture(run_id=rid)["contexts"])
    if aerial_name == "monocular_aerial":
        aerial_ctx = list(monocular_aerial_fixture(run_id=rid)["contexts"])
    elif aerial_name == "clearance_with_evidence":
        aerial_ctx = list(clearance_with_evidence_fixture(run_id=rid)["contexts"])
    elif aerial_name == "long_ball_alone":
        aerial_ctx = list(long_ball_alone_fixture(run_id=rid)["contexts"])
    elif aerial_name == "aerial_and_clearance":
        aerial_ctx = list(monocular_aerial_fixture(run_id=rid)["contexts"]) + list(
            clearance_with_evidence_fixture(run_id=rid)["contexts"]
        )
    return {
        "run_id": rid,
        "video_id": "video_synth_01",
        "take_on_contexts": take_ctx,
        "ground_contexts": ground_ctx,
        "aerial_contexts": aerial_ctx,
    }


def full_package_fixture(*, run_id: str | None = None) -> dict[str, Any]:
    return _merge(
        run_id=run_id,
        take_name="successful_take_on",
        ground_name="contested_ground",
        aerial_name="aerial_and_clearance",
    )


def nearby_and_switch_fixture(*, run_id: str | None = None) -> dict[str, Any]:
    return _merge(
        run_id=run_id,
        take_name="nearby_opponent_alone",
        ground_name="nearest_switch_alone",
        aerial_name="long_ball_alone",
    )


def take_on_only_fixture(*, run_id: str | None = None) -> dict[str, Any]:
    return _merge(
        run_id=run_id,
        take_name="successful_take_on",
        ground_name=None,
        aerial_name=None,
    )


def ground_only_fixture(*, run_id: str | None = None) -> dict[str, Any]:
    return _merge(
        run_id=run_id,
        take_name=None,
        ground_name="contested_ground",
        aerial_name=None,
    )


def aerial_only_fixture(*, run_id: str | None = None) -> dict[str, Any]:
    return _merge(
        run_id=run_id,
        take_name=None,
        ground_name=None,
        aerial_name="aerial_and_clearance",
    )


FIXTURES: dict[str, Callable[..., dict[str, Any]]] = {
    "full_package": full_package_fixture,
    "nearby_and_switch": nearby_and_switch_fixture,
    "take_on_only": take_on_only_fixture,
    "ground_only": ground_only_fixture,
    "aerial_only": aerial_only_fixture,
}


def load_pipeline_fixture(name: str, *, run_id: str | None = None) -> Mapping[str, Any]:
    if name not in FIXTURES:
        raise KeyError(f"unknown duels pipeline fixture: {name}")
    return FIXTURES[name](run_id=run_id)


__all__ = [
    "FIXTURES",
    "full_package_fixture",
    "nearby_and_switch_fixture",
    "take_on_only_fixture",
    "ground_only_fixture",
    "aerial_only_fixture",
    "load_pipeline_fixture",
]
