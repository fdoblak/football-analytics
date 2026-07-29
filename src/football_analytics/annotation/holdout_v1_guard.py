"""Guards: holdout_v1 must not be used for train/select/tune after F2-C."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from football_analytics.annotation.independent_gt import IndependentGTError

HOLDOUT_V1_CONSUMED = True


def assert_no_holdout_v1_for_development(
    frames: Sequence[Mapping[str, Any]] | None = None,
    *,
    split: str | None = None,
    purpose: str = "development",
) -> None:
    """Raise if code path attempts to use holdout_v1 for forbidden purposes."""
    forbidden = {
        "training",
        "model_selection",
        "threshold_tuning",
        "acceptance",
        "development",
    }
    if purpose in forbidden and split == "holdout":
        raise IndependentGTError(
            f"HOLDOUT_V1_ACCESS_DENIED:{purpose}:holdout_v1 is CONSUMED_FAILED_EVALUATION"
        )
    if frames is not None and purpose in {"training", "model_selection", "threshold_tuning"}:
        for fr in frames:
            if str(fr.get("split")) == "holdout":
                raise IndependentGTError("HOLDOUT_V1_ACCESS_DENIED:frame_in_forbidden_purpose")


def assert_dev_only_selection(split: str) -> None:
    if split != "dev":
        raise IndependentGTError(f"SELECTION_MUST_USE_DEV_ONLY:got={split}")


__all__ = [
    "HOLDOUT_V1_CONSUMED",
    "assert_dev_only_selection",
    "assert_no_holdout_v1_for_development",
]
