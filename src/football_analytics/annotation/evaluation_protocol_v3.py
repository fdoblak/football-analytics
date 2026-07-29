"""Evaluation Protocol v3 — lineage for AL recovery + blind holdout_v2.

Defines data roles before any new inference. Holdout_v1 remains consumed.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from football_analytics.annotation.evaluation_protocol_v2 import (
    EXPECTED_FROZEN_FP,
    HOLDOUT_V1_STATUS,
)
from football_analytics.annotation.independent_gt import utc_now

PROTOCOL_ID = "own_video_human_eval_protocol_v3"
PRIMARY_SCOPE = "on_pitch_human"


def protocol_v3_definition() -> dict[str, Any]:
    """Fingerprintable protocol body. Call before holdout_v2 selection / AL inference."""
    body: dict[str, Any] = {
        "schema": "evaluation_protocol_v3",
        "protocol_id": PROTOCOL_ID,
        "frozen_gt_v1_fingerprint_required": EXPECTED_FROZEN_FP,
        "primary_scope": PRIMARY_SCOPE,
        "inherits_from": "own_video_human_eval_protocol_v2",
        "ignore_eligibility": ["uncertain"],
        "ignore_prediction_iou": 0.5,
        "match_iou": 0.5,
        "matching": "greedy_one_to_one",
        "data_roles": {
            "development_data": {
                "description": (
                    "Frozen train40+dev20 GT plus historical holdout_v1 annotations "
                    "usable only for error analysis / AL candidate scoring"
                ),
                "may_use_for_training": "train_split_only",
                "may_use_for_threshold_tuning": "dev_split_only",
                "may_use_for_final_acceptance": False,
            },
            "active_learning_training_additions": {
                "description": "New unlabeled frames selected for human review → future train",
                "may_use_for_training": True,
                "may_use_for_final_acceptance": False,
                "proposals_allowed_during_review": True,
            },
            "dev_calibration_set": {
                "description": "Frozen dev 20 frames for fusion/conf selection only",
                "may_use_for_threshold_tuning": True,
                "may_use_for_final_acceptance": False,
            },
            "untouched_blind_holdout_v2": {
                "description": "New 30-frame blind holdout; no proposals; one-shot later",
                "may_use_for_training": False,
                "may_use_for_threshold_tuning": False,
                "may_use_for_model_selection": False,
                "may_use_for_final_acceptance": True,
                "proposals_forbidden": True,
                "inference_during_review_forbidden": True,
            },
            "consumed_historical_holdout_v1": {
                "status": HOLDOUT_V1_STATUS,
                "acceptance_reusable": False,
                "may_use_for_error_analysis": True,
                "may_use_for_training": False,
                "may_use_for_model_selection": False,
                "may_use_for_threshold_tuning": False,
                "may_produce_acceptance": False,
            },
        },
        "lineage_rules": {
            "holdout_v1_not_acceptance": True,
            "holdout_v2_locked_before_training": True,
            "holdout_v2_no_tuning": True,
            "al_and_holdout_v2_disjoint": True,
            "frozen_gt_v1_immutable": True,
            "defined_before_inference": True,
        },
        "gt_mutation_forbidden_for_frozen_v1": True,
        "written_at_utc": utc_now(),
    }
    body["protocol_fingerprint"] = hashlib.sha256(
        json.dumps(
            {k: v for k, v in body.items() if k not in {"written_at_utc", "protocol_fingerprint"}},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return body


__all__ = [
    "EXPECTED_FROZEN_FP",
    "HOLDOUT_V1_STATUS",
    "PRIMARY_SCOPE",
    "PROTOCOL_ID",
    "protocol_v3_definition",
]
