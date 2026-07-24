"""Hard-fail leakage validator for Stage 16 prediction vs reference GT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_analytics.acceptance.contracts import (
    EXTERNAL_CC_BY_REFERENCE_GT,
    LEAKAGE_SEPARATION_VIOLATION,
    NAMESPACE_EVALUATION,
    NAMESPACE_PREDICTIONS,
    NAMESPACE_REFERENCE_GT,
)


class LeakageError(RuntimeError):
    def __init__(self, message: str) -> None:
        super().__init__(f"{LEAKAGE_SEPARATION_VIOLATION}: {message}")
        self.error_code = LEAKAGE_SEPARATION_VIOLATION


FORBIDDEN_PRED_MARKERS = (
    EXTERNAL_CC_BY_REFERENCE_GT,
    "reference_ground_truth",
    "gsr_player_observation",
    "bas_reference_event",
)


def assert_namespace_layout(run_dir: Path) -> None:
    run_dir = Path(run_dir)
    for name in (
        NAMESPACE_PREDICTIONS,
        NAMESPACE_REFERENCE_GT,
        NAMESPACE_EVALUATION,
    ):
        (run_dir / name).mkdir(parents=True, exist_ok=True)


def validate_no_gt_under_predictions(run_dir: Path) -> None:
    pred = Path(run_dir) / NAMESPACE_PREDICTIONS
    if not pred.exists():
        return
    for path in pred.rglob("*"):
        if not path.is_file():
            continue
        rel = str(path.relative_to(pred)).lower()
        if "reference_ground_truth" in rel or "gsr" in rel or "bas_gt" in rel:
            raise LeakageError(f"GT-like path under predictions/: {rel}")
        if path.suffix.lower() in {".json", ".jsonl", ".parquet", ".csv"}:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")[:200_000]
            except OSError:
                continue
            for marker in FORBIDDEN_PRED_MARKERS:
                if marker in text:
                    raise LeakageError(f"marker {marker!r} in predictions file {rel}")


def validate_prediction_bundle_not_gt(bundle: dict[str, Any]) -> None:
    provenance = str(bundle.get("provenance") or bundle.get("annotation_provenance") or "")
    if provenance == EXTERNAL_CC_BY_REFERENCE_GT:
        raise LeakageError("prediction bundle marked as external GT provenance")
    origin = str(bundle.get("metric_origin") or "")
    if origin in {"external_gt_copy", "ground_truth_injection"}:
        raise LeakageError("prediction metric_origin indicates GT injection")


def validate_event_ledger_not_copied_from_gt(
    predicted_events: list[dict[str, Any]],
    reference_events: list[dict[str, Any]],
) -> None:
    """Fail if predicted ledger is an exact identity copy of reference events."""
    if not predicted_events or not reference_events:
        return
    if len(predicted_events) != len(reference_events):
        return
    pred_keys = [
        (
            e.get("label"),
            e.get("t_ms"),
            e.get("player_id"),
            e.get("source"),
        )
        for e in predicted_events
    ]
    ref_keys = [
        (
            e.get("label"),
            e.get("t_ms"),
            e.get("player_id"),
            EXTERNAL_CC_BY_REFERENCE_GT,
        )
        for e in reference_events
    ]
    # If every predicted event claims GT provenance or mirrors ref exactly with GT source
    gt_sourced = sum(1 for e in predicted_events if e.get("source") == EXTERNAL_CC_BY_REFERENCE_GT)
    if gt_sourced == len(predicted_events) and predicted_events:
        raise LeakageError("predicted event ledger entirely sourced from external GT")
    if pred_keys == [(a, b, c, EXTERNAL_CC_BY_REFERENCE_GT) for a, b, c, _ in ref_keys]:
        raise LeakageError("predicted events identical to reference GT copies")


def validate_run_dir(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    assert_namespace_layout(run_dir)
    validate_no_gt_under_predictions(run_dir)
    receipt = {
        "status": "PASS",
        "run_dir": str(run_dir),
        "checked": [
            NAMESPACE_PREDICTIONS,
            NAMESPACE_REFERENCE_GT,
            NAMESPACE_EVALUATION,
        ],
    }
    out = run_dir / NAMESPACE_EVALUATION / "leakage_validation.json"
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt
