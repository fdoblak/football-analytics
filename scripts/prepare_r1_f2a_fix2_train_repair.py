#!/usr/bin/env python3
"""R1-F2-A-FIX2: reset false-complete train frames and prepare repair mode.

Does NOT modify dev/holdout annotations. Does NOT freeze or fine-tune.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_analytics.annotation.independent_gt import (
    DEFAULT_RUNTIME,
    DEFAULT_VIDEO,
    EXPECTED_SOURCE_SHA256,
    append_audit_line,
    atomic_write_json,
    sha256_file,
    utc_now,
)
from football_analytics.annotation.train_repair import (
    FAILED_TRAIN_FRAME_INDICES,
    PROTECTED_TRAIN_FRAME_INDICES,
    REPAIR_MODE,
    assert_immutable_fingerprints,
    collect_fingerprints,
    reset_false_complete_train_frames,
)

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    ap.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    ap.add_argument(
        "--windows-dir",
        type=Path,
        default=Path("/mnt/c/Users/furka/Desktop/Football Analytics Validation/R1 Independent GT"),
    )
    args = ap.parse_args()

    digest = sha256_file(args.video)
    if digest != EXPECTED_SOURCE_SHA256:
        raise SystemExit(f"SOURCE_SHA_MISMATCH:{digest}")

    draft_path = args.runtime / "draft_annotations.json"
    audit_path = args.runtime / "review_audit.jsonl"
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    if draft.get("source_sha256") != digest:
        raise SystemExit("DRAFT_SOURCE_SHA_MISMATCH")

    audit_before = audit_path.read_text(encoding="utf-8") if audit_path.is_file() else ""
    before = collect_fingerprints(draft)
    summary = reset_false_complete_train_frames(draft)
    after = collect_fingerprints(draft)
    assert_immutable_fingerprints(before, after)

    # provenance / blind policy must remain
    for fr in draft["frames"]:
        if fr["split"] in {"dev", "holdout"}:
            for h in fr.get("humans") or []:
                if h.get("origin") != "manual_blind":
                    raise SystemExit(f"DEV_HOLDOUT_ORIGIN_CHANGED:{fr['frame_idx']}")

    atomic_write_json(draft_path, draft)
    append_audit_line(
        audit_path,
        {
            "event": "fix2_reset_false_complete_train",
            "repair_mode": REPAIR_MODE,
            "reset_n": summary["reset_n"],
            "failed_indices": summary["failed_indices"],
            "protected_indices": summary["protected_indices"],
            "fingerprints_before": before,
            "fingerprints_after": after,
            "ts": utc_now(),
        },
    )
    audit_after = audit_path.read_text(encoding="utf-8")
    if not audit_after.startswith(audit_before):
        raise SystemExit("AUDIT_NOT_APPEND_ONLY")
    # split audit lineage: prior dev/holdout events must still be present verbatim
    for line in audit_before.splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        if ev.get("split") in {"dev", "holdout"} and line not in audit_after:
            raise SystemExit(f"AUDIT_LINEAGE_LOST:{ev.get('split')}")

    # progress recompute
    by = {
        "train": {"n": 0, "complete": 0},
        "dev": {"n": 0, "complete": 0},
        "holdout": {"n": 0, "complete": 0},
    }
    n_complete = 0
    repair_pending = 0
    for fr in draft["frames"]:
        by[fr["split"]]["n"] += 1
        if fr.get("completed"):
            by[fr["split"]]["complete"] += 1
            n_complete += 1
        if fr.get("repair_required") and not fr.get("completed"):
            repair_pending += 1
    atomic_write_json(
        args.runtime / "progress.json",
        {
            "schema": "independent_gt_progress_v1",
            "n_frames": len(draft["frames"]),
            "n_complete": n_complete,
            "by_split": by,
            "repair": {
                "mode": REPAIR_MODE,
                "target_n": len(FAILED_TRAIN_FRAME_INDICES),
                "pending_n": repair_pending,
                "protected_train": list(PROTECTED_TRAIN_FRAME_INDICES),
            },
            "updated_at_utc": utc_now(),
        },
    )

    # repo evidence (small)
    ev = REPO / "artifacts/evidence/reboot_01/r1_f2a_fix2_train_repair"
    ev.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        ev / "repair_manifest.json",
        {
            "schema": "r1_f2a_fix2_repair_manifest_v1",
            "repair_mode": REPAIR_MODE,
            "failed_train_frame_indices": list(FAILED_TRAIN_FRAME_INDICES),
            "protected_train_frame_indices": list(PROTECTED_TRAIN_FRAME_INDICES),
            "fingerprints": after,
            "fingerprints_before": before,
            "root_cause": ("UI allowed Complete while pending proposals remained and humans=[]"),
            "dev_holdout_immutable": True,
            "written_at_utc": utc_now(),
        },
        mode=0o644,
    )
    atomic_write_json(
        REPO / "artifacts/evidence/reboot_01/GATE_STATUS.json",
        {
            "schema": "r1_f2a_fix2_gate_status_v1",
            "gate": "PASS — TRAIN ANNOTATION REPAIR READY",
            "acceptance_eligible": False,
            "human_approved": False,
            "reviewed_gt": False,
            "frozen": False,
            "repair_mode": REPAIR_MODE,
            "repair_target_n": len(FAILED_TRAIN_FRAME_INDICES),
            "written_at_utc": utc_now(),
        },
        mode=0o644,
    )

    # Windows package with repair launcher
    import importlib.util
    import sys

    mod_path = Path(__file__).resolve().parent / "r1_gt_windows_package.py"
    spec = importlib.util.spec_from_file_location("r1_gt_windows_package", mod_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["r1_gt_windows_package"] = mod
    spec.loader.exec_module(mod)
    info = mod.write_windows_package(args.windows_dir, include_repair=True)
    print(
        json.dumps(
            {
                "gate": "PASS — TRAIN ANNOTATION REPAIR READY",
                "reset": summary,
                "fingerprints": after,
                "windows": info,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
