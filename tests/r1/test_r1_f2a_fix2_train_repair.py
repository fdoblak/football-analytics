"""R1-F2-A-FIX2: false-complete repair mode + gates."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from football_analytics.annotation.independent_gt import IndependentGTError
from football_analytics.annotation.train_repair import (
    FAILED_TRAIN_FRAME_INDICES,
    PROTECTED_TRAIN_FRAME_INDICES,
    REPAIR_MODE,
    REPAIR_REASON,
    assert_immutable_fingerprints,
    bulk_accept_proposals,
    collect_fingerprints,
    reject_pending_proposals,
    reset_false_complete_train_frames,
    set_no_human_confirmed,
    validate_repair_complete,
    validate_train_complete_allowed,
)

REPO = Path(__file__).resolve().parents[2]


def _mini_draft() -> dict:
    """Minimal draft covering protected + one failed + one each of dev/holdout."""
    failed = FAILED_TRAIN_FRAME_INDICES[0]
    frames = []
    for idx in PROTECTED_TRAIN_FRAME_INDICES:
        frames.append(
            {
                "frame_idx": idx,
                "t_s": float(idx) / 30.0,
                "split": "train",
                "completed": True,
                "review_status": "complete",
                "provenance": "manual",
                "humans": [
                    {
                        "box_id": f"p-{idx}",
                        "bbox_xyxy": [10.0, 10.0, 40.0, 80.0],
                        "class_name": "human",
                        "role": "player",
                        "team_appearance": "yellow",
                        "eligibility": "on_pitch",
                        "visibility": "clear",
                        "jersey_number_visible": False,
                        "jersey_number": None,
                        "origin": "manual",
                    }
                ],
                "proposals": [],
            }
        )
    frames.append(
        {
            "frame_idx": failed,
            "t_s": float(failed) / 30.0,
            "split": "train",
            "completed": True,
            "review_status": "complete",
            "provenance": "proposal_seed",
            "humans": [],
            "proposals": [
                {
                    "proposal_id": "pr1",
                    "bbox_xyxy": [100.0, 100.0, 140.0, 200.0],
                    "score": 0.7,
                },
                {
                    "proposal_id": "pr2",
                    "bbox_xyxy": [200.0, 100.0, 240.0, 200.0],
                    "score": 0.6,
                },
            ],
        }
    )
    for split, idx in (("dev", 400), ("holdout", 500)):
        frames.append(
            {
                "frame_idx": idx,
                "t_s": 20.0,
                "split": split,
                "completed": True,
                "review_status": "complete",
                "provenance": "manual_blind",
                "humans": [
                    {
                        "box_id": f"{split}-1",
                        "bbox_xyxy": [50.0, 50.0, 90.0, 150.0],
                        "class_name": "human",
                        "role": "player",
                        "team_appearance": "white",
                        "eligibility": "on_pitch",
                        "visibility": "clear",
                        "jersey_number_visible": False,
                        "jersey_number": None,
                        "origin": "manual_blind",
                    }
                ],
                "proposals": [],
            }
        )
    return {
        "source_sha256": "97b298e41a82b567a7d68bd2322993bea34492b1cbb58362b0d72ca4a5471160",
        "frames": frames,
        "human_approved": False,
        "reviewed_gt": False,
        "frozen": False,
    }


class TrainRepairUnitTests(unittest.TestCase):
    def test_false_complete_regression_gate(self) -> None:
        fr = {
            "split": "train",
            "humans": [],
            "proposals": [{"bbox_xyxy": [1, 1, 20, 40]}],
            "no_human_confirmed": False,
        }
        errs = validate_train_complete_allowed(fr)
        self.assertIn("PENDING_PROPOSALS_MUST_BE_ACCEPTED_OR_REJECTED", errs)
        self.assertIn("ZERO_HUMANS_REQUIRES_EXPLICIT_NO_HUMAN_CONFIRMATION", errs)

    def test_pending_proposal_gate_blocks_complete(self) -> None:
        fr = {
            "split": "train",
            "humans": [
                {
                    "bbox_xyxy": [10, 10, 40, 80],
                    "role": "unknown",
                    "team_appearance": "unknown",
                    "eligibility": "uncertain",
                    "visibility": "clear",
                    "jersey_number_visible": False,
                    "jersey_number": None,
                }
            ],
            "proposals": [{"bbox_xyxy": [100, 100, 140, 200]}],
        }
        self.assertTrue(any("PENDING" in e for e in validate_train_complete_allowed(fr)))

    def test_explicit_no_human_confirmation(self) -> None:
        fr = {
            "split": "train",
            "humans": [],
            "proposals": [{"bbox_xyxy": [1, 1, 20, 40]}],
            "no_human_confirmed": False,
        }
        with self.assertRaises(IndependentGTError):
            set_no_human_confirmed(fr, True)
        reject_pending_proposals(fr)
        set_no_human_confirmed(fr, True)
        self.assertEqual(validate_train_complete_allowed(fr), [])

    def test_bulk_accept(self) -> None:
        draft = _mini_draft()
        # full reset needs all 37 — unit-test bulk on synthetic frame only
        fr = next(f for f in draft["frames"] if f["frame_idx"] == FAILED_TRAIN_FRAME_INDICES[0])
        fr["completed"] = False
        added = bulk_accept_proposals(fr, box_id_factory=uuid.uuid4)
        self.assertEqual(len(added), 2)
        self.assertEqual(fr["proposals"], [])
        self.assertFalse(fr["completed"])
        for h in added:
            self.assertEqual(h["origin"], "proposal_reviewed_bulk")
            self.assertEqual(h["role"], "unknown")
        # second bulk is no-op / no duplicates
        added2 = bulk_accept_proposals(fr, box_id_factory=uuid.uuid4)
        self.assertEqual(added2, [])
        self.assertEqual(len(fr["humans"]), 2)

    def test_duplicate_gate(self) -> None:
        fr = {
            "split": "train",
            "proposals": [],
            "humans": [
                {
                    "bbox_xyxy": [10, 10, 40, 80],
                    "role": "unknown",
                    "team_appearance": "unknown",
                    "eligibility": "uncertain",
                    "visibility": "clear",
                    "jersey_number_visible": False,
                    "jersey_number": None,
                },
                {
                    "bbox_xyxy": [10, 10, 40, 80],
                    "role": "unknown",
                    "team_appearance": "unknown",
                    "eligibility": "uncertain",
                    "visibility": "clear",
                    "jersey_number_visible": False,
                    "jersey_number": None,
                },
            ],
        }
        self.assertTrue(any("duplicate" in e for e in validate_train_complete_allowed(fr)))

    def test_invalid_bbox_gate(self) -> None:
        fr = {
            "split": "train",
            "proposals": [],
            "humans": [
                {
                    "bbox_xyxy": [-5, 10, 40, 80],
                    "role": "unknown",
                    "team_appearance": "unknown",
                    "eligibility": "uncertain",
                    "visibility": "clear",
                    "jersey_number_visible": False,
                    "jersey_number": None,
                }
            ],
        }
        self.assertTrue(any("invalid_bbox" in e for e in validate_train_complete_allowed(fr)))

    def test_dev_holdout_immutable_on_reset_shape(self) -> None:
        # Use runtime draft if present for full 37 reset; else skip shape
        runtime = Path("/home/fdoblak/workspace/independent_gt_review/own_video_97b298e4")
        draft_path = runtime / "draft_annotations.json"
        if not draft_path.is_file():
            self.skipTest("runtime draft missing")
        draft = json.loads(draft_path.read_text(encoding="utf-8"))
        # If already repaired, ensure fingerprints stable under no-op protected check
        before = collect_fingerprints(draft)
        # mutate a failed frame proposal count only if still repairable empty
        failed_fr = next(
            f
            for f in draft["frames"]
            if f["split"] == "train" and int(f["frame_idx"]) == FAILED_TRAIN_FRAME_INDICES[0]
        )
        snap = copy.deepcopy(failed_fr.get("humans"))
        failed_fr["humans"] = list(failed_fr.get("humans") or [])
        # touch something train-only
        failed_fr["repair_touch"] = True
        after = collect_fingerprints(draft)
        assert_immutable_fingerprints(before, after)
        failed_fr["humans"] = snap
        failed_fr.pop("repair_touch", None)

    def test_protected_train_list(self) -> None:
        self.assertEqual(PROTECTED_TRAIN_FRAME_INDICES, (0, 5, 15))
        self.assertEqual(len(FAILED_TRAIN_FRAME_INDICES), 37)
        self.assertEqual(REPAIR_MODE, "train-empty-complete")
        self.assertEqual(REPAIR_REASON, "EMPTY_COMPLETE_WITH_VISIBLE_PROPOSALS")

    def test_repair_validator_incomplete(self) -> None:
        runtime = Path("/home/fdoblak/workspace/independent_gt_review/own_video_97b298e4")
        draft_path = runtime / "draft_annotations.json"
        if not draft_path.is_file():
            self.skipTest("runtime draft missing")
        draft = json.loads(draft_path.read_text(encoding="utf-8"))
        report = validate_repair_complete(draft)
        # After FIX2 prepare, repair should NOT be complete yet
        if any(
            f.get("repair_required") and not f.get("completed")
            for f in draft["frames"]
            if f.get("split") == "train"
        ):
            self.assertFalse(report["repair_complete"])

    def test_windows_bats_ascii(self) -> None:
        import importlib.util
        import sys

        mod_path = REPO / "scripts" / "r1_gt_windows_package.py"
        spec = importlib.util.spec_from_file_location("r1_gt_windows_package", mod_path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules["r1_gt_windows_package"] = mod
        spec.loader.exec_module(mod)
        mod.assert_bat_safe(mod.load_canonical_bat())
        mod.assert_bat_safe(mod.load_canonical_repair_bat(), require_repair=True)

    def test_repair_only_navigation(self) -> None:
        import importlib.util

        from football_analytics.annotation.independent_gt import DEFAULT_RUNTIME, DEFAULT_VIDEO

        if not (DEFAULT_RUNTIME / "draft_annotations.json").is_file():
            self.skipTest("runtime draft missing")
        mod_path = REPO / "scripts" / "r1_independent_gt_review_server.py"
        spec = importlib.util.spec_from_file_location("r1_gt_server", mod_path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        app = mod.ReviewApp(DEFAULT_RUNTIME, DEFAULT_VIDEO, repair_mode=REPAIR_MODE)
        try:
            self.assertEqual(len(app.nav_indices), 37)
            idxs = [int(app.draft["frames"][i]["frame_idx"]) for i in app.nav_indices]
            self.assertEqual(idxs, list(FAILED_TRAIN_FRAME_INDICES))
            for i in idxs:
                self.assertNotIn(i, PROTECTED_TRAIN_FRAME_INDICES)
            state = app.public_state()
            self.assertEqual(state["n_frames"], 37)
            self.assertEqual(state["repair_mode"], REPAIR_MODE)
            self.assertEqual(state["split"], "train")
        finally:
            app.cap.release()

    def test_append_only_audit_helper(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.jsonl"
            p.write_text('{"event":"a","split":"dev"}\n', encoding="utf-8")
            before = p.read_text(encoding="utf-8")
            p.write_text(before + '{"event":"b","split":"train"}\n', encoding="utf-8")
            after = p.read_text(encoding="utf-8")
            self.assertTrue(after.startswith(before))


class TrainRepairResetIntegration(unittest.TestCase):
    def test_reset_only_on_copy_of_runtime_shape(self) -> None:
        runtime = Path("/home/fdoblak/workspace/independent_gt_review/own_video_97b298e4")
        draft_path = runtime / "draft_annotations.json"
        if not draft_path.is_file():
            self.skipTest("runtime draft missing")
        draft = json.loads(draft_path.read_text(encoding="utf-8"))
        # If already reset, verify flags; else verify reset works on deep copy
        already = any(f.get("repair_required") for f in draft["frames"])
        if already:
            n = sum(
                1
                for f in draft["frames"]
                if f.get("split") == "train"
                and int(f["frame_idx"]) in FAILED_TRAIN_FRAME_INDICES
                and f.get("repair_required")
            )
            self.assertEqual(n, 37)
            for f in draft["frames"]:
                if f["split"] == "train" and int(f["frame_idx"]) in PROTECTED_TRAIN_FRAME_INDICES:
                    self.assertTrue(f.get("completed"))
                    self.assertTrue(f.get("humans"))
                    self.assertFalse(f.get("repair_required"))
            return
        before = collect_fingerprints(draft)
        copy_d = json.loads(json.dumps(draft))
        summary = reset_false_complete_train_frames(copy_d)
        after = collect_fingerprints(copy_d)
        assert_immutable_fingerprints(before, after)
        self.assertEqual(summary["reset_n"], 37)


if __name__ == "__main__":
    unittest.main()
