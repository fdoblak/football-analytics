"""Tests for independent football human GT (R1-F2-A)."""

from __future__ import annotations

import json
import socket
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path

from football_analytics.annotation.coordinates import (
    SOURCE_HEIGHT,
    SOURCE_WIDTH,
    CoordinateError,
    canvas_bbox_to_source,
    make_letterbox_transform,
    roundtrip_error_px,
    source_bbox_to_canvas,
)
from football_analytics.annotation.frame_selection import build_independent_gt_selection
from football_analytics.annotation.independent_gt import (
    EXPECTED_SOURCE_SHA256,
    IndependentGTError,
    append_audit_line,
    assert_no_prediction_leakage,
    atomic_write_json,
    empty_draft,
    soft_box_warnings,
    train_proposals_are_gt,
    validate_box_geometry,
    validate_freeze_ready,
    validate_metadata,
)


class FrameSelectionTests(unittest.TestCase):
    def test_counts_and_time_isolation(self) -> None:
        sel = build_independent_gt_selection()
        self.assertEqual(sel["counts"]["train"], 40)
        self.assertEqual(sel["counts"]["dev"], 20)
        self.assertEqual(sel["counts"]["holdout"], 20)
        self.assertEqual(sel["counts"]["total"], 80)
        for fr in sel["frames"]:
            t = fr["t_s"]
            if fr["split"] == "train":
                self.assertGreaterEqual(t, 0.0)
                self.assertLess(t, 12.0)
            elif fr["split"] == "dev":
                self.assertGreaterEqual(t, 12.0)
                self.assertLess(t, 22.0)
            else:
                self.assertGreaterEqual(t, 22.0)
        # neighbor gap: no stack of near-identical frames
        idxs = [f["frame_idx"] for f in sel["frames"] if f["split"] == "train"]
        gaps = [idxs[i + 1] - idxs[i] for i in range(len(idxs) - 1)]
        self.assertTrue(min(gaps) >= 4)
        # not a near-duplicate stack (median gap should be healthy)
        self.assertGreaterEqual(sorted(gaps)[len(gaps) // 2], 8)

    def test_coverage_categories_present(self) -> None:
        sel = build_independent_gt_selection()
        cats = {c for fr in sel["frames"] for c in fr["categories"]}
        for need in ("small_distant", "crowded", "goal_area", "sideline", "dark_clothing"):
            self.assertIn(need, cats)


class CoordinateRoundTripTests(unittest.TestCase):
    def test_source_canvas_source(self) -> None:
        t = make_letterbox_transform(SOURCE_WIDTH, SOURCE_HEIGHT, 960, 540)
        box = (100.0, 80.0, 160.0, 220.0)
        err = roundtrip_error_px(box, t)
        self.assertLessEqual(err, 1.0)
        c = source_bbox_to_canvas(box, t)
        back = canvas_bbox_to_source(c, t, source_w=SOURCE_WIDTH, source_h=SOURCE_HEIGHT)
        self.assertLessEqual(max(abs(a - b) for a, b in zip(box, back, strict=True)), 1.0)


class BBoxPolicyTests(unittest.TestCase):
    def test_invalid_zero_area(self) -> None:
        with self.assertRaises(CoordinateError):
            validate_box_geometry([10, 10, 10, 50])

    def test_oob_rejected(self) -> None:
        with self.assertRaises(CoordinateError):
            validate_box_geometry([-5, 0, 40, 80])

    def test_duplicate_warning(self) -> None:
        a = [10.0, 10.0, 50.0, 100.0]
        b = [11.0, 11.0, 51.0, 101.0]
        w = soft_box_warnings(a, [b])
        self.assertTrue(any(x.startswith("duplicate") for x in w))

    def test_metadata_allowlist(self) -> None:
        errs = validate_metadata(
            {
                "class_name": "human",
                "role": "player",
                "team_appearance": "yellow",
                "eligibility": "on_pitch",
                "visibility": "clear",
                "jersey_number_visible": False,
                "jersey_number": None,
            }
        )
        self.assertEqual(errs, [])
        bad = validate_metadata(
            {
                "class_name": "human",
                "role": "striker",
                "team_appearance": "yellow",
                "eligibility": "on_pitch",
                "visibility": "clear",
                "jersey_number_visible": False,
                "jersey_number": None,
            }
        )
        self.assertTrue(any("role" in e for e in bad))

    def test_jersey_no_hallucination(self) -> None:
        errs = validate_metadata(
            {
                "class_name": "human",
                "role": "player",
                "team_appearance": "white",
                "eligibility": "on_pitch",
                "visibility": "clear",
                "jersey_number_visible": False,
                "jersey_number": 5,
            }
        )
        self.assertTrue(any("jersey_number_must_be_null" in e for e in errs))


class LeakageAndFreezeTests(unittest.TestCase):
    def test_dev_proposal_leakage_hard_fail(self) -> None:
        fr = {"split": "dev", "proposals": [{"bbox_xyxy": [1, 2, 3, 4]}], "humans": []}
        with self.assertRaises(IndependentGTError):
            assert_no_prediction_leakage(fr, split="dev")

    def test_holdout_confidence_hard_fail(self) -> None:
        fr = {
            "split": "holdout",
            "proposals": [],
            "humans": [{"bbox_xyxy": [10, 10, 40, 80], "confidence": 0.9}],
        }
        with self.assertRaises(IndependentGTError):
            assert_no_prediction_leakage(fr, split="holdout")

    def test_train_unreviewed_proposal_not_gt(self) -> None:
        fr = {
            "humans": [{"origin": "proposal_unreviewed", "bbox_xyxy": [1, 2, 30, 40]}],
        }
        self.assertTrue(train_proposals_are_gt(fr))

    def test_freeze_without_user_approval_hard_fail(self) -> None:
        draft = empty_draft(
            video=Path("/tmp/x.mp4"),
            source_sha256=EXPECTED_SOURCE_SHA256,
            frames=[
                {
                    "frame_idx": 0,
                    "t_s": 0.0,
                    "split": "train",
                    "categories": [],
                    "proposals": [],
                }
            ],
        )
        report = validate_freeze_ready(
            draft, source_sha256=EXPECTED_SOURCE_SHA256, user_approved=False
        )
        self.assertFalse(report["freeze_allowed"])
        self.assertIn("USER_APPROVAL_REQUIRED", report["errors"])
        self.assertFalse(report["human_approved"])
        self.assertFalse(report["frozen"])

    def test_source_sha_mismatch_hard_fail(self) -> None:
        draft = empty_draft(
            video=Path("/tmp/x.mp4"),
            source_sha256="0" * 64,
            frames=[
                {
                    "frame_idx": 0,
                    "t_s": 0.0,
                    "split": "train",
                    "categories": [],
                    "proposals": [],
                }
            ],
        )
        report = validate_freeze_ready(draft, source_sha256="0" * 64, user_approved=True)
        self.assertIn("SOURCE_SHA256_MISMATCH", report["errors"])


class AtomicAuditTests(unittest.TestCase):
    def test_atomic_autosave_and_audit_append(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "draft.json"
            atomic_write_json(path, {"a": 1})
            self.assertTrue(path.is_file())
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["a"], 1)
            audit = root / "audit.jsonl"
            append_audit_line(audit, {"event": "t1"})
            append_audit_line(audit, {"event": "t2"})
            lines = audit.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["event"], "t1")


class LocalhostServerSmokeTests(unittest.TestCase):
    def test_bind_localhost_only_and_no_cdn(self) -> None:
        import importlib.util

        server_path = (
            Path(__file__).resolve().parents[2] / "scripts" / "r1_independent_gt_review_server.py"
        )
        spec = importlib.util.spec_from_file_location("r1_indep_gt_server", server_path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        HTML = mod.HTML
        self.assertNotIn("cdn.", HTML.lower())
        self.assertNotIn("googleapis", HTML.lower())
        self.assertNotIn("cloudflare", HTML.lower())

        runtime = Path("/home/fdoblak/workspace/independent_gt_review/own_video_97b298e4")
        if not (runtime / "draft_annotations.json").is_file():
            self.skipTest("runtime not prepared")
        video = Path(
            "/home/fdoblak/football_data/videos/raw_matches/own_video_97b298e4/original.mp4"
        )
        app = mod.ReviewApp(runtime, video)
        handler = mod.build_handler(app)
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        from http.server import ThreadingHTTPServer

        server = ThreadingHTTPServer(("127.0.0.1", port), handler)
        th = threading.Thread(target=server.serve_forever, daemon=True)
        th.start()
        try:
            conn = HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", "/api/state")
            resp = conn.getresponse()
            self.assertEqual(resp.status, 200)
            resp.read()
            hold_i = next(i for i, f in enumerate(app.draft["frames"]) if f["split"] == "holdout")
            app.index = hold_i
            conn.request("GET", "/api/state")
            resp = conn.getresponse()
            body = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(body["split"], "holdout")
            self.assertEqual(body["proposals"], [])
            self.assertTrue(body["blind"])
            conn.close()
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
