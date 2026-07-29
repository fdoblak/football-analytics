"""Tests for R1-F2-D active learning + blind holdout_v2 package."""

from __future__ import annotations

import importlib.util
import json
import socket
import subprocess
import threading
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

from football_analytics.annotation.active_learning_selection import candidate_pool
from football_analytics.annotation.evaluation_protocol_v2 import (
    EXPECTED_FROZEN_FP,
    HOLDOUT_V1_STATUS,
)
from football_analytics.annotation.evaluation_protocol_v3 import (
    PROTOCOL_ID,
    protocol_v3_definition,
)
from football_analytics.annotation.gt_freeze import DEFAULT_FROZEN_DIR
from football_analytics.annotation.holdout_v2_blind import (
    TEMPORAL_BUFFER,
    assert_no_holdout_v2_inference,
    select_blind_holdout_v2,
)

REPO = Path(__file__).resolve().parents[2]
EV = REPO / "artifacts" / "evidence" / "reboot_01" / "r1_f2d_active_learning"
WIN_BAT = (
    Path("/mnt/c/Users/furka/Desktop/Football Analytics Validation/R1 Independent GT")
    / "START_ACTIVE_LEARNING_REVIEW.bat"
)
REPO_BAT = REPO / "scripts" / "windows" / "START_ACTIVE_LEARNING_REVIEW.bat"
WRAPPER = REPO / "scripts" / "start_r1_active_learning_review.sh"
RUNTIME = Path("/home/fdoblak/workspace/independent_gt_review/own_video_97b298e4_active_learning")
VIDEO = Path("/home/fdoblak/football_data/videos/raw_matches/own_video_97b298e4/original.mp4")


def _load_server():
    path = REPO / "scripts" / "r1_independent_gt_review_server.py"
    spec = importlib.util.spec_from_file_location("r1_indep_gt_server_f2d", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _frozen() -> dict:
    return json.loads((DEFAULT_FROZEN_DIR / "annotations.json").read_text(encoding="utf-8"))


class ProtocolV3Tests(unittest.TestCase):
    def test_roles_and_holdout_v1_consumed(self) -> None:
        p = protocol_v3_definition()
        self.assertEqual(p["protocol_id"], PROTOCOL_ID)
        self.assertEqual(
            p["data_roles"]["consumed_historical_holdout_v1"]["status"],
            HOLDOUT_V1_STATUS,
        )
        self.assertFalse(p["data_roles"]["consumed_historical_holdout_v1"]["acceptance_reusable"])
        self.assertFalse(
            p["data_roles"]["consumed_historical_holdout_v1"]["may_produce_acceptance"]
        )
        self.assertTrue(p["data_roles"]["untouched_blind_holdout_v2"]["proposals_forbidden"])
        self.assertTrue(p["lineage_rules"]["holdout_v2_no_tuning"])
        self.assertEqual(p["frozen_gt_v1_fingerprint_required"], EXPECTED_FROZEN_FP)
        q = protocol_v3_definition()
        self.assertEqual(p["protocol_fingerprint"], q["protocol_fingerprint"])


class HoldoutV2BlindTests(unittest.TestCase):
    def test_selection_disjoint_buffer_no_detector(self) -> None:
        ann = _frozen()
        old = {int(f["frame_idx"]) for f in ann["frames"]}
        sel = select_blind_holdout_v2(ann, n_target=30)
        self.assertEqual(sel["n_frames"], 30)
        idxs = set(sel["frame_indices"])
        self.assertTrue(idxs.isdisjoint(old))
        for i in idxs:
            self.assertTrue(all(abs(i - u) >= TEMPORAL_BUFFER for u in old))
        self.assertFalse(sel["selection_method"]["uses_detector"])
        self.assertFalse(sel["selection_method"]["uses_confidence"])
        self.assertEqual(len(sel["selection_fingerprint"]), 64)

    def test_inference_guard(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_no_holdout_v2_inference([10, 20], {20, 30})


class ActiveLearningPoolTests(unittest.TestCase):
    def test_pool_excludes_old_and_holdout(self) -> None:
        pool = set(candidate_pool(old_80={0, 10}, holdout_v2={100, 200}))
        self.assertNotIn(0, pool)
        self.assertNotIn(10, pool)
        self.assertNotIn(100, pool)
        self.assertNotIn(101, pool)  # holdout clearance
        self.assertIn(50, pool)
        self.assertGreater(len(pool), 500)


class PackageEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (EV / "summary.json").is_file():
            raise unittest.SkipTest("F2-D evidence not prepared yet")

    def test_package_ok_and_leakage_free(self) -> None:
        summary = json.loads((EV / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(
            summary["gate"],
            "PASS — ACTIVE LEARNING AND NEW BLIND HOLDOUT REVIEW READY",
        )
        hold = json.loads((EV / "holdout_v2_selection.json").read_text(encoding="utf-8"))
        al = json.loads((EV / "active_learning_selection.json").read_text(encoding="utf-8"))
        self.assertEqual(hold["n_frames"], 30)
        self.assertLessEqual(al["n_frames"], 100)
        self.assertGreaterEqual(al["n_frames"], 40)
        self.assertTrue(set(al["frame_indices"]).isdisjoint(set(hold["frame_indices"])))
        draft = json.loads((RUNTIME / "draft_annotations.json").read_text(encoding="utf-8"))
        for fr in draft["frames"]:
            if fr.get("section") == "holdout_v2":
                self.assertEqual(fr.get("proposals"), [])
                self.assertEqual(fr.get("split"), "holdout")
        frozen = _frozen()
        self.assertEqual(frozen["canonical_fingerprint"], EXPECTED_FROZEN_FP)
        lineage = json.loads((EV / "holdout_v1_lineage.json").read_text(encoding="utf-8"))
        self.assertEqual(lineage["holdout_v1_status"], HOLDOUT_V1_STATUS)
        self.assertFalse(lineage["acceptance_reusable"])
        cleanup = json.loads((EV / "cleanup_receipt.json").read_text(encoding="utf-8"))
        self.assertIs(cleanup["data_loss"], False)
        plan = json.loads((EV / "next_detector_plan.json").read_text(encoding="utf-8"))
        self.assertEqual(plan["status"], "PLAN_ONLY_NO_TRAINING")
        self.assertTrue(plan["holdout_v2_tuning_forbidden"])
        self.assertEqual(len(plan["candidates"]), 3)

    def test_al_quota_targets_recorded(self) -> None:
        al = json.loads((EV / "active_learning_selection.json").read_text(encoding="utf-8"))
        counts = al["counts"]
        self.assertGreaterEqual(counts["small_distant_weighted"], 30)
        self.assertGreaterEqual(counts["crowded_occlusion"], 20)
        self.assertGreaterEqual(counts["hard_negative_fp"], 15)
        self.assertGreaterEqual(counts["goal_sideline"], 10)
        for fr in al["frames"]:
            self.assertTrue(fr["selection_reasons"])
            for r in fr["selection_reasons"]:
                self.assertIn(
                    r,
                    {
                        "disagreement",
                        "likely_false_negative",
                        "likely_false_positive",
                        "small_object",
                        "crowded",
                        "hard_negative",
                        "temporal_shift",
                        "coverage_fill",
                    },
                )


class LauncherTests(unittest.TestCase):
    def test_bat_ascii_crlf_health(self) -> None:
        for bat in (REPO_BAT, WIN_BAT):
            if not bat.is_file():
                raise unittest.SkipTest(f"bat missing: {bat}")
            raw = bat.read_bytes()
            self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
            self.assertIn(b"\r\n", raw)
            raw.decode("ascii")
            self.assertIn(b"Ubuntu-22.04", raw)
            self.assertIn(b"8768", raw)
            self.assertIn(b"active_learning", raw)
            self.assertIn(b"Browser was NOT opened", raw)
            self.assertNotIn(b"nohup", raw)
            self.assertNotIn(b"pkill", raw)

    def test_wrapper_bash(self) -> None:
        if not WRAPPER.is_file():
            raise unittest.SkipTest("wrapper missing")
        r = subprocess.run(
            ["bash", "-n", str(WRAPPER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("--active-learning", text)
        self.assertIn("8768", text)


class ActiveLearningServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (RUNTIME / "draft_annotations.json").is_file():
            raise unittest.SkipTest("AL runtime draft missing")
        cls.mod = _load_server()
        cls.runtime = RUNTIME
        cls.video = VIDEO

    def test_health_blind_holdout_and_predict_forbidden(self) -> None:
        app = self.mod.ReviewApp(self.runtime, self.video, active_learning=True)
        handler = self.mod.build_handler(app)
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        server = ThreadingHTTPServer(("127.0.0.1", port), handler)
        th = threading.Thread(target=server.serve_forever, daemon=True)
        th.start()
        try:
            c = HTTPConnection("127.0.0.1", port, timeout=5)
            c.request("GET", "/health")
            health = json.loads(c.getresponse().read().decode())
            self.assertEqual(health["status"], "ok")
            self.assertTrue(health["active_learning"])

            # Jump to a holdout_v2 frame
            hold_i = next(
                i for i, fr in enumerate(app.draft["frames"]) if fr.get("section") == "holdout_v2"
            )
            with app.lock:
                app.index = hold_i
                app.save()
            c.request("GET", "/api/state")
            st = json.loads(c.getresponse().read().decode())
            self.assertEqual(st["section"], "holdout_v2")
            self.assertTrue(st["blind"])
            self.assertEqual(st["proposals"], [])
            prog = st["progress"]
            self.assertIn("active_learning", prog)
            self.assertIn("holdout_v2", prog)

            c.request(
                "POST", "/api/predict", body=b"{}", headers={"Content-Type": "application/json"}
            )
            resp = c.getresponse()
            body = resp.read().decode()
            self.assertEqual(resp.status, 400)
            self.assertIn("FORBIDDEN", body.upper())

            c.request(
                "POST",
                "/api/accept_proposal",
                body=b'{"proposal_index":0}',
                headers={"Content-Type": "application/json"},
            )
            resp2 = c.getresponse()
            self.assertEqual(resp2.status, 400)
            self.assertIn("HOLDOUT_V2", resp2.read().decode().upper())
        finally:
            server.shutdown()


class FrozenImmutableTests(unittest.TestCase):
    def test_frozen_fingerprint_unchanged(self) -> None:
        ann = _frozen()
        self.assertTrue(ann.get("frozen"))
        self.assertEqual(ann["canonical_fingerprint"], EXPECTED_FROZEN_FP)


if __name__ == "__main__":
    unittest.main()
