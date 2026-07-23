"""Pipeline and evaluation tests for Stage 4D."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

import pyarrow as pa

from football_analytics.broadcast.broadcast_evaluation import (
    evaluate_broadcast_windows,
    passes_safety_gates,
)
from football_analytics.broadcast.broadcast_pipeline import run_broadcast_integrate
from football_analytics.broadcast.playability import load_routing_policy
from football_analytics.core.run_id import generate_run_id
from football_analytics.data.compiler import compile_arrow_schema, get_contract
from football_analytics.data.parquet import read_contract_parquet, write_contract_parquet
from football_analytics.data.registry import default_project_root


def _cast(name: str, rows: list[dict[str, Any]]) -> Any:
    return pa.Table.from_pylist(rows, schema=compile_arrow_schema(get_contract(name, 1)))


def _shot(run_id: str, start: int, end: int, shot_id: str = "s1") -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": "v1",
        "shot_id": shot_id,
        "start_time_us": start,
        "end_time_us": end,
        "start_frame_index": None,
        "end_frame_index_exclusive": None,
        "start_boundary_id": None,
        "end_boundary_id": None,
        "duration_us": end - start,
        "frame_count": None,
        "timeline_mapping_quality": "exact_identity",
        "segment_status": "active",
        "provenance_json": '{"o":"t"}',
        "contract_version": 1,
    }


def _cam(run_id: str, start: int, end: int, **kwargs: Any) -> dict[str, Any]:
    row = {
        "run_id": run_id,
        "video_id": "v1",
        "camera_segment_id": kwargs.get("camera_segment_id", "c1"),
        "shot_id": kwargs.get("shot_id", "s1"),
        "start_time_us": start,
        "end_time_us": end,
        "start_frame_index": None,
        "end_frame_index_exclusive": None,
        "view_family": kwargs.get("view_family", "main_broadcast"),
        "framing_scale": kwargs.get("framing_scale", "wide"),
        "camera_position": "unknown",
        "camera_motion": "static",
        "replay_status": kwargs.get("replay_status", "live"),
        "graphics_status": kwargs.get("graphics_status", "none"),
        "playability": kwargs.get("playability", "playable"),
        "calibration_suitability": "suitable",
        "tracking_suitability": "suitable",
        "target_identity_suitability": "unknown",
        "classification_source": "manual",
        "confidence": 0.9,
        "coverage": kwargs.get("coverage", 1.0),
        "review_status": "accepted",
        "evidence_refs": [kwargs.get("camera_segment_id", "c1")],
        "provenance_json": '{"o":"t"}',
        "contract_version": 1,
    }
    return row


def _bnd(run_id: str, t: int, bid: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_id": "v1",
        "boundary_id": bid,
        "boundary_time_us": t,
        "left_frame_index": None,
        "right_frame_index": None,
        "transition_type": "hard_cut",
        "transition_duration_us": 0,
        "confidence": 1.0,
        "detection_source": "manual",
        "evidence_ref": None,
        "review_status": "accepted",
        "provenance_json": '{"o":"t"}',
        "contract_version": 1,
    }


def _frames(run_id: str, end_us: int = 1_000_000) -> list[dict[str, Any]]:
    rows = []
    t = 0
    i = 0
    while t < end_us:
        rows.append(
            {
                "run_id": run_id,
                "video_id": "v1",
                "frame_index": i,
                "pts": i,
                "video_time_us": t,
                "duration_us": 40_000,
                "is_key_frame": i == 0,
                "decode_status": "ok",
            }
        )
        i += 1
        t += 40_000
    return rows


class BroadcastPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.run_id = generate_run_id()
        self.policy = load_routing_policy(
            default_project_root() / "configs/broadcast/broadcast_routing_policy.yaml"
        )

    def test_01_pipeline_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shots = [_shot(self.run_id, 0, 400_000)]
            cams = [_cam(self.run_id, 0, 400_000)]
            bnds = [_bnd(self.run_id, 0, "b0"), _bnd(self.run_id, 400_000, "b1")]
            write_contract_parquet(
                _cast("frames", _frames(self.run_id)),
                root / "frames.parquet",
                get_contract("frames", 1),
                contain_root=root,
            )
            write_contract_parquet(
                _cast("shot_boundaries", bnds),
                root / "boundaries.parquet",
                get_contract("shot_boundaries", 1),
                contain_root=root,
            )
            write_contract_parquet(
                _cast("shot_segments", shots),
                root / "shots.parquet",
                get_contract("shot_segments", 1),
                contain_root=root,
            )
            write_contract_parquet(
                _cast("camera_view_segments", cams),
                root / "cameras.parquet",
                get_contract("camera_view_segments", 1),
                contain_root=root,
            )
            out = root / "out"
            out.mkdir()
            res = run_broadcast_integrate(
                timeline=str(root / "frames.parquet"),
                boundaries=str(root / "boundaries.parquet"),
                shots=str(root / "shots.parquet"),
                camera_views=str(root / "cameras.parquet"),
                output_dir=str(out),
                policy=self.policy,
                contain_root=root,
                run_id=self.run_id,
                video_id="v1",
            )
            self.assertTrue(res.accepted, res.error_code)
            self.assertTrue(Path(str(res.analysis_windows_parquet)).is_file())
            self.assertTrue(Path(str(res.review_queue_json)).is_file())
            self.assertTrue(Path(str(res.pipeline_receipt_json)).is_file())
            rows = read_contract_parquet(
                Path(str(res.analysis_windows_parquet)),
                get_contract("analysis_windows", 1),
                contain_root=root,
            ).to_pylist()
            self.assertGreaterEqual(len(rows), 1)
            self.assertEqual(rows[0]["tracking_eligibility"], "eligible")

    def test_02_evaluation_safety_zero_fp(self) -> None:
        pred = [
            {
                "analysis_window_id": "aw_0000",
                "start_time_us": 0,
                "end_time_us": 100,
                "playability": "non_playable",
                "replay_status": "unknown",
                "tracking_eligibility": "ineligible",
                "calibration_eligibility": "ineligible",
                "identity_eligibility": "ineligible",
                "ball_analysis_eligibility": "ineligible",
                "live_event_eligibility": "unknown",
                "physical_metric_eligibility": "ineligible",
                "decision_codes": ["GRAPHICS_NON_PLAYABLE", "REPLAY_UNKNOWN_BLOCK_LIVE_COUNTING"],
                "manual_review_required": True,
            }
        ]
        report = evaluate_broadcast_windows(pred, pred, repeat_predictions=pred)
        ok, fails = passes_safety_gates(report)
        self.assertTrue(ok, fails)
        self.assertEqual(report.unsafe_live_event_false_positive_rate, 0.0)
        self.assertTrue(report.deterministic_repeat)


class BroadcastEvaluationExtraTests(unittest.TestCase):
    def test_01_detects_live_fp(self) -> None:
        pred = [
            {
                "analysis_window_id": "aw_0000",
                "start_time_us": 0,
                "end_time_us": 10,
                "playability": "playable",
                "replay_status": "unknown",
                "tracking_eligibility": "eligible",
                "calibration_eligibility": "eligible",
                "identity_eligibility": "unknown",
                "ball_analysis_eligibility": "eligible",
                "live_event_eligibility": "eligible",
                "physical_metric_eligibility": "eligible",
                "decision_codes": [],
                "manual_review_required": False,
            }
        ]
        gt = [
            {
                **pred[0],
                "live_event_eligibility": "unknown",
                "manual_review_required": True,
                "decision_codes": ["REPLAY_UNKNOWN_BLOCK_LIVE_COUNTING"],
            }
        ]
        report = evaluate_broadcast_windows(pred, gt)
        self.assertGreater(report.unsafe_live_event_false_positive_rate or 0, 0)
        ok, _ = passes_safety_gates(report)
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
