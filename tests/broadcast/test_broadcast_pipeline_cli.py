"""CLI smoke + integration tests for Stage 4D broadcast integrate."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

import pyarrow as pa

from football_analytics.cli import main
from football_analytics.core.run_id import generate_run_id
from football_analytics.data.compiler import compile_arrow_schema, get_contract
from football_analytics.data.parquet import write_contract_parquet
from football_analytics.data.registry import default_project_root


def _cast(name: str, rows: list[dict[str, Any]]) -> Any:
    return pa.Table.from_pylist(rows, schema=compile_arrow_schema(get_contract(name, 1)))


def _write_bundle(root: Path, run_id: str) -> None:
    shots = [
        {
            "run_id": run_id,
            "video_id": "v1",
            "shot_id": "s1",
            "start_time_us": 0,
            "end_time_us": 200_000,
            "start_frame_index": None,
            "end_frame_index_exclusive": None,
            "start_boundary_id": None,
            "end_boundary_id": None,
            "duration_us": 200_000,
            "frame_count": None,
            "timeline_mapping_quality": "exact_identity",
            "segment_status": "active",
            "provenance_json": '{"o":"t"}',
            "contract_version": 1,
        }
    ]
    cams = [
        {
            "run_id": run_id,
            "video_id": "v1",
            "camera_segment_id": "c1",
            "shot_id": "s1",
            "start_time_us": 0,
            "end_time_us": 200_000,
            "start_frame_index": None,
            "end_frame_index_exclusive": None,
            "view_family": "main_broadcast",
            "framing_scale": "wide",
            "camera_position": "unknown",
            "camera_motion": "static",
            "replay_status": "live",
            "graphics_status": "none",
            "playability": "playable",
            "calibration_suitability": "suitable",
            "tracking_suitability": "suitable",
            "target_identity_suitability": "unknown",
            "classification_source": "manual",
            "confidence": 0.9,
            "coverage": 1.0,
            "review_status": "accepted",
            "evidence_refs": ["c1"],
            "provenance_json": '{"o":"t"}',
            "contract_version": 1,
        }
    ]
    bnds = [
        {
            "run_id": run_id,
            "video_id": "v1",
            "boundary_id": "b0",
            "boundary_time_us": 0,
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
    ]
    frames = [
        {
            "run_id": run_id,
            "video_id": "v1",
            "frame_index": i,
            "pts": i,
            "video_time_us": i * 40_000,
            "duration_us": 40_000,
            "is_key_frame": i == 0,
            "decode_status": "ok",
        }
        for i in range(6)
    ]
    write_contract_parquet(
        _cast("frames", frames),
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


class BroadcastPipelineCliTests(unittest.TestCase):
    def test_01_cli_integrate_smoke(self) -> None:
        run_id = generate_run_id()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_bundle(root, run_id)
            out = root / "out"
            out.mkdir()
            policy = default_project_root() / "configs/broadcast/broadcast_routing_policy.yaml"
            code = main(
                [
                    "broadcast",
                    "integrate",
                    "--timeline",
                    str(root / "frames.parquet"),
                    "--boundaries",
                    str(root / "boundaries.parquet"),
                    "--shots",
                    str(root / "shots.parquet"),
                    "--camera-views",
                    str(root / "cameras.parquet"),
                    "--output-dir",
                    str(out),
                    "--policy",
                    str(policy),
                    "--contain-root",
                    str(root),
                    "--run-id",
                    run_id,
                    "--video-id",
                    "v1",
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue((out / "analysis_windows.parquet").is_file())
            self.assertTrue((out / "review_queue.json").is_file())
            self.assertTrue((out / "pipeline_receipt.json").is_file())


class BroadcastPipelineIntegrationTests(unittest.TestCase):
    def test_01_end_to_end_coverage(self) -> None:
        from football_analytics.broadcast.broadcast_pipeline import run_broadcast_integrate
        from football_analytics.broadcast.playability import load_routing_policy
        from football_analytics.data.parquet import read_contract_parquet

        run_id = generate_run_id()
        policy = load_routing_policy(
            default_project_root() / "configs/broadcast/broadcast_routing_policy.yaml"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_bundle(root, run_id)
            out = root / "out"
            out.mkdir()
            res = run_broadcast_integrate(
                timeline=str(root / "frames.parquet"),
                boundaries=str(root / "boundaries.parquet"),
                shots=str(root / "shots.parquet"),
                camera_views=str(root / "cameras.parquet"),
                output_dir=str(out),
                policy=policy,
                contain_root=root,
                run_id=run_id,
                video_id="v1",
            )
            self.assertTrue(res.accepted)
            rows = read_contract_parquet(
                Path(str(res.analysis_windows_parquet)),
                get_contract("analysis_windows", 1),
                contain_root=root,
            ).to_pylist()
            covered = sum(int(r["end_time_us"]) - int(r["start_time_us"]) for r in rows)
            self.assertEqual(covered, 200_000)


if __name__ == "__main__":
    unittest.main()
