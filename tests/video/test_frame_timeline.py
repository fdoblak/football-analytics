"""Tests for Stage 3D streaming frame timeline parser and parquet writer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from football_analytics.core.run_id import generate_run_id
from football_analytics.data.compiler import compile_arrow_schema
from football_analytics.data.parquet import (
    read_contract_parquet,
    write_contract_parquet_streaming,
)
from football_analytics.data.registry import (
    default_project_root,
    default_registry_path,
    load_schema_registry,
)
from football_analytics.video.frame_timeline import (
    build_ffprobe_frames_argv,
    iter_mapped_frames_from_lines,
    map_raw_frame,
    mapped_frames_to_record_batches,
    parse_compact_frame_line,
)
from football_analytics.video.time_mapping import MappingStats
from football_analytics.video.types import Rational


class FrameTimelineParserTests(unittest.TestCase):
    def test_parse_compact_line(self) -> None:
        raw = parse_compact_frame_line("1|0|0|512|I")
        assert raw is not None
        self.assertEqual(raw.pts, 0)
        self.assertTrue(raw.key_frame)
        self.assertEqual(raw.pict_type, "I")

    def test_parse_side_data_noise(self) -> None:
        raw = parse_compact_frame_line("1|0|0|512|IH.26[45] User Data Unregistered SEI message")
        assert raw is not None
        self.assertEqual(raw.pict_type, "I")

    def test_parse_na_pts(self) -> None:
        raw = parse_compact_frame_line("0|N/A|N/A|512|B")
        assert raw is not None
        self.assertIsNone(raw.pts)

    def test_missing_pts_carry_forward(self) -> None:
        stats = MappingStats()
        warnings: list[tuple[str, str]] = []
        seen: set[int] = set()
        tb = Rational(1, 12800)
        first, prev_pts, prev_t = map_raw_frame(
            parse_compact_frame_line("1|512|512|512|I"),  # type: ignore[arg-type]
            frame_index=0,
            time_base=tb,
            prev_pts=None,
            prev_video_time_us=None,
            stats=stats,
            warnings=warnings,
            seen_pts=seen,
        )
        self.assertEqual(first.decode_status, "ok")
        self.assertEqual(first.video_time_us, 40000)
        missing, _, t2 = map_raw_frame(
            parse_compact_frame_line("0|N/A|N/A|512|B"),  # type: ignore[arg-type]
            frame_index=1,
            time_base=tb,
            prev_pts=prev_pts,
            prev_video_time_us=prev_t,
            stats=stats,
            warnings=warnings,
            seen_pts=seen,
        )
        self.assertEqual(missing.decode_status, "skipped")
        self.assertEqual(missing.video_time_us, 40000)
        self.assertEqual(stats.missing_pts_count, 1)

    def test_duplicate_and_non_monotonic(self) -> None:
        stats = MappingStats()
        warnings: list[tuple[str, str]] = []
        seen: set[int] = set()
        tb = Rational(1, 25)
        a, p, t = map_raw_frame(
            parse_compact_frame_line("1|100|100|1|I"),  # type: ignore[arg-type]
            frame_index=0,
            time_base=tb,
            prev_pts=None,
            prev_video_time_us=None,
            stats=stats,
            warnings=warnings,
            seen_pts=seen,
        )
        _ = a
        b, p2, t2 = map_raw_frame(
            parse_compact_frame_line("0|100|100|1|P"),  # type: ignore[arg-type]
            frame_index=1,
            time_base=tb,
            prev_pts=p,
            prev_video_time_us=t,
            stats=stats,
            warnings=warnings,
            seen_pts=seen,
        )
        self.assertEqual(stats.duplicate_pts_count, 1)
        c, _, _ = map_raw_frame(
            parse_compact_frame_line("0|50|50|1|B"),  # type: ignore[arg-type]
            frame_index=2,
            time_base=tb,
            prev_pts=p2,
            prev_video_time_us=t2,
            stats=stats,
            warnings=warnings,
            seen_pts=seen,
        )
        self.assertEqual(c.decode_status, "unknown")
        self.assertEqual(stats.non_monotonic_pts_count, 1)

    def test_streaming_batches_and_atomic_parquet(self) -> None:
        lines = iter(["1|0|0|512|I", "0|512|512|512|B", "0|1024|1024|512|B"])
        mapped, result = iter_mapped_frames_from_lines(
            lines, time_base=Rational(1, 12800), maximum_frames=100
        )
        reg = load_schema_registry(default_registry_path(), project_root=default_project_root())
        contract = reg.load_contract("frames", 1)
        schema = compile_arrow_schema(contract)
        run_id = generate_run_id()
        batches = mapped_frames_to_record_batches(
            mapped, run_id=run_id, video_id="vid_test", batch_size=2, arrow_schema=schema
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "frames.parquet"
            write_contract_parquet_streaming(
                batches, path, contract, contain_root=root, overwrite=False
            )
            self.assertTrue(path.is_file())
            table = read_contract_parquet(path, contract, contain_root=root)
            self.assertEqual(table.num_rows, 3)
            self.assertEqual(result.stats.frame_count, 3)
            # no leftover temps
            temps = list(root.glob("*.tmp"))
            self.assertEqual(temps, [])

    def test_argv_uses_stream_index(self) -> None:
        argv = build_ffprobe_frames_argv(
            Path("/usr/bin/ffprobe"), Path("/tmp/x.mp4"), video_stream_index=2
        )
        self.assertIn("-select_streams", argv)
        self.assertIn("2", argv)
        self.assertIn("compact=nk=1:p=0", argv)
        self.assertNotIn(True, [isinstance(x, list) for x in argv])


if __name__ == "__main__":
    unittest.main()
