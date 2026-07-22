"""Unit tests for Stage 3A video typed contracts."""

from __future__ import annotations

import subprocess
import sys
import unittest

from football_analytics.core.run_id import generate_run_id
from football_analytics.video.types import (
    ContractFingerprints,
    FrameCountSource,
    FrameRateMode,
    IngestMode,
    IngestReceipt,
    IngestRequest,
    ProvenanceInfo,
    Rational,
    ReceiptProvenance,
    ReceiptStatus,
    SourceKind,
    StreamDisposition,
    VideoContractError,
    VideoProbe,
    VideoSource,
    VideoStreamInfo,
    normalize_rotation_degrees,
    select_primary_video_stream,
)


def _sha(ch: str = "a") -> str:
    return ch * 64


def _source(**kwargs):
    base = dict(
        source_id="src_unit_one",
        source_kind=SourceKind.SYNTHETIC_FIXTURE,
        original_filename="clip.mp4",
        source_path="/home/fdoblak/workspace/video_contract_checks/clip.mp4",
        source_size_bytes=100,
        source_sha256=_sha(),
        media_type="video/mp4",
        container_hint="mp4",
        created_at_utc="2026-07-22T21:00:00Z",
        registered_at_utc="2026-07-22T21:00:01Z",
        immutability_policy="detect_mutation",
        provenance=ProvenanceInfo(origin="synthetic_generated", label="unit", notes=None),
    )
    base.update(kwargs)
    return VideoSource(**base)


def _stream(**kwargs) -> VideoStreamInfo:
    base = dict(
        stream_index=0,
        codec_name="h264",
        codec_long_name="H.264",
        profile="High",
        pixel_format="yuv420p",
        width=160,
        height=120,
        coded_width=160,
        coded_height=120,
        sample_aspect_ratio=Rational(1, 1),
        display_aspect_ratio=Rational(4, 3),
        rotation_degrees=0,
        time_base=Rational(1, 25_000),
        codec_time_base=Rational(1, 50),
        r_frame_rate=Rational(25, 1),
        avg_frame_rate=Rational(25, 1),
        nominal_frame_rate=Rational(25, 1),
        frame_rate_mode=FrameRateMode.CFR,
        start_pts=0,
        duration_ts=25_000,
        duration_us=1_000_000,
        frame_count=25,
        frame_count_source=FrameCountSource.NB_FRAMES,
        bit_rate_bps=1000,
        color_range=None,
        color_space=None,
        color_transfer=None,
        color_primaries=None,
        field_order=None,
        disposition=StreamDisposition(True, False, False),
    )
    base.update(kwargs)
    return VideoStreamInfo(**base)


class VideoTypesTests(unittest.TestCase):
    def test_rational_rejects_zero_denominator(self) -> None:
        with self.assertRaises(VideoContractError):
            Rational(25, 0)

    def test_rational_roundtrip(self) -> None:
        r = Rational(30000, 1001)
        self.assertEqual(Rational.from_dict(r.to_dict()), r)

    def test_source_roundtrip_immutable(self) -> None:
        src = _source()
        again = VideoSource.from_dict(src.to_dict())
        self.assertEqual(src.to_dict(), again.to_dict())
        self.assertEqual(src.fingerprint(), again.fingerprint())
        with self.assertRaises(AttributeError):
            src.source_id = "mutated"  # type: ignore[misc]

    def test_timezone_validation(self) -> None:
        with self.assertRaises(VideoContractError):
            _source(created_at_utc="2026-07-22T21:00:00")

    def test_sha_validation(self) -> None:
        with self.assertRaises(VideoContractError):
            _source(source_sha256="abc")

    def test_request_uses_run_id_validator(self) -> None:
        run_id = generate_run_id()
        req = IngestRequest(
            request_id="req_unit_one",
            run_id=run_id,
            source_id="src_unit_one",
            source_path="/home/fdoblak/workspace/video_contract_checks/clip.mp4",
            requested_at_utc="2026-07-22T21:00:02Z",
            ingest_mode=IngestMode.VALIDATE_ONLY,
            policy_version="video_ingest_policy_v1",
            probe_requested=False,
            normalization_requested=False,
            expected_source_sha256=_sha(),
            expected_source_size_bytes=100,
            output_root="/home/fdoblak/workspace/video_contract_checks/out",
            fixture_mode=True,
        )
        self.assertEqual(IngestRequest.from_dict(req.to_dict()).run_id, run_id)
        from football_analytics.core.run_id import RunIdError

        with self.assertRaises(RunIdError):
            IngestRequest.from_dict({**req.to_dict(), "run_id": "bad"})

    def test_rotation_normalize(self) -> None:
        self.assertEqual(normalize_rotation_degrees(-90), 270)
        self.assertEqual(_stream(rotation_degrees=-90).rotation_degrees, 270)

    def test_unknown_frame_count_null(self) -> None:
        stream = _stream(frame_count=None, frame_count_source=FrameCountSource.UNKNOWN)
        self.assertIsNone(stream.frame_count)

    def test_stream_selection_skips_attached_pic(self) -> None:
        pic = _stream(
            stream_index=0,
            width=10,
            height=10,
            disposition=StreamDisposition(False, True, False),
        )
        main = _stream(stream_index=1, width=160, height=120)
        self.assertEqual(select_primary_video_stream((pic, main)), 1)

    def test_probe_rejects_wrong_selection(self) -> None:
        streams = (_stream(stream_index=0),)
        with self.assertRaises(VideoContractError):
            VideoProbe(
                source_id="src_unit_one",
                source_sha256=_sha(),
                probe_tool="synthetic_metadata",
                probe_tool_version="t",
                probed_at_utc="2026-07-22T21:00:00Z",
                container="mp4",
                format_name="mp4",
                duration_us=1_000_000,
                start_time_us=0,
                bit_rate_bps=1,
                file_size_bytes=10,
                streams=streams,
                selected_video_stream_index=5,
                selected_audio_stream_index=None,
                warnings=(),
            )

    def test_receipt_no_false_success(self) -> None:
        with self.assertRaises(ValueError):
            ReceiptStatus("succeeded")
        receipt = IngestReceipt(
            receipt_id="rcpt_unit_one",
            request_id="req_unit_one",
            run_id=generate_run_id(),
            source_id="src_unit_one",
            source_sha256=_sha(),
            source_size_bytes=100,
            status=ReceiptStatus.VALIDATED,
            started_at_utc="2026-07-22T21:00:03Z",
            completed_at_utc="2026-07-22T21:00:04Z",
            probe_record_ref=None,
            normalize_plan_ref=None,
            artifact_refs=(),
            policy_version="video_ingest_policy_v1",
            contract_fingerprints=ContractFingerprints(source=_sha(), request=_sha("b")),
            warnings=(),
            errors=(),
            provenance=ReceiptProvenance(stage="3A", label="unit"),
        )
        self.assertEqual(receipt.status, ReceiptStatus.VALIDATED)
        with self.assertRaises(VideoContractError):
            IngestReceipt.from_dict(
                {
                    **receipt.to_dict(),
                    "status": "rejected",
                    "errors": [],
                }
            )

    def test_import_side_effects_no_heavy_engines(self) -> None:
        # Use a clean subprocess so we do not reload Enum classes in-process.
        from pathlib import Path

        repo = Path(__file__).resolve().parents[2]
        code = (
            "import sys\n"
            "banned={'torch','cv2','ultralytics','SoccerNet'}\n"
            "for name in list(sys.modules):\n"
            "    if name.split('.')[0] in banned:\n"
            "        del sys.modules[name]\n"
            "import football_analytics.video\n"
            "import football_analytics.video.types\n"
            "bad={m for m in sys.modules if m.split('.')[0] in banned}\n"
            "assert not bad, sorted(bad)\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_vfr_vs_cfr_semantics(self) -> None:
        cfr = _stream(frame_rate_mode=FrameRateMode.CFR)
        vfr = _stream(
            frame_rate_mode=FrameRateMode.VFR,
            avg_frame_rate=Rational(24000, 1001),
            frame_count=None,
            frame_count_source=FrameCountSource.UNKNOWN,
        )
        self.assertEqual(cfr.frame_rate_mode, FrameRateMode.CFR)
        self.assertEqual(vfr.frame_rate_mode, FrameRateMode.VFR)
        self.assertIsNone(vfr.frame_count)

    def test_nonzero_start_pts(self) -> None:
        stream = _stream(start_pts=-1000)
        self.assertEqual(stream.start_pts, -1000)


if __name__ == "__main__":
    unittest.main()
