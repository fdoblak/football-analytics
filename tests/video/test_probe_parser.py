"""Parser unit tests for Stage 3B FFprobe JSON mapping."""

from __future__ import annotations

import unittest

from football_analytics.video.ffprobe import ProbeError
from football_analytics.video.probe_parser import (
    map_ffprobe_json_to_video_probe,
    parse_rational,
    seconds_to_us,
    select_primary_audio_stream,
)
from football_analytics.video.types import (
    FrameRateMode,
    Rational,
    VideoStreamInfo,
)


def _video(**kwargs):
    base = {
        "index": 0,
        "codec_type": "video",
        "codec_name": "h264",
        "width": 320,
        "height": 240,
        "pix_fmt": "yuv420p",
        "r_frame_rate": "25/1",
        "avg_frame_rate": "25/1",
        "time_base": "1/25",
        "disposition": {"default": 1, "attached_pic": 0, "forced": 0},
    }
    base.update(kwargs)
    return base


class ProbeParserTests(unittest.TestCase):
    def test_rational_na_and_zero_over_zero(self) -> None:
        self.assertIsNone(parse_rational("N/A", label="x"))
        self.assertIsNone(parse_rational("0/0", label="x"))
        self.assertEqual(parse_rational("30/1", label="x"), Rational(30, 1))
        with self.assertRaises(ProbeError):
            parse_rational("1/0", label="x")

    def test_seconds_to_us_deterministic(self) -> None:
        self.assertEqual(seconds_to_us("1.5", label="d"), 1_500_000)
        self.assertEqual(seconds_to_us("0.000001", label="d"), 1)
        self.assertIsNone(seconds_to_us("N/A", label="d"))

    def test_minimal_valid(self) -> None:
        probe = map_ffprobe_json_to_video_probe(
            {"format": {"duration": "1.0", "format_name": "mp4"}, "streams": [_video()]},
            source_id="src_parse_one",
            source_sha256="a" * 64,
            file_size_bytes=10,
            probe_tool_version="4.4.2",
            probed_at_utc="2026-07-22T22:00:00Z",
        )
        self.assertEqual(probe.selected_video_stream_index, 0)
        self.assertEqual(probe.duration_us, 1_000_000)
        self.assertEqual(probe.streams[0].frame_rate_mode, FrameRateMode.CFR)  # type: ignore[union-attr]

    def test_vfr_unknown_and_null_frames(self) -> None:
        probe = map_ffprobe_json_to_video_probe(
            {
                "format": {"duration": "2.0", "format_name": "mp4"},
                "streams": [
                    _video(r_frame_rate="30/1", avg_frame_rate="24000/1001", nb_frames="N/A")
                ],
            },
            source_id="src_parse_vfr",
            source_sha256="b" * 64,
            file_size_bytes=10,
            probe_tool_version="4.4.2",
            probed_at_utc="2026-07-22T22:00:00Z",
        )
        vs = probe.streams[0]
        assert isinstance(vs, VideoStreamInfo)
        self.assertEqual(vs.frame_rate_mode, FrameRateMode.VFR)
        self.assertIsNone(vs.frame_count)

    def test_attached_pic_ignored(self) -> None:
        probe = map_ffprobe_json_to_video_probe(
            {
                "format": {"format_name": "mp4", "duration": "1"},
                "streams": [
                    _video(
                        index=0,
                        width=10,
                        height=10,
                        disposition={"default": 0, "attached_pic": 1, "forced": 0},
                    ),
                    _video(index=1, width=320, height=240),
                ],
            },
            source_id="src_parse_pic",
            source_sha256="c" * 64,
            file_size_bytes=10,
            probe_tool_version="4.4.2",
            probed_at_utc="2026-07-22T22:00:00Z",
        )
        self.assertEqual(probe.selected_video_stream_index, 1)

    def test_audio_only_rejected(self) -> None:
        with self.assertRaises(ProbeError) as ctx:
            map_ffprobe_json_to_video_probe(
                {
                    "format": {"format_name": "mp4", "duration": "1"},
                    "streams": [
                        {
                            "index": 0,
                            "codec_type": "audio",
                            "codec_name": "aac",
                            "sample_rate": "44100",
                            "channels": 2,
                            "time_base": "1/44100",
                            "disposition": {"default": 1, "attached_pic": 0, "forced": 0},
                        }
                    ],
                },
                source_id="src_audio",
                source_sha256="d" * 64,
                file_size_bytes=10,
                probe_tool_version="4.4.2",
                probed_at_utc="2026-07-22T22:00:00Z",
            )
        self.assertEqual(ctx.exception.code, "NO_USABLE_VIDEO_STREAM")

    def test_negative_start_and_rotation_tag(self) -> None:
        probe = map_ffprobe_json_to_video_probe(
            {
                "format": {"format_name": "mp4", "duration": "1", "start_time": "-0.040000"},
                "streams": [_video(tags={"rotate": "90"})],
            },
            source_id="src_rot",
            source_sha256="e" * 64,
            file_size_bytes=10,
            probe_tool_version="4.4.2",
            probed_at_utc="2026-07-22T22:00:00Z",
        )
        self.assertEqual(probe.start_time_us, -40_000)
        vs = probe.streams[0]
        assert isinstance(vs, VideoStreamInfo)
        self.assertEqual(vs.rotation_degrees, 90)

    def test_too_many_streams(self) -> None:
        streams = [_video(index=i) for i in range(5)]
        with self.assertRaises(ProbeError) as ctx:
            map_ffprobe_json_to_video_probe(
                {"format": {"format_name": "mp4"}, "streams": streams},
                source_id="src_many",
                source_sha256="f" * 64,
                file_size_bytes=10,
                probe_tool_version="4.4.2",
                probed_at_utc="2026-07-22T22:00:00Z",
                max_stream_count=3,
            )
        self.assertEqual(ctx.exception.code, "TOO_MANY_STREAMS")

    def test_audio_selection_default(self) -> None:
        # Build via mapped probe including two audio streams
        probe = map_ffprobe_json_to_video_probe(
            {
                "format": {"format_name": "mp4", "duration": "1"},
                "streams": [
                    _video(index=0),
                    {
                        "index": 1,
                        "codec_type": "audio",
                        "codec_name": "aac",
                        "sample_rate": "48000",
                        "channels": 2,
                        "time_base": "1/48000",
                        "disposition": {"default": 0, "attached_pic": 0, "forced": 0},
                    },
                    {
                        "index": 2,
                        "codec_type": "audio",
                        "codec_name": "aac",
                        "sample_rate": "48000",
                        "channels": 2,
                        "time_base": "1/48000",
                        "disposition": {"default": 1, "attached_pic": 0, "forced": 0},
                    },
                ],
            },
            source_id="src_audsel",
            source_sha256="1" * 64,
            file_size_bytes=10,
            probe_tool_version="4.4.2",
            probed_at_utc="2026-07-22T22:00:00Z",
        )
        self.assertEqual(probe.selected_audio_stream_index, 2)
        idx = select_primary_audio_stream(probe.streams)
        self.assertEqual(idx, 2)


if __name__ == "__main__":
    unittest.main()
