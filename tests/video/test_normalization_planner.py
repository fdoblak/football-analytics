"""Unit tests for Stage 3C normalization planner (pure, no subprocess)."""

from __future__ import annotations

import unittest
from dataclasses import replace

from football_analytics.video.contracts import default_repo_root, load_ingest_policy
from football_analytics.video.fixtures import build_synthetic_probe, synthetic_video_stream
from football_analytics.video.normalization import (
    REASON_AUDIO_TX,
    REASON_CANONICAL,
    REASON_CODEC,
    REASON_DIMS,
    REASON_FPS,
    REASON_ROTATION,
    REASON_SAR,
    compute_target_dimensions,
    compute_timeout_seconds,
    estimate_output_bytes,
    plan_normalization,
)
from football_analytics.video.types import (
    AudioStreamInfo,
    FrameCountSource,
    FrameRateMode,
    Rational,
    StreamDisposition,
)


class NormalizationPlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_ingest_policy(default_repo_root() / "configs/video/ingest_policy.yaml")

    def test_already_canonical(self) -> None:
        probe = build_synthetic_probe(
            source_id="src_canon",
            source_sha256="a" * 64,
            file_size_bytes=2048,
            streams=(synthetic_video_stream(),),
            duration_us=1_000_000,
        )
        planned = plan_normalization(
            probe=probe,
            policy=self.policy,
            output_path="/tmp/out.mp4",
            plan_id="plan_canon_one",
            source_id="src_canon",
        )
        self.assertFalse(planned.plan.required)
        self.assertEqual(planned.plan.reasons, (REASON_CANONICAL,))

    def test_mpeg4_requires_codec(self) -> None:
        probe = build_synthetic_probe(
            source_id="src_mpeg4",
            source_sha256="b" * 64,
            file_size_bytes=2048,
            streams=(synthetic_video_stream(codec_name="mpeg4"),),
            duration_us=1_000_000,
        )
        planned = plan_normalization(
            probe=probe,
            policy=self.policy,
            output_path="/tmp/out.mp4",
            plan_id="plan_mpeg4_one",
            source_id="src_mpeg4",
        )
        self.assertTrue(planned.plan.required)
        self.assertIn(REASON_CODEC, planned.plan.reasons)

    def test_rotation_and_odd_dims_and_sar(self) -> None:
        stream = replace(
            synthetic_video_stream(rotation=90, width=161, height=121),
            sample_aspect_ratio=Rational(4, 3),
        )
        probe = build_synthetic_probe(
            source_id="src_rot",
            source_sha256="c" * 64,
            file_size_bytes=2048,
            streams=(stream,),
            duration_us=1_000_000,
        )
        planned = plan_normalization(
            probe=probe,
            policy=self.policy,
            output_path="/tmp/out.mp4",
            plan_id="plan_rot_one",
            source_id="src_rot",
        )
        self.assertTrue(planned.bake_rotation)
        self.assertIn(REASON_ROTATION, planned.plan.reasons)
        self.assertIn(REASON_DIMS, planned.plan.reasons)
        self.assertIn(REASON_SAR, planned.plan.reasons)

    def test_vfr_forces_cfr(self) -> None:
        probe = build_synthetic_probe(
            source_id="src_vfr",
            source_sha256="d" * 64,
            file_size_bytes=2048,
            streams=(
                synthetic_video_stream(
                    frame_rate_mode=FrameRateMode.VFR,
                    r_num=30,
                    r_den=1,
                    avg_num=24000,
                    avg_den=1001,
                    frame_count=None,
                    frame_count_source=FrameCountSource.UNKNOWN,
                ),
            ),
            duration_us=2_000_000,
        )
        planned = plan_normalization(
            probe=probe,
            policy=self.policy,
            output_path="/tmp/out.mp4",
            plan_id="plan_vfr_one",
            source_id="src_vfr",
        )
        self.assertTrue(planned.frame_rate_conversion)
        self.assertIn(REASON_FPS, planned.plan.reasons)
        self.assertIsNotNone(planned.plan.target_frame_rate)

    def test_unknown_frame_rate_preserved(self) -> None:
        probe = build_synthetic_probe(
            source_id="src_unk",
            source_sha256="e" * 64,
            file_size_bytes=2048,
            streams=(
                synthetic_video_stream(
                    frame_rate_mode=FrameRateMode.UNKNOWN,
                    frame_count=None,
                    frame_count_source=FrameCountSource.UNKNOWN,
                ),
            ),
            duration_us=1_000_000,
        )
        planned = plan_normalization(
            probe=probe,
            policy=self.policy,
            output_path="/tmp/out.mp4",
            plan_id="plan_unk_one",
            source_id="src_unk",
        )
        self.assertFalse(planned.frame_rate_conversion)
        self.assertNotIn(REASON_FPS, planned.plan.reasons)

    def test_non_aac_audio_transcode(self) -> None:
        audio = AudioStreamInfo(
            stream_index=1,
            codec_name="mp3",
            sample_rate_hz=44100,
            channels=2,
            channel_layout="stereo",
            time_base=Rational(1, 44100),
            duration_us=1_000_000,
            bit_rate_bps=128000,
            disposition=StreamDisposition(default=True, attached_pic=False, forced=False),
        )
        probe = build_synthetic_probe(
            source_id="src_aud",
            source_sha256="f" * 64,
            file_size_bytes=4096,
            streams=(synthetic_video_stream(), audio),
            duration_us=1_000_000,
        )
        planned = plan_normalization(
            probe=probe,
            policy=self.policy,
            output_path="/tmp/out.mp4",
            plan_id="plan_aud_one",
            source_id="src_aud",
        )
        self.assertEqual(planned.audio_action, "transcode")
        self.assertIn(REASON_AUDIO_TX, planned.plan.reasons)

    def test_helpers(self) -> None:
        tw, th = compute_target_dimensions(
            1920, 1080, max_width=1280, max_height=720, even=True, upscale=False
        )
        self.assertEqual((tw, th), (1280, 720))
        tw2, th2 = compute_target_dimensions(
            100, 100, max_width=1920, max_height=1080, even=True, upscale=False
        )
        self.assertEqual((tw2, th2), (100, 100))
        probe = build_synthetic_probe(
            source_id="src_est",
            source_sha256="1" * 64,
            file_size_bytes=10_000,
            streams=(synthetic_video_stream(),),
            duration_us=1_000_000,
        )
        est = estimate_output_bytes(probe, self.policy)
        self.assertGreaterEqual(est, 10_000)
        timeout = compute_timeout_seconds(2_000_000, self.policy)
        self.assertGreaterEqual(timeout, 30.0)
        max_timeout = float(self.policy["ffmpeg_policy"]["maximum_timeout_seconds"])
        self.assertLessEqual(timeout, max_timeout)


if __name__ == "__main__":
    unittest.main()
