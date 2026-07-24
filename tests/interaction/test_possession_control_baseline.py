#!/usr/bin/env python3
"""Stage 10C possession / control baseline unit tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from football_analytics.interaction.possession_config import (
    load_possession_baseline_config,
    possession_baseline_config_fingerprint,
)
from football_analytics.interaction.possession_fixtures import load_fixture
from football_analytics.interaction.possession_service import compute_possession_control
from football_analytics.interaction.types import NOT_EVALUATED_INTERACTION


class PossessionControlBaselineTests(unittest.TestCase):
    def test_01_config_fingerprint_stable(self) -> None:
        a = load_possession_baseline_config()
        b = load_possession_baseline_config()
        self.assertEqual(
            possession_baseline_config_fingerprint(a),
            possession_baseline_config_fingerprint(b),
        )
        self.assertEqual(a["stage"], "10C")
        self.assertEqual(a["automatic_ceiling"], "provisional")

    def test_02_provisional_not_confirmed(self) -> None:
        fx = load_fixture("provisional_control")
        with tempfile.TemporaryDirectory() as tmp:
            r = compute_possession_control(output_dir=Path(tmp), points=fx["points"])
            self.assertTrue(r.accepted, r.error_code)
            states = {p["possession_state"] for p in r.possessions}
            self.assertNotIn("confirmed", states)
            self.assertTrue(states & {"provisional", "candidate"})

    def test_03_missing_ball_not_loose(self) -> None:
        fx = load_fixture("missing_ball")
        with tempfile.TemporaryDirectory() as tmp:
            r = compute_possession_control(output_dir=Path(tmp), points=fx["points"])
            self.assertTrue(r.accepted, r.error_code)
            for p in r.possessions:
                self.assertNotIn("LOOSE_BALL", p["reason_codes"])
            self.assertTrue(
                any(
                    p["possession_state"] == "not_evaluable"
                    and "MISSING_BALL_NOT_NO_POSSESSION" in p["reason_codes"]
                    for p in r.possessions
                )
            )

    def test_04_contested_and_loose(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = compute_possession_control(
                output_dir=root / "c", points=load_fixture("contested")["points"]
            )
            self.assertTrue(c.accepted, c.error_code)
            self.assertTrue(any(p["possession_state"] == "contested" for p in c.possessions))
            loose = compute_possession_control(
                output_dir=root / "l", points=load_fixture("loose_ball")["points"]
            )
            self.assertTrue(loose.accepted, loose.error_code)
            self.assertTrue(
                any(
                    p["possession_state"] == "unknown" and "LOOSE_BALL" in p["reason_codes"]
                    for p in loose.possessions
                )
            )

    def test_05_nearest_not_owner_and_not_evaluated(self) -> None:
        fx = load_fixture("nearest_not_owner")
        with tempfile.TemporaryDirectory() as tmp:
            r = compute_possession_control(output_dir=Path(tmp), points=fx["points"])
            self.assertTrue(r.accepted, r.error_code)
            owned = [
                p
                for p in r.possessions
                if p["possession_state"] in {"candidate", "provisional", "confirmed"}
                and p["owner_human_track_id"] is not None
            ]
            self.assertEqual(owned, [])
            self.assertEqual(
                r.summary["evaluation_status"],
                NOT_EVALUATED_INTERACTION,
            )


if __name__ == "__main__":
    unittest.main()
