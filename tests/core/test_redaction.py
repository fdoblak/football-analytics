#!/usr/bin/env python3
"""Redaction tests (Stage 2B)."""

from __future__ import annotations

import unittest

from football_analytics.core.redaction import (
    REDACTED,
    is_sensitive_key,
    redact_text,
    redact_value,
    sanitize_remote_url,
)


class RedactionTests(unittest.TestCase):
    def test_01_nested_secrets(self) -> None:
        src = {"a": {"api_key": "x", "ok": 1}, "token": "t"}
        out = redact_value(src)
        self.assertEqual(out["a"]["api_key"], REDACTED)
        self.assertEqual(out["token"], REDACTED)
        self.assertEqual(out["a"]["ok"], 1)
        self.assertEqual(src["token"], "t")  # no mutation

    def test_02_case_variants(self) -> None:
        self.assertTrue(is_sensitive_key("API_KEY"))
        self.assertTrue(is_sensitive_key("Password"))
        self.assertFalse(is_sensitive_key("username"))

    def test_03_bearer(self) -> None:
        sample = "Authorization: " + "Bearer " + "abcdefghijklmnop"
        self.assertIn(REDACTED, redact_text(sample))

    def test_04_url_userinfo(self) -> None:
        url = sanitize_remote_url("https://user:ghp_secret@github.com/o/r.git")
        self.assertNotIn("ghp_secret", url)
        self.assertNotIn("user:", url)
        self.assertIn("github.com", url)

    def test_05_list_tuple(self) -> None:
        out = redact_value([{"password": "p"}, ("ok",)])
        self.assertEqual(out[0]["password"], REDACTED)

    def test_06_newline_collapse(self) -> None:
        self.assertNotIn("\n", redact_text("line1\nline2"))

    def test_07_github_pat_pattern(self) -> None:
        token = "github_pat_" + "ABCDEFGHijklmnop"
        text = redact_text("leak " + token + " here")
        self.assertIn(REDACTED, text)
        self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
