#!/usr/bin/env python3
"""Structured logging tests (Stage 2B)."""

from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path

from football_analytics.core.structured_logging import (
    LOGGER_NAME,
    LoggingError,
    configure_logger,
    log_event,
)


class StructuredLoggingTests(unittest.TestCase):
    def tearDown(self) -> None:
        logger = logging.getLogger(LOGGER_NAME)
        for h in list(logger.handlers):
            logger.removeHandler(h)
            h.close()

    def test_01_valid_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            logger = configure_logger(console=False, jsonl_path=path, contain_root=Path(tmp))
            log_event(logger, "INFO", "hello", event="t", run_id="r1", stage="s", context={"k": 1})
            line = path.read_text(encoding="utf-8").strip()
            payload = json.loads(line)
            for key in (
                "schema_version",
                "timestamp_utc",
                "level",
                "logger",
                "event",
                "message",
                "run_id",
                "stage",
                "context",
                "exception",
            ):
                self.assertIn(key, payload)

    def test_02_no_duplicate_handlers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "e.jsonl"
            configure_logger(console=False, jsonl_path=path, contain_root=Path(tmp))
            configure_logger(console=False, jsonl_path=path, contain_root=Path(tmp))
            logger = logging.getLogger(LOGGER_NAME)
            self.assertEqual(len(logger.handlers), 1)

    def test_03_newline_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "e.jsonl"
            logger = configure_logger(console=False, jsonl_path=path, contain_root=Path(tmp))
            log_event(logger, "INFO", "a\nb", event="n")
            line = path.read_text(encoding="utf-8").strip()
            self.assertEqual(len(line.splitlines()), 1)
            json.loads(line)

    def test_04_exception_redaction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "e.jsonl"
            logger = configure_logger(console=False, jsonl_path=path, contain_root=Path(tmp))
            try:
                secret = "super" + "secrettokenvalue"
                raise RuntimeError("Bearer " + secret)
            except RuntimeError:
                log_event(logger, "ERROR", "boom", event="ex", exc_info=True)
            blob = path.read_text(encoding="utf-8")
            self.assertNotIn("supersecrettokenvalue", blob)
            self.assertIn("[REDACTED]", blob)

    def test_05_invalid_level(self) -> None:
        with self.assertRaises(LoggingError):
            configure_logger(level="NOPE", console=False)

    def test_06_symlink_log_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            real = Path(tmp) / "real.jsonl"
            real.write_text("", encoding="utf-8")
            link = Path(tmp) / "link.jsonl"
            link.symlink_to(real)
            with self.assertRaises(LoggingError):
                configure_logger(console=False, jsonl_path=link, contain_root=Path(tmp))

    def test_07_file_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "e.jsonl"
            logger = configure_logger(console=False, jsonl_path=path, contain_root=Path(tmp))
            log_event(logger, "INFO", "p", event="p")
            mode = path.stat().st_mode & 0o777
            self.assertEqual(mode, 0o600)

    def test_08_root_logger_untouched(self) -> None:
        before = list(logging.getLogger().handlers)
        with tempfile.TemporaryDirectory() as tmp:
            configure_logger(
                console=False, jsonl_path=Path(tmp) / "e.jsonl", contain_root=Path(tmp)
            )
        self.assertEqual(list(logging.getLogger().handlers), before)

    def test_09_secret_in_context_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "e.jsonl"
            logger = configure_logger(console=False, jsonl_path=path, contain_root=Path(tmp))
            log_event(logger, "INFO", "x", event="c", context={"api_key": "leak"})
            payload = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(payload["context"]["api_key"], "[REDACTED]")


if __name__ == "__main__":
    unittest.main()
