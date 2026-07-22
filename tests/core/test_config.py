#!/usr/bin/env python3
"""Config load / merge / fingerprint tests (Stage 2B)."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import yaml

from football_analytics.core.config import (
    ConfigError,
    apply_env_overrides,
    config_fingerprint,
    deep_merge,
    default_defaults_path,
    load_resolved_config,
    load_yaml_mapping,
    resolved_config_as_dict,
)


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.defaults = default_defaults_path()

    def test_01_default_load(self) -> None:
        cfg = load_resolved_config(defaults_path=self.defaults)
        self.assertEqual(cfg["schema_version"], 1)
        self.assertEqual(cfg["project"]["name"], "football-analytics")

    def test_02_safe_yaml_root_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.yaml"
            p.write_text("- not a mapping\n", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_yaml_mapping(p)

    def test_03_max_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "big.yaml"
            p.write_bytes(b"a: 1\n" + b"x: " + (b"y" * 300_000))
            with self.assertRaises(ConfigError):
                load_yaml_mapping(p)

    def test_04_precedence_user_over_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            user = Path(tmp) / "user.yaml"
            user.write_text("logging:\n  level: ERROR\n", encoding="utf-8")
            cfg = load_resolved_config(defaults_path=self.defaults, user_config_path=user)
            self.assertEqual(cfg["logging"]["level"], "ERROR")

    def test_05_env_override_allowlist(self) -> None:
        cfg = load_resolved_config(
            defaults_path=self.defaults,
            environ={"FOOTBALL_ANALYTICS_LOG_LEVEL": "DEBUG"},
        )
        self.assertEqual(cfg["logging"]["level"], "DEBUG")

    def test_06_forbidden_env_override(self) -> None:
        with self.assertRaises(ConfigError):
            load_resolved_config(
                defaults_path=self.defaults,
                environ={"FOOTBALL_ANALYTICS_ARBITRARY": "x"},
            )

    def test_07_explicit_overrides_win(self) -> None:
        cfg = load_resolved_config(
            defaults_path=self.defaults,
            environ={"FOOTBALL_ANALYTICS_LOG_LEVEL": "DEBUG"},
            overrides={"logging": {"level": "CRITICAL"}},
        )
        self.assertEqual(cfg["logging"]["level"], "CRITICAL")

    def test_08_deep_merge(self) -> None:
        self.assertEqual(deep_merge({"a": {"b": 1}}, {"a": {"c": 2}}), {"a": {"b": 1, "c": 2}})

    def test_09_type_mismatch(self) -> None:
        with self.assertRaises(ConfigError):
            deep_merge({"a": 1}, {"a": {"b": 2}})

    def test_10_unknown_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            user = Path(tmp) / "user.yaml"
            user.write_text("pipeline:\n  x: 1\n", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_resolved_config(defaults_path=self.defaults, user_config_path=user)

    def test_11_secret_key_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            user = Path(tmp) / "user.yaml"
            user.write_text("project:\n  api_key: leak\n", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_resolved_config(defaults_path=self.defaults, user_config_path=user)

    def test_12_nan_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "nan.yaml"
            # PyYAML may load .nan
            p.write_text("schema_version: 1\nlogging:\n  level: .nan\n", encoding="utf-8")
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
            if isinstance(data.get("logging", {}).get("level"), float) and math.isnan(
                data["logging"]["level"]
            ):
                with self.assertRaises(ConfigError):
                    load_yaml_mapping(p)

    def test_13_immutability(self) -> None:
        cfg = load_resolved_config(defaults_path=self.defaults)
        with self.assertRaises(TypeError):
            cfg["schema_version"] = 99  # type: ignore[index]

    def test_14_fingerprint_stable(self) -> None:
        a = config_fingerprint(load_resolved_config(defaults_path=self.defaults))
        b = config_fingerprint(load_resolved_config(defaults_path=self.defaults))
        self.assertEqual(a["digest"], b["digest"])
        self.assertEqual(len(a["digest"]), 64)

    def test_15_fingerprint_changes_on_list_order_via_override(self) -> None:
        # Use hashing.include_hidden_files bool flip
        a = config_fingerprint(load_resolved_config(defaults_path=self.defaults))
        b = config_fingerprint(
            load_resolved_config(
                defaults_path=self.defaults,
                overrides={"hashing": {"include_hidden_files": True}},
            )
        )
        self.assertNotEqual(a["digest"], b["digest"])

    def test_16_apply_env_format(self) -> None:
        base = resolved_config_as_dict(load_resolved_config(defaults_path=self.defaults))
        out = apply_env_overrides(base, environ={"FOOTBALL_ANALYTICS_LOG_FORMAT": "json"})
        self.assertEqual(out["logging"]["format"], "json")

    def test_17_symlink_config_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            real = Path(tmp) / "real.yaml"
            link = Path(tmp) / "link.yaml"
            real.write_text(self.defaults.read_text(encoding="utf-8"), encoding="utf-8")
            link.symlink_to(real)
            with self.assertRaises(ConfigError):
                load_yaml_mapping(link)


if __name__ == "__main__":
    unittest.main()
