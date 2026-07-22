#!/usr/bin/env python3
"""Unit tests for scripts/check_secrets.py."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def load_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "check_secrets.py"
    name = "check_secrets_under_test"
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CS = load_module()


def _pk_header() -> str:
    return "-----BEGIN " + "RSA " + "PRIVATE " + "KEY-----"


def _gh_token(ch: str = "A") -> str:
    return "ghp" + "_" + (ch * 36)


def _aws_key(ch: str = "B") -> str:
    return "AKIA" + (ch * 16)


class CheckSecretsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory(prefix="fa_sec_")
        self.root = Path(self._td.name)
        (self.root / "docs").mkdir()
        (self.root / "docs" / "readme.md").write_text(
            "Set SOCCERNET_PASSWORD in your environment.\nUse `HUGGINGFACE_TOKEN`.\n",
            encoding="utf-8",
        )
        (self.root / ".env.example").write_text(
            "SOCCERNET_PASSWORD=\nHUGGINGFACE_TOKEN=\n",
            encoding="utf-8",
        )
        (self.root / "ok.txt").write_text("hello world\n", encoding="utf-8")

    def tearDown(self) -> None:
        self._td.cleanup()

    def test_01_clean_fixture_pass(self) -> None:
        code = CS.main(["--root", str(self.root), "--quiet"])
        self.assertEqual(code, 0)

    def test_02_private_key_header(self) -> None:
        (self.root / "key.txt").write_text(_pk_header() + "\nAAAA\n", encoding="utf-8")
        code = CS.main(["--root", str(self.root), "--quiet"])
        self.assertEqual(code, 1)
        result = CS.run_scan(self.root, staged=False)
        self.assertTrue(any(f.rule == "private_key_header" for f in result.findings))

    def test_03_github_token(self) -> None:
        tok = _gh_token("A")
        (self.root / "leak.txt").write_text(f"token {tok}\n", encoding="utf-8")
        result = CS.run_scan(self.root, staged=False)
        self.assertEqual(result.exit_code, 1)
        self.assertTrue(any(f.rule == "github_token" for f in result.findings))
        joined = " ".join(f.evidence for f in result.findings)
        self.assertNotIn(tok, joined)

    def test_04_aws_key(self) -> None:
        key = _aws_key("B")
        (self.root / "aws.txt").write_text(f"key={key}\n", encoding="utf-8")
        result = CS.run_scan(self.root, staged=False)
        self.assertTrue(any(f.rule == "aws_access_key" for f in result.findings))

    def test_05_bearer_token(self) -> None:
        line = "Authorization: " + "Bearer " + "abcdefghijklmnop"
        (self.root / "b.txt").write_text(line + "\n", encoding="utf-8")
        result = CS.run_scan(self.root, staged=False)
        self.assertTrue(any(f.rule == "bearer_token" for f in result.findings))

    def test_06_url_token(self) -> None:
        secret = "supersecretvalue123"
        (self.root / "u.txt").write_text(
            "https://example.com/x?" + "token=" + secret + "\n", encoding="utf-8"
        )
        result = CS.run_scan(self.root, staged=False)
        self.assertTrue(any(f.rule == "url_query_secret" for f in result.findings))
        self.assertNotIn(secret, result.findings[0].evidence)

    def test_07_password_assignment(self) -> None:
        val = "hunter2hunter2"
        (self.root / "p.txt").write_text("password" + " = " + f'"{val}"\n', encoding="utf-8")
        result = CS.run_scan(self.root, staged=False)
        self.assertTrue(any(f.rule == "password_assignment" for f in result.findings))

    def test_08_env_example_empty_pass(self) -> None:
        code = CS.main(["--root", str(self.root), "--quiet"])
        self.assertEqual(code, 0)

    def test_09_doc_env_var_name_pass(self) -> None:
        result = CS.run_scan(self.root, staged=False)
        self.assertEqual(result.exit_code, 0)

    def test_10_binary_skip(self) -> None:
        (self.root / "w.pth").write_bytes(b"\x00\x01\x02\x03" * 100)
        result = CS.run_scan(self.root, staged=False)
        self.assertTrue(any(s.get("reason") == "binary_extension" for s in result.skipped))
        self.assertEqual(result.exit_code, 0)

    def test_11_large_file_skip(self) -> None:
        big = self.root / "big.txt"
        big.write_bytes(b"a" * (CS.MAX_SCAN_BYTES + 10))
        result = CS.run_scan(self.root, staged=False)
        self.assertTrue(any(s["path"].endswith("big.txt") for s in result.skipped))

    def test_12_symlink_outside_not_followed(self) -> None:
        outside = Path(tempfile.mkdtemp(prefix="fa_out_"))
        try:
            secret = outside / "secret.txt"
            secret.write_text("password" + " = " + '"outsideleakvalue"\n', encoding="utf-8")
            link = self.root / "link.txt"
            link.symlink_to(secret)
            result = CS.run_scan(self.root, staged=False)
            self.assertTrue(
                any(
                    s.get("reason") in {"symlink_outside_root", "symlink_skipped"}
                    for s in result.skipped
                )
            )
            self.assertFalse(any("outsideleakvalue" in f.evidence for f in result.findings))
        finally:
            secret.unlink(missing_ok=True)
            outside.rmdir()

    def test_13_redacted_output(self) -> None:
        tok = _gh_token("C")
        (self.root / "t.txt").write_text(tok + "\n", encoding="utf-8")
        result = CS.run_scan(self.root, staged=False)
        blob = json.dumps(result.to_dict())
        self.assertNotIn(tok, blob)
        self.assertIn("REDACTED", blob)

    def test_14_staged_mode(self) -> None:
        subprocess.run(["git", "init"], cwd=self.root, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "t@e.com"], cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=self.root, check=True)
        dirty = self.root / "staged_leak.txt"
        dirty.write_text(_gh_token("D") + "\n", encoding="utf-8")
        code = CS.main(["--root", str(self.root), "--staged", "--quiet"])
        self.assertEqual(code, 0)
        subprocess.run(
            ["git", "add", "staged_leak.txt"], cwd=self.root, check=True, capture_output=True
        )
        code2 = CS.main(["--root", str(self.root), "--staged", "--quiet"])
        self.assertEqual(code2, 1)

    def test_15_json_output_valid(self) -> None:
        out = self.root / "scan.json"
        code = CS.main(["--root", str(self.root), "--json-out", str(out), "--quiet"])
        self.assertEqual(code, 0)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "PASS")

    def test_16_exit_code_contract(self) -> None:
        code = CS.main(["--root", str(self.root / "missing"), "--quiet"])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
