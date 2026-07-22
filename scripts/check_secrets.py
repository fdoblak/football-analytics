#!/usr/bin/env python3
"""Lightweight secret scanner for football-analytics (read-only)."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

EXIT_PASS = 0
EXIT_FINDING = 1
EXIT_CONFIG = 2

MAX_SCAN_BYTES = 512 * 1024
BINARY_SNIFF = 8192

SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    ".ipynb_checkpoints",
}

BINARY_EXTENSIONS = {
    ".pt",
    ".pth",
    ".onnx",
    ".engine",
    ".mkv",
    ".mp4",
    ".avi",
    ".parquet",
    ".npy",
    ".npz",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".zip",
    ".gz",
    ".whl",
    ".so",
    ".bin",
    ".pkl",
    ".pickle",
}

CREDENTIAL_EXTENSIONS = {".pem", ".key", ".p12", ".pfx", ".p8"}

PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
)
GITHUB_TOKEN_RE = re.compile(r"\b(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b")
AWS_KEY_RE = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9\-._~+/]+=*")
URL_SECRET_RE = re.compile(
    r"(?i)[?&](?:token|access_token|api_key|password|secret)=([^\s&#]+)"
)
PASSWORD_ASSIGN_RE = re.compile(
    r"(?i)\b(?:password|passwd|pwd)\s*=\s*['\"]([^'\"]{8,})['\"]"
)
ENV_ASSIGN_RE = re.compile(
    r"(?m)^(?:export\s+)?"
    r"([A-Z][A-Z0-9_]*(?:_PASSWORD|_PASSWD|_TOKEN|_SECRET|_API_KEY|_KEY))"
    r"\s*=\s*(.+)$"
)
HIGH_ENTROPY_RE = re.compile(r"\b([A-Za-z0-9+/=_-]{40,})\b")

SAFE_DOC_PATTERNS = (
    re.compile(r"(?i)\b(?:set|export|use)\s+[A-Z][A-Z0-9_]+\b"),
    re.compile(r"(?i)\$\{?[A-Z][A-Z0-9_]+\}?"),
    re.compile(r"(?i)`[A-Z][A-Z0-9_]+`"),
)

CODE_PATTERN_LINE = re.compile(
    r"""(?:re\.compile\s*\(|^\s*[A-Z0-9_]+\s*=\s*re\.|^\s*[A-Z0-9_]+\s*=\s*r['\"]|^\s*r['\"].*(?:password|token|secret)\s*=)"""
)


@dataclass
class Finding:
    rule: str
    path: str
    line: Optional[int]
    evidence: str
    severity: str = "high"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule": self.rule,
            "path": self.path,
            "line": self.line,
            "evidence": self.evidence,
            "severity": self.severity,
        }


@dataclass
class ScanResult:
    status: str = "PASS"
    exit_code: int = EXIT_PASS
    findings: List[Finding] = field(default_factory=list)
    skipped: List[Dict[str, str]] = field(default_factory=list)
    scanned_files: int = 0
    errors: List[str] = field(default_factory=list)

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    def finalize(self) -> "ScanResult":
        if self.errors and not self.findings:
            self.status = "CONFIG_ERROR"
            self.exit_code = EXIT_CONFIG
        elif self.findings:
            self.status = "FINDINGS"
            self.exit_code = EXIT_FINDING
        else:
            self.status = "PASS"
            self.exit_code = EXIT_PASS
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": self.status,
            "exit_code": self.exit_code,
            "scanned_files": self.scanned_files,
            "finding_count": len(self.findings),
            "findings": [f.to_dict() for f in self.findings],
            "skipped": list(self.skipped),
            "errors": list(self.errors),
        }


def redact(value: str, keep: int = 4) -> str:
    if not value:
        return ""
    value = value.strip()
    if len(value) <= keep * 2:
        return "***REDACTED***"
    return f"{value[:keep]}…{value[-keep:]} [REDACTED]"


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq: Dict[str, int] = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


def is_probably_binary(data: bytes) -> bool:
    if b"\x00" in data:
        return True
    if not data:
        return False
    text_chars = sum(1 for b in data if 9 <= b <= 13 or 32 <= b <= 126)
    return (text_chars / len(data)) < 0.70


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path = path.resolve()
    parent = path.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"JSON parent missing: {parent}")
    if path.exists():
        raise FileExistsError(f"Refusing overwrite: {path}")
    fd, tmp = tempfile.mkstemp(prefix=".secret_scan_", dir=str(parent))
    tmp_path = Path(tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(tmp_path), str(path))
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def iter_candidate_paths(root: Path) -> Iterable[Path]:
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        # do not descend into symlink dirs
        pruned = []
        for d in dirnames:
            full = Path(dirpath) / d
            if full.is_symlink():
                continue
            pruned.append(d)
        dirnames[:] = pruned
        for name in filenames:
            yield Path(dirpath) / name


def staged_paths(root: Path) -> List[Path]:
    proc = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "git staged listing failed")
    out: List[Path] = []
    for line in (proc.stdout or "").splitlines():
        rel = line.strip()
        if not rel:
            continue
        out.append(root / rel)
    return out


def looks_like_doc_var_name_only(line: str) -> bool:
    stripped = line.strip()
    if any(p.search(stripped) for p in SAFE_DOC_PATTERNS):
        # no assignment of a real value
        if "=" not in stripped and ":" not in stripped:
            return True
        # markdown / docs mentioning NAME=
        if re.search(r"(?i)\b[A-Z][A-Z0-9_]+\s*=\s*(?:$|<|\"\"|''|\s*#)", stripped):
            return True
        if re.search(r"(?i)`[A-Z][A-Z0-9_]+`", stripped) and not re.search(
            r"[:=]\s*['\"]?[A-Za-z0-9+/=_-]{8,}", stripped
        ):
            return True
    return False


def env_value_is_empty(value: str) -> bool:
    v = value.strip().strip("'\"")
    return v == "" or v in {"changeme", "placeholder", "YOUR_TOKEN_HERE", "<token>"}


def scan_text(path: Path, text: str, result: ScanResult, *, is_env_example: bool) -> None:
    rel = str(path)
    for lineno, line in enumerate(text.splitlines(), start=1):
        if looks_like_doc_var_name_only(line):
            continue
        if CODE_PATTERN_LINE.search(line):
            continue
        if PRIVATE_KEY_RE.search(line):
            result.add(
                Finding("private_key_header", rel, lineno, "BEGIN PRIVATE KEY [REDACTED]")
            )
        for m in GITHUB_TOKEN_RE.finditer(line):
            result.add(
                Finding("github_token", rel, lineno, redact(m.group(0)))
            )
        for m in AWS_KEY_RE.finditer(line):
            result.add(Finding("aws_access_key", rel, lineno, redact(m.group(0))))
        for m in BEARER_RE.finditer(line):
            result.add(Finding("bearer_token", rel, lineno, redact(m.group(0))))
        for m in URL_SECRET_RE.finditer(line):
            result.add(
                Finding("url_query_secret", rel, lineno, "query-secret=" + redact(m.group(1)))
            )
        for m in PASSWORD_ASSIGN_RE.finditer(line):
            val = m.group(1)
            if is_env_example and env_value_is_empty(val):
                continue
            if env_value_is_empty(val) or val.lower() in {"null", "none", "false", "true"}:
                continue
            if re.fullmatch(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?", val):
                continue
            result.add(Finding("password_assignment", rel, lineno, redact(val)))
        for m in ENV_ASSIGN_RE.finditer(line):
            name, value = m.group(1), m.group(2)
            if is_env_example and env_value_is_empty(value):
                continue
            if env_value_is_empty(value):
                continue
            if value.strip().startswith("#"):
                continue
            # Skip non-assignment literals (e.g. frozenset/tuple continuations)
            if value.lstrip().startswith(("(", "[", "{", "frozenset", "re.")):
                continue
            result.add(
                Finding(
                    "env_secret_assignment",
                    rel,
                    lineno,
                    f"{name}=" + redact(value),
                )
            )
        if not is_env_example:
            for m in HIGH_ENTROPY_RE.finditer(line):
                token = m.group(1)
                if token.startswith("http") or "/" in token and token.count("/") > 2:
                    continue
                if SHA_LIKE(token):
                    continue
                if shannon_entropy(token) >= 4.5 and len(token) >= 40:
                    if re.fullmatch(r"[a-fA-F0-9]{40,64}", token):
                        continue
                    result.add(
                        Finding("high_entropy", rel, lineno, redact(token), severity="medium")
                    )


def SHA_LIKE(token: str) -> bool:
    return bool(re.fullmatch(r"[a-fA-F0-9]{40,64}", token))


def scan_file(path: Path, root: Path, result: ScanResult) -> None:
    try:
        # stay inside root; do not follow symlinks out
        if path.is_symlink():
            target = path.resolve()
            try:
                target.relative_to(root.resolve())
            except ValueError:
                result.skipped.append({"path": str(path), "reason": "symlink_outside_root"})
                return
            # if symlink points outside, already returned; if inside, still skip following
            # for content — scan the symlink path only if it is a regular file via open
            # without following? open follows. Skip symlink files entirely for safety.
            result.skipped.append({"path": str(path), "reason": "symlink_skipped"})
            return
        if not path.is_file():
            return
    except OSError as exc:
        result.skipped.append({"path": str(path), "reason": f"stat_error:{exc}"})
        return

    name = path.name
    suffix = path.suffix.lower()
    if suffix in BINARY_EXTENSIONS:
        result.skipped.append({"path": str(path), "reason": "binary_extension"})
        return
    if suffix in CREDENTIAL_EXTENSIONS or name.endswith(".pem"):
        result.add(
            Finding(
                "credential_extension",
                str(path),
                None,
                f"credential-like file extension {suffix or name} [REDACTED]",
            )
        )
        return

    # tracked .env (not example)
    if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
        # presence is a finding if under scan root (policy: should not be committed)
        result.add(
            Finding(
                "dotenv_file",
                str(path),
                None,
                ".env-like file present [path only; content not echoed]",
            )
        )

    try:
        size = path.stat().st_size
    except OSError as exc:
        result.skipped.append({"path": str(path), "reason": f"stat_error:{exc}"})
        return
    if size > MAX_SCAN_BYTES:
        result.skipped.append({"path": str(path), "reason": f"too_large:{size}"})
        return

    try:
        with open(path, "rb") as handle:
            head = handle.read(BINARY_SNIFF)
            if is_probably_binary(head):
                result.skipped.append({"path": str(path), "reason": "binary_content"})
                return
            rest = handle.read(MAX_SCAN_BYTES - len(head))
            data = head + rest
    except OSError as exc:
        result.skipped.append({"path": str(path), "reason": f"read_error:{exc}"})
        return

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = data.decode("latin-1")
        except UnicodeDecodeError:
            result.skipped.append({"path": str(path), "reason": "decode_error"})
            return

    is_env_example = name == ".env.example"
    result.scanned_files += 1
    scan_text(path, text, result, is_env_example=is_env_example)


def run_scan(root: Path, *, staged: bool) -> ScanResult:
    result = ScanResult()
    root = root.resolve()
    if not root.is_dir():
        result.errors.append(f"root is not a directory: {root}")
        return result.finalize()
    try:
        paths = staged_paths(root) if staged else list(iter_candidate_paths(root))
    except Exception as exc:  # noqa: BLE001
        result.errors.append(str(exc))
        return result.finalize()
    for path in paths:
        scan_file(path, root, result)
    return result.finalize()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Scan for accidental secrets (read-only)")
    p.add_argument("--root", required=True)
    p.add_argument("--staged", action="store_true")
    p.add_argument("--json-out", default=None)
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else EXIT_CONFIG
        return code or EXIT_CONFIG
    root = Path(args.root)
    result = run_scan(root, staged=bool(args.staged))
    payload = result.to_dict()
    if args.json_out:
        try:
            write_json_atomic(Path(args.json_out), payload)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"json-out failed: {exc}")
            result.exit_code = EXIT_CONFIG
            result.status = "CONFIG_ERROR"
            payload = result.to_dict()
    if not args.quiet:
        print(f"status={result.status} exit_code={result.exit_code} findings={len(result.findings)}")
        for finding in result.findings:
            print(
                f"FINDING: {finding.rule} path={finding.path} line={finding.line} evidence={finding.evidence}"
            )
    return int(result.exit_code)


if __name__ == "__main__":
    sys.exit(main())
