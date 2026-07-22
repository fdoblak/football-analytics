"""GitHub Actions workflow safety validator (Stage 2D).

Does not call the GitHub API. Parses workflow YAML only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

EXIT_PASS = 0
EXIT_FINDING = 1
EXIT_CONFIG = 2
EXIT_INTEGRITY = 3

DEFAULT_WORKFLOW = Path(".github/workflows/ci.yml")

_ALLOWED_ACTIONS = frozenset({"actions/checkout", "actions/setup-python"})
_ACTION_SHA_RE = re.compile(r"^([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)@([0-9a-f]{40})$")
_SECRETS_RE = re.compile(r"\$\{\{\s*secrets\.")
_FORBIDDEN_SHELL = (
    re.compile(r"(?i)\bsudo\b"),
    re.compile(r"(?i)curl\s[^|\n]*\|\s*(?:ba)?sh"),
    re.compile(r"(?i)wget\s[^|\n]*\|\s*(?:ba)?sh"),
)
_FORBIDDEN_WORDS = (
    re.compile(r"(?i)\bforce[\s_-]*push\b"),
    re.compile(r"(?i)\bdeploy\b"),
    re.compile(r"(?i)\bpublish\b"),
    re.compile(r"(?i)\btwine\b"),
    re.compile(r"(?i)\bpypi\b"),
    re.compile(r"(?i)soccernet"),
    re.compile(r"(?i)\bgit\s+clone\b"),
    re.compile(r"(?i)huggingface\.co"),
    re.compile(r"(?i)model\s*download"),
    re.compile(r"(?i)dataset\s*download"),
)
_IDEAL_REFS = (
    re.compile(r"(?i)pytest"),
    re.compile(r"(?i)check_project"),
    re.compile(r"(?i)check_secrets"),
    re.compile(r"(?i)check_ci_workflow"),
)


class Result:
    def __init__(self) -> None:
        self.status = "PASS"
        self.exit_code = EXIT_PASS
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.extras: dict[str, Any] = {}

    def err(self, msg: str, *, integrity: bool = False) -> None:
        self.errors.append(msg)
        self.exit_code = EXIT_INTEGRITY if integrity else EXIT_FINDING

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def finalize(self, *, strict: bool) -> Result:
        if self.exit_code == EXIT_INTEGRITY or self.errors:
            self.status = "FAIL"
            if self.exit_code == EXIT_PASS:
                self.exit_code = EXIT_FINDING
        elif self.warnings and strict:
            self.status = "FAIL"
            self.exit_code = EXIT_FINDING
        elif self.warnings:
            self.status = "PASS_WITH_WARNINGS"
            self.exit_code = EXIT_PASS
        else:
            self.status = "PASS"
            self.exit_code = EXIT_PASS
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "status": self.status,
            "exit_code": self.exit_code,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "extras": self.extras,
        }


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _branch_includes_main(branches: Any) -> bool:
    items = _as_list(branches)
    for item in items:
        if item == "main" or item == "refs/heads/main":
            return True
        if isinstance(item, dict) and item.get("name") == "main":
            return True
    return False


def _check_triggers(on_block: Any, result: Result) -> None:
    if not isinstance(on_block, dict):
        result.err("workflow 'on' must be a mapping", integrity=True)
        return
    if "pull_request_target" in on_block:
        result.err("pull_request_target is forbidden", integrity=True)

    push = on_block.get("push")
    pr = on_block.get("pull_request")
    if "workflow_dispatch" not in on_block:
        result.err("missing workflow_dispatch trigger", integrity=True)

    def _has_main(event: Any, label: str) -> None:
        if event is None:
            result.err(f"missing {label} trigger", integrity=True)
            return
        if event is True or event == {}:
            # Bare event without branch filter — accept with warning
            result.warn(f"{label} has no branch filter (expected main)")
            return
        if not isinstance(event, dict):
            result.err(f"{label} trigger malformed", integrity=True)
            return
        branches = event.get("branches")
        if not _branch_includes_main(branches):
            result.err(f"{label} must include main branch", integrity=True)

    _has_main(push, "push")
    _has_main(pr, "pull_request")


def _check_permissions(perms: Any, result: Result) -> None:
    if not isinstance(perms, dict):
        result.err("top-level permissions must be a mapping", integrity=True)
        return
    contents = perms.get("contents")
    if contents != "read":
        result.err("permissions.contents must be 'read'", integrity=True)
    for key, value in perms.items():
        if value in {"write", "write-all", "read-write"} or (
            isinstance(value, str) and "write" in value.lower() and value != "read"
        ):
            result.err(f"write permission forbidden: {key}={value}", integrity=True)


def _check_jobs(jobs: Any, result: Result, raw_text: str) -> None:
    if not isinstance(jobs, dict) or not jobs:
        result.err("workflow must define jobs", integrity=True)
        return
    for job_name, job in jobs.items():
        if not isinstance(job, dict):
            result.err(f"job {job_name} malformed", integrity=True)
            continue
        if "timeout-minutes" not in job:
            result.err(f"job {job_name} missing timeout-minutes", integrity=True)
        steps = job.get("steps") or []
        if not isinstance(steps, list):
            result.err(f"job {job_name} steps must be a list", integrity=True)
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            uses = step.get("uses")
            if isinstance(uses, str):
                match = _ACTION_SHA_RE.fullmatch(uses)
                if not match:
                    result.err(
                        f"action must be owner/name@40hexsha: {uses}",
                        integrity=True,
                    )
                else:
                    owner_name = f"{match.group(1)}/{match.group(2)}"
                    if owner_name not in _ALLOWED_ACTIONS:
                        result.err(f"third-party action forbidden: {owner_name}", integrity=True)
                    if owner_name == "actions/checkout":
                        with_block = step.get("with") or {}
                        if (
                            isinstance(with_block, dict)
                            and with_block.get("persist-credentials") is not False
                        ):
                            result.err(
                                "actions/checkout must set persist-credentials: false",
                                integrity=True,
                            )
                    if owner_name == "actions/setup-python":
                        with_block = step.get("with") or {}
                        if isinstance(with_block, dict):
                            pyver = with_block.get("python-version")
                            if str(pyver) != "3.10.20":
                                result.err(
                                    'python-version must be exactly "3.10.20"',
                                    integrity=True,
                                )

    # Text-level forbidden patterns across the whole workflow document.
    if _SECRETS_RE.search(raw_text):
        result.err("secret interpolation (${{ secrets. }}) is forbidden", integrity=True)
    for pattern in _FORBIDDEN_SHELL:
        if pattern.search(raw_text):
            result.err(f"forbidden shell pattern matched: {pattern.pattern}", integrity=True)
    for pattern in _FORBIDDEN_WORDS:
        if pattern.search(raw_text):
            result.err(f"forbidden workflow content: {pattern.pattern}", integrity=True)

    missing_ideal = [p.pattern for p in _IDEAL_REFS if not p.search(raw_text)]
    if missing_ideal:
        result.warn(f"workflow missing recommended references: {missing_ideal}")


def run_ci_workflow_checks(
    *,
    workflow: Path,
    project_root: Path | None = None,
    strict: bool = False,
) -> Result:
    root = project_root or Path(__file__).resolve().parents[3]
    path = Path(workflow)
    if not path.is_absolute():
        path = (root / path).resolve()

    result = Result()
    result.extras["workflow_path"] = str(path)
    if not path.is_file():
        result.errors.append(f"workflow missing: {path}")
        result.status = "FAIL"
        result.exit_code = EXIT_CONFIG
        return result

    try:
        raw_text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw_text)
    except Exception as exc:  # noqa: BLE001
        result.err(f"workflow YAML parse failed: {type(exc).__name__}", integrity=True)
        return result.finalize(strict=strict)

    if not isinstance(data, dict):
        result.err("workflow root must be a mapping", integrity=True)
        return result.finalize(strict=strict)

    on_block = data.get("on") or data.get(True)  # YAML may parse 'on' as True
    if on_block is None and True in data:
        on_block = data[True]
    _check_triggers(on_block, result)

    perms = data.get("permissions")
    if perms is None:
        result.err("top-level permissions missing", integrity=True)
    else:
        _check_permissions(perms, result)

    concurrency = data.get("concurrency")
    if isinstance(concurrency, dict):
        if concurrency.get("cancel-in-progress") is not True:
            result.warn("concurrency.cancel-in-progress should be true")
        result.extras["concurrency_present"] = True
    else:
        result.extras["concurrency_present"] = False

    _check_jobs(data.get("jobs"), result, raw_text)
    return result.finalize(strict=strict)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 2D CI workflow safety validator")
    p.add_argument("--workflow", default=str(DEFAULT_WORKFLOW))
    p.add_argument("--json-out")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = run_ci_workflow_checks(workflow=Path(args.workflow), strict=bool(args.strict))
    except Exception as exc:  # noqa: BLE001
        print(f"status=FAIL exit_code={EXIT_CONFIG} error={type(exc).__name__}", file=sys.stderr)
        return EXIT_CONFIG

    payload = result.to_dict()
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.quiet:
        for w in result.warnings:
            print(f"WARNING: {w}")
        for e in result.errors:
            print(f"ERROR: {e}")
        print(f"status={result.status} exit_code={result.exit_code}")
    return int(result.exit_code)
