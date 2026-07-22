"""Secret-safe redaction for logs, records, and exception context (Stage 2B)."""

from __future__ import annotations

import copy
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

REDACTED = "[REDACTED]"

SENSITIVE_KEY_RE = re.compile(
    r"(password|passwd|token|access[_-]?token|refresh[_-]?token|api[_-]?key|"
    r"secret|private[_-]?key|credential|authorization|cookie|passwd)",
    re.IGNORECASE,
)

BEARER_RE = re.compile(r"(?i)\b(bearer\s+)([A-Za-z0-9\-._~+/]+=*)")
TOKENISH_RE = re.compile(r"(?i)\b((?:gh[pousr]_|github_pat_|sk-|xox[baprs]-)[A-Za-z0-9_\-]{8,})")
URL_USERINFO_RE = re.compile(r"(?i)^(https?://)([^/@\s]+)@")


def is_sensitive_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    return SENSITIVE_KEY_RE.search(key) is not None


def sanitize_remote_url(url: str) -> str:
    """Strip userinfo/credentials from remote URLs; never raise with secrets."""
    if not isinstance(url, str) or not url:
        return url
    try:
        parts = urlsplit(url)
        if parts.username or parts.password or "@" in (parts.netloc or ""):
            host = parts.hostname or ""
            if parts.port:
                host = f"{host}:{parts.port}"
            return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))
    except Exception:  # noqa: BLE001
        return URL_USERINFO_RE.sub(r"\1" + REDACTED + "@", url)
    return url


def redact_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    out = BEARER_RE.sub(rf"\1{REDACTED}", text)
    out = TOKENISH_RE.sub(REDACTED, out)
    out = URL_USERINFO_RE.sub(rf"\1{REDACTED}@", out)
    # Collapse newlines that could break JSONL / log injection (keep readable marker).
    if "\n" in out or "\r" in out:
        out = out.replace("\r", "\\r").replace("\n", "\\n")
    return out


def redact_value(value: Any) -> Any:
    """Deep-copy redaction; does not mutate the input."""
    return _redact(copy.deepcopy(value))


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for key, item in value.items():
            if is_sensitive_key(key):
                out[key] = REDACTED
            else:
                out[key] = _redact(item)
        return out
    if isinstance(value, list):
        return [_redact(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_redact(v) for v in value)
    if isinstance(value, str):
        return redact_text(value)
    return value
