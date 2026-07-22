"""Shared types and constants for data contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from football_analytics.data import DataContractError

FIELD_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")

ALLOWED_SCALAR_TYPES = frozenset(
    {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
        "string",
        "large_string",
        "binary",
        "fixed_size_binary",
        "timestamp",
        "duration",
    }
)
ALLOWED_NESTED_TYPES = frozenset({"list", "fixed_size_list", "struct"})
TIMESTAMP_UNITS = frozenset({"s", "ms", "us", "ns"})
MAX_NEST_DEPTH = 4
MAX_FIXED_LIST_SIZE = 1024
MAX_SPEC_BYTES = 512 * 1024
MAX_VALIDATION_ERRORS = 50

META_CONTRACT = "football_analytics.contract_name"
META_VERSION = "football_analytics.contract_version"
META_FINGERPRINT = "football_analytics.schema_fingerprint"
META_CREATED_BY = "football_analytics.created_by_version"

# Fingerprint excludes descriptive-only keys
FINGERPRINT_EXCLUDE_KEYS = frozenset({"description", "note"})


@dataclass(frozen=True)
class FieldSpec:
    name: str
    type_name: str
    nullable: bool
    value_type: str | None = None
    list_size: int | None = None
    byte_width: int | None = None
    unit: str | None = None
    tz: str | None = None
    fields: tuple[FieldSpec, ...] = ()


@dataclass(frozen=True)
class ForeignKeySpec:
    fields: tuple[str, ...]
    ref_contract: str
    ref_fields: tuple[str, ...]
    ref_version: int
    nullable: bool = False


@dataclass(frozen=True)
class ContractSpec:
    contract_name: str
    version: int
    description: str
    fields: tuple[FieldSpec, ...]
    primary_key: tuple[str, ...]
    foreign_keys: tuple[ForeignKeySpec, ...]
    partition_by: tuple[str, ...]
    sort_by: tuple[str, ...]
    semantic_rules: tuple[dict[str, Any], ...]
    table_metadata: dict[str, str]
    # path is not part of fingerprint
    source_path: str | None = None


@dataclass
class ValidationResult:
    status: str = "PASS"
    contract: str = ""
    version: int = 0
    row_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)

    def err(self, msg: str) -> None:
        if len(self.errors) < MAX_VALIDATION_ERRORS:
            self.errors.append(msg)
        elif len(self.errors) == MAX_VALIDATION_ERRORS:
            self.errors.append("error reporting truncated")
        self.status = "FAIL"

    def warn(self, msg: str) -> None:
        if len(self.warnings) < MAX_VALIDATION_ERRORS:
            self.warnings.append(msg)

    def finalize(self) -> ValidationResult:
        if self.errors:
            self.status = "FAIL"
        elif self.warnings:
            self.status = "PASS_WITH_WARNINGS"
        else:
            self.status = "PASS"
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "contract": self.contract,
            "version": self.version,
            "row_count": self.row_count,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "statistics": dict(self.statistics),
        }


def assert_safe_identifier(value: str, *, allow_null_token: bool = False) -> None:
    if value is None:
        raise DataContractError("identifier is None")
    if not isinstance(value, str) or not value:
        raise DataContractError("identifier empty")
    if "/" in value or "\\" in value or ".." in value or "$HOME" in value or value.startswith("~"):
        raise DataContractError("identifier contains path-like content")
    if any(ord(ch) < 32 for ch in value):
        raise DataContractError("identifier contains control characters")
    if not SAFE_ID_RE.fullmatch(value):
        raise DataContractError("identifier fails safe policy")
