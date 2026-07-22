"""Stage 3B probe service: safe FFprobe → VideoProbe → policy → atomic outputs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from football_analytics.core.hashing import hash_canonical_json, sha256_file
from football_analytics.core.records import RecordError, write_json_record
from football_analytics.core.run_id import generate_run_id
from football_analytics.video.ffprobe import (
    ProbeError,
    decode_ffprobe_json,
    get_ffprobe_version,
    resolve_ffprobe_binary,
    run_ffprobe,
)
from football_analytics.video.media_validation import (
    MediaValidationResult,
    validate_probe_against_policy,
)
from football_analytics.video.probe_parser import map_ffprobe_json_to_video_probe
from football_analytics.video.types import (
    ContractFingerprints,
    IngestReceipt,
    Issue,
    ProvenanceInfo,
    ReceiptProvenance,
    ReceiptStatus,
    SourceKind,
    VideoProbe,
    VideoSource,
    VideoSourceError,
)
from football_analytics.video.validation import (
    assert_extension_allowed,
    assert_safe_output_root,
    assert_safe_source_path,
    reject_unsafe_path_string,
)


@dataclass(frozen=True)
class SourceSnapshot:
    path: str
    size_bytes: int
    sha256: str
    st_dev: int
    st_ino: int
    mtime_ns: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "st_dev": self.st_dev,
            "st_ino": self.st_ino,
            "mtime_ns": self.mtime_ns,
        }


@dataclass
class ProbeServiceResult:
    accepted: bool
    exit_code: int
    probe: VideoProbe | None
    validation: MediaValidationResult | None
    receipt: IngestReceipt
    output_dir: str
    artifacts: dict[str, str]
    error_code: str | None = None

    def to_summary(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "exit_code": self.exit_code,
            "error_code": self.error_code,
            "output_dir": self.output_dir,
            "artifacts": dict(self.artifacts),
            "receipt_status": self.receipt.status.value,
            "source_id": self.receipt.source_id,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _path_safe_id(prefix: str) -> str:
    # run-id style compact token without unsafe chars
    token = generate_run_id().split("_")[-1]
    return f"{prefix}_{token}"


def snapshot_source(path: Path) -> SourceSnapshot:
    st = path.lstat()
    digest = sha256_file(path)
    return SourceSnapshot(
        path=str(path),
        size_bytes=int(st.st_size),
        sha256=digest,
        st_dev=int(st.st_dev),
        st_ino=int(st.st_ino),
        mtime_ns=int(st.st_mtime_ns),
    )


def assert_snapshots_equal(before: SourceSnapshot, after: SourceSnapshot) -> None:
    if (
        before.size_bytes != after.size_bytes
        or before.sha256 != after.sha256
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.mtime_ns != after.mtime_ns
    ):
        raise ProbeError("SOURCE_MUTATED_DURING_PROBE", "source changed during probe")


def _sanitize_message(msg: str) -> str:
    # Avoid dumping long absolute home paths repeatedly
    return msg.replace("/home/fdoblak/", "~/").replace("\x00", "")


def run_media_probe(
    *,
    source: str,
    output_dir: str,
    policy: Mapping[str, Any],
    contain_root: str | Path | None = None,
    source_id: str | None = None,
    run_id: str | None = None,
    ffprobe_runner: Callable[..., Any] | None = None,
) -> ProbeServiceResult:
    """Execute Stage 3B probe pipeline. Returns structured result + atomic artifacts."""
    started = _utc_now()
    request_id = _path_safe_id("req")
    receipt_id = _path_safe_id("rcpt")
    sid = source_id or _path_safe_id("src")
    rid = run_id or generate_run_id()
    artifacts: dict[str, str] = {}
    ff = policy["ffprobe_policy"]
    root = Path(contain_root) if contain_root else Path(str(ff["runtime_root"]))
    root.mkdir(parents=True, exist_ok=True)

    def fail(
        code: str,
        message: str,
        *,
        exit_code: int,
        status: ReceiptStatus = ReceiptStatus.FAILED,
        probe: VideoProbe | None = None,
        validation: MediaValidationResult | None = None,
        source_sha: str = "0" * 64,
        source_size: int = 0,
    ) -> ProbeServiceResult:
        completed = _utc_now()
        fps = ContractFingerprints(
            source=source_sha if len(source_sha) == 64 else "0" * 64,
            request="0" * 64,
            probe=None if probe is None else probe.fingerprint(),
        )
        # Fix fingerprints when we have probe/source later — use zeros only for hard fail early
        receipt = IngestReceipt(
            receipt_id=receipt_id,
            request_id=request_id,
            run_id=rid,
            source_id=sid,
            source_sha256=(
                source_sha
                if len(source_sha) == 64 and all(c in "0123456789abcdef" for c in source_sha)
                else "0" * 64
            ),
            source_size_bytes=max(0, source_size),
            status=status,
            started_at_utc=started,
            completed_at_utc=completed,
            probe_record_ref=artifacts.get("video_probe.json"),
            normalize_plan_ref=None,
            artifact_refs=tuple(artifacts.values()),
            policy_version=str(policy["policy_version"]),
            contract_fingerprints=fps,
            warnings=tuple(validation.warnings) if validation else (),
            errors=(Issue(code=code, message=_sanitize_message(message)),)
            + (tuple(validation.errors) if validation else ()),
            provenance=ReceiptProvenance(
                stage="3A", label="stage3b_probe", notes="probe execution"
            ),
        )
        # Note: provenance.stage must be "3A" per Stage 3A schema const — document Stage 3B in label
        try:
            out = Path(output_dir)
            write_json_record(
                out / "probe_execution_receipt.json",
                receipt.to_dict(),
                contain_root=root,
                overwrite=False,
            )
            artifacts["probe_execution_receipt.json"] = str(out / "probe_execution_receipt.json")
        except Exception:  # noqa: BLE001
            pass
        return ProbeServiceResult(
            accepted=False,
            exit_code=exit_code,
            probe=probe,
            validation=validation,
            receipt=receipt,
            output_dir=str(output_dir),
            artifacts=artifacts,
            error_code=code,
        )

    # --- validate paths ---
    try:
        reject_unsafe_path_string(source, label="source")
        reject_unsafe_path_string(output_dir, label="output_dir")
        src_path = assert_safe_source_path(source, contain_root=root, policy=policy)
        out_path = assert_safe_output_root(
            output_dir,
            contain_root=root,
            source_path=str(src_path),
            overwrite_allowed=False,
        )
        out_path.mkdir(parents=True, exist_ok=True)
        assert_extension_allowed(src_path, policy)
    except (VideoSourceError, ProbeError, OSError, ValueError) as exc:
        code = getattr(exc, "code", None) or "SOURCE_NOT_REGULAR_FILE"
        return fail(str(code), str(exc), exit_code=3)

    # pre snapshot
    try:
        before = snapshot_source(src_path)
    except Exception as exc:  # noqa: BLE001
        return fail("SOURCE_HASH_MISMATCH", str(exc), exit_code=3)

    # binary / version
    try:
        binary = resolve_ffprobe_binary(
            ff["ffprobe_binary"], allowed_realpaths=list(ff["allowed_binary_realpaths"])
        )
        version = get_ffprobe_version(binary)
    except ProbeError as exc:
        return fail(
            exc.code,
            exc.message,
            exit_code=4,
            source_sha=before.sha256,
            source_size=before.size_bytes,
        )

    # run ffprobe
    try:
        runner = ffprobe_runner or run_ffprobe
        raw = runner(src_path, policy=policy, binary=binary, version=version)
        data = decode_ffprobe_json(raw.stdout, max_depth=int(ff["maximum_json_depth"]))
        probe = map_ffprobe_json_to_video_probe(
            data,
            source_id=sid,
            source_sha256=before.sha256,
            file_size_bytes=before.size_bytes,
            probe_tool_version=version.version_token,
            probed_at_utc=_utc_now(),
            max_stream_count=int(ff["maximum_stream_count"]),
        )
    except ProbeError as exc:
        return fail(
            exc.code,
            exc.message,
            exit_code=(
                4 if exc.code.startswith("PROBE_") or exc.code == "FFPROBE_NOT_AVAILABLE" else 1
            ),
            source_sha=before.sha256,
            source_size=before.size_bytes,
        )
    except Exception as exc:  # noqa: BLE001
        return fail(
            "PROBE_UNEXPECTED_STRUCTURE",
            str(exc),
            exit_code=4,
            source_sha=before.sha256,
            source_size=before.size_bytes,
        )

    # post snapshot
    try:
        after = snapshot_source(src_path)
        assert_snapshots_equal(before, after)
    except ProbeError as exc:
        return fail(
            exc.code,
            exc.message,
            exit_code=3,
            probe=probe,
            source_sha=before.sha256,
            source_size=before.size_bytes,
        )

    validation = validate_probe_against_policy(probe, policy, source_size_bytes=before.size_bytes)

    # Build VideoSource for fingerprint linkage
    vsource = VideoSource(
        source_id=sid,
        source_kind=(
            SourceKind.SYNTHETIC_FIXTURE
            if str(root).endswith("video_probe_checks")
            else SourceKind.USER_LOCAL_VIDEO
        ),
        original_filename=src_path.name,
        source_path=str(src_path),
        source_size_bytes=before.size_bytes,
        source_sha256=before.sha256,
        media_type="video/mp4",
        container_hint=probe.container,
        created_at_utc=started,
        registered_at_utc=started,
        immutability_policy="detect_mutation",
        provenance=ProvenanceInfo(origin="local_file", label="stage3b_probe"),
    )

    # Atomic outputs (no overwrite)
    try:
        write_json_record(out_path / "video_probe.json", probe.to_dict(), contain_root=root)
        artifacts["video_probe.json"] = str(out_path / "video_probe.json")
        write_json_record(
            out_path / "media_validation.json", validation.to_dict(), contain_root=root
        )
        artifacts["media_validation.json"] = str(out_path / "media_validation.json")
    except RecordError as exc:
        return fail(
            "OUTPUT_WRITE_FAILED",
            str(exc),
            exit_code=3,
            probe=probe,
            validation=validation,
            source_sha=before.sha256,
            source_size=before.size_bytes,
        )

    status = ReceiptStatus.VALIDATED if validation.accepted else ReceiptStatus.REJECTED
    receipt = IngestReceipt(
        receipt_id=receipt_id,
        request_id=request_id,
        run_id=rid,
        source_id=sid,
        source_sha256=before.sha256,
        source_size_bytes=before.size_bytes,
        status=status,
        started_at_utc=started,
        completed_at_utc=_utc_now(),
        probe_record_ref=artifacts.get("video_probe.json"),
        normalize_plan_ref=None,
        artifact_refs=tuple(sorted(artifacts.values())),
        policy_version=str(policy["policy_version"]),
        contract_fingerprints=ContractFingerprints(
            source=vsource.fingerprint(),
            request=hash_canonical_json(
                {
                    "request_id": request_id,
                    "run_id": rid,
                    "source_id": sid,
                    "source_sha256": before.sha256,
                    "output_dir": str(out_path),
                }
            ),
            probe=probe.fingerprint(),
        ),
        warnings=tuple(validation.warnings),
        errors=tuple(validation.errors),
        provenance=ReceiptProvenance(
            stage="3A", label="stage3b_probe", notes="ffprobe media validation"
        ),
    )
    try:
        write_json_record(
            out_path / "probe_execution_receipt.json", receipt.to_dict(), contain_root=root
        )
        artifacts["probe_execution_receipt.json"] = str(out_path / "probe_execution_receipt.json")
    except RecordError as exc:
        return fail(
            "OUTPUT_WRITE_FAILED",
            str(exc),
            exit_code=3,
            probe=probe,
            validation=validation,
            source_sha=before.sha256,
            source_size=before.size_bytes,
        )

    return ProbeServiceResult(
        accepted=validation.accepted,
        exit_code=0 if validation.accepted else 1,
        probe=probe,
        validation=validation,
        receipt=receipt,
        output_dir=str(out_path),
        artifacts=artifacts,
        error_code=(
            None
            if validation.accepted
            else (validation.errors[0].code if validation.errors else "REJECTED")
        ),
    )
