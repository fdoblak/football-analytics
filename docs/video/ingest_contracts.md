# Safe video ingest contracts (Stage 3A)

## Purpose

Define immutable, validated contracts for **local** video sources so later
stages can probe, normalize, and time-map frames without silent corruption.
Stage 3A does **not** ingest real user matches, does not run FFprobe/FFmpeg
pipelines, and does not compute player metrics.

## Contract graph

```text
VideoSource ──► IngestRequest ──► (planned) VideoProbe
                     │                    │
                     └────────► NormalizePlan
                                    │
                                    ▼
                              IngestReceipt
```

| Contract | Schema | Role |
|----------|--------|------|
| Video source | `schemas/video/video_source.schema.json` | Identity, hash/size, kind, immutability |
| Ingest request | `schemas/video/ingest_request.schema.json` | Validate-only / plan-only request |
| Video probe | `schemas/video/video_probe.schema.json` | Future FFprobe-shaped metadata |
| Normalize plan | `schemas/video/normalize_plan.schema.json` | Declarative plan (no execution) |
| Ingest receipt | `schemas/video/ingest_receipt.schema.json` | Contract-level outcome |

Policy: `configs/video/ingest_policy.yaml` (`video_ingest_policy_v1`).

## Source immutability

- Sources are treated as immutable bytes.
- `source_sha256` + `source_size_bytes` bind identity.
- Re-hash must match; size/hash drift ⇒ reject (mutation / wrong file).
- Mid-hash TOCTOU: `sha256_file` compares lstat before/after; residual race is documented — do not silently accept.

## Path safety

Reject: `..`, `~`, env placeholders, null bytes, URL schemes, symlinks,
FIFO/socket/devices, directories-as-video, dangerous roots, source/output
collision, overwrite.

Containment uses Stage 1 helpers (`assert_contained`, dangerous-root checks)
and Stage 2 `sha256_file`.

## Stream selection

Deterministic: ignore `attached_pic`; prefer largest `width*height`; tie-break
lowest `stream_index`. Audio optional.

## Codec / container policy

Allowlists in policy YAML for extensions, containers, video/audio codecs, and
pixel formats. Extension is a hint only — Stage 3B probe is authoritative.

## Time base / VFR / CFR

- Canonical unit: integer microseconds (`time_us`), video-relative.
- Broadcast clock ≠ PTS (future match-clock contract).
- Frame index is **not** canonical time.
- VFR: never use `frame_index / fps`.
- `r_frame_rate`, `avg_frame_rate`, and true cadence are distinct.
- Rational rates: `{numerator, denominator}` with `denominator >= 1`.
- Unknown `duration_us` / `frame_count` stay **null** (never coerced to 0).
- Negative / non-monotonic / duplicate PTS policies belong to Stage 3B+; contracts reserve warning codes.

## Rotation / SAR / DAR

- Rotation normalized to `{0,90,180,270}` (signed inputs mapped once).
- SAR/DAR stored as rationals; resize must keep aspect (`fit_within_keep_aspect`).
- Double-apply of rotation is forbidden by policy.

## Normalize plan

Plan-only in 3A. Default `overwrite_policy=false`, no in-place write, fingerprint
deterministic via canonical JSON. VFR→CFR must be explicit (`force_cfr`).

## Receipt statuses

Allowlist: `planned`, `validated`, `rejected`, `failed`.

**No** `succeeded` / `completed` in Stage 3A — those require real ingest
execution in later stages.

## Fixture policy

Tiny synthetic media/metadata under
`/home/fdoblak/workspace/video_contract_checks/` only. Not Git-tracked.
Cleanup required. No real match videos / network / datasets.

## Not done in Stage 3A

FFprobe runner, FFmpeg normalize execution, frame extraction, detection,
tracking, ReID, metrics, Opta, GUI.

## Boundary with Stage 3B

Stage 3B consumes these contracts to run **safe FFprobe** and emit real probe
records — without changing the Stage 3A schema semantics.
