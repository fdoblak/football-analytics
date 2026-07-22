# Stage 3A completion — Safe video ingest contracts & fixture design

## 1. Purpose

Establish immutable local-video ingest contracts, policy, typed models, path/hash
safety, synthetic fixture design, validators, and tests — without executing
FFprobe/FFmpeg ingest pipelines or computing player metrics.

## 2. Starting SHA

`b5a73f3116ca92276e3897898cb28df65d586e7a`
(`Document manual Cursor workflow after removing continuous automation.`)

## 3. Baseline

- pytest: **465 passed**
- ruff / black / isort / mypy: PASS
- storage PASS; registries PASS_WITH_WARNINGS (known license/access)
- secrets findings=0; runtime/data/stage-cache PASS
- project quick PASS_WITH_WARNINGS (git metadata warning only in sandbox)

## 4. Preservation review

Path: `/home/fdoblak/workspace/manual_handoff/stage3a_preservation_20260722T210513Z`
Manifest SHA-256: `9e690246b520bb3204526f56d10a1f7380233c81e88aea2185513f7cc31e7d9e` (matched).
No symlink escape; text-only project drafts; no video/model/secret binaries.
Preservation folder left unchanged.

## 5. Reuse / rewrite decisions

| Draft file | Classification | Notes |
|------------|----------------|-------|
| `configs/video/ingest_policy.yaml` | REUSE_WITH_CHANGES | Expanded full policy; kept safe defaults |
| `docs/stages/stage_03_roadmap.md` | REUSE_WITH_CHANGES | Rewrote to 3A–3D boundaries only |
| `schemas/video/video_source.schema.json` | REJECT_AND_REWRITE | Incomplete fields vs Stage 3A spec |
| `schemas/video/ingest_request.schema.json` | REJECT_AND_REWRITE | Incomplete; missing path/fixture fields |
| `schemas/video/ffprobe_metadata.schema.json` | REJECT_AND_REWRITE | Renamed to `video_probe.schema.json`; full stream model |
| `schemas/video/normalize_plan.schema.json` | REJECT_AND_REWRITE | Added fingerprint + allowlisted policies |
| `schemas/video/ingest_receipt.schema.json` | REJECT_AND_REWRITE | Statuses `planned/validated/rejected/failed` (no false success) |
| `src/football_analytics/video/types.py` | REJECT_AND_REWRITE | Full typed contracts + Rational + selection |

Draft was **not** bulk-copied into the repo.

## 6. Contracts created

- `schemas/video/video_source.schema.json`
- `schemas/video/ingest_request.schema.json`
- `schemas/video/video_probe.schema.json`
- `schemas/video/normalize_plan.schema.json`
- `schemas/video/ingest_receipt.schema.json`

## 7. Policy

`configs/video/ingest_policy.yaml` — `video_ingest_policy_v1`
Defaults: network/symlink/special/overwrite = false; hash=sha256; time=µs.

## 8. Typed models

`src/football_analytics/video/{types,contracts,validation,fixtures}.py`
Frozen dataclasses, strict `from_dict`/`to_dict`, canonical fingerprints,
no FFmpeg/OpenCV/Torch imports on module load.

## 9. Path / source safety

Reuses Stage 1 containment + Stage 2 `sha256_file`. Rejects `..`, `~`, env,
null byte, URL, symlink, FIFO/special, directory-as-video, collisions, overwrite.

## 10. Source hash & immutability

Streaming SHA-256 with before/after lstat mutation detection; size/hash mismatch
rejects; TOCTOU residual documented.

## 11. Time-base / VFR / CFR

Canonical integer microseconds; rationals for rates; unknown duration/frame_count
remain null; VFR must not use index/fps; CFR not assumed without evidence.

## 12. Stream selection

Ignore attached pictures; largest area; lowest index tie-break.

## 13. Rotation / SAR / DAR

Rotation normalized to {0,90,180,270}; SAR/DAR as rationals; aspect-preserving
resize policy.

## 14. Normalize plan

Plan-only; deterministic `plan_fingerprint`; overwrite forced false; no execution.

## 15. Receipt statuses

`planned` | `validated` | `rejected` | `failed` — no succeeded/completed in 3A.

## 16–18. Fixtures

Design + tiny FFmpeg CFR generators under
`/home/fdoblak/workspace/video_contract_checks/`; session cleanup verified;
not Git-tracked.

## 19. Validator

`scripts/check_video_contracts.py` — PASS
Runtime report:
`/home/fdoblak/workspace/video_contract_checks/video_contract_validation_20260722T212903Z.json`

## 20–21. Tests

New video tests: **41**
Total pytest: **506 passed** (was 465).

## 22–25. Quality

Ruff PASS · Black PASS · isort PASS · mypy PASS (51 files)

## 26. Build

`python -m build` PASS; sdist includes `schemas/video` + `configs/video`;
wheel includes `football_analytics.video`; no media/models/secrets in artifacts.

## 27. Other validators

runtime/data/stage-cache/storage PASS; registries PASS_WITH_WARNINGS (pre-existing);
secrets findings=0; project deep PASS_WITH_WARNINGS (dirty tree pre-commit).

## 28. Protected packages

Unchanged: torch 2.11.0+cu128, torchvision 0.26.0+cu128, torchaudio 2.11.0+cu128,
numpy 2.2.6, pandas 2.3.3, opencv-python/headless 5.0.0.93, ultralytics 8.4.91,
SoccerNet 0.1.62, pyarrow 25.0.0.

## 29. Changed files

See git commit file list (video schemas/configs/modules/tests/docs + MANIFEST.in).

## 30. Findings

- Registry license/access warnings remain (pre-existing; not introduced).
- Project deep WARN on dirty tree before commit (expected).

## 31. Acceptance

All Stage 3A acceptance criteria met for contracts/fixture design scope.

## 32. Gate

`PASS — SAFE VIDEO INGEST CONTRACTS ACTIVE`

## 33. Next stage (name only)

Aşama 3B — Güvenli FFprobe ve Medya Doğrulama
