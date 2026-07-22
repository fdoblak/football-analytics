# ADR-0001 — Stage-Gate Governance

- **Status:** Accepted (Stage 0, 2026-07-22)
- **Decision makers:** Furkan Doblak (owner); implementing agent(s) bound by this ADR
- **Applies to:** `/home/fdoblak/projects/football-analytics` and all dependent stages

## Context

The project mixes a thin application repo, many external SoccerNet/third-party codebases, isolated environments, limited 4 GB GPU memory, and incomplete durable storage (`/mnt/d` not mounted). Without hard stage gates, scope creep, irreversible installs/downloads, and unverified “done” claims are likely.

## Decision

All work proceeds through explicit stage gates. No stage may begin without a dedicated brief and user approval. No stage is complete without a completion document and required evidence.

## Stage-gate rules (normative)

1. **Separate brief per stage.** Each aşama starts only with a written brief that states goals, allowed actions, forbidden actions, and required artifacts.
2. **Mandatory artifacts.** Each stage lists required files, schemas, manifests, and reports. Missing artifacts mean the stage is incomplete.
3. **Tests at the required level.** Unit, schema, smoke, and regression tests run at the depth the stage brief requires. Absence of a test harness is recorded as a finding, not silently ignored.
4. **No completion without real fixture or benchmark evidence.** Claims of detection/tracking/calibration/event quality require golden clips and/or official benchmark evidence appropriate to that stage.
5. **Performance telemetry when required.** Stages that run models must record FPS, runtime, RAM, and peak VRAM when the brief requires it.
6. **Version pinning in records.** Input, config, model, external repo commit, and schema versions are recorded for reproducible runs.
7. **Honest unknowns.** Unreliable outputs use `null + confidence + reason`; values are never invented to fill gaps.
8. **Isolated external engines.** Third-party code is used via adapter/config/subprocess and dedicated environments; it is not vendored into the core package by default.
9. **Preserve user files.** Existing user modifications and data are not deleted, overwritten, reformatted, or force-cleaned by agents.
10. **VCS mutation requires approval.** `git commit`, tag, and push require separate explicit user approval.
11. **Completion certificate required.** A stage completion document with gate decision (`GO` / `PASS_WITH_FINDINGS` / `NO-GO`) is mandatory before the next stage.
12. **Human start authority.** Furkan must explicitly approve starting the next stage; agents must not auto-advance.

## Gate outcomes

| Outcome | Meaning |
|---------|---------|
| `GO` | Evidence complete; blockers for later stages are explicit; safe to request next-stage approval |
| `PASS_WITH_FINDINGS` | Stage purpose met; known issues documented; may proceed only after user acknowledges findings |
| `NO-GO` | Evidence insufficient or integrity uncertain; stop |

## Consequences

- Stage 0 produces audit, scope freeze, risk register, this ADR, and completion evidence only.
- Later stages (storage layout, schemas, detection, etc.) remain blocked until approved.
- Violations of forbidden actions in a brief invalidate the stage’s `GO` claim.

## References

- `docs/scope/product_v1_scope.md`
- `docs/risks/risk_register.md`
- `docs/audits/current_state_20260722.md`
- `docs/stages/stage_00_completion.md`
