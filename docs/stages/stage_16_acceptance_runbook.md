# Stage 16 acceptance runbook — Real-match validation

**Status:** NOT STARTED — wait for an explicit user prompt.

This runbook is documentation only. Stage 15 closed all implementation stages.
Do **not** invent Opta accuracy, legal clearance, `/mnt/d` readiness, or green remote CI.

## Preconditions

1. Stage 15 gate evidence present under `artifacts/evidence/stage_15/`
2. Reviewed ground-truth available for the target video (or explicitly mark `NOT_EVALUATED`)
3. Manual identity confirmation path ready when identity is uncertain
4. Legal review outcome recorded before any production redistribution of AGPL/GPL adapters
5. Optional: verified independent archive root (do not claim `/mnt/d` until probed)

## Acceptance scenarios

| Scenario | Expected |
|----------|----------|
| Positive real match | Metrics evaluable where coverage/identity/calibration allow |
| Negative / no-event segments | Explicit empty-with-coverage; never fake zeros as success |
| Ambiguous identity / ball / pass | Review hub required; no auto-confirm |
| Low coverage | `insufficient_coverage` / `not_evaluable` |
| Not evaluable calibration | Physical/zone metrics null with reason codes |

## Final visual (reserved)

Only after acceptance:

- `/home/fdoblak/football_data/rendered_outputs/final/single_player_analysis_summary.png`
- `artifacts/final/single_player_analysis_summary.png`

Synthetic Stage 14/15 renders must **not** be copied to these paths as customer finals.

## External findings to resolve (or re-document)

| ID | Item |
|----|------|
| Legal | AGPL Ultralytics / GPL NBJW production clearance |
| RISK-042 | GitHub API / remote CI visibility |
| Storage | Same-VHDX vs independent `/mnt/d` backup |
| Accuracy | Real football / Opta claims only with reviewed GT |

## Commands (when Stage 16 is explicitly started)

```text
# placeholders — implement only under Stage 16 prompt
python scripts/check_prerelease_hardening.py   # still green from Stage 15
# + real-match validators / evidence under artifacts/evidence/stage_16/
```

## Gate (expected shape — not claimed yet)

Real-match acceptance gate text will be defined in the Stage 16 prompt. Until then:
`ONLY REAL-MATCH ACCEPTANCE STAGE 16 REMAINS`.
