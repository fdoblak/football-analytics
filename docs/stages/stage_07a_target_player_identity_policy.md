# Target-player identity policy (Stage 7A)

Canonical machine-readable policy:

`configs/identity/identity_evidence_policy.yaml`

## Product rules

1. One paying `target_player` — not a team report.
2. Track ID is never player identity.
3. Display name / jersey / team / role / appearance alone cannot confirm.
4. Manual verified anchors are strongest but **scoped** to track + time.
5. Assignments are append-only; revoke supersedes; revoked ≠ metric-eligible.
6. Unknown is safe; never invent customer metrics for ambiguous identity.
7. Cross-video identity requires explicit manual/provenance — no auto link.
8. **Face recognition and biometric identity are forbidden.**
9. Evaluation labels must not leak into production features.
10. False target attribution is the critical product error.

Stage 7A encodes these as contracts + validators only; producers arrive later.
