# Target events (Stage 13)

Canonical append-only target-player event ledger and coverage-aware metric
aggregation over Stage 10–12 sources.

- Policy: `configs/events/events_policy.yaml`
- Contracts: `replay_candidates`, `target_event_ledger`, `event_revisions`
- Semantics: never invent live when replay uncertain; never invent team names;
  conflict → attack direction unknown; no destructive merge
