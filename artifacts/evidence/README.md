# Evidence retention — football-analytics

Small, safe, successful-stage artifacts kept in Git history for continuity.

## What goes here

Under `artifacts/evidence/stage_<id>/`:

- Validator summary JSON
- Receipt / quality summaries
- Tiny synthetic Parquet/JSON/JSONL samples
- Small deterministic SVG/PNG diagrams
- Config/schema fingerprints and SHA-256 provenance

Indexed by `artifacts/evidence/index.json`.

## What never goes in Git

- Full videos, frame/crop dumps
- Model weights / checkpoints
- Datasets / SoccerNet content
- Cache, build, venv, pip freeze
- Secrets / credentials
- User identity / reference photos
- Unlicensed real match frames
- Any single file > 10 MiB
- Backfill batch totaling > 100 MiB

Large/restricted items are recorded as **manifest-only** rows (path, size,
SHA-256, reason) — not uploaded. Do **not** auto-push weights/videos to Git LFS.

## Policy for future stages

After each successful stage:

1. Copy only the important **small** outputs into `artifacts/evidence/stage_<id>/`.
2. Update `index.json` (content-hash dedupe).
3. Stage explicit paths and normal-push to GitHub.
4. Do **not** commit every command log or temporary session directory.

Runtime remains under `/home/fdoblak/workspace/…` and stays gitignored.

## Backfill

`scripts/collect_evidence.py` reads existing workspace check directories **read-only**,
copies safe small JSON summaries, and marks missing dirs as `not_available_cleaned`.
It never deletes or modifies originals.
