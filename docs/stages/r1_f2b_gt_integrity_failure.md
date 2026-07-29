# R1-F2-B — GT integrity failure (no freeze)

**Gate:** `NO-GO — REVIEWED GT INTEGRITY FAILURE`

User declared 80/80 complete. Automatic geometry/metadata checks passed, but
**pixel-grounded QA** found critical train-split false negatives:

- 37/40 train frames are `completed` with `humans=[]`
- Those frames still contain leftover YOLO proposals and **obvious on-pitch humans**
- Only train frames `0, 5, 15` contain human boxes (43 total)
- Dev (215) / holdout (268) look properly labeled and blind

## Failed train empty frame indices

See `failed_train_empty_frames.json`.

## Next plan (do not start R2)

1. Re-open independent GT review for **TRAIN only**
2. Accept/correct/delete proposals or draw missing humans on every train frame
3. Do not mark complete while obvious humans remain unlabeled
4. Keep existing DEV/HOLDOUT unless user requests re-check
5. Re-run R1-F2-B freeze after train repair

Freeze, fine-tuning, and holdout acceptance were **not** started.
