# R1-F2-B-R1 — Repair validation, GT freeze, fine-tune, holdout

| Field | Value |
|-------|-------|
| Gate | **NO-GO — FINE-TUNED HUMAN DETECTOR HOLDOUT FAILURE** |
| Frozen GT | `annotations/own_video_97b298e4/human_detection_v1/` |
| Holdout | one-shot after dev selection; no retrain |

Repair 37/37 passed; dev/holdout fingerprints unchanged; GT frozen.
Fine-tuned YOLO11n (A) and YOLO11s (B); selected **B** on dev.
Holdout did **not** meet acceptance thresholds (clip-specific failure).
No acceptance MP4/PNG retained. R2 not started.
