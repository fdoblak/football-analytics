# Third-party notices (Stage 15B)

Technical notices for adapters and locked third-party sources used by
`football-analytics`. This file is **not** a legal opinion and does **not**
grant production redistribution or SaaS clearance.

## Copyleft / evaluation-only adapters

| Component | License signal | Registry posture | Notes |
|-----------|----------------|------------------|-------|
| Ultralytics YOLO11n (human/ball detection) | AGPL-3.0 (runtime/package) | `approval: evaluation_only`, `production_approved: false` | Weights not in Git; do not invent production approval |
| NBJW / SoccerNet Banner SV_kp / SV_lines | GPL-2.0 (architecture source) | `evaluation_only`, `production_approved: false` | Lazy import from locked absolute paths; weight license `review_required` |
| PnLCalib | GPL-2.0 | reference / candidate only | Integration boundary review before deep use |

## Fallback / no-model behavior

When a gated model is unavailable or license-blocked:

- Prefer CPU stub / skip path
- Emit metrics as `not_evaluable` with reason `MODEL_UNAVAILABLE_OR_LICENSE_GATED`
- Do not invent successful zeros or fake detections

## Authoritative inventory

See `docs/legal/license_inventory.md` and root `model_registry.yaml`.
Legal clearance remains a **Stage 16** external item.
