# Schema migrations (Stage 2C)

## Versioning

Each contract has explicit integer versions in `configs/data/schema_registry.yaml`.
Migrations are **adjacent explicit edges** only. Downgrades are rejected by default.

## Legacy detections v0

`schemas/data/v0/detections.json` is a **legacy compatibility fixture contract**
(xywh). It is **not** claimed to be historical production data.

## detections 0 → 1

```text
bbox_x1 = bbox_x
bbox_y1 = bbox_y
bbox_x2 = bbox_x + bbox_width
bbox_y2 = bbox_y + bbox_height
class_id = mapped(class_name) or -1
is_interpolated = false
quality_flags = []
```

Rules:

- Width/height must be positive; NaN/Infinity rejected
- Row count, order, and primary key preserved
- Source Parquet unchanged (hash verified)
- Destination no-overwrite; failure cleans destination
- In-place source mutation forbidden

## Receipt

`schemas/data/migration_receipt.schema.json` — atomic JSON beside destination /
report root with hashes, fingerprints, counts, steps, `lossy=false` for this path.

## CLI

```bash
football-analytics contracts migrate detections src.parquet dst.parquet \
  --from-version 0 --to-version 1
```

## Validator

```bash
python scripts/check_data_contracts.py \
  --registry configs/data/schema_registry.yaml \
  --synthetic-roundtrip --migration-smoke \
  --json-out /home/fdoblak/workspace/data_contract_checks/report.json
```

Runtime fixtures live under `/home/fdoblak/workspace/data_contract_checks/` (Git-outside).
