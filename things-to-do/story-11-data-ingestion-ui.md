# Story 11 — Data Ingestion Pipeline & UI

## Problem
All master data (item, location, customer, DFU) and fact data (sales, forecast, inventory) is loaded via CLI commands (`make normalize-<domain>` + `make load-<domain>`). There is no frontend upload UI, no validation preview before commit, no batch monitoring dashboard, and no quarantine resolution workflow. This makes data onboarding manual, error-prone, and inaccessible to non-technical users.

## Current State
- CLI pipeline exists: `normalize_dataset_csv.py` → `load_dataset_postgres.py`
- Medallion pipeline exists: Bronze (TEXT) → Silver (typed + DQ gate + auto-fixes) → Gold (production)
- DQ/Medallion APIs exist (`/data-quality/lineage/batches`, `/quarantine/*`, `/corrections/*`) but no frontend components wired to them
- Only 1 upload endpoint exists: `POST /supply/open-pos/upload` (PO-specific, not generic)
- ExplorerTab (read-only domain browser) and DataQualityTab (health scores) exist

## Missing Capabilities
- Generic file upload API for any domain
- Validation preview before committing data
- Upsert mode for dimension tables (current pipeline always TRUNCATEs)
- Batch monitoring with pipeline progress visualization
- Quarantine resolution UI (inline edit, bulk resolve)
- Lineage visualization (upload → bronze → silver → gold flow)

## Incremental Implementation

### Phase 1: Backend Core (1 sprint)

**New files:**
- `config/ingestion_config.yaml` — max upload size, preview rows, allowed extensions, per-domain load mode (upsert/replace/append)
- `common/normalize.py` — extracted from `scripts/normalize_dataset_csv.py`:
  - `normalize_stream(spec, input_stream, delimiter)` → CSV StringIO ready for bronze
  - `detect_delimiter(sample)` → auto-detect from first 5 lines
  - `validate_headers(spec, headers)` → matched/unmatched/missing report
- `api/routers/ingestion.py` — new router, prefix `/ingestion`, uses `get_conn()` pattern

**Modify:**
- `common/medallion.py` — extend `promote_to_gold()` with `mode` param: `"replace"` (existing TRUNCATE), `"upsert"` (INSERT ON CONFLICT DO UPDATE), `"append"` (INSERT without TRUNCATE)
- `api/main.py` — mount ingestion router before `domains.py`
- `frontend/vite.config.ts` — add `/ingestion` proxy entry

**API Endpoints:**

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/ingestion/domains` | List uploadable domains with last upload date, row count |
| GET | `/ingestion/domains/{domain}/spec` | Column spec (names, types, required, business keys) |
| POST | `/ingestion/{domain}/preview` | Dry-run: parse file → column mapping + sample rows + DQ preview |
| POST | `/ingestion/{domain}/confirm/{preview_id}` | Run full medallion pipeline on previewed data → returns batch_id |
| GET | `/ingestion/batch/{batch_id}/status` | Poll pipeline progress (bronze/silver/gold stage) |
| GET | `/ingestion/batch/{batch_id}/summary` | Final row counts, DQ gate results, corrections applied |

**Tests:**
- `tests/unit/test_normalize.py` — normalize_stream, detect_delimiter, validate_headers
- `tests/unit/test_medallion_upsert.py` — upsert mode SQL generation
- `tests/api/test_ingestion.py` — all 6 endpoints with mocked DB

### Phase 2: Frontend Upload & Monitoring (1 sprint)

**New files:**
- `frontend/src/api/queries/ingestion.ts` — query keys, fetch functions, TypeScript interfaces
- `frontend/src/components/FileDropZone.tsx` — reusable drag-and-drop (HTML5 native, Tailwind-styled)
- `frontend/src/tabs/DataIngestionTab.tsx` — main tab with 4 sub-views:
  1. **DomainSelector** — grid of 8 domain cards (last upload, row count, health score)
  2. **UploadPanel** — FileDropZone + file metadata display
  3. **PreviewPanel** — column mapping table, data preview (first N rows), DQ summary bar, confirm/cancel
  4. **BatchMonitorPanel** — pipeline stepper (bronze→silver→gold), row counts per stage, polling

**Modify:**
- `frontend/src/App.tsx` — lazy import DataIngestionTab
- `frontend/src/components/AppSidebar.tsx` — nav item in "system" section
- `frontend/src/api/queries/index.ts` — re-export ingestion module

**Tests:**
- `frontend/src/components/__tests__/FileDropZone.test.tsx`
- `frontend/src/tabs/__tests__/DataIngestionTab.test.tsx`

### Phase 3: Quarantine Resolution & Lineage (1 sprint)

**Extend `api/routers/medallion.py`:**
- `POST /data-quality/quarantine/{id}/fix` — accept corrected row, validate, insert to silver
- `POST /data-quality/quarantine/bulk-resolve` — bulk resolve multiple entries
- `GET /data-quality/quarantine/summary` — counts by domain, reason, resolution status

**New files:**
- `frontend/src/components/LineageFlowDiagram.tsx` — SVG/CSS flow diagram: Upload → Bronze (N) → Silver (M) → Gold (K) with quarantine branch

**Enhance DataQualityTab quarantine section:**
- Inline row editor (field validation per DomainSpec types)
- Bulk selection checkboxes + bulk resolve/dismiss
- Filter by rejection reason, domain, batch
- Re-submit fixed rows

### Phase 4: Domain Rollout (1 sprint)
- Verify all 8 domains work through the UI with zero new code (config-driven)
- Load mode per domain: dimensions (item, location, customer, dfu) = upsert; facts (sales, forecast) = replace/append
- Add domain-specific hints in preview (e.g., "Sales file must include TYPE column")
- Large file warnings for fact tables (inventory ~190M rows → recommend CLI)

## Dependencies
- Existing medallion pipeline (`common/medallion.py`) — extended, not replaced
- `DomainSpec` in `common/domain_specs.py` — drives all generic behavior
- Existing DQ APIs (`/data-quality/*`) — quarantine, corrections, lineage endpoints
- `config/medallion_config.yaml` — existing per-domain DQ rules

## Business Value
- Non-technical users can upload and validate data without CLI access
- Preview-before-commit prevents bad data from reaching production tables
- Upsert mode enables incremental dimension updates (no full reload required)
- Quarantine resolution workflow turns DQ failures into actionable fixes
- Lineage visualization provides full audit trail for compliance
- Template pattern: build once for dim_item, works for all 8 domains automatically
