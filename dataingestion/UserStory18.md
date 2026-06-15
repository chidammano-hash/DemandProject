# User Story 18: API endpoints for full + incremental run

**Phase:** 5 — Unified Orchestration
**Depends on:** US16
**Complexity:** M  **Risk:** MEDIUM

## Story
As a **frontend/UI**, I want **HTTP endpoints to trigger a full load or an incremental refresh (per-domain or all)**, so that **the whole pipeline is operable from the browser, not just per-domain single loads**.

## Background / Current State
Today the UI can submit **single-domain** loads (`POST /integration/jobs`) and chains (`POST /integration/chains`), but there is **no** endpoint that runs the unified `mode=full` / `mode=refresh` pipeline across all domains. US16 added the job type; this story exposes it over HTTP.

## Acceptance Criteria
- [ ] **AC1** — `POST /integration/pipeline` accepts `{mode: "full"|"refresh", domains?: string[], parallel?: bool}` and submits the `etl_pipeline` job (returns 202 + job id).
- [ ] **AC2** — Endpoint is in the correct router (`api/routers/platform/integration*.py`), uses `APIRouter(prefix=...)`, short decorator paths, and `Depends(require_api_key)` (it's a write).
- [ ] **AC3** — `GET` reuses the unified job status/logs endpoints (US17) — no new bespoke status store.
- [ ] **AC4** — Mounted in `api/main.py` **before** `domains.py`; prefix added to `frontend/vite.config.ts` AND `frontend/src/api/queries/index.ts` barrel in the same change.
- [ ] **AC5** — 5xx details are short verb-phrases; no exception text interpolation.

## TDD Plan
### Write first (red)
- `tests/api/test_integration_pipeline.py::test_post_pipeline_full_returns_202`
- `::test_post_pipeline_refresh_with_domains`
- `::test_pipeline_requires_api_key`
- `::test_invalid_mode_422`
- `::test_status_via_unified_job_endpoint`
(httpx AsyncClient + ASGITransport, `make_pool`, patch `_get_pool`/`_get_async_pool` as appropriate)
### Then implement (green) → Refactor
- Add endpoint; wire to `JobManager.submit_job`; update Vite proxy + barrel.

## Implementation Notes
- Run `make audit-routers` to verify proxy/barrel parity.
- Reuse existing `KNOWN_DOMAINS` validation.

## Definition of Done
- [ ] Full + refresh runnable over HTTP; status via unified endpoints.
- [ ] `make audit-routers`, `tests/api/test_integration_pipeline.py`, `make test-all` green.
