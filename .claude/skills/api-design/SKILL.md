---
name: api-design
description: FastAPI router conventions for DemandProject — prefixes, auth on writes, get_conn, safe 5xx, read-replica routing, and the register-a-router checklist.
origin: ECC
---

# API Design (DemandProject)

FastAPI + psycopg3 + Pydantic v2 conventions. CLAUDE.md "API / Routers" is authoritative;
this reinforces it and covers the router-registration checklist.

## When to Activate
Adding or reviewing a router under `api/routers/**`, or an endpoint contract.

## Router rules (violations are defects)
- **`APIRouter(prefix="/...")` + short paths** in decorators. Never put the full path in
  `@router.get(...)`.
- **`get_conn()` directly** in `inv_planning_*.py` routers — never `Depends(_get_pool)`
  (FastAPI inspects the MagicMock signature → 422 in tests).
- **Every write** (`post`/`put`/`delete`/`patch`) carries
  `dependencies=[Depends(require_api_key)]` at router or route level.
- **5xx never interpolate exception text.** Pattern:
  ```python
  logger.exception("generating forecast failed")
  raise HTTPException(status_code=500, detail="forecast generation failed")
  ```
  No `f"...{exc}"`, no `str(exc)` in a 500 detail.
- **SQL:** `%s` for values, `psycopg.sql.Identifier` for identifiers. Mock tuple column
  count must equal the SELECT column count.
- **Read-heavy analytics** use both cache and read-only routing: apply
  `@cached_sync(...)` / `@cached_async(...)` to repeated GETs and use
  `get_read_only_conn()` / `get_async_read_only_conn()` when the query tolerates
  replica lag. Never use read-only helpers for read-after-write flows.
- **Dashboard + Accuracy GETs are the reference pattern.** New or modified hot GETs in
  `api/routers/core/dashboard.py` and `api/routers/forecasting/accuracy.py` must keep
  `@cached_sync(...)` and `get_read_only_conn()` together; `scripts/ai_checks/check_unenforced_rules.sh`
  enforces this scoped rule.
- **`domains.py` is mounted LAST** in `api/main.py` — its `{domain}` catch-all shadows
  anything mounted after it.

## Status codes (semantic, not 200-for-everything)
`200` read · `201` created (+ `Location`) · `400`/`422` validation with field detail ·
`401`/`403` auth · `404` missing · `409` conflict · `500` unexpected (opaque detail, logged).

## Registering a new router (all three, same change)
1. `app.include_router(...)` in `api/main.py` **before** `domains.py`.
2. Add the path prefix to `frontend/vite.config.ts` `API_PATH_PREFIXES`.
3. Add it to the `frontend/src/api/queries/index.ts` barrel.
Then `make audit-routers` to verify main.py ↔ vite parity. Missing #2/#3 → frontend gets
HTML instead of JSON. Scaffold with `make new-router DOMAIN=<d> NAME=<n>`.

## Related
- Agent: `python-reviewer` · Skill: `python-patterns`, `postgres-patterns`, `security-review`
