# 09 — AI Intelligence Operations

This section covers running and operating the LLM-powered features of the Supply Chain Command Center: the **AI Planner** (proactive exception agent), the **Chat** natural-language SQL surface, the **Tuning Chat** assistant for LGBM hyperparameter tuning, the document **Insights / Embeddings** ingestion pipeline, and the **admin tooling** required to keep clients healthy across key rotations.

All components share a single LLM client layer (`api/llm.py`) that prefers OpenAI and silently fails over to Anthropic when configured. Every LLM turn (and tool call) is logged to `ai_call_log` for cost / latency observability.

---

## 9.1 LLM Client Layer

### 9.1.1 What it is

`api/llm.py` is the only module that constructs LLM SDK clients. Every router and background script that needs an LLM must import from here so that singletons, failover, and the admin reset path stay consistent.

| Function | Purpose |
|---|---|
| `get_openai()` | Returns a lazy-init `openai.OpenAI` singleton. Raises `HTTPException(503)` if `OPENAI_API_KEY` is missing or shorter than 20 characters. |
| `get_anthropic()` | Returns a lazy-init `anthropic.Anthropic` singleton, or `None` when `ANTHROPIC_API_KEY` is unset / the SDK is not installed. |
| `chat_completion(messages, model, temperature, max_tokens)` | Unified completion helper. Tries OpenAI first; on any non-`HTTPException` failure falls back to Anthropic; raises `HTTPException(503)` if both providers fail. |
| `reset_llm_client()` | Closes both clients (best-effort `client.close()`), nulls the singletons, returns `{"openai_reset": bool, "anthropic_reset": bool}`. |

### 9.1.2 Required Environment Variables

| Variable | Required for | Notes |
|---|---|---|
| `OPENAI_API_KEY` | Chat, AI Planner (default provider), embeddings via `text-embedding-3-small` | Must be ≥ 20 characters or `get_openai()` returns 503. |
| `ANTHROPIC_API_KEY` | Failover from OpenAI, AI Planner when `provider: "anthropic"` is set in `config/ai_planner_config.yaml` | Optional. Without it, `chat_completion` will raise 503 if OpenAI is down. |
| `API_KEY` | Auth gate on `/admin/*`, `/ai-planner/*` write endpoints, `/chat` | When unset, `require_api_key` is a no-op (development only). |

### 9.1.3 Model Defaults & Overrides

| Surface | Default model | Override |
|---|---|---|
| `chat_completion()` | `gpt-4o-mini`, `temperature=0.3`, `max_tokens=2000` | Per-call kwargs. |
| `/chat` endpoint | `gpt-4o`, `temperature=0.1`, `max_tokens=2000`, `response_format=json_object` | Hard-coded in `api/routers/intelligence/chat.py`. |
| AI Planner agent | `gpt-4o`, `temperature=0.2`, `max_tokens=4096` | `config/ai_planner_config.yaml` keys: `provider`, `model`, `max_tokens`, `temperature`. |
| Anthropic failover model | `claude-sonnet-4-20250514` | Hard-coded in `chat_completion`. |
| Embeddings (Chat RAG) | `text-embedding-3-small` (1536 dims) | Hard-coded in `chat.py`; matches `chat_embeddings.embedding` column. |

---

## 9.2 Admin LLM Reset

### 9.2.1 Endpoint

```
POST /admin/llm/reset
Headers: X-API-Key: <API_KEY>
```

Defined in `api/routers/platform/admin.py`. Mounted with `dependencies=[Depends(require_api_key)]`, so the entire `/admin` prefix is unreachable when `API_KEY` is unset in production.

### 9.2.2 What it does

1. Calls `reset_llm_client()` from `api/llm.py`.
2. Best-effort closes any open `openai.OpenAI` and `anthropic.Anthropic` clients.
3. Clears the module-level `_openai_client` and `_anthropic_client` singletons.
4. Returns `{"status": "ok", "openai_reset": <bool>, "anthropic_reset": <bool>}`.

### 9.2.3 When to call

- **After rotating `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`** in the API process environment. Without a reset, the cached client will keep authenticating with the old key until the API restarts.
- **After a 401 from the LLM provider** caused by suspected key revocation.
- **As part of an incident runbook** when the LLM appears to be consistently failing for auth-shaped reasons.

```bash
# Example
curl -X POST http://api:8000/admin/llm/reset \
     -H "X-API-Key: $API_KEY" | jq .
```

The next call to `get_openai()` / `get_anthropic()` rebuilds the client from the current environment.

### 9.2.4 Related admin endpoint

`POST /admin/tuning/invalidate-stale` clears the `stale` flag on `cluster_tuning_profile` rows so the tuning scheduler re-tunes them on the next cycle. Returns `{"status": "noop", "reason": "..."}` when the column is not yet present (Stream F not landed) or when the DB raises `psycopg.Error` — degrading rather than 500-ing keeps the admin UI safe to poll.

---

## 9.3 AI Planner

### 9.3.1 What it is

The AI Planner is a **proactive batch agent**, not a chatbot. It scans the demand+inventory portfolio, traces causal chains across forecast → inventory → EOQ → policy layers, and writes structured insights back to the database.

- Module: `common/ai/ai_planner.py` (class `AIPlannerAgent`)
- Router: `api/routers/intelligence/ai_planner.py`
- Frontend tab: `frontend/src/tabs/AIPlannerTab.tsx` (sub-panels in `frontend/src/tabs/ai-planner/`)
- Config: `config/ai_planner_config.yaml`
- Provider: configurable — `provider: "openai"` (default, uses `OPENAI_API_KEY`) or `provider: "anthropic"` (uses `ANTHROPIC_API_KEY`)

### 9.3.2 Endpoints

| Method & Path | Purpose | Auth |
|---|---|---|
| `POST /ai-planner/analyze` | Synchronous DFU-level analysis (~5–15s); returns inserted insights inline | `require_api_key` |
| `POST /ai-planner/portfolio-scan` | Fire-and-forget portfolio scan, returns `202 {scan_run_id}` | `require_api_key` |
| `GET  /ai-planner/insights` | Paginated insight list with severity / status / item / loc / brand / category / market / channel filters | open |
| `POST /ai-planner/insights/{insight_id}/status` | Acknowledge / resolve; writes outcome row for feedback tracking | `require_api_key` |
| `POST /ai-planner/insights/{insight_id}/snooze` | Snooze for N days (1–365); sets `snoozed_until` | `require_api_key` |
| `POST /ai-planner/auto-accept` | Bulk-accept open insights matching severity/type rules; supports `dry_run` | `require_api_key` |
| `GET  /ai-planner/metrics` | Per-model / per-tool aggregates from `ai_call_log` for the last N days (default 7) | open |
| `GET  /ai-planner/memos` | List planning memos (`portfolio` or `sku` scope) latest-first | open |

### 9.3.3 Inputs (LLM tools)

The agent uses provider-native tool calling. Tool definitions live in `_TOOL_DEFINITIONS` in `common/ai/ai_planner.py`. Each tool issues a direct psycopg query — no HTTP round trips.

| Tool | Returns |
|---|---|
| `get_dfu_full_context` | DFU attributes, latest inventory, EOQ, policy, champion WAPE — single-row snapshot |
| `get_forecast_performance` | Per-month champion forecast bias / abs-err for trailing window |
| `get_portfolio_exceptions` | Top-N DFUs ranked by stockout / excess / WAPE signals |
| `compute_bias_trend` | 3-month + 6-month rolling bias, over/under-forecast month counts |
| `get_inventory_trend` | Monthly DOS / on-hand / lead time |
| `get_eoq_context` | EOQ vs effective EOQ, months of supply, total annual cost |
| `get_similar_dfus` | Peer DFUs in same cluster + ABC class for benchmarking |
| `check_stockout_history` | Stockout / excess month counts over last 6 months from `mv_inventory_forecast_monthly` |
| `get_portfolio_health_summary` | Aggregated portfolio metrics for grounding portfolio memos |
| `create_insight` | Validates with Pydantic, INSERTs into `ai_insights`, returns `insight_id` |

### 9.3.4 Outputs

- `ai_insights` — validated structured insights (severity, summary, recommendation, financial impact, causal-chain reasoning).
- `ai_planning_memos` — narrative memos (`scope = 'portfolio' | 'sku'`).
- `ai_recommendation_outcomes` — feedback rows written when planners acknowledge / resolve / auto-accept; used to evaluate AI quality post-hoc.
- `ai_call_log` — per-turn observability (provider, model, tokens, latency, tool name, success flag).

### 9.3.5 Circuit breakers

Two hard caps protect cost / latency:

```python
MAX_TURNS    = 40
TOKEN_BUDGET = 100_000   # cumulative tokens per scan
```

When either is hit the agent logs a warning and exits cleanly with whatever insights it managed to write.

### 9.3.6 Running

| Surface | How |
|---|---|
| Single DFU (sync) | `POST /ai-planner/analyze` from the UI / curl |
| Portfolio (background) | `POST /ai-planner/portfolio-scan` — runs on the in-process `ThreadPoolExecutor(max_workers=2)` defined in the router |
| Scheduled scan | Wire to APScheduler via `common/services/job_scheduler.py` (call `AIPlannerAgent.run_portfolio_scan(...)` directly inside a job) |

---

## 9.4 Chat (NL → SQL)

### 9.4.1 What it is

A conversational, read-only natural-language SQL surface backed by RAG over `chat_embeddings`.

- Router: `api/routers/intelligence/chat.py`
- Frontend: `frontend/src/tabs/ChatPanel.tsx` (rendered as a side panel on every tab except `lgbmTuning`, which has its own Tuning Chat)
- Endpoint: `POST /chat` (guarded by `require_api_key`)

### 9.4.2 Pipeline

1. Embed the user question with `text-embedding-3-small`.
2. pgvector cosine-similarity search against `chat_embeddings` (`ORDER BY embedding <=> %s::vector LIMIT 10`).
3. Build a system prompt that includes the compact `DOMAIN_SPECS` schema summary, retrieved context, and project business rules (accuracy / bias / WAPE formulas, type=1 sales filter, lag semantics, etc.).
4. Call `gpt-4o` with `response_format=json_object`, expecting `{"answer": ..., "sql": ...}`.
5. If the model returns SQL, run it through `_is_safe_sql()`:
   - Must start with `SELECT` after stripping comments.
   - No `;` (blocks multi-statement attacks).
   - Forbidden tokens: `INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, GRANT, REVOKE, COPY`.
6. If safe, execute inside `SET LOCAL statement_timeout = '5000'` + `SET TRANSACTION READ ONLY`, capped at 500 rows.

### 9.4.3 Data sources it can query

- All tables registered in `common/domain_specs.py` (sales, forecast, inventory, dimensions, etc.).
- The compact schema and business rules in the system prompt steer the model toward correct columns (`item_id`, `basefcst_pref`, `tothist_dmd`, `model_id`, `lag`).

### 9.4.4 Failure modes returned to the UI

- Empty question → `422`.
- Generated DDL/DML → `error: "Generated SQL was blocked for safety reasons (only SELECT allowed)."`, `sql: null`.
- Execution error → `error: "SQL execution error: ..."`.
- Otherwise: `{answer, sql, data, columns, row_count}`.

---

## 9.5 Insights Generation

### 9.5.1 Where insights come from

There is **no standalone CLI script** for insight generation. Insights are produced by the AI Planner agent (`AIPlannerAgent.run_portfolio_scan` / `run_sku_analysis`) and reach the `ai_insights` table only via `create_insight`.

### 9.5.2 When to run

| Trigger | How |
|---|---|
| Weekly portfolio scan | `POST /ai-planner/portfolio-scan` (typically scheduled via APScheduler) |
| On-demand DFU drill-down | `POST /ai-planner/analyze` from the UI |
| Memo generation | `AIPlannerAgent.generate_portfolio_memo(period)` (writes to `ai_planning_memos`) |

### 9.5.3 Validation rules — `CreateInsightInput`

`common/ai/ai_planner.py` enforces a Pydantic model on every `create_insight` tool call. **Common gotchas you must respect when writing tests or new producers:**

- `summary` must be 20–300 chars **and contain at least one digit** (`@field_validator("summary")`). Plain prose with no metric value is rejected with a warning and the INSERT is skipped (returns `-1`).
- `recommendation` must be 30–600 chars.
- `insight_type` must be one of `stockout_risk | excess_inventory | forecast_bias | policy_gap | champion_degradation`.
- `severity` must be one of `critical | high | medium | low`.
- `financial_impact_estimate` must be in `[0, 10_000_000]`.

### 9.5.4 Status updates — 11-column `RETURNING`

`POST /ai-planner/insights/{insight_id}/status` runs:

```sql
UPDATE ai_insights SET status = %s, updated_at = NOW() [, acknowledged_at | resolved_at = NOW()]
WHERE insight_id = %s
RETURNING insight_id, status, insight_type, item_id, loc, abc_vol,
          financial_impact_estimate, dos, total_lt_days,
          champion_wape, forecast_bias_pct
```

Tests must mock this RETURNING with an **11-tuple** (see `MEMORY.md` "Important Test Mocks"). When `status ∈ {acknowledged, resolved}`, the router also writes an `ai_recommendation_outcomes` row (decision = `accepted` for acknowledged, `resolved` for resolved) for feedback tracking.

---

## 9.6 Embeddings

### 9.6.1 Two embedding stores

| Store | Populated by | Used by |
|---|---|---|
| `chat_embeddings` (1536-dim pgvector) | One-time bootstrap of the schema / business-glossary corpus | `_vector_search()` in `/chat` |
| `rag_chunks` (1536-dim pgvector + tsvector) | `scripts/ai/ingest_docs.py` | `common/ai/rag.upsert_chunks` / RAG retrieval (Gen-4 AI-5) |

### 9.6.2 `scripts/ai/ingest_docs.py`

Walks a directory of markdown (e.g., `docs/sop`), chunks each file into overlapping fixed-size windows (default `chunk_size=500`, `chunk_overlap=50`), and upserts via `common.ai.rag.upsert_chunks`.

```bash
python -m scripts.ai.ingest_docs --root docs/sop --source sop --dry-run
```

**Operational notes:**

- The script currently writes **zero-vector placeholder embeddings** — chunks still get their `tsvector` populated by the DB trigger so lexical retrieval works. The TODO at line 116 marks where the real embedding provider will plug in.
- The non-`--dry-run` path is intentionally not wired yet (returns `1` and logs `connection wiring not implemented; rerun with --dry-run.`). Treat the script as a scaffold for content extraction; do not rely on it for production embedding refresh until the connection wiring lands.
- Re-ingesting the same file is safe: chunks are upserted via `(doc_id, chunk_index)` unique key.

### 9.6.3 Refreshing `chat_embeddings`

`chat_embeddings` is currently bootstrapped out-of-band. If RAG retrieval starts returning `(no embeddings available)` chunks, repopulate the table from the schema / glossary source and verify the dim matches the `text-embedding-3-small` 1536-dim vector or the `<=>` operator will error.

---

## 9.7 Tuning Chat

### 9.7.1 What it is

An interactive debugging assistant for LGBM hyperparameter tuning runs. It seeds a chat session with the last 10 `lgbm_tuning_run` rows so the model has context, then lets a planner converse about why a tune did/didn't move WAPE.

- Router: `api/routers/forecasting/tuning_chat.py`
- Frontend: `frontend/src/tabs/lgbm-tuning/TuningChatPanel.tsx` (inside `ModelTuningTab` / LgbmTuningTab)
- Tables: `tuning_chat_session`, `tuning_chat_message`

### 9.7.2 Endpoints (prefix `/lgbm-tuning/chat`)

| Method & Path | Purpose |
|---|---|
| `POST /lgbm-tuning/chat/sessions` | Create session, seed `context` JSON with recent runs |
| `GET  /lgbm-tuning/chat/sessions?status=active&limit=20` | List sessions with message counts |
| (`POST /sessions/{id}/messages`, `POST /sessions/{id}/confirm-run` and the rest of the surface live in the same file) | Send messages / confirm a recommended re-run |

### 9.7.3 Frontend integration note

`App.tsx` hides the global ChatPanel on `lgbmTuning` to avoid two competing assistants on the same screen — Tuning Chat is shown instead.

---

## 9.8 Operational Concerns — Cost, Tokens, Rate Limits

### 9.8.1 Observability via `ai_call_log`

Every LLM turn and every tool dispatch writes a row through `log_ai_call()` (best-effort, swallows failures so the agent never fails because logging failed):

```
scan_run_id, dfu_key, provider, model, turn_number,
prompt_tokens, completion_tokens, total_tokens, latency_ms,
tool_name, tool_success, error_type
```

Query the rollup via `GET /ai-planner/metrics?days=7`:

- Per-model: LLM turns vs tool calls, total tokens, avg / p95 latency, error rate %.
- Per-tool: total calls, failures, avg latency.

### 9.8.2 Cost guard rails

- AI Planner: `MAX_TURNS=40`, `TOKEN_BUDGET=100_000` per scan (hard caps in `common/ai/ai_planner.py`).
- Chat: `max_tokens=2000`, single round trip, no tool loop.
- Tune defaults via `config/ai_planner_config.yaml` (`portfolio_scan_limit`, `forecast_lookback_months`, `max_tokens`).

### 9.8.3 Rate limits

OpenAI / Anthropic rate-limit errors surface as exceptions inside `chat_completion` and trigger the failover path. The portfolio-scan loop is **single-threaded per scan** (`ThreadPoolExecutor(max_workers=2)` shared across scans) so it cannot fan-out beyond two concurrent agentic loops.

### 9.8.4 OpenAI mock pattern (tests)

When mocking `client.chat.completions.create` in unit tests, the response object's `usage` field MUST expose **integer** attributes:

```python
resp.usage.total_tokens      = 123   # int, not MagicMock
resp.usage.prompt_tokens     = 100
resp.usage.completion_tokens = 23
```

Otherwise `total_tokens += usage.total_tokens or 0` raises `TypeError` and the loop dies on turn 1. See the `tests/api/` AI Planner tests for the canonical fixture.

---

## 9.9 Frontend Integration

| Surface | File | Sidebar location |
|---|---|---|
| AI Planner tab | `frontend/src/tabs/AIPlannerTab.tsx` (+ sub-panels in `ai-planner/`) | First entry — keyboard shortcut `1` (see `KeyboardShortcutHelp.tsx`) |
| Chat side-panel | `frontend/src/tabs/ChatPanel.tsx` | Floating panel, rendered on every tab except `lgbmTuning` (`App.tsx:317-318`) |
| Tuning Chat | `frontend/src/tabs/lgbm-tuning/TuningChatPanel.tsx` | Inside `ModelTuningTab` / LGBM Tuning tab |

### 9.9.1 Vite proxy entries

`frontend/vite.config.ts` must proxy these prefixes to `:8000` (already wired):

- `/chat`
- `/ai-planner`
- `/lgbm-tuning` (covers Tuning Chat)
- `/admin` (covers `/admin/llm/reset`, `/admin/tuning/invalidate-stale`)

If you add a new AI router under a new prefix, update the proxy and run `make audit-routers`.

### 9.9.2 API mounts (in `api/main.py`)

```
chat.router            (line 238)
ai_planner.router      (line 262)
admin_router.router    (line 284)
tuning_chat.router     (line 301)
```

All four are mounted **before** `domains.py` (which holds catch-all path params and must remain last).

---

## 9.10 Troubleshooting

### Symptom: `503 OPENAI_API_KEY not configured`

`get_openai()` rejects a missing or short key. Either set `OPENAI_API_KEY` in the API process environment or, if it was just rotated, run `POST /admin/llm/reset` to drop the cached client and force re-init on the next call.

### Symptom: 503 immediately after rotating the OpenAI / Anthropic key

The cached singleton still holds the old key. Call `POST /admin/llm/reset` and re-invoke. The response will tell you which clients were actually closed (`{"openai_reset": true, "anthropic_reset": false}` etc.).

### Symptom: AI Planner returns no insights even though it ran

Check `ai_call_log` for the `scan_run_id` returned by the endpoint:

- Lots of rows but zero `tool_name = create_insight` entries → the model decided nothing crossed a severity threshold; rare but possible.
- `create_insight` calls present but no rows in `ai_insights` → the Pydantic validator rejected the payload. The most common cause is `summary must contain at least one metric value (number)`. Check the API logs for the `create_insight validation failed:` warning and see §9.5.3.

### Symptom: Portfolio scan logs `MAX_TURNS` or `TOKEN_BUDGET` warning

The agent hit a circuit breaker. Either the prompt is sending the model in circles or the portfolio is too large. Tune `portfolio_scan_limit` in `config/ai_planner_config.yaml`, or split the scan into multiple smaller runs.

### Symptom: `/chat` returns `(no embeddings available)`

`chat_embeddings` is empty or the pgvector extension is missing. Verify `SELECT COUNT(*) FROM chat_embeddings;` and rebuild the corpus. Falling back to lexical-only context is non-fatal — answers will be lower quality but the endpoint still functions.

### Symptom: `/chat` returns `Generated SQL was blocked for safety reasons`

The model returned non-SELECT SQL. Re-prompt with a more constrained question; do not relax `_is_safe_sql()` — its block-list is the only thing keeping the read-only contract.

### Symptom: LLM timeouts / hangs

`chat_completion` does not set an explicit timeout — relies on SDK defaults. If you see hangs, set `OPENAI_TIMEOUT` / `ANTHROPIC_TIMEOUT` in the environment or wrap calls in `asyncio.wait_for` for the async paths. The `/chat` endpoint also bounds the SQL it generates with `SET LOCAL statement_timeout = '5000'` (5 s) so DB-side hangs cannot pile up.

### Symptom: Anthropic failover not triggering

Verify (a) `anthropic` is installed (`uv pip show anthropic`), (b) `ANTHROPIC_API_KEY` is set and ≥ 20 chars, (c) `get_anthropic()` is being called — note that `chat_completion` only falls over when OpenAI raises, *not* when it returns a 200 with bad content.

### Symptom: Test fails with `TypeError: unsupported operand type(s) for +=: 'int' and 'MagicMock'`

The mocked `response.usage` did not expose integer fields. Fix the fixture per §9.8.4.

### Symptom: Test fails on insight status update with index error / wrong tuple length

The mock for the `RETURNING` clause must be an **11-tuple** matching the column list in §9.5.4.

### Symptom: `admin/tuning/invalidate-stale` always returns `noop`

Expected until the `cluster_tuning_profile.stale` column lands (Stream F). The endpoint deliberately degrades (rather than 500-ing) so the admin UI can poll it safely. Once the column is migrated, the next call returns `{"status": "ok", "invalidated": <n>}`.

---

## 9.11 Quick Reference

```bash
# Rotate keys then refresh in-process clients
export OPENAI_API_KEY=sk-...
curl -X POST http://api:8000/admin/llm/reset -H "X-API-Key: $API_KEY"

# Kick a portfolio scan
curl -X POST http://api:8000/ai-planner/portfolio-scan -H "X-API-Key: $API_KEY"

# Drill into one DFU
curl -X POST http://api:8000/ai-planner/analyze \
     -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
     -d '{"item_id":"ABC123","loc":"DC1"}'

# Inspect AI cost / latency over the last week
curl http://api:8000/ai-planner/metrics?days=7 | jq .

# Ask a question (NL → SQL)
curl -X POST http://api:8000/chat \
     -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
     -d '{"question":"What is the YoY sales change for brand X in Q1?"}'

# Dry-run doc ingest into RAG corpus
python -m scripts.ai.ingest_docs --root docs/sop --source sop --dry-run
```
