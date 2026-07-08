# SKU Chatbot

> A conversational, tool-using assistant that lets a planner ask free-form questions about a single SKU (or a small set) and get grounded, cited answers drawn from the platform's own sales, forecast, inventory, feature, and accuracy data. Default runtime is the **Claude Agent SDK** with tiered model routing (Haiku / Sonnet / Opus), and `runtime.provider: codex` can instead use **Codex CLI** (`codex exec`) with a read-only SKU context snapshot. Both support local subscription/session auth for development and API-key auth for standalone automation.

| | |
|---|---|
| **Status** | Partial — Phases 1 + 3 + agentic adjust + global page-aware chat (backend, persistence, standalone tab + a chat drawer on every tab, champion-adjust approval flow; 62 tests green). The live model path requires `uv sync --extra agent`; persistence requires `sql/196`–`197`. |
| **UI Tab** | SkuChatTab (standalone) **+ a global, page-aware chat drawer (`GlobalChatDrawer`) on every tab** |
| **Key Files** | `frontend/src/tabs/SkuChatTab.tsx`, `frontend/src/tabs/sku-chat/{SkuChatPanel,GlobalChatDrawer}.tsx` + `pageChatConfig.ts`, `frontend/src/context/ActiveSkuContext.tsx` (chat inherits the active page's SKU; `ItemAnalysisTab` publishes), `frontend/src/App.tsx` (mounts the global drawer + `ActiveSkuProvider`), `frontend/src/api/queries/sku-chat.ts`, `api/routers/intelligence/sku_chat.py`, `common/ai/sku_chat/` (config, auth, model_router, sku_data, tools, agent, prompts, store, champion_adjust), `config/ai/sku_chat_config.yaml`, `sql/196`, `sql/197` |
| **API Prefixes** | `/sku-chat` |
| **Depends On** | Claude Agent SDK (`claude-agent-sdk`, `agent` extra) for `runtime.provider: claude`; Codex CLI for `runtime.provider: codex`; `dim_sku` + SKU features (03-02), production/candidate forecast (02), demand history workbench (03-06), accuracy KPIs (02), AI Planning Agent (06-01) as the architectural sibling |

> **Implementation status (Phases 1 + 3 + agentic adjust + Codex runtime switch).** Built: the read-only tool layer (`sku_data.py` → 7 `@tool` MCP tools for Claude and a read-only JSON context snapshot for Codex), per-turn model-tier routing, `runtime.provider` (`claude` / `codex`) plus `auth.mode` resolution, the SSE `/sku-chat/{config,session,stream}` router + `GET /session/{id}` history (mounted before `domains.py`), **best-effort persistence** (`store.py` → `sku_chat_session|message|call_log`, `sql/196`), the standalone React tab, and a **global, page-aware chat drawer (`GlobalChatDrawer`) present on every tab** — its focus (sent to the agent as page context), suggested-prompt chips, and SKU scope adapt to the active page via `pageChatConfig`. Plus the **agentic champion-forecast adjustment** on the Claude runtime: an `apply_champion_adjustment` staging tool + an "AI Adjust" button + an in-chat **approval card** + `POST /sku-chat/adjustment/{id}` that reuses the existing guarded `save_adjustment` (`champion_adjust.py`, `sql/197`). The Claude Agent SDK and Codex CLI are both lazy runtime dependencies, so the API and test suite run without them; the selected runtime returns a single `error` event if its local dependency/auth is unavailable.

---

## Problem

Planners can already *see* every per-SKU signal the platform computes — sales history, forecast overlays, inventory position, demand-behaviour features, cluster membership, accuracy/bias/WAPE — but each lives behind a different tab, chart, or endpoint. Answering a plain-English question like *"why did the forecast for ABC-123 in the North DC miss last quarter, and is the safety stock still right?"* means manually stitching together five views and doing the reasoning in your head.

The existing **AI Planning Agent (06-01)** is explicitly *not a chatbot* — it proactively scans the portfolio and writes structured insights to a queue. There is no interactive, ask-anything surface scoped to one SKU where a planner can drill in, follow up, and get a grounded narrative answer with the numbers cited.

We also have a deployment constraint the rest of the AI stack does not address: today the team runs everything inside **Claude Code**, where model access is already authenticated via the local subscription — we should not have to wire an `ANTHROPIC_API_KEY` to ship this. But the same feature must be deployable **outside** Claude Code later (standalone container, CI, customer environment), where an explicit API key (or Bedrock/Vertex) is required. The design must make that an operational config switch, not a rewrite.

---

## Solution

A backend **`SkuChatAgent`** with a selectable runtime. `runtime.provider: claude` uses the **Claude Agent SDK** (`claude-agent-sdk`, the Python SDK that the Claude Code harness exposes — *not* the bare `anthropic` Messages SDK used by 06-01). `runtime.provider: codex` invokes **Codex CLI** (`codex exec`) in a read-only sandbox. For each incoming chat turn the agent:

1. **Routes to a model tier** — a cheap Haiku classifier picks Haiku / Sonnet / Opus based on the question's complexity (overridable per-request and via config).
2. **Runs the configured runtime** — Claude gets the SKU's data as **read-only SDK MCP tools** (`@tool` functions registered through `create_sdk_mcp_server`) that call our existing query layer in-process (no subprocess, no HTTP hop). Codex gets a pre-fetched, read-only JSON context snapshot and is invoked with `codex exec --sandbox read-only --ask-for-approval never`.
3. **Streams a grounded answer** back to the React chat UI over Server-Sent Events, with the tool calls surfaced as a "evidence trail" so the answer is auditable.

Auth is delegated to the selected local runtime. Claude supports `auth.mode: auto` (inherit Claude Code's existing session — no key needed), `api_key` (`ANTHROPIC_API_KEY`), `bedrock`, or `vertex`. Codex supports `auth.mode: auto` (inherit saved Codex CLI auth / ChatGPT sign-in) or `api_key` (`CODEX_API_KEY` for the `codex exec` invocation). The Claude tool surface is **strictly read-only** except for the approval-gated staging tool; the Codex path receives data up front and runs read-only/non-interactively.

This is a sibling to 06-01, not a replacement: 06-01 is proactive + write-capable + portfolio-wide; this is interactive + read-only + SKU-scoped.

---

## How It Works

### Architecture

```
React SkuChatTab ──POST /sku-chat/stream (SSE)──▶ api/routers/intelligence/sku_chat.py
                                                          │
                                                          ▼
                                          common/ai/sku_chat/agent.py  (SkuChatAgent)
                                                          │
                  ┌───────────────────────────────────────┼───────────────────────────────┐
                  ▼                                        ▼                                ▼
        model_router.py                         Claude Agent SDK                    sku_tools.py
   (Haiku intent classifier →            query()/ClaudeSDKClient with         @tool read-only fns →
    opus|sonnet|haiku tier)              ClaudeAgentOptions(model, tools,      existing query layer
                                          system_prompt, setting_sources=[],   (dim_sku, facts, MVs)
                                          permission_mode="bypassPermissions")
                                                          │
                                                          ▼
                                              auth.py (resolve auth.mode)
                                       auto | api_key | bedrock | vertex → env for SDK runtime
```

### Conversation flow (per turn)

1. UI sends the user message plus **SKU context** (`item_id`, `customer_group`, `loc`) and the `session_id`.
2. `model_router` classifies intent (single fast Haiku call) → picks a tier (see Model Routing).
3. `SkuChatAgent` builds `ClaudeAgentOptions` for the chosen model with: our read-only MCP tool server, the SKU-specialist system prompt, `setting_sources=[]` (ignore the developer's local `~/.claude`), and `permission_mode="bypassPermissions"`.
4. The agent loops: Claude calls SKU tools to gather evidence, then composes a grounded answer. Tool handlers run **in-process** against the existing read paths.
5. Streamed `StreamEvent` text deltas are relayed to the browser as SSE `text` events; tool calls are relayed as `tool` events (evidence trail).
6. On `ResultMessage`, the agent emits a final SSE `result` event carrying the answer text + token/cost usage, and (Phase 3) persists the turn to `sku_chat_message` / `sku_chat_call_log` for audit.

### Model Routing — "use Opus, Sonnet, Haiku wisely"

Each chat turn is an independent Agent SDK invocation, so the model is chosen **per turn**. A small Haiku classifier (≈1 cheap call) maps the question to a tier; the result is overridable.

| Tier | Model (default) | Use for | Examples |
|---|---|---|---|
| `fast` | `claude-haiku-4-5` | Single-field lookups, definitions, the intent classifier itself, greetings | "What's the lead time?" "Which cluster is this in?" "What does XYZ class mean?" |
| `standard` | `claude-sonnet-4-6` | One-SKU analytical Q&A spanning a few tool calls | "Summarise demand and the current forecast." "Is bias trending up?" |
| `deep` | `claude-opus-4-8` | Multi-signal diagnosis, "why" questions, cross-SKU/cluster reasoning, recommendations | "Why did Q3 miss and is safety stock still right?" "Compare this SKU to its cluster peers." |

- Tier→model mapping lives in `config/ai/sku_chat_config.yaml` (`models.<tier>`). Aliases (`opus`/`sonnet`/`haiku`) and full IDs are both accepted by the SDK runtime; we pin **full IDs** in config to avoid drift.
- Per-request override: the UI may send `model_tier` (or an explicit `model`) to force a tier — e.g. a "Deep analysis" toggle.
- Routing rules (keyword/length heuristics + the Haiku classifier's label) are config-driven (`routing.*`), never hardcoded.
- The Agent SDK also honours `ANTHROPIC_DEFAULT_OPUS_MODEL` / `ANTHROPIC_DEFAULT_SONNET_MODEL` / `ANTHROPIC_DEFAULT_HAIKU_MODEL` env overrides; we set these from config at process start so the same YAML governs both explicit and alias-resolved selection.

### Tools (read-only SDK MCP server)

Defined with the SDK `@tool` decorator and registered via `create_sdk_mcp_server(name="sku", ...)`; allow-listed as `mcp__sku__*`. Every handler runs in-process and reuses the existing chunked read helpers / router service functions — **no new SQL patterns**, no writes.

| Tool | Reads from | Purpose |
|---|---|---|
| `search_skus` | `dim_sku` | Resolve a fuzzy item/brand/description to concrete `(item_id, customer_group, loc)` keys |
| `get_sku_profile` | `dim_sku` (features) | Demand stats, CV, intermittency, seasonality, ABC-XYZ, `ml_cluster`, execution lag |
| `get_sku_sales_history` | `fact_sales_monthly` / `agg_sales_monthly` | Monthly demand history (chunked read) |
| `get_sku_forecast` | `fact_production_forecast` (staging) + `fact_candidate_forecast` (backtest) | Forward forecast + CI bands, per-model backtest overlay |
| `get_sku_inventory` | `agg_inventory_monthly` | On-hand, on-order, DOS, lead time trend |
| `get_sku_accuracy` | `agg_accuracy_by_dim` / `agg_accuracy_by_dfu` (193) | WAPE, bias, accuracy by model + lag; error contribution |
| `get_sku_cluster_peers` | `dim_sku` | Similar SKUs in the same `ml_cluster` for comparison |
| `apply_champion_adjustment` | — (writes a `pending` row only) | **Write-staging tool** (only when `champion_adjust.enabled`): stages a guardrail-validated champion-forecast adjustment for planner approval. It does **not** write the forecast — see below. |

**Join rule (carried from CLAUDE.md):** any tool that joins a forecast/accuracy fact to `dim_sku` must match on the full grain `item_id AND customer_group AND loc` — a 2-key join fans rows across customer groups and inflates WAPE/bias. Tools return compact, pre-aggregated payloads (not raw fact dumps) to keep the context window small and answers grounded.

### Champion-forecast adjustment (agentic, approval-gated)

When `champion_adjust.enabled`, the chatbot can **drive a champion-forecast adjustment** — but it reuses the existing AI Champion adjuster engine (`common/ai/champion_adjust_service.py`, spec [02-27](../02-forecasting/27-ai-champion-forecast.md)) rather than reimplementing the math or guardrails, and the agent never writes the forecast itself:

1. The planner clicks **"AI Adjust"** (or asks in chat). The agent gathers context via its read tools, then calls `apply_champion_adjustment(item_id, loc, rationale)`.
2. The tool calls `adjust_dfu()` (the adjuster's tested preview path: builds context, runs its LLM, **re-applies guardrails** — `max_abs_pct_change`, `min_confidence`, `require_evidence`, horizon cap) and **stages** the validated preview as a `pending` row in `sku_chat_pending_adjustment`. No forecast is written.
3. After the turn, the router surfaces the staged proposal as an SSE `approval_request` event; the UI renders an **approval card** (recommendation, %, confidence, rationale) with **Approve / Reject**.
4. **Approve** → `POST /sku-chat/adjustment/{id}` → `champion_adjust.apply_adjustment` → the existing guarded `save_adjustment` writes `fact_ai_champion_forecast` under `model_id='ai_champion'` (quantities re-derived from the baseline, guardrails re-applied server-side). **Reject** → the row is marked rejected; nothing is written.

This is the user-chosen "full agentic write tool gated by an in-chat approval round-trip", implemented as **deferred execution** (stage → human-approve → guarded write) rather than a blocking permission callback — robust under multi-worker SSE and reusing the proven adjuster. The chatbot thus *absorbs* the adjuster's interaction surface while *reusing* its engine.

### Authentication & deployment modes

`runtime.provider` selects the local agent runtime, and `auth.mode` is interpreted by that runtime. This keeps "use the local developer subscription/session now; use API credentials later" as a config switch instead of a rewrite.

| Config | What happens | When |
|---|---|---|
| `runtime.provider=claude`, `auth.mode=auto` | No `ANTHROPIC_API_KEY` injected; the agent inherits whatever the surrounding Claude Code session is authenticated with (subscription login / `CLAUDE_CODE_OAUTH_TOKEN`). | Local dev in Claude Code |
| `runtime.provider=claude`, `auth.mode=api_key` | Sets `ANTHROPIC_API_KEY` from the secret store before invoking the SDK. | Standalone container / CI / customer env |
| `runtime.provider=claude`, `auth.mode=bedrock` | Sets `CLAUDE_CODE_USE_BEDROCK=1` (+ AWS creds). | AWS-native deploy |
| `runtime.provider=claude`, `auth.mode=vertex` | Sets `CLAUDE_CODE_USE_VERTEX=1` (+ GCP creds). | GCP-native deploy |
| `runtime.provider=codex`, `auth.mode=auto` | Runs `codex exec` with no API key injected; Codex reuses saved CLI auth (for example ChatGPT sign-in). | Local dev in Codex |
| `runtime.provider=codex`, `auth.mode=api_key` | Requires `CODEX_API_KEY` and injects it only into the `codex exec` process. | Standalone trusted automation |

`auth.py` resolves the runtime/mode pair, validates that the required credentials are present, and fails loud at request time if a selected non-`auto` mode lacks credentials.

> **Implementation note.** The exact subscription-auth inheritance behaviour and any ToS limits on using local subscription/OAuth credentials from a long-running server should be confirmed against the installed Claude Code / Codex CLI version before production use. The `auto` paths are intended for local developer-hosted use; production standalone deployments should use `api_key`, Bedrock, or Vertex as appropriate.

### Settings isolation & guardrails

Because the agent runs as a server, it must not pick up the developer's local config or prompt for permissions:

- `setting_sources=[]` — Claude runtime: do not load `~/.claude` / project `CLAUDE.md` / local settings.
- `system_prompt=<SKU specialist prompt>` — replaces the default; defines tone, citation discipline ("every numeric claim must come from a tool result"), and refusal behaviour for out-of-scope questions.
- `permission_mode="bypassPermissions"` — Claude runtime: fully non-interactive. Safe **only because** the allow-listed tool surface is read-only (`mcp__sku__*`) except for approval-gated staging; no Bash/Write/Edit/Read-file tools are exposed.
- `codex exec --sandbox read-only --ask-for-approval never` — Codex runtime: receives a pre-fetched JSON context snapshot and cannot write files.
- Per-turn circuit breakers (mirroring 06-01): `max_turns`, `token_budget`, and a wall-clock `timeout_seconds`. On breach the turn ends gracefully with a partial answer + a "truncated" flag.

---

## Data Model

Persistence (**Phase 3, built**) is **best-effort**: `common/ai/sku_chat/store.py` writes sessions/messages/call-logs, but every write catches `psycopg.Error` and no-ops, so the chat streams whether or not `sql/196` has been applied (`persistence.enabled` in config toggles it off entirely). DDL at `sql/196_create_sku_chat_log.sql`; tables added to the `db-truncate-data` Make target.

| Table | Purpose | Key Columns |
|---|---|---|
| `sku_chat_session` | One conversation, scoped to a SKU | `session_id`, `item_id`, `customer_group`, `loc`, `created_by`, `created_at`, `last_active_at` |
| `sku_chat_message` | One turn | `id`, `session_id`, `role` (user/assistant), `content`, `model`, `tier`, `created_at` |
| `sku_chat_call_log` | Per-turn observability | `id`, `session_id`, `message_id`, `model`, `input_tokens`, `output_tokens`, `cache_read_tokens`, `total_cost_usd`, `tool_calls`, `latency_ms`, `truncated`, `created_at` |
| `sku_chat_pending_adjustment` | Staged champion adjustments awaiting approval (`sql/197`) | `approval_id`, `session_id`, `item_id`, `customer_group`, `loc`, `preview` (JSONB), `status` (pending/approved/rejected), `created_by`, `created_at`, `decided_at` |

`total_cost_usd` is the SDK's **client-side estimate** (`ResultMessage.total_cost_usd`) — stored for trend insight, not billing truth. Add the three tables to `db-truncate-data` + the cleanup section of `docs/operations-manual/11-maintenance-troubleshooting.md` per the feature checklist.

---

## API

Router `api/routers/intelligence/sku_chat.py`, prefix `/sku-chat`, `get_conn()` + `%s`, mounted **before** `domains.py`. Read paths use `get_read_only_conn()` where eligible.

| Method | Path | Purpose |
|---|---|---|
| POST | `/sku-chat/stream` | Send a turn; returns an SSE stream (`text` deltas, `tool` evidence events, terminal `result` with usage). Write-class → `Depends(require_api_key)`. |
| POST | `/sku-chat/session` | Create a session bound to a SKU; returns `session_id`. `Depends(require_api_key)`. |
| GET | `/sku-chat/session/{session_id}` | Fetch a session + its ordered message history (404 if unknown). |
| GET | `/sku-chat/config` | Return `runtime.provider`, active model-tier mappings, `auth.mode` (without secrets), and guardrail limits — drives the UI's tier toggle and health badge. |
| POST | `/sku-chat/adjustment/{approval_id}` | Approve (apply) or reject a staged champion-forecast adjustment (`{decision: "approve"\|"reject"}`). Approve calls the guarded `save_adjustment`. `Depends(require_api_key)`. |

- SSE event envelope: `{"type": "text"|"tool"|"result"|"error", ...}`. Mapped from the SDK stream — `StreamEvent` `content_block_delta`/`text_delta` → `text`; `ToolUseBlock` → `tool`; `ResultMessage` → `result`.
- 5xx responses follow the rule: `logger.exception(...)` then `raise HTTPException(500, detail="<short verb-phrase>")` — never interpolate exception text.
- New prefix `/sku-chat` added to **both** `frontend/vite.config.ts` `API_PATH_PREFIXES` and the `frontend/src/api/queries/index.ts` barrel in the same change; verified by `make audit-routers`.

---

## Configuration

File: `config/ai/sku_chat_config.yaml` (loaded via `load_config`). Inline comments required on every key.

| Key | Purpose | Default |
|---|---|---|
| `runtime.provider` | `claude` \| `codex` | `claude` |
| `auth.mode` | `auto` \| `api_key` \| `bedrock` \| `vertex` | `auto` |
| `models.fast` | Model for the `fast` tier | `claude-haiku-4-5` |
| `models.standard` | Model for the `standard` tier | `claude-sonnet-4-6` |
| `models.deep` | Model for the `deep` tier | `claude-opus-4-8` |
| `codex_models.fast` | Codex model for the `fast` tier | `gpt-5.4-mini` |
| `codex_models.standard` | Codex model for the `standard` tier | `gpt-5.5` |
| `codex_models.deep` | Codex model for the `deep` tier | `gpt-5.5` |
| `codex.binary` | Codex CLI executable | `codex` |
| `codex.sandbox` | Sandbox for `codex exec` | `read-only` |
| `codex.approval_policy` | Approval policy for `codex exec` | `never` |
| `routing.classifier_model` | Model that classifies intent → tier | `claude-haiku-4-5` |
| `routing.default_tier` | Tier when classification is inconclusive | `standard` |
| `routing.allow_user_override` | Let the UI force a tier/model | `true` |
| `champion_adjust.enabled` | Register the `apply_champion_adjustment` staging tool + approval flow | `true` |
| `guardrails.max_turns` | Tool-call rounds per turn | `12` |
| `guardrails.token_budget` | Token cap per turn | `60000` |
| `guardrails.timeout_seconds` | Wall-clock cap per turn | `60` |
| `tools.allowed` | Allow-listed tool names (`mcp__sku__*`) | all 7 read-only tools |
| `context.history_lookback_months` | Default sales/forecast window for tools | `24` |
| `system_prompt` | SKU-specialist instructions (citation discipline, scope, refusal) | inline block |

Reuses the existing `config/ai/` directory and `load_config` convention (no new loader). No magic numbers in Python.

---

## Frontend

- **`sku-chat/SkuChatPanel.tsx`** (reusable): the conversation thread (streaming bubble, per-answer tool "evidence" chips, tier/model + cost meta), the composer (input + "Deep analysis" toggle), the routing badge from `/sku-chat/config`, and session continuity (captures `session_id` from the `meta` event and re-sends it). Also an **"AI Adjust" button** (sends a canned deep-tier prompt to propose a champion adjustment) and an **approval card** rendered from `approval_request` events (recommendation / % / confidence / rationale + Approve & apply / Reject, calling `POST /sku-chat/adjustment/{id}`). Scoped entirely by `{ itemId, loc, customerGroup }` props.
- **`SkuChatTab.tsx`** (standalone tab): a SKU selector row (`item_id` / `customer_group` / `loc`) wrapping `SkuChatPanel`.
- **`sku-chat/GlobalChatDrawer.tsx`** (global, page-aware): a floating "Assistant" button → right-side slide-out drawer rendering `SkuChatPanel`, **present on every tab** (mounted once in `App.tsx`; hidden only on the standalone SKU Chat tab). It reads the active tab and **re-contextualizes per page** — title, `page_focus`, and suggested prompts come from **`pageChatConfig.ts`** (a per-tab registry). **SKU scope is inherited from the SKU the active page is showing** (via `ActiveSkuContext` — e.g. the Item Analysis local selector, which can diverge from the global filter), **falling back to the single-valued global filter `{item, location}`** when no page publishes one. **Persistent per-page threads:** one `SkuChatPanel` is kept alive per thread key (`tab|item|loc`) and only the active one is shown — inactive threads are `hidden` (the HTML attribute, so they survive in jsdom and the browser alike), so React preserves each conversation and it **resumes when you return**. Because the SKU is in the key, a SKU change on a SKU-scoped page starts a fresh thread. The aside is kept mounted once opened (hidden when closed) so threads persist across open/close; retained threads are **LRU-capped (12)** — an evicted thread loses its in-memory history (the backend session row remains; auto-rehydration is a future addition).
- **`context/ActiveSkuContext.tsx`** (scope inheritance): a tolerant focus channel — a page calls `usePublishActiveSku(item, loc)` to announce the SKU it is displaying (auto-clears on unmount); the drawer reads it via `useActiveSku()`. `ItemAnalysisTab` publishes its debounced SKU here so the assistant inherits exactly what the chart shows. Kept separate from `GlobalFilterContext` because it is transient display focus, not a persistent filter, and side-effect-free (writing into the global filter would leak to other tabs and fight the existing global→local seeding).
- **Per-page customization** = (a) `page_focus` sent to the agent so answers are tailored to the page, (b) page-specific suggested-prompt chips, (c) SKU scope inherited from the active page's shown SKU, falling back to the global filter. The tool surface stays the same SKU read tools (+ the staging adjust tool); deeper per-page aggregate tools (portfolio/inventory) are a future addition.
- **`src/api/queries/sku-chat.ts`**: typed interfaces mirroring the Pydantic schemas (no `: any`); `streamSkuChat` SSE reader (`res.body.getReader()`, since the stream is a POST); `fetchSkuChatConfig` / `createSkuChatSession` / `fetchSkuChatSession`; `skuChatKeys`.
- No inline hex colors; theme via context.

---

## Security

- **No direct fact writes.** Only `mcp__sku__*` tools are exposed; no filesystem/Bash/Write tools. The one write-capable tool, `apply_champion_adjustment`, **only stages** a `pending` proposal — it cannot touch `fact_ai_champion_forecast`. The actual forecast write happens **only** when a planner approves via the key-guarded `POST /sku-chat/adjustment/{id}`, which calls the existing guarded `save_adjustment` (guardrails re-applied, quantities re-derived from the baseline). So `bypassPermissions` stays safe: the worst the agent can do autonomously is stage a proposal a human must approve.
- All endpoints that write or that return conversation content carry `Depends(require_api_key)` — `/stream`, `/session` (POST), `/session/{id}` (GET history), and `/adjustment/{id}`. Only `/config` (no secrets, no user data) is open. Note: `require_api_key` is a single shared key, so there is no per-principal ownership check — `created_by` comes from an untrusted `X-User` header and is audit metadata only; per-user history isolation would require the platform RBAC (08-02), not this router's shared-key guard.
- Secrets (`ANTHROPIC_API_KEY`, cloud creds) never logged, never returned by `/sku-chat/config`, never placed in the system prompt or message history.
- Prompt-injection posture: tool results are data, not instructions; the system prompt instructs the agent to treat SKU data as evidence and to refuse out-of-scope/destructive requests. Tools take typed args and bind values via `%s` — no SQL string interpolation.
- Per-turn token/turn/time budgets bound cost and runaway loops.

---

## Testing

| Layer | Test | Notes |
|---|---|---|
| Tools | `tests/unit/test_sku_chat_tools.py` | Each `@tool` handler against `make_pool`-style mocks; full-grain join asserted |
| Router | `tests/api/test_sku_chat.py` | httpx `AsyncClient` + `ASGITransport`, `patch("api.core._get_pool")`; the Agent SDK `query` is mocked to yield canned `StreamEvent`/`ResultMessage` objects |
| Routing | `tests/unit/test_sku_chat_router.py` | Tier selection from classifier label + override precedence |
| Auth | `tests/unit/test_sku_chat_auth.py` | Each `auth.mode` sets the right env / fails loud when creds missing |
| Frontend | `src/tabs/__tests__/SkuChatTab.test.tsx` | `TestQueryWrapper`, barrel mock, SSE stream stubbed |
| E2E | `frontend/e2e/tests/navigation.spec.ts` | New sidebar tab via `navigateToTab()`, semantic selectors |

Run `make test-all`. The SDK is always mocked in tests — no live model calls in CI.

---

## Rollout / Phasing

1. **Phase 1 — grounded single-SKU chat (built).** Tiered routing, the 7 read-only tools, SSE streaming, `auth.mode`, the standalone tab.
2. **Phase 2 — tiered routing + deep diagnosis (built in Phase 1).** Heuristic tier router (`deep`/Opus tier, cluster-peer reasoning, "Deep analysis" toggle). An LLM intent classifier remains a drop-in upgrade to `classify_tier`.
3. **Phase 3 — persistence + global page-aware chat + agentic adjust (built).** `sku_chat_*` tables (`sql/196`–`197`), best-effort `store.py`, `GET /session/{id}` history, per-turn cost/usage logging, a **global `GlobalChatDrawer` on every tab** (per-page focus/suggestions/scope via `pageChatConfig`), and the champion-adjust approval flow. Remaining: cost/usage *dashboards* over `sku_chat_call_log`, **deeper per-page aggregate tools** (portfolio/inventory) beyond the SKU read tools, and a live validation of `auth.mode: api_key|bedrock|vertex` for an out-of-Claude-Code deployment.

---

## Open Questions / Decisions

| # | Question | Default taken |
|---|---|---|
| 1 | Streaming vs request/response? | **SSE streaming** — chat UX; falls back to a buffered `result` if SSE unsupported |
| 2 | Persist conversations or stateless? | **Stateless (Redis session) for Phase 1**, DB persistence in Phase 3 |
| 3 | Where does the spec live / what name? | **`06-ai-platform/07-sku-chatbot.md`** (AI platform sibling to 06-01) |
| 4 | Multi-SKU in scope? | **Single-SKU first**; cluster-peer comparison is a bounded multi-SKU case in Phase 2 |
| 5 | Subscription-auth ToS for long-running servers | **Flagged** — `auto` for local Claude-Code use; production uses `api_key`/Bedrock/Vertex. Confirm against installed SDK/CLI version before Phase 3. |

---

## Dependencies

| Dependency | Reason |
|---|---|
| `claude-agent-sdk` (Python) | Agent loop, in-process MCP tools, model selection, streaming, runtime auth delegation |
| Claude Code CLI runtime (bundled with the SDK) | Resolves model auth (`auto`/api-key/Bedrock/Vertex) |
| `dim_sku` + SKU features (03-02) | SKU profile, cluster, ABC-XYZ, execution lag |
| Production + candidate forecast (02) | Forward forecast + backtest overlay tools |
| Demand history (`fact_sales_monthly` / `agg_sales_monthly`, 03-06) | Sales-history tool |
| Inventory aggregates (`agg_inventory_monthly`) | Inventory tool |
| Accuracy MVs (`agg_accuracy_by_dim`, `agg_accuracy_by_dfu` / sql 193) | Accuracy/bias/WAPE tool |
| Redis (Phase 1) / Postgres (Phase 3) | Session continuity + audit |

---

## See Also

- `06-ai-platform/01-ai-planning-agent.md` — proactive, write-capable, portfolio-wide AI agent (this is its interactive, read-only, SKU-scoped sibling)
- `03-demand-intelligence/02-sku-feature-engineering.md` — the `dim_sku` features the profile tool surfaces
- `03-demand-intelligence/06-demand-history-workbench.md` — per-SKU demand decomposition reused by the sales tool
- `02-forecasting/` — production/candidate forecast and accuracy endpoints reused as tools
- `08-integration/02-rbac.md` — `require_api_key` guard applied to write-class chat endpoints
