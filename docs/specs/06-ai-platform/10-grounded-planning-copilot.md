# Grounded Planning Copilot

## Outcome

The global assistant uses an owner-scoped, evidence-grounded session contract. It may explain
planning data, but it cannot mutate forecasts, inventory, or workflow state. Every answer must cite
evidence produced during the same turn, in retrieval order, with a SHA-256 content hash.

## Trust boundary

- The server resolves page, DFU, opportunity, exception, and workflow context. Clients cannot submit
  forecast-promotion or inventory-run lineage identifiers.
- Local Ollama inference on a loopback URL is the default. Cloud OpenAI use requires both an explicit
  configuration mode and `DEMAND_AI_ALLOW_CLOUD`; provider fallback is prohibited.
- Session and turn writes require the API key and authenticated owner. Reads are owner-scoped.
- Prompts are limited to 4,000 characters; tool calls, turns, conversation history, concurrency,
  retention, and session TTL are bounded in `config/ai/copilot_config.yaml`.
- General answers have `action_request: null`. Existing approval workflows remain separate.

## Persistence

Migration `204_create_grounded_copilot_inventory_opportunities.sql` introduces owner-scoped session,
turn, and evidence tables. Evidence stores its source, business key, freshness, exact value snapshot,
content hash, and forecast/inventory lineage. Session creation and turns use idempotency keys.

## API and UI

- `POST /ai-copilot/sessions`
- `GET /ai-copilot/sessions/{session_id}`
- `POST /ai-copilot/sessions/{session_id}/turns`

The global drawer uses these contracts and displays expandable evidence metadata. The standalone SKU
chat remains available for its existing specialized workflow.

## Failure behavior

Invalid context or ungrounded output is rejected. Missing configuration/provider/database returns a
stable unavailable response without leaking exception text. No ungrounded answer is returned.
