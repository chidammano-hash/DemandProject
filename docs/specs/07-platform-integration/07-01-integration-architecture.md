# 07-01 — Platform Integration Architecture

## Overview

This spec covers the four bidirectional integration vectors that connect Demand Studio to external systems: ERP/WMS, cloud data pipelines, REST API consumers, and notification/alerting channels.

All vectors are independently deployable, reuse the existing job scheduler (APScheduler), config YAML pattern, `_run_*` callable pattern, and `require_api_key` auth — no replacement of existing infrastructure.

---

## Priority Order (by impact/effort ratio)

| # | Vector | Effort | Why first |
|---|---|---|---|
| 1 | Notifications & Alerting | Low | Zero infra deps; pure outbound HTTP; unblocks AI agent alerting immediately |
| 2 | REST API Consumers | Medium | Needs CORS + rate limiting first; enables partner integrations |
| 3 | Data Pipeline Connectors | Medium | Replaces CSV ETL; enables cloud-scale data ingest/export |
| 4 | ERP / WMS Integration | High | Adapter per system; highest complexity; depends on vectors 1-3 |

---

## Vector 1 — Notifications & Alerting

### Supported Channels
- **Slack** — incoming webhook URL per workspace channel
- **Microsoft Teams** — Adaptive Card webhooks (plain text webhooks deprecated)
- **Email** — SMTP relay (SendGrid, SES, or corporate SMTP)
- **PagerDuty** — Events API v2

### New Files

| File | Purpose |
|---|---|
| `common/notification_engine.py` | Core dispatcher: `NotificationEvent` dataclass, `NotificationEngine` class with per-channel adapters (`_send_slack`, `_send_teams`, `_send_email`, `_send_pagerduty`), env-var interpolation of secrets (`${VAR}` → `os.environ`), per-channel sliding-window rate limiter (deque), `_send_with_retry()` with exponential backoff, `get_notification_engine()` singleton |
| `config/notification_config.yaml` | Channels (slack_ops, slack_critical, teams, email, pagerduty) with `"${ENV_VAR}"` secret refs; routing rules: `event_type → severity → [channel_ids]`; retry config; per-channel rate limit |
| `api/routers/notifications.py` | `GET /notifications/channels`, `POST /notifications/test` (auth), `GET /notifications/history` |
| `sql/059_create_notification_log.sql` | `notification_log` table: event_type, severity, channel_id, channel_type, status (sent/failed/skipped), error, sent_at, source, metadata JSONB; indexes on (event_type, sent_at DESC), (status, sent_at DESC) |

### Existing Files Modified

| File | Change |
|---|---|
| `common/job_registry.py` | In `_execute_job()`: inject `get_notification_engine().notify(NotificationEvent(...))` after success + failure branches (best-effort, wrapped in try/except) |
| `common/ai_planner.py` | After `create_insight` writes a critical/high insight, emit `NotificationEvent(event_type="insight_critical", severity=insight.severity, ...)` |
| `api/main.py` | Add `notifications.router` import + `app.include_router()` |

### NotificationEvent Schema

```python
@dataclass
class NotificationEvent:
    event_type: str      # "job_completed" | "job_failed" | "insight_critical" | "stockout_alert"
    title: str
    body: str
    severity: str        # "critical" | "high" | "medium" | "low"
    metadata: dict       # job_id, item_no, loc, financial_impact, etc.
    source: str          # "job_engine" | "ai_planner" | "storyboard" | "control_tower"
```

### Routing Rules (config-driven)

```yaml
routing:
  job_failed:
    critical: [slack_critical, pagerduty_critical]
    high:     [slack_ops]
  insight_critical:
    critical: [slack_critical, email_planning, pagerduty_critical]
    high:     [slack_ops, email_planning]
  stockout_alert:
    critical: [slack_critical, pagerduty_critical]
```

### Key Constraints
- All notification calls are **non-fatal** — wrapped in try/except; a failed Slack webhook never crashes a job
- Secrets never in YAML; `${VAR}` refs resolved from `os.environ` at load time
- Per-channel deque tracks send count to prevent webhook 429 storms
- Teams uses Adaptive Card JSON payload (required by Microsoft)

---

## Vector 2 — REST API Consumers

Enables external apps (ERP portals, partner dashboards, BI tools) to consume Demand Studio data via its REST API.

### New Files

| File | Purpose |
|---|---|
| `common/rate_limiter.py` | Sliding window rate limiter keyed by `(api_key_hash, endpoint_group)`. Thread-safe deque. Raises `HTTPException(429)` with `Retry-After` header. Tiers: default (60 req/min), premium (300 req/min), internal (unlimited) |
| `config/api_consumer_config.yaml` | CORS allowed_origins (env var refs), rate limit tiers + key→tier mapping, public endpoint surface definitions, webhook signing secret ref |
| `common/webhook_dispatcher.py` | `WebhookDispatcher`: loads registrations from DB, fans out to subscribed URLs, signs payloads HMAC-SHA256 (`X-Demand-Signature: sha256=<hex>`), retries with backoff, logs to `webhook_delivery_log` |
| `api/routers/webhooks.py` | `POST /webhooks/register` (auth), `DELETE /webhooks/{id}` (auth), `GET /webhooks` (auth), `POST /webhooks/{id}/test`, `POST /webhooks/receive/{source}` (inbound ERP push receiver) |
| `sql/060_create_webhook_registrations.sql` | `webhook_registrations` (webhook_id PK, consumer_name, target_url, event_types[], api_key_hash, enabled, secret, delivery_count, failure_count) + `webhook_delivery_log` (webhook_id FK, event_type, status, http_status, attempt_count, error) |

### Existing Files Modified

| File | Change |
|---|---|
| `api/main.py` | Replace hardcoded `CORSMiddleware` origins with `_load_cors_origins()` reading `api_consumer_config.yaml` + resolving env vars; add `RateLimitMiddleware(BaseHTTPMiddleware)` applied globally before route dispatch; add `webhooks.router` |
| `frontend/vite.config.ts` | Add `/webhooks` proxy entry |
| `api/routers/production_forecast.py` | After forecast publication, call `get_webhook_dispatcher().dispatch("forecast_published", {...})` |

### Webhook Signing (GitHub/Stripe pattern)

```
X-Demand-Signature: sha256=HMAC-SHA256(signing_secret, timestamp + "." + body_bytes)
X-Demand-Timestamp: <unix epoch>
```

Consumers verify by computing the same HMAC and comparing with `hmac.compare_digest()`.

### Event Types Dispatched

| Event | Trigger |
|---|---|
| `forecast_published` | New production forecast plan version written |
| `insight_created` | AI planner creates a new insight |
| `exception_generated` | Replenishment exception queue regenerated |
| `job_completed` | Any scheduled job completes |

### Key Constraints
- CORS: `localhost:5173` stays hardcoded as baseline; additional origins loaded from config (non-breaking)
- Rate limiting as Starlette middleware — applies globally, no per-router changes needed
- `POST /webhooks/receive/{source}` validates ERP-specific inbound signatures before any processing

---

## Vector 3 — Data Pipeline Connectors

Replaces or augments CSV-based ETL with direct cloud data warehouse connectors.

### Supported Systems
- **Snowflake** — `snowflake-connector-python`
- **Google BigQuery** — `google-cloud-bigquery`
- **AWS S3** — `boto3` (Parquet/CSV)
- **Databricks** — `databricks-sql-connector` (Unity Catalog support)

### New Files

| File | Purpose |
|---|---|
| `common/cloud_connector.py` | Abstract `CloudConnector` base (`read_table()`, `write_dataframe()`, `test_connection()`); concrete `SnowflakeConnector`, `BigQueryConnector`, `S3Connector`, `DatabricksConnector`; `get_connector(source_id)` factory; transform registry (normalize_sales, normalize_forecast, normalize_inventory reuse existing logic); incremental load watermark support |
| `config/data_pipeline_config.yaml` | Sources (snowflake_sales, s3_inventory, bigquery_forecast) + sinks (snowflake_forecast_output, s3_insights_export) with env var secret refs, SQL queries, target tables, transform names, incremental column; `pipeline_schedule` block with cron per source/sink |
| `scripts/run_cloud_pull.py` | CLI: `--source <source_id> [--dry-run]` — fetch via connector → apply transform → validate against DomainSpec → load to Postgres |
| `scripts/run_cloud_push.py` | CLI: `--sink <sink_id> [--dry-run]` — query from Postgres → write to cloud sink |
| `api/routers/data_pipeline.py` | `GET /data-pipeline/sources`, `GET /data-pipeline/sinks`, `POST /data-pipeline/pull/{source_id}` (auth), `POST /data-pipeline/push/{sink_id}` (auth), `GET /data-pipeline/runs`, `POST /data-pipeline/test/{source_id}` |

### Existing Files Modified

| File | Change |
|---|---|
| `common/job_state.py` | Add `_run_cloud_pull(params, progress_cb)` and `_run_cloud_push(params, progress_cb)` wrapping the CLI scripts via `_run_subprocess()` |
| `common/job_registry.py` | Register `cloud_pull` and `cloud_push` in `JOB_TYPE_REGISTRY` with group `"etl"` |
| `api/main.py` | Add `data_pipeline.router` |
| `frontend/vite.config.ts` | Add `/data-pipeline` proxy entry |
| `pyproject.toml` | Add optional dep groups: `[project.optional-dependencies]` snowflake, bigquery, s3, databricks |

### Data Pipeline Config Pattern

```yaml
sources:
  snowflake_sales:
    type: snowflake
    account: "${SNOWFLAKE_ACCOUNT}"
    user: "${SNOWFLAKE_USER}"
    password: "${SNOWFLAKE_PASSWORD}"
    warehouse: "DEMAND_WH"
    database: "PROD_DB"
    schema: "SUPPLY_CHAIN"
    query: "SELECT ... FROM fact_sales WHERE month_start >= ..."
    target_table: fact_sales_monthly
    transform: normalize_sales
    incremental_column: month_start    # optional: only fetch rows newer than watermark

sinks:
  snowflake_forecast_output:
    type: snowflake
    target_table: FACT_PRODUCTION_FORECAST
    source_query: "SELECT ... FROM fact_production_forecast WHERE plan_version = %(plan_version)s"

pipeline_schedule:
  pull_snowflake_sales:
    source_id: snowflake_sales
    cron: "0 2 * * *"
    job_type: cloud_pull
```

### Key Constraints
- Cloud connectors are **optional installs** — factory raises `ImportError` with install instructions if package not present; no import error at module load
- Schema validation: DataFrame columns validated against `DomainSpec` for target table before any Postgres write
- Incremental load: `watermark_table` stores last-loaded timestamp per source to avoid full reloads

---

## Vector 4 — ERP / WMS Integration

### Supported Systems
- **SAP S/4HANA / ECC** — OData v2 REST, IDoc, optional RFC via `pyrfc`
- **Oracle Fusion Cloud / EBS** — REST Data Services (ORDS), OAuth2
- **NetSuite** — SuiteTalk REST, Token-Based Authentication (OAuth 1.0a)
- **Manhattan Associates WMS** — REST API + optional SFTP file drop

### New Files

| File | Purpose |
|---|---|
| `common/erp_adapter.py` | Abstract `ERPAdapter` base: `parse_purchase_orders()`, `parse_item_master()`, `parse_receipts()`, `format_planned_order()`; concrete `SAPAdapter`, `OracleAdapter`, `NetSuiteAdapter`, `ManhattanAdapter`; `get_erp_adapter(system_id)` factory |
| `config/erp_config.yaml` | Per-system: connection type (odata/rest/sftp), credentials as env var refs, inbound entity configs with field_map + poll_interval, outbound config with approval gate; `field_validation` rules |
| `scripts/run_erp_pull.py` | CLI: `--system <id> --entity purchase_orders\|receipts [--since DATE]` — poll ERP → transform via adapter → load to Postgres supply tables → log to `erp_sync_log` |
| `scripts/run_erp_push.py` | CLI: `--system <id> [--dry-run]` — query `fact_planned_orders WHERE status='approved'` → translate via adapter → POST to ERP; **never pushes unapproved orders** |
| `api/routers/erp_integration.py` | `GET /erp/systems`, `POST /erp/{id}/test` (auth), `POST /erp/{id}/pull/pos` (auth), `POST /erp/{id}/pull/receipts` (auth), `POST /erp/{id}/push/orders` (auth), `GET /erp/sync-log` |
| `sql/061_create_erp_sync_log.sql` | `erp_sync_log`: system_id, direction (inbound/outbound), entity_type, status, records_in, records_out, error_count, errors JSONB, started_at, completed_at, triggered_by |

### Existing Files Modified

| File | Change |
|---|---|
| `common/job_state.py` | Add `_run_erp_pull(params, progress_cb)` and `_run_erp_push(params, progress_cb)` |
| `common/job_registry.py` | Register `erp_pull` and `erp_push` in `JOB_TYPE_REGISTRY` with group `"etl"` |
| `api/routers/supply.py` | `POST /supply/planned-orders/{id}/release` triggers ERP push as side effect when ERP system is configured + enabled |
| `api/main.py` | Add `erp_integration.router` |
| `frontend/vite.config.ts` | Add `/erp` proxy entry |

### ERP Config Pattern

```yaml
systems:
  sap_prod:
    system: sap
    label: "SAP S/4HANA Production"
    enabled: true
    connection:
      type: odata
      base_url: "${SAP_ODATA_BASE_URL}"
      username: "${SAP_USER}"
      password: "${SAP_PASSWORD}"
    inbound:
      purchase_orders:
        endpoint: "MM_PUR_PO_MIGO_SRV/PurchaseOrderSet"
        poll_interval_minutes: 60
        field_map:
          po_number: "PurchaseOrder"
          item_no:   "Material"
          loc:       "Plant"
    outbound:
      planned_orders:
        type: odata_post
        endpoint: "MM_PUR_REQ_MAINT_SRV/PurchaseRequisitionSet"
        require_approval: true    # only push status='approved' orders
```

### System-Specific Notes

| System | Auth | Inbound Format | Outbound Format |
|---|---|---|---|
| SAP S/4HANA | Basic / mTLS | OData v2 JSON, IDoc XML | BAPI POST, OData POST |
| Oracle Fusion | OAuth 2.0 (client credentials) | REST JSON (ORDS) | REST POST |
| NetSuite | OAuth 1.0a TBA (`requests-oauthlib`) | SuiteTalk REST JSON | REST POST `/record/v1/purchaseOrder` |
| Manhattan WMS | API key header | REST JSON or SFTP XML | REST POST or SFTP file drop |

### Key Constraints
- Adapter isolation: all ERP-specific complexity (IDoc parsing, OData pagination, OAuth variants) stays inside the adapter class; the rest of the codebase only calls `parse_purchase_orders(payload)`
- Outbound approval gate: `run_erp_push.py` validates `status='approved'` — skips and logs warnings for unapproved orders; this gate cannot be bypassed via the REST API
- SAP OData v2 uses `$skip + $top` for pagination; handled transparently by `SAPAdapter`
- NetSuite uses OAuth 1.0a TBA (not 2.0); `requests-oauthlib` constructs the `Authorization` header

---

## New Makefile Targets

```makefile
# Notifications
notify-schema          # Apply sql/059_create_notification_log.sql
notify-test            # POST /notifications/test to configured Slack channel

# Webhooks / API Consumers
webhooks-schema        # Apply sql/060_create_webhook_registrations.sql
webhooks-list          # GET /webhooks (lists registered consumers)

# Data Pipeline
pipeline-pull          # make pipeline-pull SOURCE=snowflake_sales
pipeline-push          # make pipeline-push SINK=snowflake_forecast_output
pipeline-test          # make pipeline-test SOURCE=snowflake_sales

# ERP Integration
erp-schema             # Apply sql/061_create_erp_sync_log.sql
erp-pull               # make erp-pull SYSTEM=sap_prod ENTITY=purchase_orders
erp-push               # make erp-push SYSTEM=sap_prod (--dry-run by default)
erp-push-live          # make erp-push-live SYSTEM=sap_prod
erp-test               # make erp-test SYSTEM=sap_prod
```

---

## Required Tests

| Test File | Vector | Covers |
|---|---|---|
| `tests/unit/test_notification_engine.py` | 1 | Channel dispatch, routing rules, retry, rate limiting, env var interpolation |
| `tests/api/test_notifications.py` | 1 | GET /channels, POST /test, GET /history |
| `tests/unit/test_rate_limiter.py` | 2 | Sliding window, tier enforcement, 429 behavior |
| `tests/api/test_webhooks.py` | 2 | Register, list, delete, test delivery, inbound receiver |
| `tests/unit/test_cloud_connector.py` | 3 | Factory, connector instantiation (mocked), transform registry, schema validation |
| `tests/api/test_data_pipeline.py` | 3 | Sources list, pull trigger, push trigger, run history |
| `tests/unit/test_erp_adapter.py` | 4 | Each adapter parse/format with fixture payloads (SAP, Oracle, NetSuite, Manhattan JSON) |
| `tests/api/test_erp_integration.py` | 4 | Systems list, pull/push endpoints, sync-log |

All HTTP calls mocked; all DB calls use `make_pool()` from `tests/api/conftest.py`.

---

## Environment Variables

```bash
# Vector 1 — Notifications
SLACK_WEBHOOK_OPS=https://hooks.slack.com/services/...
SLACK_WEBHOOK_CRITICAL=...
TEAMS_WEBHOOK_SUPPLY=...
SMTP_HOST=smtp.sendgrid.net
SMTP_USER=apikey
SMTP_PASSWORD=...
PAGERDUTY_INTEGRATION_KEY=...

# Vector 2 — API Consumers
WEBHOOK_SIGNING_SECRET=<32 bytes hex>
CONSUMER_ORIGIN_1=https://erp.company.com

# Vector 3 — Data Pipeline
SNOWFLAKE_ACCOUNT=xy12345.us-east-1
SNOWFLAKE_USER=demand_studio
SNOWFLAKE_PASSWORD=...
S3_BUCKET=company-supply-chain-data
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
GCP_PROJECT=company-prod
GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json

# Vector 4 — ERP Integration
SAP_ODATA_BASE_URL=https://sap.company.com/sap/opu/odata/sap/
SAP_USER=demand_studio_integration
SAP_PASSWORD=...
ORACLE_FUSION_URL=https://oracle.company.com
ORACLE_CLIENT_ID=...
ORACLE_CLIENT_SECRET=...
NETSUITE_ACCOUNT_ID=...
NETSUITE_CONSUMER_KEY=...
NETSUITE_TOKEN_ID=...
MANHATTAN_BASE_URL=https://manhattan.company.com
MANHATTAN_API_KEY=...
```

---

## Verification

| Vector | End-to-End Test |
|---|---|
| Notifications | `POST /notifications/test` → message appears in Slack; `notification_log` row with `status='sent'` |
| API Consumers | Set `API_KEY`; call from external origin → 200 with key, 401 without; exceed rate → 429 with `Retry-After`; register webhook → trigger forecast publish → check `webhook_delivery_log` |
| Data Pipeline | `POST /data-pipeline/test/snowflake_sales` → success; trigger pull → job in `GET /jobs/active` → completes → rows in `fact_sales_monthly` |
| ERP Integration | `POST /erp/sap_prod/test` → success; trigger PO pull → POs visible in `GET /supply/open-pos`; approve planned order → trigger push → ERP receives order (dry-run log confirms) |