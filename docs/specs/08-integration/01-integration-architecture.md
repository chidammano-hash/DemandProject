# Integration Architecture

> Connects Supply Chain Command Center to external systems through two shipped integration vectors - notifications and REST API consumers - plus two proposed vectors not yet implemented: cloud data pipelines and ERP/WMS adapters.

| | |
|---|---|
| **Status** | Partial - Vectors 1-2 (Notifications, REST API Consumers) implemented; Vectors 3-4 (Cloud Data Pipelines, ERP/WMS) proposed, not yet implemented |
| **UI Tab** | N/A (infrastructure layer) |
| **Key Files** | `common/services/notification_engine.py`, `common/services/rate_limiter.py`, `common/services/webhook_dispatcher.py`, `api/routers/platform/webhooks.py`, `api/routers/platform/notifications.py` |

---

## Problem

Supply Chain Command Center runs as a standalone application. Planners cannot receive alerts in Slack when a job fails, ERP systems cannot push purchase orders into the platform, and BI tools cannot pull forecast data via API. Without integration points, the platform stays siloed from the tools teams already use daily.

## Solution

Two integration vectors are shipped today; two more are proposed for a future phase. Each reuses (or would reuse) the existing job scheduler, YAML config pattern, and API key auth. Notifications push alerts outbound. REST API governance controls external consumers. The proposed cloud connectors would replace CSV-based ETL, and the proposed ERP adapters would handle bidirectional data exchange with enterprise systems - neither is implemented yet.

## How It Works

### Vector 1 -- Notifications & Alerting

1. Platform events (job failures, AI insights, stockout alerts) create a `NotificationEvent`
2. The notification engine routes the event to channels based on severity rules in config
3. Supported channels: Slack (incoming webhooks), Microsoft Teams (Adaptive Cards), Email (SMTP), PagerDuty (Events API v2)
4. Failed deliveries retry with exponential backoff; all sends are non-fatal (never crash a job)
5. Secrets use `${ENV_VAR}` references in YAML, resolved from `os.environ` at load time

### Vector 2 -- REST API Consumers

1. External apps authenticate with API key (`X-API-Key` header)
2. Sliding-window rate limiter enforces per-client-IP request limits on write endpoints (POST/PUT/DELETE; default 300 req/min for the standard tier)
3. CORS origins loaded from config (localhost:5173 always included as baseline)
4. Webhook registrations allow external systems to subscribe to platform events
5. Outbound webhooks are signed with HMAC-SHA256 for payload verification

### Vector 3 - Cloud Data Pipeline Connectors (Proposed, not yet implemented)

**Status: proposed.** No `CloudConnector` base class or Snowflake/BigQuery/S3/Databricks integration exists in the codebase today. The steps below describe the intended design for a future phase, not shipped behavior.

1. Abstract `CloudConnector` base class with concrete implementations per platform
2. Supported: Snowflake, Google BigQuery, AWS S3 (Parquet/CSV), Databricks
3. Pull flow: fetch from cloud source, apply transform, validate against DomainSpec, load to Postgres
4. Push flow: query from Postgres, write to cloud sink
5. Incremental load via watermark column avoids full reloads
6. Cloud connectors are optional installs -- missing packages raise clear install instructions

### Vector 4 - ERP / WMS Integration (Proposed, not yet implemented)

**Status: proposed.** No `ERPAdapter` base class or SAP/Oracle/NetSuite/Manhattan integration exists in the codebase today. The steps below describe the intended design for a future phase, not shipped behavior.

1. Abstract `ERPAdapter` base class with per-system implementations
2. Supported: SAP S/4HANA (OData), Oracle Fusion (REST/OAuth2), NetSuite (SuiteTalk), Manhattan WMS (REST/SFTP)
3. Inbound: poll ERP for purchase orders, receipts, item master updates
4. Outbound: push approved planned orders back to ERP (unapproved orders are never pushed)
5. All ERP-specific complexity (IDoc parsing, OData pagination, OAuth variants) stays inside adapters

## Priority Order

| # | Vector | Status | Effort | Rationale |
|---|---|---|---|---|
| 1 | Notifications & Alerting | Implemented | Low | No infra deps; unblocks AI agent alerting |
| 2 | REST API Consumers | Implemented | Medium | Needs CORS + rate limiting; enables partner integrations |
| 3 | Data Pipeline Connectors | Proposed | Medium | Replaces CSV ETL; enables cloud-scale ingest/export |
| 4 | ERP / WMS Integration | Proposed | High | Adapter per system; depends on vectors 1-3 |

## Event Types

| Event | Trigger | Vectors |
|---|---|---|
| `job_completed` / `job_failed` | Any scheduled job finishes | Notifications, Webhooks |
| `forecast_published` | New production forecast version | Webhooks |
| `insight_created` | AI planner creates insight | Notifications, Webhooks |
| `exception_generated` | Replenishment exception queue regenerated | Notifications, Webhooks |
| `stockout_alert` | Stockout detected | Notifications |
| `threshold_breach` | KPI crosses configured limit | Notifications, Webhooks |
| `approval_required` | S&OP stage needs planner action | Notifications, Webhooks |
| `data_loaded` | Dataset loaded into Postgres | Webhooks |

## Configuration

Each implemented vector has its own config file. The proposed vectors (3 and 4) have no config file, since no code reads one yet:

| Vector | Config File |
|---|---|
| Notifications | `config/platform/notification_config.yaml` |
| API Consumers | Rate limiting and versioning config lives in `config/platform/auth_config.yaml` under the `governance:` block - see [API Governance](./09-api-governance.md) |
| Data Pipelines (proposed) | Not yet implemented - no config file exists |
| ERP Integration (proposed) | Not yet implemented - no config file exists |

All secrets use `${ENV_VAR}` references -- never stored in YAML.

## Environment Variables

| Variable | Vector | Purpose |
|---|---|---|
| `SLACK_WEBHOOK_OPS` | Notifications | Slack ops channel webhook URL |
| `SLACK_WEBHOOK_CRITICAL` | Notifications | Slack critical channel webhook URL |
| `SMTP_HOST`, `SMTP_USER`, `SMTP_PASSWORD` | Notifications | Email relay credentials |
| `PAGERDUTY_INTEGRATION_KEY` | Notifications | PagerDuty Events API key |
| `WEBHOOK_SIGNING_SECRET` | API Consumers | HMAC signing secret for outbound webhooks |
| `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_PASSWORD` | Data Pipelines (proposed) | Snowflake connection - not read by any code today |
| `SAP_ODATA_BASE_URL`, `SAP_USER`, `SAP_PASSWORD` | ERP (proposed) | SAP S/4HANA connection - not read by any code today |

## Dependencies

- Vector 1 (Notifications): No external deps beyond `httpx`
- Vector 2 (API Consumers): No external deps (stdlib `threading`, `hmac`)
- Vector 3 (Data Pipelines, proposed): would require `snowflake-connector-python`, `google-cloud-bigquery`, `boto3`, `databricks-sql-connector` - none installed today
- Vector 4 (ERP, proposed): would require `pyrfc` (SAP RFC), `requests-oauthlib` (NetSuite OAuth 1.0a) - none installed today

## See Also

- [Notifications](./04-notifications.md) -- channel dispatch details
- [Webhooks](./10-webhooks.md) -- outbound event delivery
- [API Governance](./09-api-governance.md) -- rate limiting and usage tracking
- [RBAC](./02-rbac.md) -- authentication and authorization
- [Caching](./03-caching.md) -- query performance optimization
