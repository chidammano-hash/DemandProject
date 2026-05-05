# Integration Architecture

> Connects Supply Chain Command Center to external systems through four integration vectors: notifications, REST API consumers, cloud data pipelines, and ERP/WMS adapters.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | N/A (infrastructure layer) |
| **Key Files** | `common/notification_engine.py`, `common/rate_limiter.py`, `common/webhook_dispatcher.py`, `api/routers/webhooks.py`, `api/routers/notifications.py` |

---

## Problem

Supply Chain Command Center runs as a standalone application. Planners cannot receive alerts in Slack when a job fails, ERP systems cannot push purchase orders into the platform, and BI tools cannot pull forecast data via API. Without integration points, the platform stays siloed from the tools teams already use daily.

## Solution

Four independently deployable integration vectors connect the platform to external systems. Each vector reuses the existing job scheduler, YAML config pattern, and API key auth. Notifications push alerts outbound. REST API governance controls external consumers. Cloud connectors replace CSV-based ETL. ERP adapters handle bidirectional data exchange with enterprise systems.

## How It Works

### Vector 1 -- Notifications & Alerting

1. Platform events (job failures, AI insights, stockout alerts) create a `NotificationEvent`
2. The notification engine routes the event to channels based on severity rules in config
3. Supported channels: Slack (incoming webhooks), Microsoft Teams (Adaptive Cards), Email (SMTP), PagerDuty (Events API v2)
4. Failed deliveries retry with exponential backoff; all sends are non-fatal (never crash a job)
5. Secrets use `${ENV_VAR}` references in YAML, resolved from `os.environ` at load time

### Vector 2 -- REST API Consumers

1. External apps authenticate with API key (`X-API-Key` header)
2. Token bucket rate limiter enforces per-client request limits (default 100 req/min)
3. CORS origins loaded from config (localhost:5173 always included as baseline)
4. Webhook registrations allow external systems to subscribe to platform events
5. Outbound webhooks are signed with HMAC-SHA256 for payload verification

### Vector 3 -- Cloud Data Pipeline Connectors

1. Abstract `CloudConnector` base class with concrete implementations per platform
2. Supported: Snowflake, Google BigQuery, AWS S3 (Parquet/CSV), Databricks
3. Pull flow: fetch from cloud source, apply transform, validate against DomainSpec, load to Postgres
4. Push flow: query from Postgres, write to cloud sink
5. Incremental load via watermark column avoids full reloads
6. Cloud connectors are optional installs -- missing packages raise clear install instructions

### Vector 4 -- ERP / WMS Integration

1. Abstract `ERPAdapter` base class with per-system implementations
2. Supported: SAP S/4HANA (OData), Oracle Fusion (REST/OAuth2), NetSuite (SuiteTalk), Manhattan WMS (REST/SFTP)
3. Inbound: poll ERP for purchase orders, receipts, item master updates
4. Outbound: push approved planned orders back to ERP (unapproved orders are never pushed)
5. All ERP-specific complexity (IDoc parsing, OData pagination, OAuth variants) stays inside adapters

## Priority Order

| # | Vector | Effort | Rationale |
|---|---|---|---|
| 1 | Notifications & Alerting | Low | No infra deps; unblocks AI agent alerting |
| 2 | REST API Consumers | Medium | Needs CORS + rate limiting; enables partner integrations |
| 3 | Data Pipeline Connectors | Medium | Replaces CSV ETL; enables cloud-scale ingest/export |
| 4 | ERP / WMS Integration | High | Adapter per system; depends on vectors 1-3 |

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

Each vector has its own config file:

| Vector | Config File |
|---|---|
| Notifications | `config/platform/notification_config.yaml` |
| API Consumers | `config/api_governance_config.yaml` |
| Data Pipelines | `config/data_pipeline_config.yaml` |
| ERP Integration | `config/erp_config.yaml` |

All secrets use `${ENV_VAR}` references -- never stored in YAML.

## Environment Variables

| Variable | Vector | Purpose |
|---|---|---|
| `SLACK_WEBHOOK_OPS` | Notifications | Slack ops channel webhook URL |
| `SLACK_WEBHOOK_CRITICAL` | Notifications | Slack critical channel webhook URL |
| `SMTP_HOST`, `SMTP_USER`, `SMTP_PASSWORD` | Notifications | Email relay credentials |
| `PAGERDUTY_INTEGRATION_KEY` | Notifications | PagerDuty Events API key |
| `WEBHOOK_SIGNING_SECRET` | API Consumers | HMAC signing secret for outbound webhooks |
| `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_PASSWORD` | Data Pipelines | Snowflake connection |
| `SAP_ODATA_BASE_URL`, `SAP_USER`, `SAP_PASSWORD` | ERP | SAP S/4HANA connection |

## Dependencies

- Vector 1 (Notifications): No external deps beyond `httpx`
- Vector 2 (API Consumers): No external deps (stdlib `threading`, `hmac`)
- Vector 3 (Data Pipelines): Optional `snowflake-connector-python`, `google-cloud-bigquery`, `boto3`, `databricks-sql-connector`
- Vector 4 (ERP): Optional `pyrfc` (SAP RFC), `requests-oauthlib` (NetSuite OAuth 1.0a)

## See Also

- [Notifications](./04-notifications.md) -- channel dispatch details
- [Webhooks](./10-webhooks.md) -- outbound event delivery
- [API Governance](./09-api-governance.md) -- rate limiting and usage tracking
- [RBAC](./02-rbac.md) -- authentication and authorization
- [Caching](./03-caching.md) -- query performance optimization
