# Commercial Readiness Audit — Supply Chain Command Center

**Author:** Fractional CTO / Tech DD
**Date:** 2026-05-13
**Status:** Initial assessment
**Audience:** Founder, board, prospective investors, prospective design partners

---

## 1. Executive Readiness Verdict

**Score: 3 / 10 on a "ship to a paying external customer" scale.**

What is here is impressive on the planning-science axis: 70+ FastAPI routers, 88 SQL DDL migrations, 38 YAML configs, a real ML competition harness with champion selection, an AI agent with circuit breakers, materialized-view tiering for a 198M-row inventory fact, Redis-backed multi-worker caching, an opt-in read replica, and a pg-queue worker for long jobs. Architecturally this is closer to RELEX/o9 than to the typical "demo with five charts." See [docs/ARCHITECTURE.md](../ARCHITECTURE.md) and [docs/ENTERPRISE_ARCHITECTURE.md](../ENTERPRISE_ARCHITECTURE.md).

But on the commercialization axis the platform is a single-tenant internal tool. There is no `tenant_id` anywhere in the schema — `grep` for "tenant" across [sql/](../../sql/), [api/](../../api/), and [common/](../../common/) returns one match in a comment. There is no SSO, no SCIM, no per-row data security model, no billing, no admin/CS console, no tenant provisioning, no customer-managed keys, no DPA template, no SOC2 work-in-progress, no formal SLA/observability stack. The auth layer ([common/auth.py](../../common/auth.py)) is a single-user-pool JWT with a four-role enum, and when both `JWT_SECRET` and `API_KEY` are unset it returns an anonymous user with **role=admin** (line 169) — that is a "delete in week 1" defect for any commercial deployment.

**The single biggest blocker is not technical: it is IP ownership.** The codebase contains explicit references to `sgwsagility.atlassian.net` Jira tickets in [docs/specs/PRD/PRD-CSCS-11725-brand-performance-chatbot.md](../specs/PRD/PRD-CSCS-11725-brand-performance-chatbot.md) and the CLAUDE.md notes the platform was "originally built as an internal tool for a multi-billion-dollar B2B distributor." Until a clean IP-assignment / carve-out / license agreement with Southern Glazer's is in writing, every other item on this punch list is moot. No investor, no design partner, no enterprise procurement team will sign anything without that letter.

The runner-up technical blocker is multi-tenancy. Everything else (SSO, billing, admin console, SOC2) is vendor-able / well-trodden. Multi-tenancy is the architectural decision that the next 50 design choices will inherit, and getting it wrong on a 198M-row fact table is expensive to undo.

---

## 2. Multi-Tenancy

### Current state

Zero. The schema in [sql/](../../sql/) has no `tenant_id` column on any of the 88 migrations. [common/core/db.py](../../common/core/db.py) returns a single set of connection params from environment variables. [common/auth.py](../../common/auth.py) `dim_user` table has no tenant FK. The seed admin in [sql/062_create_users_rbac.sql](../../sql/062_create_users_rbac.sql) is `admin@demandstudio.local` with bcrypt-hashed `admin123` — fine for a dev image, fatal in shared infrastructure.

The cache layer ([common/services/cache.py](../../common/services/cache.py)) is keyed without any tenant scoping, MV refresh in the Makefile is global, the `data/` artifact tree (champion CSVs, serialized model files, clustering features) lives at fixed filesystem paths via [common/core/paths.py](../../common/core/paths.py), and APScheduler jobs in [common/services/job_registry.py](../../common/services/job_registry.py) are global to the process.

### Options for SC planning workloads

| Option | Pros | Cons | Verdict for this product |
|---|---|---|---|
| (a) Silo'd-DB per tenant | Hardest data boundary; per-tenant Postgres tunings; per-tenant MV refresh windows; trivial blast-radius story; trivial GDPR-delete; per-tenant ML model artifacts naturally isolated. | Operational cost — N Postgres instances, N pools, N migration runs, N MV refresh jobs. Cross-tenant analytics impossible. Connection multiplexing problem at 50+ tenants. | **Recommended for first 1-20 customers.** |
| (b) Shared-DB + `tenant_id` column on every fact + RLS | Single Postgres, lowest infra cost. Easy upserts. Easy cross-tenant analytics for internal product team. | Massive schema rewrite — 88 DDLs touched, 70 routers touched, every MV rebuilt with `(tenant_id, …)` PKs, every index re-keyed. RLS performance degrades on the 198M-row tables (planner can't prune without explicit filter). One bad query plan leaks data. ML pipelines (champion CSVs, MLflow runs, cluster experiments) must be re-keyed too. Single-MV stampede affects all tenants. | High risk at this scale. Defer. |
| (c) Hybrid: shared dim tables + silo'd fact tables (schema-per-tenant) | Fact-table data isolation; shared `dim_*` tables for analytics + ML feature stores; one Postgres instance until you outgrow it. | Schema-search-path discipline required everywhere; tooling has to know which schema to query. MV refresh becomes per-schema. Backups become per-schema. | **Recommended for scale 20-200 tenants.** Migration from (a) → (c) is straightforward. |

### Recommendation

Start with **Option (a) — silo'd Postgres-per-tenant on shared K8s/RDS infrastructure**. Each tenant gets:

- A dedicated Postgres logical database (single RDS cluster, multiple databases initially; promote to dedicated cluster for paranoia tier).
- A dedicated Redis logical DB (`redis://…/<tenant_idx>`).
- A tenant-scoped object-store prefix for `data/` artifacts (`s3://<bucket>/tenants/<tenant_id>/champion/…`).
- A tenant-scoped MLflow experiment namespace.
- The same FastAPI process serving multiple tenants via a `X-Tenant-ID` header → `set_tenant_context()` middleware that swaps the pool reference per-request.

This avoids the 88-DDL rewrite, gets the platform to revenue in months not quarters, and matches what Blue Yonder Luminate / o9 actually do for their first cohort.

When the customer count crosses ~20 the operational pain (N migrations, N MV refreshes) forces a move to **Option (c)**. The RLS option is not recommended — fact-table sizes mean the optimizer needs explicit `tenant_id` predicates on every query, and one missing predicate is a data breach.

### Effort estimate

- **Option (a) end-to-end:** 4-6 engineering-months (one strong backend eng + ops). Includes per-request pool routing, tenant provisioning script, tenant-scoped artifact paths, scheduler tenancy, MV refresh-per-tenant, and operations runbook. Does **not** include the admin UI to provision tenants (separate punch-list item).
- **Migration (a) → (c) when needed:** 2 months.

---

## 3. Authentication & Authorization

### Current state

[common/auth.py](../../common/auth.py) provides:

- bcrypt password hashing (good).
- JWT HS256 with 30-min access + 7-day refresh ([config/platform/auth_config.yaml](../../config/platform/auth_config.yaml)).
- 4-tier role enum: viewer / planner / manager / admin (sql/062).
- `require_role()` factory and `require_api_key()` legacy header guard.
- `fact_audit_log` table with old/new value JSONB columns.

### Gaps for enterprise

| Gap | Severity | Notes |
|---|---|---|
| **Anonymous-as-admin fallback** | P0 critical | [common/auth.py:166-169](../../common/auth.py) — when `JWT_SECRET` is the dev default and `API_KEY` is unset, the request is granted role=admin. Must be removed or hard-fail in production; "auth disabled" mode must be impossible to reach in a deployed image. |
| **HS256 symmetric JWTs** | P1 | Migrate to RS256/EdDSA so the public key can be published and rotation does not require redeploying every consumer. |
| **No SSO** | P0 for enterprise | No SAML 2.0, no OIDC. Every Fortune 1000 procurement asks "do you support our IdP?" before they will buy. Recommend `python-social-auth` or Authlib + a hosted IdP-broker (WorkOS, Auth0, Stytch, Frontegg). The build-vs-buy answer is **buy** for the first 18 months. |
| **No SCIM 2.0** | P1 | Required by Okta/Azure AD enterprise tier. Same vendors above ship this. |
| **Coarse RBAC** | P1 | 4 fixed roles is too few. Need: per-module roles (forecasting-only planner, inventory-only planner), per-tenant entitlements (this tenant has the AI-Planner add-on, that one doesn't), per-record sharing (planner X can only see warehouse 4502). The data exists — `dim_location.state_id` and warehouses would map naturally to a row-security predicate — but no plumbing reads it. |
| **API-key auth is global, not per-tenant** | P0 with tenancy | The legacy `API_KEY` env var grants admin on the whole instance. Per-tenant API keys with rotation, scopes, and per-key audit are required for any customer that wants to integrate from their warehouse. |
| **Audit log retention is unbounded** | P1 | `fact_audit_log` has no partitioning or retention policy. SOC2 wants 1-year online + 7-year archive. |
| **Audit log writes can silently fail** | P2 | [common/auth.py:267-268](../../common/auth.py) — bare `except Exception: pass`. Audit failures must at minimum log; many compliance frameworks require audit-write failures to fail the request. |
| **No MFA / passwordless / WebAuthn** | P2 | Customers will ask. Vendor SSO covers this. |

### Effort estimate

- **Remove anonymous-admin + harden defaults + per-tenant API keys:** 2 weeks.
- **WorkOS or Frontegg SSO + SCIM integration:** 3-4 weeks.
- **Per-module RBAC + entitlements model:** 1.5 months.
- **Row-level security (per-warehouse / per-customer-group):** 1-2 months (touches every router that returns inventory or customer data).
- **Audit-log partitioning + retention + archive-to-S3:** 2-3 weeks.

**Total: 4-5 engineering-months.**

---

## 4. Data Isolation & Security

### Current state

- **In transit:** Nginx CSP headers in [frontend/nginx.conf](../../frontend/nginx.conf), but no TLS termination configured in compose. Production must front with ALB/CloudFront/Nginx with valid certs (table stakes).
- **At rest:** Postgres data volume is plain `pg_data:/var/lib/postgresql/data` in [docker-compose.yml](../../docker-compose.yml) — relies on disk-level encryption (RDS gp3 default). Adequate for SaaS V1.
- **Secrets:** `.env` file pattern. [.env.example](../../.env.example) documents `POSTGRES_PASSWORD`, `JWT_SECRET`, `API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`. No vault integration. ADR mentions "migrate to AWS Secrets Manager / HashiCorp Vault" as Phase 1 work ([docs/ENTERPRISE_ARCHITECTURE.md L1783](../ENTERPRISE_ARCHITECTURE.md)). Nothing built.
- **Key rotation:** None. JWT_SECRET rotation forces every active session out; not a procedure today.
- **PII:** [sql/003_create_dim_customer.sql](../../sql/003_create_dim_customer.sql) and the customer demand fact table contain customer names, addresses, ship-to identifiers. No tagging, no masking, no field-level encryption. The AI-planner can and does send customer-level data to OpenAI / Anthropic — see [api/llm.py](../../api/llm.py) and [common/ai/ai_planner.py](../../common/ai/ai_planner.py). Many enterprise SC buyers will refuse this without DPA-level controls.
- **Data residency:** Single-region. No EU partition.

### What is needed

| Capability | Effort | Priority |
|---|---|---|
| TLS termination + cert mgmt (ACM + ALB) | S | P0 |
| Move secrets to AWS Secrets Manager / Vault, rotate quarterly | S-M | P0 |
| Postgres TDE (RDS-provided) + encrypted EBS for any non-RDS state | S | P0 |
| BYOK (customer-managed KMS keys for their tenant DB + S3 prefix) | M | P1 (enterprise tier) |
| PrivateLink / VPC-peered ingest (customer pushes to our VPC, never the public internet) | M | P1 |
| Field-level PII tagging + LLM redaction layer (PII never leaves our VPC unless customer opts in) | L | P0 if any healthcare-distribution prospect; P1 otherwise |
| EU data residency (separate AWS account, separate DB cluster, geo-routed) | L | P2, deferred until first EU prospect |
| Quarterly key rotation runbook | S | P1 |
| Pen test before first paying customer | M (third-party) | P0 |

### Effort estimate

**3-4 engineering-months** for the P0 items + a third-party pen test (~$25-40K, 4 weeks calendar). BYOK and EU residency add another **2-3 months** when triggered by a prospect.

---

## 5. Compliance

### What matters for SC-planning buyers

| Cert | Buyer expectation | Recommended action |
|---|---|---|
| **SOC 2 Type II** | Table stakes. Every Fortune 5000 procurement will ask for the report before contract signing. Average sale stalls 90+ days without it. | Begin Type I in month 1, Type II observation window starts immediately, full Type II report ~12 months after audit start. |
| **ISO 27001** | Required for most European enterprise customers; nice-to-have in the US. | Defer until ARR > $3M or first EU enterprise prospect. ISO 27001 work largely overlaps SOC 2. |
| **GDPR (Article 28 DPA)** | Required to even talk to an EU prospect. | DPA template + sub-processor list + data-processing record before first EU pilot. |
| **HIPAA** | Only relevant if you sell into healthcare distribution (McKesson, Cardinal). | Skip for V1. If pursued, requires BAAs with all sub-processors (OpenAI does not currently sign HIPAA BAAs on their consumer API — moves you to Azure OpenAI). |
| **FedRAMP** | Not relevant initially; only for federal/DoD. | Skip indefinitely. |
| **CCPA** | If you have any California consumer data — likely yes if your customers ship to consumers. | Covered by GDPR-style privacy policy. |

### Cost & timeline to SOC 2 Type II

- **Auditor:** $25-50K Type I + $35-65K Type II (Vanta/Drata/Secureframe-supported audit). Add $20-40K/year for the GRC tool itself (Vanta/Drata/Secureframe).
- **Internal effort:** 0.25 FTE security/IT for 12 months minimum. Founder will spend 2-4 weeks total in audit interviews.
- **Calendar:** Type I in 3-4 months from kickoff. Type II observation window 6 months minimum (recommended 9-12). Total **~12 months from start to a clean Type II report**.
- **Prerequisites:** SSO, MFA on all admin tools, secrets manager, encrypted backups, written incident-response plan, written change-mgmt process, written access-review cadence. Most of this is paperwork once the SSO + secrets-manager infrastructure exists.

### Recommended compliance roadmap

1. **Month 0:** Pick a GRC vendor (Vanta is the most opinionated — start there). Sign DPA template + privacy policy + sub-processor list. Adopt SSO and MFA on every internal tool.
2. **Month 1-3:** Type I audit. Use this as the forcing function for the secrets-manager / pen-test work in Section 4.
3. **Month 4-12:** Type II observation window. Customer-facing trust portal at `trust.<product>.com`.
4. **Month 12+:** ISO 27001 if pulled by a European deal. HIPAA if pulled by a healthcare deal.

---

## 6. Onboarding & Data Integration

### Current state

The pipeline is pure CSV. [scripts/etl/normalize_dataset_csv.py](../../scripts/etl/normalize_dataset_csv.py) and [scripts/etl/load_dataset_postgres.py](../../scripts/etl/load_dataset_postgres.py) are driven by [common/core/domain_specs.py](../../common/core/domain_specs.py). Inputs land in `data/input/`, normalize to `data/staged/`, load to Postgres. There is a domain registry pattern that makes adding new sources cheap as long as the source is a CSV.

There is real promise here that has not been built out: [docs/specs/08-integration/01-integration-architecture.md](../specs/08-integration/01-integration-architecture.md) describes four "vectors" — notifications, REST consumers, cloud connectors (Snowflake/BigQuery/S3/Databricks), ERP adapters (SAP/Oracle/NetSuite/Manhattan). The notifications vector is implemented ([common/services/notification_engine.py](../../common/services/notification_engine.py)). The connector + adapter vectors are documented but not built; spec calls them "Implemented" but the code is not there. There is a generic [api/routers/platform/integration.py](../../api/routers/platform/integration.py) and [integration_chain.py](../../api/routers/platform/integration_chain.py), and the runner in [common/services/integration_runner.py](../../common/services/integration_runner.py), but they orchestrate the existing CSV pipelines, not external sources.

### What enterprise needs

Each enterprise customer will want **at least one** of:

| Source | Build effort | Notes |
|---|---|---|
| **SAP S/4HANA via OData** | L | The hard one. IDoc parsing, OAuth2 + SAP-specific token quirks, change-tracking via CDS view triggers. Plan 2-3 months for a competent first version. Most customers will accept "we extract via cron + their basis team" for V1. |
| **SAP ECC (legacy)** | L | Still ~50% of installed base. RFC adapter (PyRFC) or via SAP PI/PO. |
| **Oracle Fusion Cloud** | M | REST/OAuth2 — easier than SAP. |
| **NetSuite** | M | SuiteTalk SOAP/REST — well-documented. Mid-market staple. |
| **Microsoft Dynamics 365** | M | OData. |
| **Snowflake / Databricks / BigQuery** | S | One per warehouse via the python connector + JDBC. The "modern data stack" customers — fastest to onboard. |
| **S3 / GCS / Azure Blob (Parquet/CSV)** | S | The escape hatch. Every customer has a way to drop a file in a bucket. Build this first. |
| **EDI 850/852/856/810** | M | Drop-in for any retailer-side customer (J.B. Hunt, Walmart, target retailers). Use a third-party (SPS Commerce, TrueCommerce) translation layer initially. |
| **REST/GraphQL pull from buyers' OMS/WMS/TMS** | M | Per-customer; mostly a pattern (token auth + paginated GET + idempotent UPSERT) once one is built. |
| **Webhook receiver for buyer-pushed events** | S | Already partly there in [api/routers/platform/webhooks.py](../../api/routers/platform/webhooks.py) (outbound). Inbound receiver with HMAC verification ~1 week. |
| **Per-customer schema mapping UI** | L | This is the under-estimated one. Every customer's "item master" looks different. You need a planner-friendly UI to map *their* columns to *your* `dim_item` schema, with a preview, validation, dry-run. RudderStack and Hightouch sell standalone tools for this; build the simplest version with a JSON mapping + visual diff. |

### Effort estimate

- **S3/GCS/Azure Blob ingest + 1 cloud DW connector (Snowflake) + per-customer schema mapping UI:** 2-3 months. Unlocks ~60% of the "modern data stack" prospects.
- **SAP S/4HANA OData adapter:** 2-3 months once the first SAP customer signs.
- **NetSuite + Dynamics:** 1-2 months each, once the first prospect appears.
- **EDI:** Outsource to SPS Commerce / TrueCommerce in Y1; build native in Y2 if volume justifies.
- **Inbound webhook receiver + per-tenant API key scopes for ingest:** 3-4 weeks.

**Total to "first three enterprise customer integration patterns work": 4-6 months.**

---

## 7. Observability, SLAs, and Operations

### Current state

- Structured request logging in [api/main.py](../../api/main.py) middleware (request-id, duration ms, status code). Good baseline.
- `log_min_duration_statement=5000` set in [docker-compose.yml](../../docker-compose.yml). Good.
- pg_stat_statements is on ([sql/170_enable_pg_stat_statements.sql](../../sql/170_enable_pg_stat_statements.sql)).
- `perf_profiling` table ([sql/094_create_perf_profiling.sql](../../sql/094_create_perf_profiling.sql)) and `query_performance` ([sql/064_create_query_performance.sql](../../sql/064_create_query_performance.sql)) capture script-level timings.
- `/health` endpoint exists but is not visible from `api/main.py` (likely in a router).
- No metrics endpoint, no Prometheus, no Grafana, no APM (Datadog/New Relic/Honeycomb).
- No error tracking (Sentry).
- No customer-facing status page.
- No on-call rotation, no PagerDuty/Opsgenie integration.
- Backup/restore: nothing in compose. RDS auto-backups assumed in production but not documented.
- DR: none. Single-region.
- Deploys: docker-compose. No CI/CD. ENTERPRISE_ARCHITECTURE.md flags this in Phase 1 ([L1782](../ENTERPRISE_ARCHITECTURE.md)).

### What enterprise SLA requires

For a 99.9% SLA (8.76 hours downtime/year):

| Component | Today | Required |
|---|---|---|
| Uptime monitoring | None | Pingdom / BetterUptime / Datadog Synthetics — synthetic checks every 60s on `/health` + 3 critical endpoints |
| Error tracking | None | Sentry on backend + frontend |
| Metrics | pg_stat_statements only | Prometheus + Grafana, or Datadog. Track p50/p95/p99 latency per endpoint, pool saturation, MV refresh duration, job queue depth, cache hit rate (Redis) |
| Distributed tracing | None | OpenTelemetry → Honeycomb or Datadog APM. Critical for the AI-planner (multi-tool agent) and the multi-stage backtest pipeline |
| Log aggregation | stdout to docker | Ship to ELK/Loki/Datadog Logs. Retention 30d hot + 1y archive |
| Status page | None | statuspage.io or Atlassian Statuspage at `status.<product>.com` |
| On-call | None | PagerDuty or Opsgenie, 1 primary + 1 secondary, 24×7 once first paying customer |
| Backups | Compose volume | RDS automated backups (35-day point-in-time restore) + weekly snapshot to cross-region S3 |
| **RPO** | undefined | 5-minute RPO from RDS PITR + cross-region snapshot for DR |
| **RTO** | undefined | 4-hour RTO; documented runbook to bring up a fresh RDS from snapshot in DR region |
| Chaos / DR drill | None | Quarterly DR exercise once paying customers > 0 |

### Effort estimate

- **Sentry + Datadog APM + status page + uptime checks + structured logs to Loki:** 3-4 weeks.
- **Prometheus/Grafana dashboards for the SC-specific metrics (MV refresh lag, forecast pipeline duration, AI agent token spend):** 2-3 weeks.
- **CI/CD pipeline (GitHub Actions: lint → typecheck → test → build → deploy to staging → canary):** 3-4 weeks.
- **Backup/DR runbook + first DR drill:** 2 weeks.
- **On-call rotation setup:** ongoing operational cost — needs a second engineer minimum to be sustainable.

**Total: 2-3 engineering-months.**

---

## 8. Billing & Subscription

### Current state

**Nothing.** No `stripe` import anywhere in the codebase. No `subscription` / `billing` / `invoice` / `metering` references in [api/](../../api/), [common/](../../common/), or [sql/](../../sql/).

### What is needed

For a SaaS supply-chain product the natural metering dimensions are:

- Per-tenant base subscription (tier — Standard / Pro / Enterprise).
- Add-on modules (AI Planner, S&OP, Cluster Tuning Studio, FVA — each a packageable line item).
- **SKU count** (most natural unit; aligns to compute cost — clustering, backtest, MV refresh all scale with SKU count).
- DFU count (variant; closer to forecast-line cost).
- **Active planner seats** (RBAC `planner` + `manager` roles in [common/auth.py](../../common/auth.py)).
- API call volume (for customers integrating into their own apps).
- Optional overages on AI-planner LLM token spend (this is real cost — see the $10M financial cap circuit-breaker in [common/ai/ai_planner.py](../../common/ai/ai_planner.py); LLM cost is genuinely variable per tenant).

| Component | Recommended vendor | Build effort |
|---|---|---|
| Subscription billing engine | Stripe Billing or Maxio (formerly Chargify) — Maxio is purpose-built for B2B SaaS metering | 3-4 weeks integration |
| Usage metering pipeline | Roll your own or use Orb / Lago / Metronome | 3-4 weeks (the "send usage events to billing" plumbing) |
| Tax | Stripe Tax (US, EU VAT) | 1 week |
| Invoicing | Stripe Invoicing or Maxio | included |
| Dunning + retry + write-off | Stripe Smart Retries | included |
| Customer billing portal | Stripe Customer Portal | 1 week |
| Internal CRM ↔ billing sync | HubSpot ↔ Stripe via Zapier or Tray | 1 week |
| Accounting handoff | Stripe → QuickBooks/NetSuite | 1 week |

### Effort estimate

**6-8 engineering-weeks** for a full Stripe Billing + metering integration that supports per-SKU + per-seat + per-AI-token pricing. Recommend launching with **simple per-seat + per-SKU tiered pricing** for the first 3 customers (manually invoiced, no Stripe needed), then adding Stripe Billing once you have 4-5 paying customers and a real pricing model.

---

## 9. Admin / Customer-Success Tooling

### Current state

There is an [api/routers/platform/admin.py](../../api/routers/platform/admin.py) that handles LLM client reset and tuning-stale invalidation — internal ops endpoints, not a customer-success console. There is no UI for any of the items below.

### What is needed (an internal-only "Backstage" UI)

| Capability | Detail |
|---|---|
| **Tenant provisioning** | Create tenant → provision DB → seed schema → run migrations → create admin user → email invite. Today this is a wiki page at best. |
| **Tenant overview** | One row per tenant: created-at, plan, MAU, SKU count, last-login, last-data-refresh, MRR, health score, NPS. |
| **Impersonation (with audit)** | "Login as <tenant>" — every action by the impersonator written to that tenant's `fact_audit_log` with a flag `impersonator_email`. Required for support and absolutely required for SOC 2 (impersonation must be logged). |
| **Per-tenant model performance** | Surface champion accuracy, FVA, forecast-pipeline runtime per tenant so CS can flag a tenant whose accuracy is degrading before they churn. The data is there in [sql/068_create_fva_tracking.sql](../../sql/068_create_fva_tracking.sql) and the champion-experiments tables; the dashboard is not. |
| **Entitlements management** | Toggle modules (AI Planner on/off, S&OP on/off) per tenant without a deploy. |
| **Job & queue inspection** | View pg-queue depth, APScheduler state, MV refresh status across tenants. |
| **Usage dashboard** | API calls / SKU count / seat count per tenant per month. Drives invoicing if you do not use a metering platform. |
| **"Send broadcast"** | Push announcements to all tenants. Trivial, and CS will ask for it on day 1. |

### Effort estimate

- **MVP Backstage (tenant list + impersonation + entitlements + usage):** 4-6 weeks.
- **Full version with per-tenant model performance + tenant provisioning automation:** 2-3 months.

This is the unglamorous work that makes the difference between "we have a product" and "we have a business."

---

## 10. Performance & Scale Per-Tenant

### Current state

| Component | Current sizing | Per-CLAUDE.md |
|---|---|---|
| Postgres | docker-compose `shared_buffers=4GB`, `work_mem=128MB`, `max_connections=100` | Single-tenant single-node. 198M rows in `fact_inventory_snapshot` (now weekly-partitioned, see [sql/184](../../sql/184_partition_inventory_snapshot_weekly_cutover.sql)). |
| Redis | 256MB max, allkeys-lru | Shared cache across gunicorn workers. |
| API | gunicorn + 4 uvicorn workers, anyio threadpool 100 | [Dockerfile](../../Dockerfile) |
| DB pool | min=5, max=50, statement_timeout=30s | [api/pool.py](../../api/pool.py) |
| Read replica | Opt-in via `READ_REPLICA_URL`; 7 customer-analytics endpoints opted in | [api/core.py](../../api/core.py) `get_async_read_only_conn` |
| Long jobs | pg-queue worker + APScheduler | [common/services/pg_queue.py](../../common/services/pg_queue.py) |

### Where it breaks

- **At ~5-10 tenants on shared Postgres**, the connection pool math fails. 4 gunicorn workers × max_size=50 = 200 connections per tenant pool. Multi-tenant on shared Postgres requires **PgBouncer** in transaction-pooling mode in front of every per-tenant DB. Today there is none.
- **At ~20+ tenants**, MV refresh windows collide. The Makefile assumes serialized refresh. Per-tenant scheduling needed.
- **At ~50+ tenants**, single-node RDS becomes the bottleneck for ML training. Move ML training off the API box entirely (separate task pool) — pg-queue worker is the right pattern but needs to be split off into its own deployable.
- **At any tenant** with > 1M SKUs, the backtest pipeline (per-SKU per-cluster) needs partition-aware sharding. Weekly partitions on `fact_inventory_snapshot` are in (sql/184), but `fact_external_forecast_monthly` and `fact_customer_demand_monthly` are not partitioned. Tenant-by-tenant sharding will help; horizontal sharding (Citus) is overkill for this stage.

### Architectural changes needed

| Change | When | Effort |
|---|---|---|
| PgBouncer in transaction pooling mode | Before tenant #2 | 1 week |
| Move long-running ML training to a dedicated worker pool (k8s Job, separate from API) | Before tenant #5 | 3-4 weeks |
| Per-tenant MV refresh scheduler with concurrency control | Before tenant #10 | 2-3 weeks |
| Partition `fact_external_forecast_monthly` and `fact_customer_demand_monthly` by month + tenant | Before tenant #15 | 3-4 weeks each |
| Move MLflow off the API box, point at S3 artifact root | Before tenant #5 | 1 week |
| Read replica enforced for all reporting endpoints (today only 7 of ~70 routers) | Continuous | 2-3 weeks |

---

## 11. Hosting Strategy

### Recommended sequence

1. **Phase 1 — Pure SaaS in your AWS (months 0-12).** Single-region (us-east-2), single-AZ-with-failover RDS, EKS or ECS Fargate for the API, ElastiCache for Redis, S3 for artifacts. Per-tenant DBs on shared cluster. Targets: design partners + first 5-10 paying customers.
2. **Phase 2 — Single-tenant hosted in your AWS (month 12-24).** For "enterprise paranoia" customers who want their own VPC. Same Terraform module per customer (a "tenant cell"). Higher price, longer sales cycle. Targets: 2-3 enterprise logos.
3. **Phase 3 — On-prem / private cloud only by exception (month 24+).** Helm chart + Terraform-for-airgapped variant. Only when a 7-figure deal requires it. The SAP install base will pull this eventually. Treat as a custom service engagement, not a self-serve install.

Resist the temptation to ship Phase 2 or Phase 3 first. Each one triples your ops surface area; you cannot afford that pre-PMF.

---

## 12. Code / Architecture Remediations

The good news is the codebase is clean by internal-tool standards. The architecture is consistent (CLAUDE.md is unusually disciplined), the SQL is parameterized everywhere, the auth layer exists, the audit log exists, the config pattern is uniform. The bad news is that "internal tool quality" surfaces in specific places that must be hardened before anything ships externally:

| Item | File | Why it matters |
|---|---|---|
| Anonymous-as-admin fallback | [common/auth.py:166-169](../../common/auth.py) | Hard-fail if `JWT_SECRET` is the dev default in any non-`development` environment. Add a `APP_ENV` env var checked at startup. |
| Seed admin password `admin123` | [sql/062_create_users_rbac.sql:51-52](../../sql/062_create_users_rbac.sql) | Either remove the seed entirely from production migrations, or randomize at first boot and email the new admin. |
| `audit_log` swallows all exceptions | [common/auth.py:267-268](../../common/auth.py) | At minimum, `logger.exception()`. SOC2-relevant. |
| `default admin@demandstudio.local` | [sql/062_create_users_rbac.sql:50](../../sql/062_create_users_rbac.sql) | Brand the product name out of the schema. The product is not "demand studio." |
| `frontend/nginx.conf` allow-list | [frontend/nginx.conf](../../frontend/nginx.conf) | The 60+ path-prefix list is a known maintenance hazard; CLAUDE.md flags it. Consider one prefix `/api/v1/...` plus `make audit-routers` enforcement. |
| CORS `allow_origins=["http://localhost:5173"]` only | [api/main.py:144-147](../../api/main.py) | Move CORS allowlist to env / per-tenant config; today every deploy is a code change. |
| No API versioning | All routers | Every router is mounted at its bare prefix. Adding `/v1` later breaks every client. Add `/v1/` now. |
| Filesystem artifact paths | [common/core/paths.py](../../common/core/paths.py), `data/` tree | Replace with object-store abstraction (`s3://...` / `gs://...`) and tenant-scoped prefix before tenancy lands. |
| Hardcoded planning date fallback | `common/core/planning_date.py` (per CLAUDE.md) | Verify per-tenant override possible (some customers run on a fiscal calendar). |
| Catch-all `/domains/{domain}` router | [api/routers/domains.py](../../api/routers/domains.py) | Convenient for an internal tool, dangerous for an external API. Each domain endpoint should be explicit so OpenAPI types are stable for customers consuming the API. |
| MLflow with `--serve-artifacts` no auth | [docker-compose.yml:48-66](../../docker-compose.yml) | MLflow has no auth in this config. Either firewall it off or use an SSO proxy in front. |
| `SMTP_PASSWORD` only in env | [config/platform/notification_config.yaml](../../config/platform/notification_config.yaml) | OK, but per-tenant SMTP credentials are not modeled — every tenant sends from your address today. |
| No request-body size limits | [api/main.py](../../api/main.py) middleware stack | Add a max-body-size middleware (10MB default) before any file upload endpoints exist. |
| `print()` audit allowlists in [scripts/ai_checks/](../../scripts/ai_checks/) | per CLAUDE.md memory | These pinned-existing-violation allowlists are fine for an internal tool. For commercial ship, every entry must be triaged. |
| 22,000+ LoC in `api/routers/forecasting/` and `api/routers/inventory/` (per `wc -l`) | Across both router subdirectories | Code is well-split per CLAUDE.md's 800-LoC rule, but the surface area is large. Each public-facing endpoint needs an OpenAPI description, an example, and a Pydantic response model. Today many do not. |
| Internal Jira links + SGWS branding | [docs/specs/PRD/PRD-CSCS-11725-brand-performance-chatbot.md](../specs/PRD/PRD-CSCS-11725-brand-performance-chatbot.md) | Scrub all internal references before any external doc share — including PRDs, ADRs, the README, the data fixture filenames (`itemdata.csv`, `dfu_lvl2_hist.txt` are SGWS-shaped). |

None of these are "rip-and-replace" rewrites. All are weeks-of-work hardening, not months. The architecture is sound.

---

## 13. Legal & IP

### #1 blocker: ownership

The platform was built as an internal tool at SGWS (Southern Glazer's). The SGWS Jira ticket references in [docs/specs/PRD/PRD-CSCS-11725-brand-performance-chatbot.md](../specs/PRD/PRD-CSCS-11725-brand-performance-chatbot.md) and the SGWS Confluence link in the same document are direct evidence. The code does not contain a `LICENSE` file at the repo root — none was found.

**Until this is in writing, no commercial activity is safe:**

- A clean **assignment, license-back, or carve-out agreement with SGWS**, executed by their corporate counsel and recorded with their IP custodian.
- Confirmation that the build was on personal time and personal equipment (if applicable) and that no SGWS proprietary data, models, or third-party licensed components were embedded.
- A **clean-room rebuild assessment** — if the agreement is unfavorable, identify which modules need to be rewritten from spec rather than carried over.

This is a 4-12 week legal exercise depending on SGWS's responsiveness. **Do not raise a priced round, do not sign a design-partner LOI, do not put up a marketing site, until this is closed.** Investors will diligence this on day one and a "we're working on it" answer is a deal-killer.

### Other legal must-do items

| Item | Effort | When |
|---|---|---|
| Standard MSA template (built off CSAA / IAPP templates) | 2-3 weeks with SaaS-experienced counsel | Before first design-partner conversation |
| DPA template (GDPR Art. 28) | 1-2 weeks | Before first EU prospect |
| Privacy policy + cookie policy | 1 week | Before first marketing-site launch |
| Terms of service | 1 week | Same |
| Sub-processor list (OpenAI, Anthropic, AWS, Stripe, Sentry, Datadog, etc.) | 2-3 days, ongoing maintenance | Same |
| OSS license audit (run `pip-licenses` + `license-checker` for npm) | 1 week | Before first paying customer |
| EULA for any future on-prem variant | 2-3 weeks | Phase 3 only |
| Trademark search + filing on the product name | $1-3K + 6-9 months wait | Before public launch |
| Insurance: Tech E&O + Cyber + General Liability | 2-3 weeks | Before first paying customer signs |

---

## 14. Prioritized Punch List

Effort key: **S** = ≤ 2 weeks, **M** = 2-6 weeks, **L** = 6-12 weeks, **XL** = > 3 months.
Priority key: **P0** = blocker for stated milestone, **P1** = required, **P2** = nice-to-have.
Milestones: **DP** = pre-design-partner, **PC** = pre-paying-customer, **EE** = pre-enterprise-deal.

| # | Item | Effort | Pri | Milestone |
|---|---|---|---|---|
| 1 | Resolve SGWS IP ownership in writing | M (legal) | P0 | DP |
| 2 | Remove anonymous-as-admin fallback in [common/auth.py](../../common/auth.py); fail-closed defaults; remove seed admin password from prod migrations | S | P0 | DP |
| 3 | Scrub SGWS-internal references from docs, PRDs, fixture filenames, schema branding | S | P0 | DP |
| 4 | TLS termination, secrets manager, encrypted backups, secrets rotation runbook | M | P0 | DP |
| 5 | Multi-tenancy via per-tenant Postgres + per-request pool routing + tenant-scoped artifact paths + per-tenant Redis | L | P0 | PC |
| 6 | Tenant provisioning script + minimal Backstage UI (list / impersonate / entitlements) | M | P0 | PC |
| 7 | SSO via WorkOS or Frontegg (SAML + OIDC) + SCIM | M | P0 | EE |
| 8 | Per-tenant API keys with scopes + per-tenant rate limits | M | P0 | PC |
| 9 | Per-module RBAC + entitlements model | L | P1 | EE |
| 10 | Audit-log partitioning + retention + archive to S3; fix swallowed exceptions in `log_audit` | S | P1 | DP |
| 11 | Sentry + Datadog APM + status page + uptime checks + structured log shipping | M | P0 | PC |
| 12 | CI/CD pipeline (lint → typecheck → test → build → deploy → canary) | M | P0 | DP |
| 13 | Backup/PITR + cross-region snapshot + documented DR runbook + first DR drill | M | P1 | PC |
| 14 | API versioning (`/v1/` prefix on every router) + OpenAPI response models on every public endpoint | M | P1 | PC |
| 15 | S3/GCS/Azure Blob inbound + Snowflake connector + per-customer schema-mapping UI | L | P0 | PC |
| 16 | SAP S/4HANA OData adapter | L | P0 | EE (after first SAP prospect) |
| 17 | NetSuite + Dynamics + Oracle Fusion adapters | L each | P1 | EE (per prospect) |
| 18 | Inbound webhook receiver with HMAC verification | S | P1 | PC |
| 19 | EDI 850/852/856/810 via SPS Commerce passthrough | M | P2 | EE |
| 20 | SOC 2 Type I (engagement + tooling + initial audit) | M (calendar 3-4mo) | P0 | PC |
| 21 | SOC 2 Type II observation window completes | XL (12 mo) | P0 | EE |
| 22 | Privacy policy + ToS + MSA + DPA + sub-processor list | M (legal) | P0 | DP |
| 23 | Pen test by reputable vendor (Bishop Fox / Latacora / Doyensec) | M | P0 | PC |
| 24 | OSS license audit | S | P1 | PC |
| 25 | Stripe Billing + metering pipeline (per-SKU + per-seat + per-AI-token tiers) | M | P1 | PC (manual invoice for first 3) |
| 26 | Tax (Stripe Tax) + dunning + customer billing portal | S | P1 | PC |
| 27 | PgBouncer in transaction-pooling mode in front of per-tenant DBs | S | P0 | PC |
| 28 | Move ML training to dedicated worker pool (k8s Job), separate from API; MLflow on S3 | M | P1 | PC |
| 29 | Read-replica routing extended to all reporting endpoints (today 7/70) | M | P2 | EE |
| 30 | Per-tenant MV refresh scheduler with concurrency control | M | P1 | EE |
| 31 | BYOK (customer-managed KMS keys) | M | P2 | EE |
| 32 | PrivateLink / VPC-peered ingest path | M | P2 | EE |
| 33 | EU data residency (separate region/cluster) | L | P2 | EE (when first EU prospect appears) |
| 34 | PII tagging + LLM redaction layer (PII never leaves tenant VPC unless opted in) | L | P0 if healthcare prospect; P1 otherwise | EE |
| 35 | Trademark filing on product name | S (cost) | P1 | DP |
| 36 | Tech E&O + Cyber + GL insurance | S | P0 | PC |
| 37 | Internal CRM ↔ billing ↔ accounting handoff | S | P2 | post-PC |
| 38 | Brand the product (today schema/seed/email use "demand studio" / "demand_mvp") | S | P0 | DP |

### Critical-path summary

- **Pre-design-partner (DP) gate (months 0-3):** items 1-4, 10, 12, 22, 35, 38. Mostly legal + hardening + brand.
- **Pre-paying-customer (PC) gate (months 3-9):** items 5, 6, 8, 11, 13-15, 18, 20, 23-27, 36. Multi-tenancy + integrations + observability + SOC2 Type I + commercial paperwork.
- **Pre-enterprise-deal (EE) gate (months 9-15+):** items 7, 9, 16-17, 21, 30-34. SSO, ERP adapters, SOC2 Type II, BYOK, tier-2 integrations.

Each gate represents a real change in commercial posture: DP unlocks unpaid pilots, PC unlocks first paid customer, EE unlocks Fortune-500 procurement.

---

*End of audit.*
