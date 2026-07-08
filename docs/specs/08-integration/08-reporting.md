# Reporting & Scheduled Exports

> Templates and delivery schedules for planning reports are managed via API. Actual generation and rendering of report output (PDF, Excel, CSV) is not yet built, so this does not deliver reports end-to-end today.

| | |
|---|---|
| **Status** | Partial - template listing and schedule CRUD (create/list/delete) read and write real rows; delivery-history reads real rows but nothing currently populates `fact_report_delivery`; `POST /reports/generate` is a stub that returns a `queued` status without rendering any output |
| **UI Tab** | N/A (API-driven) |
| **Key Files** | `api/routers/platform/reports.py` |

---

## Problem

Executives and cross-functional stakeholders need regular planning summaries but do not log into the platform daily. Without automated reporting, planners manually export data, build slides, and email them out -- a repetitive task that delays information sharing and introduces formatting inconsistencies.

## Solution

A report-template and schedule registry lets planners define what to generate (`dim_report_template`), when (`fact_report_schedule`, cron-based), and track delivery outcomes (`fact_report_delivery`). Today only the registry half is implemented: template listing and schedule create/list/delete read and write real database rows. `POST /reports/generate` is a stub - it validates that the template exists and returns a `queued` status message, but no rendering engine exists in the codebase (no PDF/Excel library is even a project dependency) and no code path ever inserts a row into `fact_report_delivery`, so the delivery-history endpoint returns an empty list in practice.

## How It Works

1. `dim_report_template` holds the report catalog, but nothing in the shipped code seeds it - the templates listed below are illustrative `report_type` values, not rows that exist out of the box
2. `GET /reports/templates` lists whatever templates exist in the table
3. `POST /reports/generate` looks up the template and returns `{"status": "queued", ...}` - the handler's own comment reads "In a full implementation, this would invoke the report engine"; no query execution, rendering, or file output happens
4. `POST /reports/schedules` and `DELETE /reports/schedules/{schedule_id}` create and remove rows in `fact_report_schedule` (real read/write, manager role required)
5. `GET /reports/deliveries` reads `fact_report_delivery`, but no code path in the repo ever inserts into that table, so it returns an empty list today

## Data Model

### `dim_report_template`

| Column | Type | Description |
|---|---|---|
| `template_id` | `SERIAL PK` | Auto-increment ID |
| `name` | `TEXT UNIQUE` | Human-readable template name |
| `report_type` | `TEXT NOT NULL` | e.g., accuracy_summary, inventory_health, exception_digest |
| `query_config` | `JSONB` | SQL queries or API endpoints to pull data |
| `layout` | `JSONB` | Sections, charts, tables, page orientation |
| `created_by` | `UUID` | User who created the template (nullable) |
| `is_system` | `BOOLEAN` | Whether it is a built-in vs. user-created template (default false) |
| `created_at` | `TIMESTAMPTZ` | Row creation time |

### `fact_report_schedule`

| Column | Type | Description |
|---|---|---|
| `schedule_id` | `SERIAL PK` | Auto-increment ID |
| `template_id` | `INTEGER FK` | Which template to generate |
| `recipients` | `JSONB` | Email addresses or user IDs |
| `cron` | `TEXT` | e.g., "0 8 * * 1" (weekly Monday 8 AM) |
| `format` | `TEXT` | pdf, csv, html (default: pdf) |
| `enabled` | `BOOLEAN` | Whether the schedule is running (default true) |
| `last_run_at` | `TIMESTAMPTZ` | Most recent execution |
| `next_run_at` | `TIMESTAMPTZ` | Next scheduled execution |
| `created_at` | `TIMESTAMPTZ` | Row creation time |

### `fact_report_delivery`

| Column | Type | Description |
|---|---|---|
| `delivery_id` | `BIGSERIAL PK` | Auto-increment ID |
| `schedule_id` | `INTEGER FK` | Which schedule produced this delivery |
| `status` | `TEXT` | Delivery status (default: pending; no producer currently writes any other value) |
| `file_path` | `TEXT` | File path to the generated report |
| `error` | `TEXT` | Error details if failed |
| `created_at` | `TIMESTAMPTZ` | Row creation time |
| `delivered_at` | `TIMESTAMPTZ` | When the report was delivered |

## Built-in Report Templates (illustrative, not seeded)

These are the `report_type` values the schema anticipates. No seed script or endpoint currently inserts them into `dim_report_template` - the table is empty until something writes to it directly.

| Template | Contents |
|---|---|
| Accuracy Summary | KPIs (WAPE, bias, accuracy%), model comparison, trend chart |
| Inventory Health | DOS distribution, stockout risk, excess inventory, health scores |
| Exception Digest | Open exceptions by severity, top-critical items, resolution rate |
| S&OP Status | Current cycle stage, gap cards, approved plan summary |
| AI Insights Briefing | Open insights, financial risk, recommendations digest |

## API

| Method | Path | Description |
|---|---|---|
| GET | `/reports/templates` | List available report templates (real read; table is empty by default) |
| POST | `/reports/generate` | Stub - validates the template and returns `{"status": "queued", ...}`; does not render or store an output file |
| GET | `/reports/schedules` | List active report schedules (real read) |
| POST | `/reports/schedules` | Create a recurring schedule (real write, manager role required, returns 201) |
| DELETE | `/reports/schedules/{schedule_id}` | Delete a schedule (real write, manager role required) |
| GET | `/reports/deliveries` | List report delivery history (real read; empty today - nothing writes to `fact_report_delivery`) |

## Configuration

> `config/reporting_config.yaml` has been deleted (dead config, no consumers). Report template and schedule metadata live in `dim_report_template` / `fact_report_schedule`. There is no output-file storage path, retention policy, or auto-purge implemented - those depend on the report-generation engine, which does not exist yet.

## Dependencies

- No PDF/Excel/CSV rendering dependency is installed today (`openpyxl`, `weasyprint`, and `reportlab` are not in `pyproject.toml`) - report generation has no rendering engine behind it
- Schedule rows (`fact_report_schedule`) are stored, but no scheduler job currently reads them to trigger generation
- [Notifications](./04-notifications.md) would be the natural delivery channel once generation and delivery are implemented

## See Also

- [Notifications](./04-notifications.md) - prospective email delivery channel for reports
- [FVA](./07-fva.md) - accuracy data the Accuracy Summary template would draw from, once generation exists
- [Integration Architecture](./01-integration-architecture.md) -- reporting as an output integration
