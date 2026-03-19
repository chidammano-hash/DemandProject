# Reporting & Scheduled Exports

> Generates templated reports (PDF, Excel, CSV) on a schedule and delivers them via email or download link, so stakeholders get regular planning summaries without logging in.

| | |
|---|---|
| **Status** | ✅ Implemented |
| **UI Tab** | N/A (API-driven) |
| **Key Files** | `api/routers/reports.py`, `config/reporting_config.yaml` |

---

## Problem

Executives and cross-functional stakeholders need regular planning summaries but do not log into the platform daily. Without automated reporting, planners manually export data, build slides, and email them out -- a repetitive task that delays information sharing and introduces formatting inconsistencies.

## Solution

A report generation engine with built-in templates for common planning reports. Planners configure what to generate (template + filters), when to generate it (cron schedule), and where to deliver it (download link or email). The engine pulls data via configured queries, renders the template, and stores the output file. Scheduled reports run automatically via the job scheduler.

## How It Works

1. Admin seeds built-in report templates (accuracy summary, inventory health, exception digest, etc.)
2. Planner selects a template and sets parameters (date range, item/location filters)
3. `POST /reports/generate` queues the generation job and returns 202 immediately
4. The job pulls data, renders the template into the chosen format (PDF, Excel, or CSV), and writes the output file
5. `fact_report_generation` is updated with status `completed` and the output file path
6. Planner downloads the report via `/reports/deliveries` or receives it by email
7. For recurring reports, a schedule (cron expression) triggers automatic generation and delivery

## Data Model

### `dim_report_template`

| Column | Type | Description |
|---|---|---|
| `template_id` | `SERIAL PK` | Auto-increment ID |
| `template_name` | `TEXT UNIQUE` | Human-readable template name |
| `description` | `TEXT` | What the report covers |
| `report_type` | `TEXT` | accuracy_summary, inventory_health, exception_digest, sop_status, custom |
| `output_format` | `TEXT` | pdf, xlsx, csv (default: pdf) |
| `query_config` | `JSONB` | SQL queries or API endpoints to pull data |
| `layout_config` | `JSONB` | Sections, charts, tables, page orientation |
| `is_active` | `BOOLEAN` | Whether the template is available |

### `fact_report_generation`

| Column | Type | Description |
|---|---|---|
| `generation_id` | `BIGSERIAL PK` | Auto-increment ID |
| `template_id` | `INTEGER FK` | Which template was used |
| `parameters` | `JSONB` | Filters applied (date range, items, locations) |
| `status` | `TEXT` | pending, generating, completed, failed |
| `output_path` | `TEXT` | File path to the generated report |
| `output_size_bytes` | `BIGINT` | File size |
| `generated_by` | `TEXT` | User or schedule that triggered generation |
| `started_at` | `TIMESTAMPTZ` | When generation began |
| `completed_at` | `TIMESTAMPTZ` | When generation finished |
| `error_message` | `TEXT` | Error details if failed |

### `fact_report_schedule`

| Column | Type | Description |
|---|---|---|
| `schedule_id` | `SERIAL PK` | Auto-increment ID |
| `template_id` | `INTEGER FK` | Which template to generate |
| `schedule_name` | `TEXT` | Human-readable schedule name |
| `cron_expression` | `TEXT` | e.g., "0 8 * * MON" (weekly Monday 8 AM) |
| `parameters` | `JSONB` | Default filters for scheduled runs |
| `recipients` | `TEXT[]` | Email addresses or user IDs |
| `delivery_channel` | `TEXT` | download or email (default: download) |
| `is_active` | `BOOLEAN` | Whether the schedule is running |
| `last_run_at` | `TIMESTAMPTZ` | Most recent execution |
| `next_run_at` | `TIMESTAMPTZ` | Next scheduled execution |

## Built-in Report Templates

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
| GET | `/reports/templates` | List available report templates |
| POST | `/reports/generate` | Queue report generation (returns 202) |
| GET | `/reports/schedules` | List active report schedules |
| POST | `/reports/schedules` | Create a recurring schedule |
| DELETE | `/reports/schedules/{id}` | Delete a schedule |
| GET | `/reports/deliveries` | List generated reports with download links |

## Configuration

`config/reporting_config.yaml` controls output directory, retention policy, and delivery settings. Reports are stored in `data/reports/` (auto-created, gitignored). Auto-purge removes reports older than 90 days by default.

## Dependencies

- `openpyxl>=3.1` for Excel generation
- `weasyprint>=60` or `reportlab>=4.0` for PDF generation (optional, falls back to HTML)
- APScheduler (`common/job_registry.py`) executes scheduled generation
- [Notifications](./04-notifications.md) delivers email reports when configured

## See Also

- [Notifications](./04-notifications.md) -- email delivery channel for reports
- [FVA](./07-fva.md) -- accuracy data used by the Accuracy Summary template
- [Integration Architecture](./01-integration-architecture.md) -- reporting as an output integration
