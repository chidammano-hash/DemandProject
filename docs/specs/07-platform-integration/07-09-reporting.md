# 07-09 Reporting & Scheduled Exports

## Overview

Report generation engine supporting templated PDF/Excel/CSV exports with scheduled delivery. Planners define report templates, schedule recurring generation, and receive reports via email or download link.

## Components

| Component | Path | Purpose |
|---|---|---|
| Router | `api/routers/reports.py` | 5 REST endpoints for report lifecycle management |
| DDL | `sql/068_create_reports.sql` | `dim_report_template`, `fact_report_generation`, `fact_report_schedule` |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/reports/templates` | List available report templates |
| POST | `/reports/generate` | Generate a report from template + parameters (returns 202) |
| GET | `/reports/schedules` | List active report schedules |
| POST | `/reports/schedules` | Create a recurring report schedule |
| DELETE | `/reports/schedules/{id}` | Delete a report schedule |
| GET | `/reports/deliveries` | List generated reports with download links |

## Database Schema

### `dim_report_template`
- `template_id SERIAL PRIMARY KEY`
- `template_name TEXT UNIQUE NOT NULL`
- `description TEXT`
- `report_type TEXT NOT NULL` (accuracy_summary, inventory_health, exception_digest, sop_status, custom)
- `output_format TEXT DEFAULT 'pdf'` (pdf, xlsx, csv)
- `query_config JSONB NOT NULL` (SQL queries or API endpoints to pull data)
- `layout_config JSONB` (sections, charts, tables, page orientation)
- `is_active BOOLEAN DEFAULT true`
- `created_at TIMESTAMPTZ DEFAULT NOW()`

### `fact_report_generation`
- `generation_id BIGSERIAL PRIMARY KEY`
- `template_id INTEGER REFERENCES dim_report_template(template_id)`
- `parameters JSONB` (filters: date range, items, locations)
- `status TEXT DEFAULT 'pending'` (pending, generating, completed, failed)
- `output_path TEXT` (file path to generated report)
- `output_size_bytes BIGINT`
- `generated_by TEXT` (user or schedule)
- `started_at TIMESTAMPTZ`, `completed_at TIMESTAMPTZ`
- `error_message TEXT`
- `created_at TIMESTAMPTZ DEFAULT NOW()`
- Indexes: `(template_id, created_at DESC)`, `(status)`

### `fact_report_schedule`
- `schedule_id SERIAL PRIMARY KEY`
- `template_id INTEGER REFERENCES dim_report_template(template_id)`
- `schedule_name TEXT NOT NULL`
- `cron_expression TEXT NOT NULL` (e.g., "0 8 * * MON" for weekly Monday 8AM)
- `parameters JSONB` (default filters for scheduled runs)
- `recipients TEXT[]` (email addresses or user IDs)
- `delivery_channel TEXT DEFAULT 'download'` (download, email)
- `is_active BOOLEAN DEFAULT true`
- `last_run_at TIMESTAMPTZ`, `next_run_at TIMESTAMPTZ`
- `created_at TIMESTAMPTZ DEFAULT NOW()`

## Built-in Report Templates

1. **Accuracy Summary** — KPIs (WAPE, bias, accuracy%), model comparison, trend chart
2. **Inventory Health** — DOS distribution, stockout risk, excess inventory, health scores
3. **Exception Digest** — Open exceptions by severity, top-critical items, resolution rate
4. **S&OP Status** — Current cycle stage, gap cards, approved plan summary
5. **AI Insights Briefing** — Open insights, financial risk, recommendations digest

## Report Generation Flow

1. User selects template and sets parameters (date range, filters)
2. `POST /reports/generate` queues generation job (returns 202)
3. Job pulls data via configured queries, renders template, writes output file
4. `fact_report_generation` updated with `completed` status and `output_path`
5. User downloads via `/reports/deliveries` or receives email notification

## Integration Points

- APScheduler (`common/job_registry.py`) executes scheduled report generation
- Notification engine (07-05) delivers email reports when configured
- Output directory: `data/reports/` (auto-created, gitignored)
- Report retention: configurable auto-purge (default 90 days)

## Make Targets

```bash
make reports-schema     # Apply DDL (one-time)
make reports-seed       # Seed built-in report templates
make reports-all        # reports-schema + reports-seed
```

## Dependencies

- `openpyxl>=3.1` for Excel generation
- `weasyprint>=60` or `reportlab>=4.0` for PDF generation (optional, falls back to HTML)
- `papaparse` (frontend, already in deps) for CSV export
