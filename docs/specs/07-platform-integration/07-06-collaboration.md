# 07-06 Collaboration & Annotations

## Overview

In-app collaboration layer enabling planners to annotate forecasts, DFUs, and exceptions with notes, tag colleagues via @mentions, share filtered views, and resolve discussion threads. All annotations are persisted and linked to their source entity.

## Components

| Component | Path | Purpose |
|---|---|---|
| Router | `api/routers/collaboration.py` | 8 REST endpoints for annotations, mentions, shared views |
| DDL | `sql/065_create_collaboration.sql` | `fact_annotation`, `fact_annotation_mention`, `dim_shared_view` |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/collaboration/annotations` | List annotations by entity (type + id), paginated |
| POST | `/collaboration/annotations` | Create annotation with optional @mentions |
| PUT | `/collaboration/annotations/{id}` | Edit annotation text |
| DELETE | `/collaboration/annotations/{id}` | Soft-delete annotation |
| PUT | `/collaboration/annotations/{id}/resolve` | Mark annotation thread as resolved |
| GET | `/collaboration/mentions` | List unresolved mentions for a user |
| POST | `/collaboration/shared-views` | Save a shared view (filters + tab state as URL) |
| GET | `/collaboration/shared-views/{token}` | Retrieve shared view by token |

## Database Schema

### `fact_annotation`
- `annotation_id BIGSERIAL PRIMARY KEY`
- `entity_type TEXT NOT NULL` (dfu, forecast, exception, insight, scenario)
- `entity_id TEXT NOT NULL` (composite key or ID of the referenced entity)
- `author_id INTEGER` (user_id, nullable for unauthenticated mode)
- `author_name TEXT NOT NULL DEFAULT 'Planner'`
- `body TEXT NOT NULL`
- `is_resolved BOOLEAN DEFAULT false`
- `is_deleted BOOLEAN DEFAULT false`
- `created_at TIMESTAMPTZ DEFAULT NOW()`, `updated_at TIMESTAMPTZ`
- Indexes: `(entity_type, entity_id, created_at DESC)`, `(author_id)`

### `fact_annotation_mention`
- `mention_id BIGSERIAL PRIMARY KEY`
- `annotation_id BIGINT REFERENCES fact_annotation(annotation_id)`
- `mentioned_user_id INTEGER`
- `mentioned_name TEXT NOT NULL`
- `is_read BOOLEAN DEFAULT false`
- `created_at TIMESTAMPTZ DEFAULT NOW()`
- Index: `(mentioned_user_id, is_read) WHERE is_read = false`

### `dim_shared_view`
- `view_id SERIAL PRIMARY KEY`
- `token TEXT UNIQUE NOT NULL` (UUID, used in shareable URL)
- `created_by INTEGER` (user_id)
- `label TEXT`
- `tab TEXT NOT NULL`, `filters JSONB NOT NULL`
- `url_state TEXT` (full URL query string)
- `expires_at TIMESTAMPTZ` (optional, default 30 days)
- `created_at TIMESTAMPTZ DEFAULT NOW()`
- Index: `(token)`

## Annotation Workflow

1. Planner views a DFU, exception, or forecast in the UI
2. Clicks "Add Note" to create an annotation, optionally tagging colleagues with `@name`
3. @mentioned users see unread mention count in sidebar and can navigate to the annotation
4. Annotations can be resolved (marking the discussion as complete) or edited
5. Shared views allow planners to snapshot their current filter state and share via URL token

## Frontend Integration

- Annotation panel component (collapsible) embedded in DFU Analysis, Storyboard, and AI Planner tabs
- `@` autocomplete from user list (or static planner names in unauthenticated mode)
- Shared view: "Share" button in GlobalFilterBar generates a token URL via `POST /collaboration/shared-views`
- Incoming shared view: `?shared=<token>` URL param triggers filter restoration on load

## Conventions

- All endpoints use `get_conn()` directly (consistent with `inv_planning_*.py` pattern)
- Soft-delete for annotations (set `is_deleted = true`, excluded from queries)
- No auth dependency in v1 (uses `author_name` field); integrates with RBAC when 07-03 is implemented
