# Collaboration & Annotations

> Lets planners annotate forecasts, exceptions, and insights with notes, tag colleagues with @mentions, and share filtered views via URL tokens.

| | |
|---|---|
| **Status** | ✅ Implemented |
| **UI Tab** | Embedded panels in DFU Analysis, Storyboard, and AI Planner tabs |
| **Key Files** | `api/routers/collaboration.py` |

---

## Problem

Planning decisions happen in context -- a planner notices an anomaly in a DFU forecast, adjusts a safety stock target, or reviews an AI insight. Today, the reasoning behind these decisions lives in email threads, chat messages, or not at all. When another planner picks up the same item next month, they have no record of what was discussed or decided.

## Solution

An in-app annotation system lets planners attach notes directly to forecasts, DFUs, exceptions, and AI insights. Notes support @mentions to tag colleagues, who see an unread count in the sidebar. Discussion threads can be marked as resolved when the issue is addressed. Shared views let planners snapshot their current filter state and share it via a URL token, so a colleague sees exactly the same data view.

## How It Works

1. Planner views a DFU, exception, or forecast in the UI
2. Clicks "Add Note" to create an annotation, optionally tagging colleagues with `@name`
3. @mentioned users see an unread mention count in the sidebar and can navigate directly to the annotation
4. Annotations can be edited, resolved (marking the discussion as complete), or soft-deleted
5. To share a view, a planner clicks "Share" in the filter bar -- this saves the current tab + filters as a URL token
6. A colleague opens the `?shared=<token>` URL, which restores the exact filter state

## Data Model

### `fact_annotation`

| Column | Type | Description |
|---|---|---|
| `annotation_id` | `BIGSERIAL PK` | Auto-increment ID |
| `entity_type` | `TEXT` | What the note is attached to: dfu, forecast, exception, insight, scenario |
| `entity_id` | `TEXT` | Composite key or ID of the referenced entity |
| `author_id` | `INTEGER` | User ID (nullable for unauthenticated mode) |
| `author_name` | `TEXT` | Display name (default: "Planner") |
| `body` | `TEXT` | Annotation text |
| `is_resolved` | `BOOLEAN` | Whether the discussion thread is closed |
| `is_deleted` | `BOOLEAN` | Soft-delete flag (excluded from queries) |
| `created_at` | `TIMESTAMPTZ` | When the annotation was created |
| `updated_at` | `TIMESTAMPTZ` | Last edit timestamp |

### `fact_annotation_mention`

| Column | Type | Description |
|---|---|---|
| `mention_id` | `BIGSERIAL PK` | Auto-increment ID |
| `annotation_id` | `BIGINT FK` | References the parent annotation |
| `mentioned_user_id` | `INTEGER` | User who was tagged |
| `mentioned_name` | `TEXT` | Display name of tagged user |
| `is_read` | `BOOLEAN` | Whether the mention has been read |
| `created_at` | `TIMESTAMPTZ` | When the mention was created |

### `dim_shared_view`

| Column | Type | Description |
|---|---|---|
| `view_id` | `SERIAL PK` | Auto-increment ID |
| `token` | `TEXT UNIQUE` | UUID used in shareable URL |
| `created_by` | `INTEGER` | User who created the share |
| `label` | `TEXT` | Optional description |
| `tab` | `TEXT` | Active tab name |
| `filters` | `JSONB` | Filter state snapshot |
| `url_state` | `TEXT` | Full URL query string |
| `expires_at` | `TIMESTAMPTZ` | Expiration (default 30 days) |

## API

| Method | Path | Description |
|---|---|---|
| GET | `/collaboration/annotations` | List annotations by entity type + ID (paginated) |
| POST | `/collaboration/annotations` | Create annotation with optional @mentions |
| PUT | `/collaboration/annotations/{id}` | Edit annotation text |
| DELETE | `/collaboration/annotations/{id}` | Soft-delete an annotation |
| PUT | `/collaboration/annotations/{id}/resolve` | Mark thread as resolved |
| GET | `/collaboration/mentions` | List unresolved mentions for a user |
| POST | `/collaboration/shared-views` | Save current filter state as shareable URL token |
| GET | `/collaboration/shared-views/{token}` | Retrieve shared view by token |

## Configuration

No config file required. All endpoints use `get_conn()` directly (consistent with `inv_planning_*.py` pattern). In v1, authentication is optional -- annotations use the `author_name` field. When RBAC is enabled, `author_id` is populated from the JWT token.

## Dependencies

- No external dependencies
- Integrates with [RBAC](./02-rbac.md) for user identity when available

## See Also

- [RBAC](./02-rbac.md) -- provides user identity for annotations
- [Notifications](./04-notifications.md) -- @mentions can trigger notification delivery
- [FVA](./07-fva.md) -- annotations provide context for intervention tracking
