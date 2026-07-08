# Collaboration & Annotations

> Lets planners annotate resources (DFUs, forecasts, exceptions, insights) with threaded notes, tag colleagues with @mentions, and save filter/layout state as shared views.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | None - backend-only. The fetcher functions in `frontend/src/api/queries/platform.ts` (`fetchAnnotations`, `fetchMyMentions`, `fetchSharedViews`) are defined but not imported or called from any tab, component, or hook. |
| **Key Files** | `api/routers/platform/collaboration.py` |

---

## Problem

Planning decisions happen in context -- a planner notices an anomaly in a DFU forecast, adjusts a safety stock target, or reviews an AI insight. Today, the reasoning behind these decisions lives in email threads, chat messages, or not at all. When another planner picks up the same item next month, they have no record of what was discussed or decided.

## Solution

An annotation system lets callers attach threaded notes to any resource - forecasts, DFUs, exceptions, AI insights - identified by a `resource_type` + `resource_id` pair. Notes support `@mentions` (by email or user identifier); mentioning someone fires a best-effort notification through the [Notification Engine](./04-notifications.md). Threads can be edited, marked resolved, or hard-deleted. Shared views let a caller snapshot the current tab + filters + layout as a row in `fact_shared_view`, identified by a plain sequential integer `view_id` - there is no share token, expiration, or access control on the read endpoint.

## How It Works

1. A caller creates an annotation via `POST /collaboration/annotations`, attached to a `resource_type` + `resource_id`, optionally with a list of `mentions` (recipient identifiers, e.g. emails) and a `parent_id` for threaded replies.
2. If `mentions` is non-empty, the endpoint fires one `NotificationEngine.send(event_type="mention", ...)` call per mentioned recipient, best-effort - a delivery failure is logged but never blocks the annotation from being created.
3. `GET /collaboration/annotations` returns the full thread for a `resource_type` + `resource_id`, joined against `dim_user` for the author's email/display name.
4. A thread can be edited (`PUT`), marked resolved (`POST .../resolve`), or hard-deleted (`DELETE` - a real `DELETE FROM fact_annotation`, not a soft-delete flag).
5. `GET /collaboration/mentions/me` returns every annotation whose `mentions` JSONB contains the caller's email, newest first - there is no read/unread state.
6. `POST /collaboration/shared-views` stores the current `tab` + `filters` + `layout` and an `is_public` flag, returning an integer `view_id`. `GET /collaboration/shared-views/{view_id}` fetches by that ID directly, with no authentication check and no expiration.

## Data Model

### `fact_annotation`

| Column | Type | Description |
|---|---|---|
| `annotation_id` | `BIGSERIAL PK` | Auto-increment ID |
| `user_id` | `UUID` | Author (nullable for anonymous/API-key callers) |
| `resource_type` | `TEXT` | What the note is attached to (e.g. `sku`, `forecast`, `exception`, `insight`) |
| `resource_id` | `TEXT` | ID of the referenced resource |
| `parent_id` | `BIGINT FK` | Self-reference to `fact_annotation(annotation_id)`, for threaded replies |
| `body` | `TEXT` | Annotation text |
| `mentions` | `JSONB` | List of mentioned recipients (e.g. emails), stored inline |
| `is_resolved` | `BOOLEAN` | Whether the discussion thread is closed (default `FALSE`) |
| `created_at` | `TIMESTAMPTZ` | When the annotation was created |
| `updated_at` | `TIMESTAMPTZ` | Last edit timestamp |

Mentions are a JSONB column on `fact_annotation` itself - there is no separate mentions table. `GET /collaboration/mentions/me` matches by scanning `mentions::text` with `ILIKE` against the caller's email.

### `fact_shared_view`

| Column | Type | Description |
|---|---|---|
| `view_id` | `BIGSERIAL PK` | Auto-increment ID |
| `user_id` | `UUID` | Owner (nullable for anonymous/API-key callers) |
| `title` | `TEXT` | View title |
| `tab` | `TEXT` | Active tab name |
| `filters` | `JSONB` | Filter state snapshot |
| `layout` | `JSONB` | Layout snapshot |
| `is_public` | `BOOLEAN` | Whether the view is visible to all users (default `FALSE`) |
| `created_at` | `TIMESTAMPTZ` | When the view was saved |

There is no `token`, `url_state`, or `expires_at` column, and no UUID share token or `?shared=<token>` URL mechanism - a view is addressed by its plain integer `view_id`, and `GET /collaboration/shared-views/{view_id}` has no auth check.

## API

| Method | Path | Description |
|---|---|---|
| POST | `/collaboration/annotations` | Create an annotation on a resource, with optional `@mentions` (API key required) |
| GET | `/collaboration/annotations` | List the annotation thread for a `resource_type` + `resource_id` |
| PUT | `/collaboration/annotations/{annotation_id}` | Edit annotation text (API key required) |
| POST | `/collaboration/annotations/{annotation_id}/resolve` | Mark a thread as resolved (API key required) |
| DELETE | `/collaboration/annotations/{annotation_id}` | Hard-delete an annotation (API key required) |
| GET | `/collaboration/mentions/me` | List annotations that `@mention` the current user, newest first |
| POST | `/collaboration/shared-views` | Save the current tab + filters + layout as a shared view (API key required) |
| GET | `/collaboration/shared-views` | List shared views visible to the current user (public, or owned) |
| GET | `/collaboration/shared-views/{view_id}` | Get a shared view by ID (no auth check) |

## Configuration

No config file required. All endpoints use `get_conn()` directly (consistent with the `inv_planning_*.py` pattern). Authentication is optional - an anonymous or API-key-only caller gets a `NULL` `user_id`, while an authenticated caller's `user_id` comes from `get_current_user()`. There is no separate `author_name` column; author email/display name are joined from `dim_user` at read time.

## Dependencies

- No external dependencies
- Calls the [Notification Engine](./04-notifications.md) (best-effort) to deliver `@mention` alerts
- Integrates with [RBAC](./02-rbac.md) for user identity when available

## See Also

- [RBAC](./02-rbac.md) -- provides user identity for annotations
- [Notifications](./04-notifications.md) - @mentions trigger notification delivery
- [FVA](./07-fva.md) -- annotations provide context for intervention tracking
