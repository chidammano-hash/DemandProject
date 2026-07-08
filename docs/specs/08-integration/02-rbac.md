# Role-Based Access Control

> Controls who can do what in Supply Chain Command Center through JWT authentication (JSON Web Tokens) and a hierarchical role-level check (`require_role`).

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | N/A (infrastructure layer) |
| **Key Files** | `api/routers/platform/auth_router.py`, `api/routers/platform/users.py`, `common/auth.py`, `api/auth.py`, `config/platform/auth_config.yaml` |

---

## Problem

Without access control, every user can run jobs, change policies, and modify forecasts. There is no way to limit viewers to read-only, restrict job execution to planners and above, or audit who changed what. In a multi-user planning team, this creates risk of accidental or unauthorized changes.

## Solution

A JWT-based authentication layer with four hierarchical roles: `viewer` < `planner` < `manager` < `admin`. Users log in with email and password, receive a token, and the system checks the caller's role level against the minimum level required for the endpoint on every gated request. All mutations (changes to data) are logged to an audit trail. The existing API key auth (`X-API-Key` header) remains for backward compatibility - RBAC is additive.

## How It Works

1. User submits email + password to `POST /auth/login`
2. Server verifies password (bcrypt hash), issues a JWT access token (30 min) and refresh token (7 days)
3. Client includes `Authorization: Bearer <token>` on subsequent requests
4. `require_role(min_role)` (`common/auth.py`) decodes the token via the `get_current_user` dependency, looks up the caller's role level in `role_level` (`config/platform/auth_config.yaml`), and compares it against the minimum level required for the endpoint
5. If the caller's role level is below the endpoint's minimum, the request is rejected with 403; a missing or invalid token is rejected with 401
6. Refresh tokens allow seamless token renewal without re-entering credentials
7. All mutations are written to `fact_audit_log` with user ID, action, resource, and timestamp

## Data Model

### `dim_user`

| Column | Type | Description |
|---|---|---|
| `user_id` | `SERIAL PK` | Auto-increment user ID |
| `email` | `TEXT UNIQUE` | Login email address |
| `display_name` | `TEXT` | User's display name |
| `password_hash` | `TEXT` | bcrypt-hashed password |
| `role` | `user_role` (ENUM) | One of `viewer`, `planner`, `manager`, `admin` (default: `viewer`) |
| `is_active` | `BOOLEAN` | Soft-delete flag |
| `created_at` | `TIMESTAMPTZ` | Account creation time |
| `last_login_at` | `TIMESTAMPTZ` | Most recent login |

There is no `dim_role` table. Roles are a fixed Postgres enum (`user_role`, defined in `sql/062_create_users_rbac.sql`), not a data-driven lookup table, and the hierarchy between them lives in `role_level` in `config/platform/auth_config.yaml` (see "Roles and Hierarchy" below) rather than in a `permissions` column.

### `fact_audit_log`

| Column | Type | Description |
|---|---|---|
| `log_id` | `BIGSERIAL PK` | Auto-increment log ID |
| `user_id` | `INTEGER FK` | Who performed the action |
| `action` | `TEXT` | Action type: create, update, delete, login, logout |
| `resource` | `TEXT` | Table or endpoint affected |
| `resource_id` | `TEXT` | Specific entity identifier |
| `details` | `JSONB` | Additional context |
| `ip_address` | `INET` | Client IP |
| `created_at` | `TIMESTAMPTZ` | When the action occurred |

## Roles and Hierarchy

Roles are ordered, not a flat permission-string array. Each role's numeric level is defined in `role_level` (`config/platform/auth_config.yaml`), and a role implicitly has access to everything a lower-level role has:

| Role | Level | Typical User |
|---|---|---|
| `viewer` | 1 (lowest) | Executive, stakeholder - read-only access |
| `planner` | 2 | Supply chain planner - can create/edit forecasts, policies, exceptions |
| `manager` | 3 | Supply chain manager - can approve/delete, plus everything a planner can |
| `admin` | 4 (highest) | System administrator - full access, including user management |

Endpoints declare a minimum role with `require_role(min_role)` (`common/auth.py`). A caller passes only if their role's level is greater than or equal to the endpoint's minimum level; otherwise the request is rejected with 403. For example, `api/routers/intelligence/external_signals.py` gates its manual source-refresh endpoint with `Depends(require_role("manager"))`, so `manager` and `admin` users can trigger a refresh but `planner` and `viewer` users cannot.

## API

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/auth/login` | None | Authenticate, returns access + refresh tokens |
| POST | `/auth/refresh` | None | Exchange refresh token for new access token |
| POST | `/auth/logout` | Any | Invalidate refresh token |
| GET | `/users` | admin | List all users with roles |
| POST | `/users` | admin | Create a new user |
| GET | `/users/me` | Any | Current user profile from token |
| PUT | `/users/{id}` | admin | Update user fields |
| PUT | `/users/{id}/role` | admin | Change a user's role |
| DELETE | `/users/{id}` | admin | Deactivate user (soft delete) |

## Configuration

`config/platform/auth_config.yaml`:

```yaml
jwt:
  algorithm: HS256
  access_token_expire_minutes: 30
  refresh_token_expire_days: 7

role_level:
  viewer: 1                              # Level 1 - lowest privilege
  planner: 2                             # Level 2 - can create/edit
  manager: 3                             # Level 3 - can approve/delete
  admin: 4                               # Level 4 - full system access
```

JWT secret is loaded from the `JWT_SECRET` environment variable (`common/auth.py`). If `JWT_SECRET` is unset, `common/auth.py` falls back to a hardcoded insecure default and logs a warning - it is not auto-generated - so `JWT_SECRET` must be set explicitly in any non-dev environment.

## Dependencies

- `PyJWT>=2.8` for token creation and verification
- `bcrypt>=4.0` for password hashing
- Existing `api/auth.py` (`require_api_key`) remains for backward compatibility

## See Also

- [Integration Architecture](./01-integration-architecture.md) -- overall integration overview
- [API Governance](./09-api-governance.md) -- rate limiting per client/role
- [Collaboration](./05-collaboration.md) -- annotations use author identity from RBAC
